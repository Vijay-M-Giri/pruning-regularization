import functools
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops
import utils


class PruneBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, input_units, input_rank, **kwargs):
        super().__init__(**kwargs)
        ndims = input_rank
        self.axis = [ndims + self.axis]
        self.fused = ndims == 4
        if self.fused:
            if self.axis == [1] and ndims == 4:
                self._data_format = 'NCHW'
            elif self.axis == [3] and ndims == 4:
                self._data_format = 'NHWC'
        param_shape = (input_units, )
        self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                var_shape=(None, ),
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
        self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                var_shape=(None, ),
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        try:
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name='moving_mean',
                shape=param_shape,
                var_shape=(None, ),
                initializer=self.moving_mean_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN,
                experimental_autocast=False)

            self.moving_variance = self.add_weight(
                name='moving_variance',
                shape=param_shape,
                var_shape=(None, ),
                initializer=self.moving_variance_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN,
                experimental_autocast=False)
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def add_weight(self,
                   name=None,
                   shape=None,
                   var_shape=None,
                   dtype=tf.float32,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None,
                   use_resource=True,
                   synchronization=tf_variables.VariableSynchronization.AUTO,
                   aggregation=tf_variables.VariableAggregation.NONE,
                   **kwargs):
        initializer = initializers.get(initializer)
        regularizer = regularizers.get(regularizer)
        constraint = constraints.get(constraint)

        if initializer is None:
            if dtype.is_floating:
                initializer = initializers.get('glorot_uniform')
        init_val = functools.partial(initializer, shape, dtype=dtype)
        variable_dtype = dtype.base_dtype
        variable_shape = tensor_shape.TensorShape(var_shape)
        variable = tf.Variable(
                initial_value=init_val,
                name=name,
                trainable=trainable,
                caching_device=None,
                dtype=variable_dtype,
                validate_shape=False,
                constraint=constraint,
                synchronization=synchronization,
                aggregation=aggregation,
                shape=variable_shape)
        backend.track_variable(variable)
        if regularizer is not None:
            name_in_scope = variable.name[:variable.name.find(':')]
            self._handle_weight_regularization(name_in_scope,
                                               variable,
                                               regularizer)
        if trainable:
            self._trainable_weights.append(variable)
        else:
            self._non_trainable_weights.append(variable)
        return variable

    def build(self, input_shape):
        None

    def set_mask(self, m):
        self.gamma.assign(tf.boolean_mask(self.gamma, m))
        self.beta.assign(tf.boolean_mask(self.beta, m))
        self.moving_mean.assign(tf.boolean_mask(self.moving_mean, m))
        self.moving_variance.assign(tf.boolean_mask(self.moving_variance, m))


class PruneConv2D(tf.keras.layers.Layer):
    def __init__(self, initial_filters, input_units, kernel_size, prune=True,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 kernel_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = 2
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(
            strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, self.rank, 'dilation_rate')
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self.units = initial_filters
        self.activation = activations.get(activation)
        self.ks0 = tf.Variable(input_units, trainable=False, dtype=tf.float32)
        self.ks1 = tf.Variable(self.units, trainable=False, dtype=tf.float32)
        self.kernel = self.add_weight(
            'kernel',
            list(self.kernel_size) + [input_units, self.units],
            list(self.kernel_size) + [None, None if prune else self.units],
            regularizer=kernel_regularizer,
            trainable=True
        )
        self.bias = self.add_weight('bias', [self.units, ],
                                    [None if prune else self.units, ],
                                    trainable=True)

        tf_padding = self.padding.upper()
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)
        tf_op_name = self.__class__.__name__
        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def build(self, input_shape):
        None

    def add_weight(self,
                   name=None,
                   shape=None,
                   var_shape=None,
                   dtype=tf.float32,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None,
                   use_resource=True,
                   synchronization=tf_variables.VariableSynchronization.AUTO,
                   aggregation=tf_variables.VariableAggregation.NONE,
                   **kwargs):
        initializer = initializers.get(initializer)
        regularizer = regularizers.get(regularizer)
        constraint = constraints.get(constraint)

        if initializer is None:
            if dtype.is_floating:
                initializer = initializers.get('glorot_uniform')
        init_val = functools.partial(initializer, shape, dtype=dtype)
        # init_val = initializer(shape=shape)
        variable_dtype = dtype.base_dtype
        variable_shape = tensor_shape.TensorShape(var_shape)
        variable = tf.Variable(
                initial_value=init_val,
                name=name,
                trainable=trainable,
                caching_device=None,
                dtype=variable_dtype,
                validate_shape=False,
                constraint=constraint,
                synchronization=synchronization,
                aggregation=aggregation,
                shape=variable_shape)
        backend.track_variable(variable)
        if regularizer is not None:
            name_in_scope = variable.name[:variable.name.find(':')]
            self._handle_weight_regularization(name_in_scope,
                                               variable, [self.ks0, self.ks1],
                                               regularizer)
        if trainable:
            self._trainable_weights.append(variable)
        else:
            self._non_trainable_weights.append(variable)
        return variable

    def _handle_weight_regularization(self, name, variable, shape,
                                      regularizer):
        """Create lambdas which compute regularization losses."""

        def _loss_for_variable(v, shape):
            """Creates a regularization loss `Tensor` for variable `v`."""
            with backend.name_scope(name + '/Regularizer'):
                regularization = regularizer(v, shape)
            return regularization

        self.add_loss(functools.partial(_loss_for_variable, variable, shape))

    def call(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel)
        outputs = nn_ops.bias_add(
            outputs, self.bias, data_format=self._tf_data_format)
        return self.activation(outputs)

    def set_kernel(self, k):
        self.kernel.assign(k)
        self.ks0.assign(utils.get_dynamic_shape(k[0][0], 0))
        self.ks1.assign(utils.get_dynamic_shape(k[0][0], 1))


class PruneDense(tf.keras.layers.Layer):
    def __init__(self, inital_units, input_units, prune=True, activation=None,
                 kernel_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = inital_units
        self.activation = activations.get(activation)
        self.ks0 = tf.Variable(input_units, trainable=False, dtype=tf.float32)
        self.ks1 = tf.Variable(self.units, trainable=False, dtype=tf.float32)
        self.kernel = self.add_weight('kernel', [input_units, self.units],
                                      [None, None if prune else self.units],
                                      regularizer=kernel_regularizer,
                                      trainable=True)
        self.bias = self.add_weight('bias', [self.units, ],
                                    [None if prune else self.units, ],
                                    trainable=True)
        self.built = True

    def build(self, input_shape):
        None

    def add_weight(self,
                   name=None,
                   shape=None,
                   var_shape=None,
                   dtype=tf.float32,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None,
                   use_resource=True,
                   synchronization=tf_variables.VariableSynchronization.AUTO,
                   aggregation=tf_variables.VariableAggregation.NONE,
                   **kwargs):
        initializer = initializers.get(initializer)
        regularizer = regularizers.get(regularizer)
        constraint = constraints.get(constraint)

        if initializer is None:
            if dtype.is_floating:
                initializer = initializers.get('glorot_uniform')
        init_val = functools.partial(initializer, shape, dtype=dtype)
        # init_val = initializer(shape=shape)
        variable_dtype = dtype.base_dtype
        variable_shape = tensor_shape.TensorShape(var_shape)
        variable = tf.Variable(
                initial_value=init_val,
                name=name,
                trainable=trainable,
                caching_device=None,
                dtype=variable_dtype,
                validate_shape=False,
                constraint=constraint,
                synchronization=synchronization,
                aggregation=aggregation,
                shape=variable_shape)
        backend.track_variable(variable)
        if regularizer is not None:
            name_in_scope = variable.name[:variable.name.find(':')]
            self._handle_weight_regularization(name_in_scope,
                                               variable, [self.ks0, self.ks1],
                                               regularizer)
        if trainable:
            self._trainable_weights.append(variable)
        else:
            self._non_trainable_weights.append(variable)
        return variable

    def _handle_weight_regularization(self, name, variable, shape,
                                      regularizer):
        """Create lambdas which compute regularization losses."""

        def _loss_for_variable(v, shape):
            """Creates a regularization loss `Tensor` for variable `v`."""
            with backend.name_scope(name + '/Regularizer'):
                regularization = regularizer(v, shape)
            return regularization

        self.add_loss(functools.partial(_loss_for_variable, variable, shape))

    def call(self, inputs):
        outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        outputs = nn_ops.bias_add(outputs, self.bias)
        # return self.activation(tf.matmul(inputs, self.kernel) + self.bias)
        return self.activation(outputs)

    def set_kernel(self, k):
        self.kernel.assign(k)
        self.ks0.assign(utils.get_dynamic_shape(k, 0))
        self.ks1.assign(utils.get_dynamic_shape(k, 1))


class InputIdentityLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.kernel = tf.Variable(tf.eye(units), shape=[None, None],
                                  trainable=False, dtype=tf.float32)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


class InputMaskLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.mask = tf.Variable(tf.ones(units, tf.bool),
                                dtype=tf.bool, trainable=False)
        self.built = True

    def call(self, inputs):
        return tf.boolean_mask(inputs, self.mask, axis=1)

    def update_mask(self, m):
        self.mask.assign(tf.scatter_nd(tf.where(self.mask == True),
                                       m, [self.units, ]))


class BiDirectionalMaskLayer(tf.keras.layers.Layer):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.m1 = tf.Variable(tf.ones(in_units, tf.bool),
                              dtype=tf.bool, trainable=False)
        self.m12 = tf.Variable(tf.ones(out_units, tf.bool),
                               dtype=tf.bool, trainable=False, shape=(None, ))
        self.m2 = tf.Variable(tf.ones(out_units, tf.bool),
                              dtype=tf.bool, trainable=False)
        self.built = True

    def call(self, inputs):
        return tf.boolean_mask(inputs, self.m12, axis=1)

    def update_and_get_mask(self, in_m, out_m):
        self.m1.assign(tf.scatter_nd(tf.where(self.m1 == True),
                                     in_m, [self.in_units, ]))
        flat_m1 = tf.reshape(
            tf.broadcast_to(
                tf.reshape(self.m1, (-1, 1, 1)), shape=(self.in_units, 4, 4)),
            shape=(self.out_units, ))
        out_m = tf.logical_and(
            tf.boolean_mask(tf.logical_and(flat_m1, self.m2), self.m2),
            out_m)
        self.m2.assign(tf.scatter_nd(tf.where(self.m2 == True),
                                     out_m, [self.out_units, ]))
        self.m12.assign(tf.boolean_mask(self.m2, flat_m1))
        return out_m
