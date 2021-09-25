import tensorflow as tf
from layers import PruneConv2D, PruneDense, PruneBatchNormalization,\
    InputMaskLayer, InputIdentityLayer, BiDirectionalMaskLayer
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from regularizers import PruneRegularizer
import utils


class PruneVggLike(tf.keras.models.Model):
    def __init__(self, data_shape, activation=None):
        super().__init__()
        self.data_shape = data_shape
        self.activation = activation
        self.custom_layers = (
            [tf.keras.layers.InputLayer(input_shape=data_shape)] +
            self._get_conv_block(64, data_shape[-1], 3, 2,
                                 PruneRegularizer()) +
            self._get_conv_block(128, 64, 3, 2, PruneRegularizer()) +
            self._get_conv_block(256, 128, 3, 3, PruneRegularizer()) +
            self._get_conv_block(512, 256, 3, 3, PruneRegularizer()) +
            self._get_conv_block(512, 512, 3, 3, PruneRegularizer()) +
            [tf.keras.layers.Flatten()] +
            [PruneDense(512, 512,
                        kernel_regularizer=PruneRegularizer())] +
            [PruneBatchNormalization(512, 2)] +
            [tf.keras.layers.Activation(self.activation)] +
            [PruneDense(10, 512, prune=False,
                        kernel_regularizer=PruneRegularizer())]
        )
        self.built = True

    def _get_conv_block(self, filters, prev_filters, kernel_size, count,
                        kernel_regularizer):
        res = []
        for i in range(count):
            res += [
                PruneConv2D(filters, prev_filters, kernel_size, padding='same',
                            kernel_regularizer=kernel_regularizer),
                PruneBatchNormalization(filters, len(self.data_shape) + 1),
                tf.keras.layers.Activation(self.activation)
            ]
            prev_filters = filters
        res.append(tf.keras.layers.MaxPool2D(2))
        return res

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for l in self.custom_layers:
            x = l(x)
        return x

    def prune(self):
        prev_l, prev_m, prev_k, prev_a0, prev_a1, prev_i = [None] * 6
        for i, l in enumerate(self.custom_layers):
            if 'conv' in l.name or 'dense' in l.name:
                a0, a1, ker_f = (2, 3, 9) if 'conv' in l.name else (0, 1, 1)
                m = utils.get_nodes_to_keep(l.kernel, l.ks1, a1, ker_f)
                if prev_l is not None:
                    m = tf.logical_and(m, prev_m)
                    prev_k = tf.boolean_mask(prev_k, m, axis=prev_a1)
                    prev_l.set_kernel(prev_k)
                    prev_l.bias.assign(tf.boolean_mask(prev_l.bias, m))
                    self.custom_layers[prev_i + 1].set_mask(m)
                prev_k = tf.boolean_mask(l.kernel, m, axis=a0)
                prev_m = utils.get_nodes_to_keep(l.kernel, l.ks0, a0, ker_f)
                prev_l, prev_a0, prev_a1, prev_i = l, a0, a1, i
        self.custom_layers[-1].set_kernel(prev_k)

    def print_nodes_count(self):
        p = []
        for l in self.custom_layers:
            if 'conv' in l.name or 'dense' in l.name:
                p.append(l.ks1)
        tf.print(p)


class PruneLeNet5(tf.keras.models.Model):
    def __init__(self, data_shape, activation=None):
        super().__init__()
        self.activation = activation
        self.l0 = tf.keras.layers.InputLayer(input_shape=data_shape)
        self.l1 = PruneConv2D(20, data_shape[-1], 5, True, activation=self.activation,
                              kernel_regularizer=PruneRegularizer())
        self.l2 = tf.keras.layers.MaxPool2D(2)
        self.l3 = PruneConv2D(50, 20, 5, True, activation=self.activation,
                              kernel_regularizer=PruneRegularizer())
        self.l4 = tf.keras.layers.MaxPool2D(2)
        self.l5 = tf.keras.layers.Flatten()
        self.mask_l = BiDirectionalMaskLayer(50, 800)
        self.l6 = PruneDense(500, 800, True, activation=self.activation,
                             kernel_regularizer=PruneRegularizer())
        self.l7 = PruneDense(10, 500, False,
                             kernel_regularizer=PruneRegularizer())

        self.l1_m = LayerNodesMetric("l1")
        self.l2_m = LayerNodesMetric("l2")
        self.l3_m = LayerNodesMetric("l3")
        self.l4_m = LayerNodesMetric("l4")
        self.out_m = LayerNodesMetric("out")

    def call(self, inputs, training=None, mask=None):
        x = self.l0(inputs)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        # x = self.mask_l(x)
        x = self.l6(x)
        x = self.l7(x)
        return x

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        result = {m.name: m.result() for m in self.metrics}
        self.l1_m.update_state(self.l1.ks1)
        self.l2_m.update_state(self.l3.ks1)
        self.l3_m.update_state(self.l6.ks0)
        self.l4_m.update_state(self.l6.ks1)
        self.out_m.update_state(self.l7.ks1)
        result['l1'] = self.l1_m.result()
        result['l2'] = self.l2_m.result()
        result['l3'] = self.l3_m.result()
        result['l4'] = self.l4_m.result()
        result['out'] = self.out_m.result()
        return result

    def prune(self):
        ker_f = 25  # Hardcoded for now
        l3_k = self.l3.kernel
        l6_k = self.l6.kernel
        l7_k = self.l7.kernel
        m30 = utils.get_nodes_to_keep(l3_k, self.l3.ks1, 3, ker_f)
        m31 = utils.get_nodes_to_keep(l3_k, self.l3.ks0, 2, ker_f)
        ker_f = 1
        # m60 = utils.get_nodes_to_keep(l6_k, self.l6.ks1, 1, ker_f)
        # m60 = self.mask_l.update_and_get_mask(m31, m60)
        m60 = tf.reshape(
            tf.broadcast_to(tf.reshape(m31, (-1, 1, 1)), (m31.shape[0] ,4, 4)),
            (-1))
        m61 = tf.logical_and(
            utils.get_nodes_to_keep(l6_k, self.l6.ks0, 0, ker_f),
            utils.get_nodes_to_keep(l7_k, self.l7.ks1, 1, ker_f)
        )

        k1 = tf.boolean_mask(self.l1.kernel, m30, axis=3)
        k3 = tf.boolean_mask(self.l3.kernel, m30, axis=2)
        k3 = tf.boolean_mask(k3, m31, axis=3)

        k6 = tf.boolean_mask(l6_k, m60, axis=0)
        k6 = tf.boolean_mask(k6, m61, axis=1)
        k7 = tf.boolean_mask(l7_k, m61, axis=0)

        b1 = tf.boolean_mask(self.l1.bias, m30)
        b3 = tf.boolean_mask(self.l3.bias, m31)
        b6 = tf.boolean_mask(self.l6.bias, m61)

        self.l1.set_kernel(k1)
        self.l3.set_kernel(k3)
        self.l6.set_kernel(k6)
        self.l7.set_kernel(k7)

        self.l1.bias.assign(b1)
        self.l3.bias.assign(b3)
        self.l6.bias.assign(b6)


class PruneLeNet300(tf.keras.models.Model):
    def __init__(self, data_shape, activation=None):
        super().__init__()
        self.activation = activation
        self.m1 = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=data_shape),
            tf.keras.layers.Flatten()
        ])
        # self.l0 = InputIdentityLayer(784)
        self.l0 = InputMaskLayer(784)
        self.l1 = PruneDense(300, 784, True, self.activation,
                             kernel_regularizer=PruneRegularizer())
        self.l2 = PruneDense(100, 300, True, self.activation,
                             kernel_regularizer=PruneRegularizer())
        self.l3 = PruneDense(10, 100, False,
                             kernel_regularizer=PruneRegularizer())

        self.in_m = LayerNodesMetric("in")
        self.l1_m = LayerNodesMetric("l1")
        self.l2_m = LayerNodesMetric("l2")
        self.out_m = LayerNodesMetric("out")

    def call(self, inputs, training=None, mask=None):
        x = self.m1(inputs)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # self.prune()

        result = {m.name: m.result() for m in self.metrics}
        self.in_m.update_state(self.l1.ks0)
        self.l1_m.update_state(self.l1.ks1)
        self.l2_m.update_state(self.l2.ks1)
        self.out_m.update_state(self.l3.ks1)
        result['in'] = self.in_m.result()
        result['l1'] = self.l1_m.result()
        result['l2'] = self.l2_m.result()
        result['out'] = self.out_m.result()
        return result

    def prune(self):
        m10 = utils.get_nodes_to_keep(self.l1.kernel, self.l1.ks1, 1, 1)
        m11 = tf.logical_and(
            utils.get_nodes_to_keep(self.l1.kernel, self.l1.ks0, 0, 1),
            utils.get_nodes_to_keep(self.l2.kernel, self.l2.ks1, 1, 1))
        m2 = tf.logical_and(
            utils.get_nodes_to_keep(self.l2.kernel, self.l2.ks0, 0, 1),
            utils.get_nodes_to_keep(self.l3.kernel, self.l3.ks1, 1, 1))

        k1 = tf.boolean_mask(self.l1.kernel, m10, axis=0)
        k1 = tf.boolean_mask(k1, m11, axis=1)
        k2 = tf.boolean_mask(self.l2.kernel, m11, axis=0)
        k2 = tf.boolean_mask(k2, m2, axis=1)
        k3 = tf.boolean_mask(self.l3.kernel, m2, axis=0)

        b1 = tf.boolean_mask(self.l1.bias, m11)
        b2 = tf.boolean_mask(self.l2.bias, m2)

        self.l0.update_mask(m10)
        # self.l0.kernel.assign(tf.boolean_mask(self.l0.kernel, m10, axis=1))

        self.l1.set_kernel(k1)
        self.l2.set_kernel(k2)
        self.l3.set_kernel(k3)

        self.l1.bias.assign(b1)
        self.l2.bias.assign(b2)


class LayerNodesMetric(tf.keras.metrics.Metric):
    def __init__(self, name='layer_nodes', **kwargs):
        super().__init__(name=name, **kwargs)
        self.nodes = self.add_weight(name='nodes', initializer='zeros')

    def update_state(self, nodes):
        self.nodes.assign(nodes)

    def result(self):
        return self.nodes

