import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist as input_data
from zero_reg.zero_reg import ZeroRegularizer
from zero_opt.zero_opt_adam import ZeroOptAdam
from zero_opt.zero_opt_sgd import ZeroOptSGD

tfd = tfp.distributions


def pre_process_images(images):
    images = images / 255.
    # return np.where(images > .3, 1.0, 0.0).astype('float32')
    return images[..., np.newaxis]


def build_model_300_100(data_shape):
    activation = 'relu'
    cons_kernel = tf.keras.constraints.max_norm(max_value=3)
    cons_bias = tf.keras.constraints.max_norm(max_value=3)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=data_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(300, activation=activation,
                              kernel_regularizer=ZeroRegularizer(),
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=None),
        tf.keras.layers.Dense(100, activation=activation,
                              kernel_regularizer=ZeroRegularizer(),
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=None),
        tf.keras.layers.Dense(10, activation=activation,
                              kernel_regularizer=ZeroRegularizer(),
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=None)
    ])
    return model


def build_model_lenet_5(data_shape):
    activation = 'relu'
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=data_shape),
        tf.keras.layers.Conv2D(20, 5, 1, activation=activation,
                               kernel_regularizer=ZeroRegularizer()),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(50, 5, 1, activation=activation,
                               kernel_regularizer=ZeroRegularizer()),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation=activation,
                              kernel_regularizer=ZeroRegularizer()),
        tf.keras.layers.Dense(10, activation=activation,
                              kernel_regularizer=ZeroRegularizer()),
    ])
    return model


def zero_weights(model):
    def func(batch, logs):
        c_near_zero = sum([np.count_nonzero(np.abs(w.numpy()) <= 1e-2)
                           for w in model.weights])
        print("abs less than 1e-2", c_near_zero)
        print_prune(model)
    return func


def print_prune(model):
    c_zero = sum([np.count_nonzero(np.abs(w.numpy()) == 0.0)
                  for w in model.weights])
    print("Number of zero weights", c_zero)
    layers = list(filter(lambda x: "conv" in x.name in x.name, model.layers))
    c_total_nodes = print_prune_layers(layers)
    layers = list(filter(lambda x: "dense" in x.name in x.name, model.layers))
    c_total_nodes += print_prune_layers(layers)
    print()
    print("Total nodes pruned", c_total_nodes)


def print_prune_layers(layers):
    c_total_nodes = 0
    k = len(layers)
    for i in range(k):
        l = layers[i]
        kernel = l.weights[0]
        kernel = tf.reduce_sum(kernel,
                               axis=list(range(len(kernel.shape) - 2)))
        if i == 0:
            c_nodes = 0
            in_layer, _ = kernel.shape
            for u in range(in_layer):
                c_out = np.count_nonzero(kernel[u, :] != 0)
                if c_out <= 0:
                    c_nodes += 1
            c_total_nodes += c_nodes
            print(in_layer - c_nodes, end='-')

        c_nodes = 0
        units = kernel.shape[1]
        if i + 1 < k:
            l_next = layers[i + 1]
            kernel_next = l_next.weights[0]
            kernel_next = tf.reduce_sum(
                kernel_next, axis=list(range(len(kernel_next.shape) - 2)))
        for u in range(units):
            is_prune = np.count_nonzero(kernel[:, u] != 0) == 0
            if i + 1 < k:
                is_prune = is_prune or np.count_nonzero(
                    kernel_next[u, :] != 0) == 0
            c_nodes += is_prune
        print(units - c_nodes, end='-')
        c_total_nodes += c_nodes
    return c_total_nodes


def model_callback(model):
    model.summary()
    print_prune(model)
    weights = np.array([])
    for w in model.weights:
        weights = np.concatenate(
            (weights, w.numpy()[w.numpy() != 0.0].flatten()))
    print("abs less than 1e-2", np.count_nonzero(np.abs(weights) < 1e-2))
    plt.hist(weights, bins=np.linspace(-1, 1, 40))
    plt.show()


(x_train, y_train), (x_test, y_test) = input_data.load_data()
x_train = pre_process_images(x_train)
x_test = pre_process_images(x_test)
data_shape = (28, 28, 1)

# model = build_model_300_100(data_shape)
model = build_model_lenet_5(data_shape)
model_callback(model)
weight_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=zero_weights(model))
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', min_delta=1e-4, patience=2
)
model.compile(
    optimizer=ZeroOptSGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, epochs=20,
                    callbacks=[weight_callback, early_stop_callback])
train_error = 100. * (1. - history.history['accuracy'][-1])

loss, acc = model.evaluate(x_test, y_test)
print("Train error rate", train_error)
print("Test error rate", str((1. - acc) * 100))
model_callback(model)
