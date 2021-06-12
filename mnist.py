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
    return images


def build_model_300_100(data_shape):
    activation = 'relu'
    cons_kernel = tf.keras.constraints.max_norm(max_value=3)
    cons_bias = tf.keras.constraints.max_norm(max_value=3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=data_shape),
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
    L = len(model.layers)
    c_total_nodes = 0
    for i in range(L):
        if "dense" in model.layers[i].name:
            l = model.layers[i]
            if i == 1:
                c_nodes = 0
                kernel = l.weights[0]
                in_layer, _ = kernel.shape
                for u in range(in_layer):
                    c_out = np.count_nonzero(kernel[u, :] != 0)
                    if c_out <= 0:
                        c_nodes += 1
                c_total_nodes += c_nodes
                print(in_layer - c_nodes, end='-')

            c_nodes = 0
            if i < L - 1:
                l_next = model.layers[i + 1]
            for u in range(l.units):
                is_prune = np.count_nonzero(l.weights[0][:, u] != 0) == 0
                if i < L - 1:
                    c_out = np.count_nonzero(l_next.weights[0][u, :] != 0)
                    is_prune = is_prune or np.count_nonzero(
                        l_next.weights[0][u, :] != 0) == 0
                c_nodes += is_prune
            print(l.units - c_nodes, end='-')
            c_total_nodes += c_nodes
    print()
    print("Total nodes pruned", c_total_nodes)


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
data_shape = (28, 28)

model = build_model_300_100(data_shape)
# model_callback(model)
weight_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=zero_weights(model))

model.compile(
    optimizer=ZeroOptSGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, callbacks=weight_callback, epochs=20)
train_error = 100. * (1. - history.history['accuracy'][-1])

loss, acc = model.evaluate(x_test, y_test)
print("Train error rate", train_error)
print("Test error rate", str((1. - acc) * 100))
model_callback(model)
