import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist as input_data
from zero_reg.zero_reg import ZeroRegularizer
from zero_opt.zero_opt import ZeroOpt

tfd = tfp.distributions


def pre_process_images(images):
    images = images / 255.
    # return np.where(images > .3, 1.0, 0.0).astype('float32')
    return images


def build_model(data_shape):
    activation = 'relu'
    regs = [None, tf.keras.regularizers.L1(8 * 1e-7), ZeroRegularizer(8 * 1e-7)]
    reg_kernel = regs[2]
    reg_bias = regs[0]
    cons_kernel = tf.keras.constraints.max_norm(max_value=3)
    cons_bias = tf.keras.constraints.max_norm(max_value=3)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=data_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=activation,
                              kernel_regularizer=reg_kernel,
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=reg_bias),
        tf.keras.layers.Dense(1024, activation=activation,
                              kernel_regularizer=reg_kernel,
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=reg_bias),
        tf.keras.layers.Dense(1024, activation=activation,
                              kernel_regularizer=reg_kernel,
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=reg_bias),
        tf.keras.layers.Dense(10, activation=activation,
                              kernel_regularizer=reg_kernel,
                              kernel_constraint=cons_kernel,
                              bias_constraint=cons_bias,
                              bias_regularizer=reg_bias)
    ])
    return model


def model_callback(model):
    model.summary()
    c_zero = sum([np.count_nonzero(np.abs(w.numpy()) <= 0.0) for w in model.weights])
    print("Number of zero weights", c_zero)
    weights = np.array([])
    for w in model.weights:
        weights = np.concatenate((weights, w.numpy()[w.numpy() != 0.0].flatten()))
    print("abs less than 1e-2", np.count_nonzero(np.abs(weights) < 1e-2))
    plt.hist(weights, bins=np.linspace(-1, 1, 40))
    plt.show()


def zero_weights(model):
    def func(batch, logs):
        c_zero = sum([np.count_nonzero(np.abs(w.numpy()) <= 0.0) for w in model.weights])
        c_near_zero = sum([np.count_nonzero(np.abs(w.numpy()) <= 1e-2) for w in model.weights])
        print("Number of zero weights", c_zero)
        print("abs less than 1e-2", c_near_zero)
    return func

(x_train, y_train), (x_test, y_test) = input_data.load_data()
x_train = pre_process_images(x_train)
x_test = pre_process_images(x_test)
data_shape = (28, 28)

model = build_model(data_shape)
model_callback(model)
weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=zero_weights(model))
model.compile(
    optimizer=tf.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(x_train, y_train, callbacks=weight_callback, epochs=1)
model.compile(
    optimizer=ZeroOpt(learning_rate=0.1, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(x_train, y_train, callbacks=weight_callback, epochs=18)
model.compile(
    optimizer=ZeroOpt(learning_rate=0.1, momentum=0.9, prune=False),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(x_train, y_train, callbacks=weight_callback, epochs=1)
train_error = 100. * (1. - history.history['accuracy'][-1])

loss, acc = model.evaluate(x_test, y_test)
print("Train error rate", train_error)
print("Test error rate", str((1. - acc) * 100))
model_callback(model)
