import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from zero_reg.zero_reg import ZeroRegularizer
from zero_opt.zero_opt import ZeroOpt

N = 100
w0 = 0.125
b0 = 5.
x_range = [-20, 60]


def plot_linear(model, x, y, x_tst):
    y_hat = model(x_tst)
    plt.plot(x, y, 'b.', label='observed')
    plt.plot(x_tst, y_hat, 'r', label='test')
    plt.show()


def load_dataset(n=150, n_test=150):
    def s(x):
        e = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + e ** 2.)
    rng = np.random.default_rng()
    x = (x_range[1] - x_range[0]) * rng.random(n) + x_range[0]
    eps = rng.random(n) * s(x)
    y = w0 * x * (1. + np.sin(x)) + b0 + eps
    x = x[..., np.newaxis]
    x_tst = np.linspace(*x_range, num=n_test).astype(np.float32)
    x_tst = x_tst[..., np.newaxis]
    return x, y, x_tst


def load_sin_dataset(n=150, n_test=150):
    low = 0.0
    high = 0.5
    rng = np.random.default_rng()
    x = rng.random(n) * (high - low) + low
    eps = rng.normal(loc=0.0, scale=0.02, size=n)
    y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + \
        0.3 * np.sin(4 * np.pi * (x + eps)) + eps
    x_test = np.linspace(low - 0.4, high + 0.8, n_test)
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x, y, x_test


def build_model(data_shape):
    activation = 'tanh'
    regs = [None, tf.keras.regularizers.L1(0.01), ZeroRegularizer(0.01)]
    reg_kernel = regs[0]
    reg_bias = regs[0]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=data_shape),
        tf.keras.layers.Dense(100, activation=activation, kernel_regularizer=reg_kernel, bias_regularizer=reg_bias),
        tf.keras.layers.Dense(100, activation=activation, kernel_regularizer=reg_kernel, bias_regularizer=reg_bias),
        tf.keras.layers.Dense(1)
    ])
    return model


def model_callback(model):
    model.summary()
    c_zero = sum([np.count_nonzero(np.abs(w.numpy()) <= 0.0) for w in model.weights])
    print("Number of zero weights", c_zero)
    weights = np.array([])
    for w in model.weights:
        weights = np.concatenate((weights, w.numpy().flatten()))
    plt.hist(weights)
    plt.show()


x, y, x_test = load_sin_dataset(N)
model = build_model((1,))
model_callback(model)
model.compile(
    # optimizer=tf.optimizers.Adam(learning_rate=0.01),
    optimizer=ZeroOpt(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError()
)
model.fit(x, y, epochs=200)
plot_linear(model, x, y, x_test)
model_callback(model)
