import tensorflow as tf
from models import PruneLeNet5, PruneLeNet300
from tensorflow.keras.datasets import mnist as input_data
from mnist import pre_process_images
import datetime


def build_lenet_300(data_shape):
    activation = 'relu'
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=data_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(300, activation=activation),
        tf.keras.layers.Dense(100, activation=activation),
        tf.keras.layers.Dense(10)
    ])
    return model


def build_lenet_5(data_shape):
    activation = 'relu'
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=data_shape),
        tf.keras.layers.Conv2D(20, 5, 1, activation=activation),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(50, 5, 1, activation=activation),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation=activation),
        tf.keras.layers.Dense(10, activation=activation)
    ])
    return model


def prune_nodes(model):
    def func(batch, logs):
        if batch != 20:
            model.prune()
            None
        None
    return func


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = input_data.load_data()
    x_train = pre_process_images(x_train)
    x_test = pre_process_images(x_test)
    data_shape = (28, 28, 1)

    # model = PruneLeNet300(data_shape, 'relu')
    model = PruneLeNet5(data_shape, 'relu')
    # model = build_lenet_300(data_shape)
    # model = build_lenet_5(data_shape)
    prune_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=prune_nodes(model))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        # run_eagerly=True
    )
    t0 = datetime.datetime.now()
    history = model.fit(x_train, y_train, epochs=20, callbacks=prune_callback)
    t1 = datetime.datetime.now()
    train_error = 100. * (1. - history.history['accuracy'][-1])

    metrics = model.evaluate(x_test, y_test)
    print("Train error rate", train_error)
    print("Test error rate", str((1. - metrics[1]) * 100))
    print("Time taken for training", str(t1 - t0))
