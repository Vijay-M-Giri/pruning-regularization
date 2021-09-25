import tensorflow as tf
from models import PruneVggLike
from tensorflow.keras.datasets import cifar10 as input_data
import datetime


def pre_process_images(images):
    images = images / 255.
    # return np.where(images > .3, 1.0, 0.0).astype('float32')
    return images


def get_conv_block(filters, kernel_size, count):
    res = []
    for i in range(count):
        res += [
            tf.keras.layers.Conv2D(
                filters, kernel_size, 1, padding='same',
                kernel_regularizer=tf.keras.regularizers.L1L2(1e-6, 5*1e-5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ]
    res.append(tf.keras.layers.MaxPool2D(2))
    return res


def build_vgg_like(data_shape):
    activation = 'relu'
    model = tf.keras.Sequential(
        [tf.keras.layers.InputLayer(input_shape=data_shape)] +
        get_conv_block(64, 3, 2) +
        get_conv_block(128, 3, 2) +
        get_conv_block(256, 3, 3) +
        get_conv_block(512, 3, 3) +
        get_conv_block(512, 3, 3) +
        [tf.keras.layers.Flatten()] +
        [tf.keras.layers.Dense(
            512,
            kernel_regularizer=tf.keras.regularizers.L1L2(1e-6, 5*1e-5))] +
        [tf.keras.layers.BatchNormalization()] +
        [tf.keras.layers.Activation(activation)] +
        [tf.keras.layers.Dense(
            10, activation=activation,
            kernel_regularizer=tf.keras.regularizers.L1L2(1e-6, 5*1e-5))]
    )
    return model


def prune_nodes(model):
    def func(batch, logs):
        if batch != 20:
            model.prune()
            model.print_nodes_count()
            None
        None
    return func


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = input_data.load_data()
    x_train = pre_process_images(x_train)
    x_test = pre_process_images(x_test)
    data_shape = (32, 32, 3)

    model = PruneVggLike(data_shape)
    # model = build_vgg_like(data_shape)
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
