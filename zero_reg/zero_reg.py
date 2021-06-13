import tensorflow as tf


class ZeroRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, z=1e-4, wf=50, nf=5):
        self.z = z
        self.nf = nf
        self.wf = wf

    def __call__(self, x):
        x_sum = tf.reduce_sum(x, axis=list(range(len(x.shape) - 2)))
        return (
            self.z * x_sum.shape[1] * tf.reduce_sum(tf.tanh(
                self.nf *
                tf.reduce_sum(tf.abs(tf.tanh(self.wf * x_sum)), axis=1)
                / x_sum.shape[1]))
        )
