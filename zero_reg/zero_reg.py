import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Custom', name='zero')
class ZeroRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, z=0.01, f=50):
        self.z = z
        self.f = f

    def __call__(self, x):
        return self.z * tf.reduce_sum(tf.abs(tf.tanh(self.f * x)))
        # return self.z * tf.reduce_sum(tf.abs(self._fun(x)))

    def _fun(self, x):
        r = tf.where(tf.abs(x) < 5 * 1e-3, tf.zeros_like(x), tf.tanh(self.f * x))
        return r
