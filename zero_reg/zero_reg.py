import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Custom', name='zero')
class ZeroRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, z=1e-4, wf=50, nf=5):
        self.z = z
        self.nf = nf
        self.wf = wf

    def __call__(self, x):
        return (
            self.z * x.shape[1] * tf.reduce_sum(tf.tanh(
                self.nf *
                tf.reduce_sum(tf.abs(tf.tanh(self.wf * x)), axis=1)
                / x.shape[1]))
        )
