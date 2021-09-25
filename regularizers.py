import tensorflow as tf

class PruneRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, z=1e-5, nf=3, wf=50):
        self.z = z
        self.nf = nf
        self.wf = wf

    def __call__(self, x, shape):
        rank = x.shape.rank
        ker_f = 1 if rank == 2 else x.shape[0] * x.shape[1]
        s0, s1 = shape
        weights = tf.abs(tf.tanh(self.wf * x))
        # if rank > 2:
        #     weights = tf.reduce_sum(weights, axis=[0, 1])
        # return (
        #     self.z * ker_f * s1 * tf.reduce_sum(tf.tanh(
        #         self.nf * tf.reduce_sum(weights, axis=1) / (ker_f * s1))) +
        #     self.z * ker_f * s0 * tf.reduce_sum(tf.tanh(
        #         self.nf * tf.reduce_sum(weights, axis=0) / (ker_f * s0)))
        # )
        return self.z * tf.reduce_sum(weights)
