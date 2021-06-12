import tensorflow as tf


class ZeroOptSGD(tf.optimizers.SGD):

    def __init__(self, learning_rate=0.1, momentum=0.0, prune=True,
                 threshold=0.05):
        super().__init__(learning_rate=learning_rate, momentum=momentum)
        self.prune = prune
        self.threshold = threshold

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        gradients, variables = zip(*grads_and_vars)
        gradients = self._gradient_mask(list(gradients), list(variables))
        ans = super().apply_gradients(
            zip(gradients, variables), name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients)
        self._prune(list(variables))
        return ans

    def _prune(self, variables):
        if self.prune:
            t_vars = list(filter(lambda x: "kernel" in x.name, variables))
            k = len(t_vars)
            for i in range(k - 1):
                n_prev, n_cur = t_vars[i].shape
                if i == 0:
                    prune_nodes = self.get_nodes_to_prune(t_vars[i], 1)
                    keep_nodes = tf.logical_not(prune_nodes)
                    prune_mask = self.get_prune_mask(t_vars[i], keep_nodes, 0)
                    var = tf.multiply(t_vars[i], prune_mask)
                    t_vars[i].assign(var)
                prune_nodes = self.get_nodes_to_prune(t_vars[i], 0)
                if i + 1 < k:
                    prune_nodes = tf.math.logical_or(
                                    self.get_nodes_to_prune(t_vars[i + 1], 1),
                                    prune_nodes)
                keep_nodes = tf.logical_not(prune_nodes)
                prune_mask = self.get_prune_mask(t_vars[i], keep_nodes, 1)
                t_vars[i].assign(tf.multiply(t_vars[i], prune_mask))
                if i + 1 < k:
                    prune_mask = self.get_prune_mask(t_vars[i + 1],
                                                     keep_nodes, 0)
                    t_vars[i + 1].assign(tf.multiply(t_vars[i + 1],
                                                     prune_mask))

    def _gradient_mask(self, gradients, variables):
        gzt = []
        L = len(gradients)
        for i in range(L):
            if "bias" in variables[i].name:
                gzt.append(gradients[i])
            else:
                gzt.append(tf.where(tf.abs(variables[i]) == 0.0,
                                    tf.zeros_like(gradients[i]), gradients[i]))
        return gzt

    def get_prune_mask(self, var, keep_nodes, axis):
        shape = var.shape.as_list()
        shape[1 - axis] = 1
        return tf.cast(
            tf.broadcast_to(tf.reshape(keep_nodes, shape), var.shape),
            var.dtype)

    def get_nodes_to_prune(self, var, axis):
        n = var.shape[axis]
        return tf.less_equal(
            tf.reduce_sum(tf.abs(tf.tanh(50 * var)), axis=axis),
            self.threshold * n)
