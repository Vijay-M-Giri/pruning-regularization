import tensorflow as tf


class ZeroOptSGD(tf.optimizers.SGD):

    def __init__(self, learning_rate=0.01, momentum=0.0, prune=True,
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
            for i in range(k):
                cur_var = tf.reduce_sum(
                    t_vars[i], axis=list(range(len(t_vars[i].shape) - 2)))
                prune_nodes = self.get_nodes_to_prune(cur_var, 1)
                keep_nodes = tf.logical_not(prune_nodes)
                prune_mask = self.get_prune_mask(cur_var, keep_nodes, 0)
                prune_sum = tf.multiply(cur_var, prune_mask)
                var = self.get_var_from_sum(t_vars[i], prune_sum)
                t_vars[i].assign(var)
                if i < k - 1:
                    prune_nodes = self.get_nodes_to_prune(cur_var, 0)
                    keep_nodes = tf.logical_not(prune_nodes)
                    prune_mask = self.get_prune_mask(cur_var, keep_nodes, 1)
                    prune_sum = tf.multiply(cur_var, prune_mask)
                    var = self.get_var_from_sum(t_vars[i], prune_sum)
                    t_vars[i].assign(var)

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

    def get_var_from_sum(self, var, prune_sum):
        if len(var.shape) <= 2:
            return tf.where(prune_sum == 0, tf.zeros_like(var), var)
        broad = tf.broadcast_to(
            tf.reshape(
                prune_sum,
                tf.TensorShape([1, 1]).concatenate(prune_sum.shape)),
            var.shape)
        return tf.where(broad == 0, tf.zeros_like(var), var)
