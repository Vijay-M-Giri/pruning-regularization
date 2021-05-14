import tensorflow as tf


class ZeroOpt(tf.optimizers.SGD):
    def __init__(self, learning_rate=0.1, momentum=0.9, prune=True):
        super().__init__(learning_rate=learning_rate, momentum=momentum)
        self.prune = prune

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        gradients, variables = zip(*grads_and_vars)
        gradients, variables = self._zero_opt(list(gradients), list(variables))
        return super().apply_gradients(
            zip(gradients, variables), name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _zero_opt(self, gradients, variables):
        gzt = []
        t = 1e-2
        for i, g in enumerate(gradients):
            if "bias" in variables[i].name:
                gzt.append(g)
            else:
                gzt.append(tf.where(tf.abs(variables[i]) < t, tf.zeros_like(g), g))
                if self.prune:
                    var = tf.where(tf.abs(variables[i]) < t, tf.zeros_like(variables[i]), variables[i])
                    variables[i].assign(var)
        return gzt, variables
