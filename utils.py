import tensorflow as tf


def var_mask(var, cur_var, axis, ker_f):
    prune_nodes = get_nodes_to_prune(cur_var, axis, ker_f)
    keep_nodes = tf.logical_not(prune_nodes)
    prune_mask = get_prune_mask(cur_var, keep_nodes, 1 - axis)
    prune_sum = tf.multiply(cur_var, prune_mask)
    return get_var_from_sum(var, prune_sum)


def gradient_mask(gradients, variables):
    gzt = []
    L = len(gradients)
    for i in range(L):
        if "bias" in variables[i].name:
            gzt.append(gradients[i])
        else:
            gzt.append(tf.where(tf.abs(variables[i]) == 0.0,
                                tf.zeros_like(gradients[i]), gradients[i]))
    return gzt


def get_prune_mask(var, keep_nodes, axis):
    shape = var.shape.as_list()
    shape[1 - axis] = 1
    return tf.cast(
        tf.broadcast_to(tf.reshape(keep_nodes, shape), var.shape),
        var.dtype)


def get_var_from_sum(var, prune_sum):
    if len(var.shape) <= 2:
        return tf.where(prune_sum == 0, tf.zeros_like(var), var)
    broad = tf.broadcast_to(
        tf.reshape(
            prune_sum,
            tf.TensorShape([1, 1]).concatenate(prune_sum.shape)),
        var.shape)
    return tf.where(broad == 0, tf.zeros_like(var), var)


def get_nodes_to_keep(var, axis_shape, axis, ker_f):
    threshold = 0.1
    return tf.greater(
        tf.reduce_sum(tf.abs(tf.tanh(50 * var)),
                      axis=(axis if axis <= 1 else [0, 1, axis])),
        tf.minimum(10, threshold * axis_shape * ker_f))


def get_dynamic_shape(t, axis):
    return tf.reduce_sum(tf.ones_like(tf.gather(t, 0, axis=1-axis)))
