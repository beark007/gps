import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap_withPre
import numpy as np

def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))

def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))

def get_input_layer(dim_input, dim_output):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    action_pre = tf.placeholder('float', [None, dim_output], name='action_pre')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, action_pre, precision

def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases

def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result

def euclidean_loss_layer(a, b, precision, batch_size):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = tf.reduce_sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor

def get_loss_layer_pre(mlp_out, action, action_pre, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    loss_current = euclidean_loss_layer(a=action, b=mlp_out, precision=precision, batch_size=batch_size)
    loss_pre = euclidean_loss_layer(a=action_pre, b=mlp_out, precision=precision, batch_size=batch_size)
    return tf.add(loss_current, loss_pre, name='loss_total')

def example_tf_with_previous(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    A NN with previous actions
    Args:
        dim_input:
        dim_output:
        batch_size:
        network_config:

    Returns:

    """
    n_layers = 2
    dim_hidden = (n_layers - 1) * [40]
    dim_hidden.append(dim_output)

    nn_input, action,action_pre, precision = get_input_layer(dim_input, dim_output)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)

    fc_vars = weights_FC + biases_FC
    loss_out = get_loss_layer_pre(mlp_out=mlp_applied, action=action, action_pre=action_pre,
                                  precision=precision, batch_size=batch_size)

    return TfMap_withPre.init_from_lists([nn_input, action, action_pre, precision], [mlp_applied], [loss_out]), fc_vars, []

