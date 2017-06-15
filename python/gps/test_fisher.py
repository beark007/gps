# import tensorflow as tf
#
# def init_weights(shape, name=None):
#     return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))
#
#
# def init_bias(shape, name=None):
#     return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))
#
# dim_input = 26
# dim_out = 7
# input = tf.placeholder(dtype=tf.float32, shape=[None, dim_input], name='input')
# target = tf.placeholder(dtype=tf.float32, shape=[None, dim_out], name='output')
#
# nn_layers = 3
# cur_top = input
# hidden_layer = [40, 40, dim_out]
# var_lists = list()
# for layer in range(nn_layers):
#     in_shape = cur_top.get_shape().dims[1].value
#     cur_weight = init_weights([in_shape, hidden_layer[layer]], name='w_'+str(layer))
#     cur_bias = init_bias([hidden_layer[layer]], name='b'+str(layer))
#     var_lists.append(cur_weight)
#     var_lists.append(cur_bias)
#
#     if layer != nn_layers-1:
#         cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
#     else:
#         cur_top = tf.matmul(cur_top, cur_weight) + cur_bias
#
# output = cur_top
#
# loss = tf.reduce_sum(tf.square(target - output), name='euclidean loss')
#
# variables = tf.trainable_variables()
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=variables)
#
# weight_copies = [tf.identity(var_lists) for itr in dim_out]
# multi_output = tf.stack(output[0, itr] for itr in dim_out)
# per_gradient = tf.gradients(multi_output, weight_copies)
#
# """ load data"""
# from gps.utility.data_logger import DataLogger
# data_logger = DataLogger()
# total_mu = data_logger.unpickle('./position/total_mu.pkl')
# total_obs = data_logger.unpickle('./position/total_obs.pkl')
# train_mu = total_mu[0]
# train_obs = total_obs[0]
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# a = train_obs[0]
# b = train_mu[0]
# import time
# for i in range(100):
#     feed_dict = {input: train_obs[i], target: train_mu[i]}
#     time_start = time.time()
#     sess.run(per_gradient, feed_dict=feed_dict)
#     time_end = time.time()
#     print('compute time:', time_end - time_start)

"""
test numpy shape
"""
# import numpy as np
# a = np.load('./position/good_trajectory_mu_1.npy')
# num = a.shape[0]
# print('num:', num)

"""
test multinormal
"""
# import tensorflow as tf
# import tensorflow.contrib.distributions as ds
# import numpy as np
# sess = tf.Session()
#
# """ multi variable"""
# # a = np.array([[1.0], [2.0], [3.0]])
# # b = np.array([[1.0], [2.0], [3.0]])
# # c = np.array([[0.0], [0.0], [0.0]])
# # with sess.as_default():
# #     mvn = ds.MultivariateNormalDiag(
# #         loc=a,
# #         scale_diag=b
# #     )
# #     print mvn.prob(c).eval()
# """ single variable"""
# # a = np.array([1.0, 2.0, 3.0])
# # b = np.array([1.0, 2.0, 3.0])
# # c = np.array([[0.0, 0.0, 0.0],
# #              [1.0, 2.0, 3.0]])
# # with sess.as_default():
# #     dist = ds.Normal(loc=a, scale=b)
# #     print(sess.run(dist.mean()))
# #     print(sess.run(dist.prob(c)))

"""
test numpy convert to list
"""
# import numpy as np
# a = np.array([1,2,3,4])
# print('shape:', a.shape)
# a = a.reshape((4, 1))
# print('shape:', a)
# b = a.tolist()
# print(b)

"""
test the fisher information matrix
"""
import tensorflow as tf
import numpy as np

def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))
def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))

dim_input = 26
dim_out = 7
input = tf.placeholder(dtype=tf.float32, shape=[None, dim_input], name='input')
target = tf.placeholder(dtype=tf.float32, shape=[None, dim_out], name='output')

nn_layers = 3
cur_top = input
hidden_layer = [40, 40, dim_out]
var_lists = list()
for layer in range(nn_layers):
    in_shape = cur_top.get_shape().dims[1].value
    cur_weight = init_weights([in_shape, hidden_layer[layer]], name='w_'+str(layer))
    cur_bias = init_bias([hidden_layer[layer]], name='b'+str(layer))
    var_lists.append(cur_weight)
    var_lists.append(cur_bias)

    if layer != nn_layers-1:
        cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
    else:
        cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

output = cur_top

loss = tf.reduce_sum(tf.square(target - output), name='euclidean_loss')

variables = tf.trainable_variables()
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=variables)

# hessian = tf.hessians(output, var_lists)
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

""" load data"""
traj_mu = np.load('./position/good_trajectory_mu_0.npy')
traj_obs = np.load('./position/good_trajectory_obs_0.npy')

with sess.as_default():
    for i in range(10):
        idx = [i]
        # mu = traj_mu[i]
        # obs = traj_obs[i]
        # mu = np.expand_dims(mu, axis=0)
        # obs = np.expand_dims(obs, axis=0)
        mu = np.float32(traj_mu[idx])
        obs = np.float32(traj_obs[idx])
        sess.run(train_op, feed_dict={input: obs, target: mu})
    print(len(var_lists))
    print(sess.run(var_lists[5]))

    saver = tf.train.Saver(var_lists)
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss, var_list=variables)
    saver.save(sess, './position/temp.ckpt')
    sess.run(tf.global_variables_initializer())
    print('initial: ', sess.run(var_lists[5]))
    saver.restore(sess, './position/temp.ckpt')
    print('restore: ', sess.run(var_lists[5]))
    for i in range(10):
        idx = [i]
        # mu = traj_mu[i]
        # obs = traj_obs[i]
        # mu = np.expand_dims(mu, axis=0)
        # obs = np.expand_dims(obs, axis=0)
        mu = np.float32(traj_mu[idx])
        obs = np.float32(traj_obs[idx])
        sess.run(train_op, feed_dict={input: obs, target: mu})



