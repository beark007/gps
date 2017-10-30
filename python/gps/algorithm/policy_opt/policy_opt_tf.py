""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf
import tensorflow.contrib.distributions as ds

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver

try:
    import cPickle as pickle
except:
    import pickle



LOGGER = logging.getLogger(__name__)


class PolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.feat_op = None # features
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.feat_vals = None

        self.decay_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.01, global_step=self.decay_step,
                                                        decay_steps=1000,
                                                        decay_rate=0.7)

        self.iteration = self._hyperparams['iterations']

        self.fisher_info = list()   # fisher information for each condition
        self.var_lists = None   # variables of NN
        self.var_lists_pre = [] # variables of previous NN
        self.lam = np.zeros(0)

        self.path_name = '/home/sun/work/gps/position/'

        """ init graph"""
        self.sess = tf.Session()
        """ create a log dir to save the log result"""
        if tf.gfile.Exists('./log'):
            tf.gfile.DeleteRecursively('./log')
        tf.gfile.MakeDirs('./log')
        self.writer = tf.summary.FileWriter('./log')
        self.writer.add_graph(self.sess.graph)

        self.init_network()

        self.summery = tf.summary.merge_all()

        self.init_solver()
        self.var = self._hyperparams['init_var'] * np.ones(dU)

        self.policy = TfPolicy(dU, self.obs_tensor, self.act_op, self.feat_op,
                               np.zeros(dU), self.sess, self.device_string, copy_param_scope=self._hyperparams['copy_param_scope'])
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.var_saver = tf.train.Saver(self.var_lists)





        self.num_task = 0 # use for count the number of learnt task

    # def compute_fisher_info(self, obs, obs_idx):
    #     """
    #     compute fisher information
    #     sess:
    #     obs: the observation
    #     obs_idx: the shuffle idx
    #     """
    #     #
    #     #wait to add
    #
    #     num_samples = 500
    #     # initialize Fisher information for most recent task
    #     self.F_mat = []
    #     for v in range(len(self.var_lists)):
    #         self.F_mat.append(np.zeros(self.var_lists[v].get_shape().as_list()))
    #
    #     F_prev = copy.deepcopy(self.F_mat)
    #     mean_diffs = np.zeros(0)
    #     ax_len = np.zeros(0)
    #
    #     # calculate the probability
    #     # probs = tf.nn.softmax(self.act_op) # the output of NN
    #     probs = self.act_op
    #
    #     # calculate the fisher information with samples
    #     import time
    #     for i in range(num_samples):
    #
    #         idx = obs_idx[i:i+1]
    #         # compute first-order derivatives
    #         ders_list = list()
    #         print('act_dimension:', self.act_op.get_shape()[1])
    #         print('idx:', idx)
    #         time_start = time.time()
    #         # for act_dim in range(self.act_op.get_shape()[1]):
    #         #     # ders_list.append(self.sess.run(tf.gradients(probs[0, act_dim], self.var_lists),
    #         #     #                                feed_dict={self.obs_tensor: obs[idx]}))
    #         #     ders_list.append(self.sess.run(tf.gradients(tf.log(probs[0, act_dim]), self.var_lists),
    #         #                                    feed_dict={self.obs_tensor: obs[idx]}))
    #         # ders_list = [self.sess.run(tf.gradients(probs[0, act_dim], self.var_lists),
    #         #                       feed_dict={self.obs_tensor: obs[idx]}) for act_dim in range(self.act_op.get_shape()[1])]
    #
    #         act_dim = np.random.randint(0,7)
    #         ders_list.append(self.sess.run(tf.gradients(tf.log(probs[0, act_dim]), self.var_lists),
    #                                       feed_dict={self.obs_tensor: obs[idx]}))
    #
    #         time_end = time.time()
    #         print('compute %d..., time is %f ms' % (i, (time_end - time_start)))
    #         for v in range(len(self.F_mat)):
    #             for num_act in range(len(ders_list)):
    #                 self.F_mat[v] += np.square(ders_list[num_act][v])
    #
    #         # prepare for plot
    #         F_diff = 0
    #
    #         if i > 0:
    #             for v in range(len(self.F_mat)):
    #                 F_diff += np.sum(np.absolute(self.F_mat[v]/(i+1) - F_prev[v]))
    #             mean_diff = np.mean(F_diff)
    #             mean_diffs = np.append(mean_diffs, mean_diff)
    #             ax_len = np.append(ax_len, i)
    #             for v in range(len(self.F_mat)):
    #                 F_prev[v] = self.F_mat[v] / (i+1)
    #
    #     import matplotlib.pyplot as plt
    #     plt.plot(ax_len, mean_diffs)
    #     plt.show()
    #     plt.clf()
    #     for v in range(len(self.F_mat)):
    #         self.F_mat[v] /= num_samples
    #     self.fisher_info.append(self.F_mat)

    def compute_traj_fisher(self, traj_obs=None):
        """ compute a trajectory fisher information"""

        """ load trajectory data"""
        if traj_obs is None:
            print('num_task:', self.num_task)
            traj_obs = np.load(self.path_name + 'good_trajectory_obs_%d.npy' % self.num_task)
            # traj_K = np.load(self.path_name + 'good_trajectory_K_%d.npy' % self.num_task)
            # traj_k = np.load(self.path_name + '/good_trajectory_k_%d.npy' % self.num_task)
            # traj_covar = np.load(self.path_name + '/good_trajectory_covar_%d.npy' % self.num_task)

        traj_obs = np.float32(traj_obs)
        # traj_K = np.float32(traj_K)
        # traj_k = np.float32(traj_k)
        # traj_covar = np.float32(traj_covar)




        num_samples = traj_obs.shape[0]
        # initialize Fisher information for most recent task
        self.F_mat = []
        for v in range(len(self.var_lists)):
            self.F_mat.append(np.zeros(self.var_lists[v].get_shape().as_list()))

        F_prev = copy.deepcopy(self.F_mat)
        abs_grad = copy.deepcopy(self.F_mat)
        F_cur = copy.deepcopy(self.F_mat)
        mean_diffs = np.zeros(0)
        ax_len = np.zeros(0)

        """calculate the probability"""
        # probs = tf.nn.softmax(self.act_op) # the output of NN

        # act_abs = tf.abs(self.act_op)
        # probs = tf.nn.softmax(act_abs)

        probs = self.act_op

        save_ders = []
        save_action = np.zeros(0)
        save_prob = np.zeros(0)
        total_fisher = []

        # calculate the fisher information with samples
        import time
        weight_fisher = 1
        for i in range(num_samples):
        # for i in range(3):

            idx = [i]
            # compute first-order derivatives
            ders_list = []
            time_start = time.time()

            """ construct normal distribution"""
            # K = traj_K[i]
            # k = traj_k[i]
            # mu = np.dot(K, traj_obs[i]) + k
            # covar = np.diag(traj_covar[i])

            # normal_gaussian = ds.Normal(
            #     loc=mu,
            #     scale=covar
            # )
            # probs = normal_gaussian.prob(self.act_op)


            for act_dim in range(self.act_op.get_shape()[1]):
                # ders_list.append(self.sess.run(tf.gradients(tf.log(probs[0, act_dim]), self.var_lists),
                #                                feed_dict={self.obs_tensor: traj_obs[idx]}))
                # probs_alter = None
                # if act_dim == 0:
                #     probs_alter = tf.multiply(tf.log(probs[0, act_dim]), tf.constant(1.0))
                # else:
                #     probs_alter = tf.log(probs[0, act_dim])

                # probs_alter = tf.log(probs[0, act_dim])
                probs_alter = probs[0, act_dim]

                # probs = normal_gaussian.prob(self.act_op[0][act_dim])
                # probs_alter = tf.log(probs)


                probs_gradient = self.sess.run(tf.gradients(probs_alter, self.var_lists),
                                               feed_dict={self.obs_tensor: traj_obs[idx]})
                ders_list.append(probs_gradient)
            time_end = time.time()
            print('compute %d..., time is %f ms' % (i, (time_end - time_start)))


            """ construct the list of gradient, action, probability"""
            save_ders.append(ders_list)

            action = self.sess.run(self.act_op, feed_dict={self.obs_tensor: traj_obs[idx]})
            prob = self.sess.run(probs, feed_dict={self.obs_tensor: traj_obs[idx]})
            if i == 0:
                save_action = action
                save_prob = prob
            else:
                save_action = np.concatenate((save_action, action), axis=0)
                save_prob = np.concatenate((save_prob, prob), axis=0)


            """ compute fisher information"""


            # for v in range(len(self.F_mat)):
            #     for num_act in range(len(ders_list)):
            #         save_fisher[num_act][v] += np.square(ders_list[num_act][v] / len(ders_list))
            #         # save the absolute gradients
            #         self.F_mat[v] += np.square(ders_list[num_act][v]) / len(ders_list)

            """ compute using max value of each action"""
            if i > num_samples*0.5:
                if i > num_samples*0.8:
                    weight_fisher = 6
                else:
                    weight_fisher = 3
                print('weight_fisher:', weight_fisher)
            for v in range(len(self.F_mat)):
                max_fisher_value = np.full(self.F_mat[v].shape, -np.inf)
                for num_act in range(len(ders_list)):
                    gradient_square = np.square(ders_list[num_act][v])
                    max_fisher_value = np.maximum(max_fisher_value, gradient_square)
                self.F_mat[v] += max_fisher_value * weight_fisher
                F_cur[v] = max_fisher_value

            """ compute sum of gradient and then square"""
            # for v in range(len(self.F_mat)):
            #     for num_act in range(1, len(ders_list)):
            #         ders_list[0][v] += ders_list[num_act][v]
            #         save_fisher[num_act][v] += np.square(ders_list[num_act][v] / len(ders_list))
            #     self.F_mat[v] += np.square(ders_list[0][v] / len(ders_list))

            ####### save F_mat for each obs
            total_fisher.append(copy.deepcopy(F_cur))

            # prepare for plot
            F_diff = 0

            if i > 0:
                for v in range(len(self.F_mat)):
                    F_diff += np.sum(np.absolute(self.F_mat[v]/(i+1) - F_prev[v]))
                mean_diff = np.mean(F_diff)
                mean_diffs = np.append(mean_diffs, mean_diff)
                ax_len = np.append(ax_len, i)
                for v in range(len(self.F_mat)):
                    F_prev[v] = self.F_mat[v] / (i+1)

        # import matplotlib.pyplot as plt
        # plt.plot(ax_len, mean_diffs)
        # plt.show()
        # plt.clf()

        """ save data"""

        # pickle.dump(save_ders, open(self.path_name + 'gradient.pkl', 'wb'))
        # pickle.dump(save_action, open(self.path_name + 'action.pkl', 'wb'))
        # pickle.dump(save_prob, open(self.path_name + 'prob.pkl', 'wb'))

        fisher_init = total_fisher[0]
        for num_samples in range(len(total_fisher)):
            fisher = total_fisher[num_samples]
            each_weight = []
            for v in range(len(fisher_init)):
                mean_value = np.mean(fisher_init[v] - fisher_init[v])
                each_weight.append(mean_value)
            print('weight mean differ:', each_weight)

        """ attain fisher information for each task"""
        for v in range(len(self.F_mat)):
            self.F_mat[v] /= num_samples
        self.fisher_info.append(copy.deepcopy(self.F_mat))
        pickle.dump(self.F_mat, open(self.path_name + 'fisher_%d.pkl' % self.num_task, 'wb'))
        pickle.dump(self.fisher_info, open(self.path_name + 'fisher_info_%d.pkl' % self.num_task, 'wb'))
        pickle.dump(total_fisher, open(self.path_name + 'total_fisher_%d.pkl' % self.num_task, 'wb'))
        """ save var list used to compare with fisher information"""
        pickle.dump(self.sess.run(self.var_lists), open(self.path_name + 'var_lists_%d.pkl' % self.num_task, 'wb'))

        self.num_task = self.num_task +1

    def compute_fisher_only(self, traj_obs):
        """ only compute fisher and doesn't not save anything"""
        traj_obs = np.float32(traj_obs)
        num_samples = traj_obs.shape[0]
        F_mat = []
        for v in range(len(self.var_lists)):
            F_mat.append(np.zeros(self.var_lists[v].get_shape().as_list()))

        import time
        for i in range(num_samples):
            # for i in range(3):

            idx = [i]
            ders_list = []
            time_start = time.time()

            probs = self.act_op

            for act_dim in range(self.act_op.get_shape()[1]):
                probs_alter = probs[0, act_dim]
                probs_gradient = self.sess.run(tf.gradients(probs_alter, self.var_lists),
                                               feed_dict={self.obs_tensor: traj_obs[idx]})
                ders_list.append(probs_gradient)
            time_end = time.time()
            print('compute %d..., time is %f s' % (i, (time_end - time_start)))

            """ compute fisher information"""
            # for v in range(len(self.F_mat)):
            #     for num_act in range(len(ders_list)):
            #         save_fisher[num_act][v] += np.square(ders_list[num_act][v] / len(ders_list))
            #         # save the absolute gradients
            #         self.F_mat[v] += np.square(ders_list[num_act][v]) / len(ders_list)

            """ compute using max value of each action"""
            weight_fisher = 1
            for v in range(len(F_mat)):
                max_fisher_value = np.full(F_mat[v].shape, -np.inf)
                for num_act in range(len(ders_list)):
                    gradient_square = np.square(ders_list[num_act][v])
                    max_fisher_value = np.maximum(max_fisher_value, gradient_square)
                F_mat[v] += max_fisher_value * weight_fisher

        return F_mat

    def keep_pre_vars(self):
        self.var_lists_pre = []
        for v in range(len(self.var_lists)):
            self.var_lists_pre.append(self.sess.run(self.var_lists[v]))

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                                  network_config=self._hyperparams['network_params'])
        self.obs_tensor = tf_map.get_input_tensor()
        self.precision_tensor = tf_map.get_precision_tensor()
        self.action_tensor = tf_map.get_target_output_tensor()
        self.act_op = tf_map.get_output_op()
        self.feat_op = tf_map.get_feature_op()
        self.loss_scalar = tf_map.get_loss_op()
        self.fc_vars = fc_vars
        self.last_conv_vars = last_conv_vars

        self.var_lists = tf_map.get_var_lists_tensor()

        # Setup the gradients
        self.grads = [tf.gradients(self.act_op[:,u], self.obs_tensor)[0]
                for u in range(self._dU)]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver = TfSolver(loss_scalar=self.loss_scalar,
                               solver_name=self._hyperparams['solver_type'],
                               base_lr=self._hyperparams['lr'],
                               lr_policy=self._hyperparams['lr_policy'],
                               momentum=self._hyperparams['momentum'],
                               weight_decay=self._hyperparams['weight_decay'],
                               fc_vars=self.fc_vars,
                               last_conv_vars=self.last_conv_vars,
                               summery=self.summery)
        self.saver = tf.train.Saver()

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations'] ):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx+self.batch_size]
                feed_dict = {self.last_conv_vars: conv_values[idx_i],
                             self.action_tensor: tgt_mu[idx_i],
                             self.precision_tensor: tgt_prc[idx_i]}
                train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                if (i+1) % 500 == 0:
                    LOGGER.info('tensorflow iteration %d, average loss %f',
                                    i+1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.obs_tensor: obs[idx_i],
                         self.action_tensor: tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            train_loss = self.solver(feed_dict, self.sess, device_string=self.device_string)

            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                average_loss = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.feat_op is not None:
            self.feat_vals = self.solver.get_var_values(self.sess, self.feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))


        """ save the variable"""
        self.save_variable(self.num_task)

        return self.policy

    def start(self):
        """
        append the current variables to self.var_lists_pre
        Returns:

        """
        self.star_var=[]
        for v in range(len(self.var_lists)):
            self.star_var.append(self.sess.run(self.var_lists[v]))
        self.var_lists_pre.append(self.star_var)
        pickle.dump(self.var_lists_pre, open(self.path_name + 'var_lists_pre_%d.pkl' % (self.num_task-1), 'wb'))

    def restore(self, num_task=None):
        """
        restore the previous weight from the ckpt file
        """
        self.sess.run(tf.global_variables_initializer())
        # print('before restore:', self.sess.run(self.var_lists[5]))
        if num_task:
            self.var_saver.restore(self.sess, self.path_name + 'var_list_%d.ckpt' % num_task)
        else:
            self.var_saver.restore(self.sess, self.path_name + 'var_list.ckpt')
        # print('after restore:', self.sess.run(self.var_lists[5]))

    def save_variable(self, num_task=None):
        """
        save weights to file
        """
        if num_task is not None:
            self.var_saver.save(self.sess, self.path_name + 'var_list_%d.ckpt' % num_task)
        else:
            self.var_saver.save(self.sess, self.path_name + 'var_list.ckpt')

    def save_ckpt(self, path):
        self.saver.save(self.sess, path)

    def restore_ckpt(self, path):
        self.saver.restore(self.sess, path)

    def reset_iteration(self, itr):
        self.iteration = itr

    def update_ewc(self, obs, tgt_mu, tgt_prc, tgt_wt, lam,
                   with_ewc=False, compute_fisher=False, with_traj=True):
        """
        Update policy with ewc loss
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        if with_ewc:
            self.save_variable()
            self.start()
            ####### reset learning rate
            self.reset_lr()
            ####### use all var_lists
            self.solver.update_loss(self.fisher_info, self.var_lists, self.var_lists_pre, lam)
            ####### use previous var_lists
            # self.solver.update_loss(self.fisher_info, self.var_lists, self.star_var, 20)
            self.solver.solver_op = self.solver.reset_solver_op(loss=self.solver.ewc_loss,
                                                                var_lists=self.var_lists,
                                                                learning_rate=self.learning_rate,
                                                                global_step=self.decay_step)
            self.restore()
        else:
            self.save_variable()
            self.reset_lr()
            self.solver.update_loss(var_lists=self.var_lists)
            self.solver.solver_op = self.solver.reset_solver_op(loss=self.solver.ewc_loss,
                                                                var_lists=self.var_lists,
                                                                learning_rate=self.learning_rate,
                                                                global_step=self.decay_step)
            self.restore()

        # if self.num_task == 2:
        #     print('num_task 1: ', self.var_lists_pre[0][5])
        #     print('num_task 2: ', self.var_lists_pre[1][5])

        # self.solver.update_loss()

        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N * T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N * T, dO))
        tgt_mu = np.reshape(tgt_mu, (N * T, dU))
        tgt_prc = np.reshape(tgt_prc, (N * T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N * T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N * T / self.batch_size)
        idx = range(N * T)
        average_loss = 0
        np.random.shuffle(idx)

        if self._hyperparams['fc_only_iterations'] > 0:
            feed_dict = {self.obs_tensor: obs}
            num_values = obs.shape[0]
            conv_values = self.solver.get_last_conv_values(self.sess, feed_dict, num_values, self.batch_size)
            for i in range(self._hyperparams['fc_only_iterations']):
                start_idx = int(i * self.batch_size %
                                (batches_per_epoch * self.batch_size))
                idx_i = idx[start_idx:start_idx + self.batch_size]
                feed_dict = {self.last_conv_vars: conv_values[idx_i],
                             self.action_tensor: tgt_mu[idx_i],
                             self.precision_tensor: tgt_prc[idx_i]}
                train_loss, summery = self.solver(feed_dict, self.sess, device_string=self.device_string, use_fc_solver=True)
                average_loss += train_loss

                if (i + 1) % 500 == 0:
                    LOGGER.info('tensorflow iteration %d, average loss %f',
                                i + 1, average_loss / 500)
                    average_loss = 0
            average_loss = 0

        # actual training.

        # use for tensorboard
        base_iter = self.num_task * 7000

        average_loss_nature = 0

        # for i in range(self._hyperparams['iterations']):
        # for i in range(self.iteration):
        for i in range(8000):

            # if i < 4000:
            #     self.step_add(self.decay_step)

            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx + self.batch_size]
            feed_dict = {self.obs_tensor: obs[idx_i],
                         self.action_tensor: tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            # train_loss, summery = self.solver(feed_dict, self.sess, device_string=self.device_string)
            # self.writer.add_summary(summery, i+base_iter)

            """ add the fisher value"""
            loss = self.solver(feed_dict, self.sess, device_string=self.device_string)
            # loss_nature = self.sess.run(self.loss_scalar, feed_dict=feed_dict)
            loss_nature = loss[1]
            train_loss = loss[0]

            average_loss += train_loss
            average_loss_nature += loss_nature

            if (i + 1) % 50 == 0:
                LOGGER.info('tensorflow iteration %d, average loss %f',
                            i + 1, average_loss / 50)
                LOGGER.info('average nature loss: %f', average_loss_nature / 50)
                LOGGER.info('learning rate: %f', self.sess.run(self.learning_rate))
                if average_loss_nature < 1*50:
                    break
                # if with_ewc:
                #     if len(self.fisher_info) > 0:
                #         fisher_value = tf.constant(0, dtype=tf.float32)
                #         for num_task in range(len(self.fisher_info)):
                #             for v in range(len(self.var_lists)):
                #                 fisher_value += tf.reduce_sum(tf.multiply(self.fisher_info[num_task][v].astype(np.float32),
                #                                                           tf.square(self.var_lists[v] -
                #                                                                     self.var_lists_pre[num_task][v])))
                #                 # fisher_value += tf.reduce_sum(tf.multiply(self.fisher_info[num_task][v].astype(np.float32),
                #                 #                                           tf.square(self.var_lists[v] -
                #                 #                                                     self.var_lists_pre[2][v])))
                #                 # fisher_value += tf.reduce_sum(tf.multiply(self.fisher_info[num_task][v].astype(np.float32),
                #                 #                                           tf.square(self.var_lists[v] -
                #                 #                                                     self.star_var[v])))
                #         LOGGER.info('fisher value: %f', self.sess.run(fisher_value))

                        # """ print loss differents"""
                        # loss_ewc = self.sess.run(self.solver.ewc_loss, feed_dict=feed_dict)
                        # print('loss difference:', loss_ewc - loss_nature)

                average_loss = 0
                average_loss_nature = 0

        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.feat_op is not None:
            self.feat_vals = self.solver.get_var_values(self.sess, self.feat_op, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                                      self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

        print('computing fisher information .......')
        if compute_fisher:
            # self.compute_traj_fisher()
            if with_traj:
                self.compute_traj_fisher()
            else:
                self.compute_fisher_info(obs, idx)

        """ save varibles """
        self.save_variable(self.num_task)
        policy = self.policy
        policy_var = []
        policy_var.append(policy.bias)
        policy_var.append(policy.chol_pol_covar)
        policy_var.append(policy.dU)
        policy_var.append(policy.scale)
        policy_var.append(policy.x_idx)
        pickle.dump(policy_var, open(self.path_name + 'policy_var_%d.pkl' % self.num_task, 'wb'))


        return self.policy

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if self.policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(self.policy.scale)
                                         + self.policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.act_op, feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    def save_model(self, fname):
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.sess, fname, write_meta_graph=False)

    def restore_model(self, fname):
        self.saver.restore(self.sess, fname)
        LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name) # TODO - is this implemented.
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'tf_iter': self.tf_iter,
            'x_idx': self.policy.x_idx,
            'chol_pol_covar': self.policy.chol_pol_covar,
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.policy.x_idx = state['x_idx']
        self.policy.chol_pol_covar = state['chol_pol_covar']
        self.tf_iter = state['tf_iter']

        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)

    def step_add(self, decay_step, step=1):
        """ add the length to decay_step"""
        add_op = decay_step.assign_add(step)
        self.sess.run(add_op)

    def reset_lr(self):
        """ reset learning rate"""
        del self.learning_rate
        self.decay_step.assign(0)
        self.learning_rate = tf.train.exponential_decay(0.01, global_step=self.decay_step,
                                                        decay_steps=500,
                                                        decay_rate=0.9)
    def restore_trained_policy(self, num_task):
        """ restore the policy trained at num_task position"""
        self.restore(num_task)
        self.fisher_info = pickle.load(open(self.path_name + 'fisher_info_%d.pkl' % (num_task), 'rb'))

        policy = self.policy
        policy_var = pickle.load(open(self.path_name + 'policy_var_%d.pkl' % num_task, 'rb'))
        policy.bias = policy_var[0]
        policy.chol_pol_covar = policy_var[1]
        policy.dU = policy_var[2]
        policy.scale = policy_var[3]
        policy.x_idx = policy_var[4]

        self.var_lists_pre = []
        for i in range(num_task + 1):
            self.var_lists_pre.append(pickle.load(open(self.path_name + 'var_lists_%d.pkl' % i, 'rb')))
        self.num_task = num_task + 1

