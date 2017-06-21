""" This file defines the main object that runs experiments. """
"""
policy learn in three area and test in six area
OLGPS paper experiment 1
"""

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
import numpy as np


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

        self.init_alpha(self)

    def run(self, config, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        self.target_points = self.agent._hyperparams['target_ee_points'][:3]
        itr_start = self._initialize(itr_load)

        # """ set pre"""
        # position_train = self.data_logger.unpickle('./position/position_train.pkl')



        # print('training position.....')
        # print(position_train)

        # print('test all testing position....')
        # for i in xrange(position_train.shape[0]):
        #     test_positions = self.generate_position_radius(position_train[i], 0.03, 5, 0.01)
        #     if i == 0:
        #         all_test_positions = test_positions
        #     else:
        #         all_test_positions = np.concatenate((all_test_positions, test_positions))

        T = self.algorithm.T
        N = self._hyperparams['num_samples']
        dU = self.algorithm.dU

        flag_fail = True
        flag_suc_peg = False
        num_suc = 4
        num_good_save = 6
        error_acc = 0.02


        # position_train = self.data_logger.unpickle('./position/all_train_position.pkl')
        position_train = self.data_logger.unpickle('./position/position_train.pkl')
        # test_position = self.data_logger.unpickle('./position/all_test_position.pkl')
        all_position_train = np.zeros(position_train.shape)

        for num_pos in range(position_train.shape[0]):
        # for num_pos in range(2):

            """ alter the iteration when it far away target hole"""
            if num_pos >= 2:
                num_suc = 6
            """ load train position and reset agent model. """
            flag_fail = True
            while(flag_fail):

                """ test OLGPS only"""
                position_temp = position_train[num_pos]
                disturbe = np.random.uniform(-0.005, 0.005, 1)
                position_temp[0] = position_temp[0] + disturbe
                position_temp[1] = position_temp[1] - disturbe


                for cond in self._train_idx:
                    # self._hyperparams['agent']['pos_body_offset'][cond] = position_train[num_pos]
                    self._hyperparams['agent']['pos_body_offset'][cond] = position_temp
                self.agent.reset_model(self._hyperparams)

                """ test OLGPS generalization"""
                # for cond in self._train_idx:
                #     self._hyperparams['agent']['pos_body_offset'][cond] = position_train[num_pos]
                #     # self._hyperparams['agent']['pos_body_offset'][cond] = position_temp
                # self.agent.reset_model(self._hyperparams)

                # initial train array
                train_prc = np.zeros((0, T, dU, dU))
                train_mu = np.zeros((0, T, dU))
                train_obs_data = np.zeros((0, T, self.algorithm.dO))
                train_wt = np.zeros((0, T))

                # initial variables
                count_suc = 0

                for itr in range(itr_start, self._hyperparams['iterations']):
                    print('******************num_pos:************', num_pos)
                    print('______________________itr:____________', itr)
                    print('>>>>>>>>>>>>>count_suc:>>>>>>>>>>>>>>>', count_suc)
                    for cond in self._train_idx:
                        for i in range(self._hyperparams['num_samples']):
                            if num_pos == 0:
                                self._take_sample(itr, cond, i)
                            elif itr == 0:
                                self._take_sample(itr, cond, i)
                            else:
                                # self._take_train_sample(itr, cond, i)
                                self._take_sample(itr, cond, i)

                    traj_sample_lists = [
                        self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                        for cond in self._train_idx
                    ]

                    # calculate the distance of  the end-effector to target position
                    ee_pos = self.agent.get_ee_pos(cond)[:3]
                    target_pos = self.agent._hyperparams['target_ee_pos'][:3]
                    distance_pos = ee_pos - target_pos
                    distance_ee = np.sqrt(distance_pos.dot(distance_pos))
                    print('distance ee:', distance_ee)

                    # collect the successful sample to train global policy
                    if distance_ee <= error_acc:
                        count_suc += 1
                        if count_suc > num_suc:
                            flag_suc_peg = True
                            tgt_mu, tgt_prc, obs_data, tgt_wt = self.train_prepare(traj_sample_lists)
                            train_mu = np.concatenate((train_mu, tgt_mu))
                            train_prc = np.concatenate((train_prc, tgt_prc))
                            train_obs_data = np.concatenate((train_obs_data, obs_data))
                            train_wt = np.concatenate((train_wt, tgt_wt))
                        if count_suc > num_suc + num_good_save:
                            """ save the last good trajectory"""
                            save_mu = tgt_mu[0, :, :]
                            save_prc = tgt_prc[0, :, :, :]
                            save_obs = obs_data[0, :, :]
                            save_wt = tgt_wt[0, :]
                            save_K = self.algorithm.cur[0].traj_distr.K
                            save_k = self.algorithm.cur[0].traj_distr.k
                            save_covar = self.algorithm.cur[0].traj_distr.chol_pol_covar
                            np.save('./position/good_trajectory_mu_%d.npy' % num_pos, save_mu)
                            np.save('./position/good_trajectory_prc_%d.npy' % num_pos, save_prc)
                            np.save('./position/good_trajectory_obs_%d.npy' % num_pos, save_obs)
                            np.save('./position/good_trajectory_wt_%d.npy' % num_pos, save_wt)
                            np.save('./position/good_trajectory_K_%d.npy' % num_pos, save_K)
                            np.save('./position/good_trajectory_k_%d.npy' % num_pos, save_k)
                            np.save('./position/good_trajectory_covar_%d.npy' % num_pos, save_covar)
                            self.data_logger.pickle('./position/good_trajectory_mu_%d.pkl' % num_pos, save_mu)
                            self.data_logger.pickle('./position/good_trajectory_prc_%d.pkl' % num_pos, save_prc)
                            self.data_logger.pickle('./position/good_trajectory_obs_%d.pkl' % num_pos, save_obs)
                            self.data_logger.pickle('./position/good_trajectory_wt_%d.pkl' % num_pos, save_wt)

                        # """ save the successful insertion trajectory """
                        # np.save('./position/step_mu_%d.npy' % count_suc, tgt_mu)



                    # Clear agent samples.
                    self.agent.clear_samples()

                    # if get enough sample, then break
                    if count_suc > num_suc + num_good_save:
                        break
                    if not flag_suc_peg:
                        self._take_iteration(itr, traj_sample_lists)
                    if self.algorithm.traj_opt.flag_reset:
                        break
                    # pol_sample_lists = self._take_policy_samples()
                    # self._log_data(itr, traj_sample_lists, pol_sample_lists)
                    # if num_pos > 0:
                    #     self.algorithm.fit_global_linear_policy(traj_sample_lists)

                if not self.algorithm.traj_opt.flag_reset and flag_suc_peg:
                    flag_suc_peg = False
                    # save train position
                    all_position_train[num_pos] = position_temp

                    flag_fail = False


                    self.data_logger.pickle('./position/mu_%d.pkl' % num_pos, train_mu)
                    self.data_logger.pickle('./position/prc_%d.pkl' % num_pos, train_prc)
                    self.data_logger.pickle('./position/obs_%d.pkl' % num_pos, train_obs_data)
                    self.data_logger.pickle('./position/wt_%d.pkl' % num_pos, train_wt)

                    if num_pos == 0:
                        self.algorithm.policy_opt.update_ewc(train_obs_data, train_mu, train_prc, train_wt,
                                                             with_ewc=False, compute_fisher=False)
                    else:
                        if num_pos == 2:
                            self.algorithm.policy_opt.update_ewc(train_obs_data, train_mu, train_prc, train_wt,
                                                                 with_ewc=False, compute_fisher=False)
                        else:
                            self.algorithm.policy_opt.update_ewc(train_obs_data, train_mu, train_prc, train_wt,
                                                                 with_ewc=False, compute_fisher=False)
                    # test the trained in the current position
                    print('test current policy.....')
                    if num_pos == 2:
                        print('123')
                    self.test_current_policy()

                    """ test OLGPS only"""
                    # test_position = self.data_logger.unpickle('./position/test_position.pkl')
                    test_position = None
                    if num_pos == 0:
                        test_position = self.data_logger.unpickle('./position/test_position_%d.pkl' % (num_pos))
                    else:
                        test_position1 = self.data_logger.unpickle('./position/test_position_%d.pkl' % (num_pos - 1))
                        test_position2 = self.data_logger.unpickle('./position/test_position_%d.pkl' % (num_pos ))
                        test_position = np.concatenate((test_position1, test_position2), axis=0)

                    cost, pos_suc_count, ee_distance = self.test_cost(test_position)
                    # self.data_logger.pickle('./position/all_distance_%d.pkl' % num_pos, ee_distance)
                    # self.data_logger.pickle('./position/all_cost_%d.pkl' % num_pos, cost)
                    # self.data_logger.pickle('./position/all_suc_count_%d.pkl' % num_pos, pos_suc_count)
                    #
                    # self.data_logger.pickle('./position/all_train_position.pkl', all_position_train)
                    # self.data_logger.pickle('./position/all_test_position.pkl', test_position)

                    # if ns


                else:
                    print("a o!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # raw_input()

                self.algorithm.reset_alg()
                self.next_iteration_prepare()

            # reset the algorithm to the initial algorithm for the next position
            # del self.algorithm
            # config['algorithm']['agent'] = self.agent
            # self.algorithm = config['algorithm']['type'](config['algorithm'])
            # self.algorithm.reset_alg()
            # self.next_iteration_prepare()



        print("finish......")
        """ test OLGPS only"""
        # ee_distance1 = np.reshape(ee_distance1, (6, ee_distance1.shape[0] / 6))
        # ee_distance2 = np.reshape(ee_distance2, (6, ee_distance2.shape[0] / 6))
        # ee_distance3 = np.reshape(ee_distance3, (6, ee_distance3.shape[0] / 6))
        # self.data_logger.pickle('all_distance_3.pkl', ee_distance3)
        # self.data_logger.pickle('./position/all_distance_3.pkl', ee_distance3)
        # self.data_logger.pickle('./position/all_distance_2.pkl', ee_distance2)
        # self.data_logger.pickle('./position/all_distance_1.pkl', ee_distance1)
        # self.data_logger.pickle('./position/all_cost_3.pkl', cost3)
        # self.data_logger.pickle('./position/all_cost_2.pkl', cost2)
        # self.data_logger.pickle('./position/all_cost_1.pkl', cost1)
        # self.data_logger.pickle('./position/all_suc_3.pkl', pos_suc_count3)
        # self.data_logger.pickle('./position/all_suc_2.pkl', pos_suc_count2)
        # self.data_logger.pickle('./position/all_suc_1.pkl', pos_suc_count1)

        self._end()

    def generate_test_position(self):
        """
        generate special test positions
        Returns:

        """
        center_position = 0.02
        radius = 0.02
        max_error_bound = 0.02
        for pos_count in range(6):
            if pos_count == 2:
                center_position = center_position - 0.01
            position = self.generate_position(center_position, radius, 30, max_error_bound)
            if pos_count == 0:
                all_positions = position
            else:
                all_positions = np.concatenate((all_positions, position), axis=0)
            if pos_count == 2:
                center_position = center_position + 0.01
            center_position = center_position + radius * 2

        return all_positions

    def position_shuffle(self, positions, sort_idx):
        """
        re range the positions
        Args:
            positions:
            sort_idx:

        Returns:

        """
        return positions[sort_idx]


    def generate_position_radius(self, position_ori, radius, conditions, max_error_bound):
        """

        Args:
            position_ori: original center position of generated positions
            radius:     area's radius
            conditions: the quantity of generating positions
            max_error_bound: the mean of generated positions' error around cposition

        Returns:

        """
        c_x = position_ori[0]
        c_y = position_ori[1]
        while True:
            all_positions = np.zeros(0)
            center_position = np.array([c_x, c_y, 0])
            for i in range(conditions):
                position = np.random.uniform(radius, radius, 3)
                while True:
                    position[2] = 0
                    position[1] = (position[1] + c_y)
                    position[0] = position[0] + c_x
                    area = (position - center_position).dot(position - center_position)
                    if area <= (np.pi * radius ** 2) / 4.0:
                        break
                    position = np.random.uniform(-radius, radius, 3)
                if i == 0:
                    all_positions = position
                    all_positions = np.expand_dims(all_positions, axis=0)
                else:
                    all_positions = np.vstack((all_positions, position))

            mean_position = np.mean(all_positions, axis=0)
            mean_error = np.fabs(center_position - mean_position)
            print('mean_error:', mean_error)
            if mean_error[0] < max_error_bound and mean_error[1] < max_error_bound:
                break

        all_positions = np.floor(all_positions * 1000) / 1000.0
        # print('all_position:', all_positions)
        return all_positions

    def generate_position(self, cposition, radius, conditions, max_error_bound):
        """

        Args:
            cposition:  center position
            radius:     area radius
            conditions: the quantity of generated posion
            max_error_bound:

        Returns:
            positions

        """
        while True:
            all_positions = np.array([cposition, -cposition, 0])
            center_position = np.array([cposition, -cposition, 0])
            for i in range(conditions):
                position = np.random.uniform(cposition - radius, cposition + radius, 3)
                while True:
                    position[2] = 0
                    position[1] = -position[1]
                    area = (position - center_position).dot(position - center_position)
                    # area = np.sum(np.multiply(position - center_position, position - center_position))
                    if area <= radius ** 2:
                        # print(area)
                        break
                    position = np.random.uniform(cposition - radius, cposition + radius, 3)
                position = np.floor(position * 1000) / 1000.0
                all_positions = np.concatenate((all_positions, position))
            all_positions = np.reshape(all_positions, [all_positions.shape[0] / 3, 3])
            # print(all_positions[:, 1])
            # print('mean:')
            # print(np.mean(all_positions, axis=0))
            mean_position = np.mean(all_positions, axis=0)
            # mean_error1 = np.fabs(mean_position[0] - 0.11)
            # mean_error2 = np.fabs(mean_position[1] + 0.11)
            mean_error1 = np.fabs(mean_position[0] - (cposition - max_error_bound))
            mean_error2 = np.fabs(mean_position[1] + (cposition - max_error_bound))
            if mean_error1 < max_error_bound and mean_error2 < max_error_bound:
                print('mean:')
                print(np.mean(all_positions, axis=0))
                break
        # print(all_positions)
        # print(all_positions.shape)
        return all_positions

    def test_cost(self, position):
        """
        test the NN policy at all position
        Args:
            position:

        Returns:

        """
        total_costs = np.zeros(0)
        total_distance = np.zeros(0)
        total_suc = np.zeros(0)
        print 'calculate cost_________________'
        for itr in range(position.shape[0]):
            for cond in self._train_idx:
                self._hyperparams['agent']['pos_body_offset'][cond] = position[itr]
            self.agent.reset_model(self._hyperparams)
            _, cost, ee_points = self.take_nn_samples()
            ee_error = ee_points[:3] - self.target_points
            distance = np.sqrt(ee_error.dot(ee_error))
            error = np.sum(np.fabs(ee_error))
            if (error < 0.06):
                total_suc = np.concatenate((total_suc, np.array([1])))
            else:
                total_suc = np.concatenate((total_suc, np.array([0])))
            total_costs = np.concatenate((total_costs, np.array(cost)))
            total_distance = np.concatenate((total_distance, np.array([distance])))
        # return np.mean(total_costs), total_suc, total_distance
        return total_costs, total_suc, total_distance


    def test_old_cost(self, position):
        """
        test the NN policy at all position
        Args:
            position:

        Returns:

        """
        total_costs = np.zeros(0)
        total_distance = np.zeros(0)
        total_suc = np.zeros(0)
        print 'calculate cost_________________'
        for itr in range(position.shape[0]):
            if itr % 51 == 0:
                print('****************')
            for cond in self._train_idx:
                self._hyperparams['agent']['pos_body_offset'][cond] = position[itr]
            self.agent.reset_model(self._hyperparams)
            _, cost, ee_points = self.take_nn_samples()
            ee_error = ee_points[:3] - self.target_points
            distance = np.sqrt(ee_error.dot(ee_error))
            error = np.sum(np.fabs(ee_error))
            if (error < 0.06):
                total_suc = np.concatenate((total_suc, np.array([1])))
            else:
                total_suc = np.concatenate((total_suc, np.array([0])))
            total_costs = np.concatenate((total_costs, np.array(cost)))
            total_distance = np.concatenate((total_distance, np.array([distance])))
        # return np.mean(total_costs), total_suc, total_distance
        return np.mean(total_costs), total_suc, total_distance

    def next_iteration_prepare(self):
        """
        prepare for the next iteration
        Returns:

        """
        self.init_alpha()

    def init_alpha(self, val=None):
        """
        initialize the alpha1, 2, the default is 0.7, 0.3
        Args:
            val:

        Returns:

        """
        if val is None:
            self.alpha1 = 1
            self.alpha2 = 0
        else:
            self.alpha1 = 1
            self.alpha2 = 0
    def pol_alpha(self):
        return self.alpha1, self.alpha2

    def adjust_alpha(self, step):
        self.alpha1 = self.alpha1 - step
        self.alpha2 = self.alpha2 + step

        if self.alpha1 < 0.7:
            self.alpha1 = 0.7
            self.alpha2 = 0.3

    def train_prepare(self, sample_lists):
        """
        prepare the train data of the sample lists
        Args:
            sample_lists: sample list from agent

        Returns:
            target mu, prc, obs_data, wt

        """
        algorithm = self.algorithm
        dU, dO, T = algorithm.dU, algorithm.dO, algorithm.T
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc = np.zeros((0, T, dU, dU))
        tgt_wt = np.zeros((0, T))
        wt_origin = 0.01 * np.ones(T)
        for m in range(algorithm.M):
            samples = sample_lists[m]
            X = samples.get_X()
            N = len(samples)
            prc = np.zeros((N, T, dU, dU))
            mu = np.zeros((N, T, dU))
            wt = np.zeros((N, T))

            traj = algorithm.cur[m].traj_distr
            for t in range(T):
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t].fill(wt_origin[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
            tgt_wt = np.concatenate((tgt_wt, wt))

        return tgt_mu, tgt_prc, obs_data, tgt_wt

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread

            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                if self.algorithm.cur[0].pol_info:
                    pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                        ('pol_sample_itr_%02d.pkl' % itr_load))
                else:
                    pol_sample_lists = None
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] \
                and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy
        else:
            pol = self.algorithm.cur[cond].traj_distr
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_train_sample(self, itr, cond, i):
        """
        collect sample with merge policy
        Args:
            itr:
            cond:
            i:

        Returns:

        """
        alpha1, alpha2 = self.pol_alpha()
        print("alpha:********%03f, %03f******" %(alpha1, alpha2))
        pol1 = self.algorithm.cur[cond].traj_distr
        pol2 = self.algorithm.cur[cond].last_pol
        if not self.gui:
            self.agent.merge_controller(pol1, alpha1, pol2, alpha2, cond,
                                        verbose=(i < self._hyperparams['verbose_trials']))

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        self.algorithm.iteration(sample_lists)
        if self.gui:
            self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False)
        return [SampleList(samples) for samples in pol_samples]

    def take_nn_samples(self, N=None):
        """
        take the NN policy
        Args:
            N:

        Returns:
            samples, costs, ee_points

        """
        """
            Take samples from the policy to see how it's doing.
            Args:
                N  : number of policy samples to take per condition
            Returns: None
        """

        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        costs = list()
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False)
            policy_cost = self.algorithm.cost[0].eval(pol_samples[cond][0])[0]
            policy_cost = np.sum(policy_cost)
            print "cost: %d" % policy_cost  # wait to plot in gui in gps_training_gui.py
            costs.append(policy_cost)

            ee_points = self.agent.get_ee_point(cond)

        return [SampleList(samples) for samples in pol_samples], costs, ee_points

    def take_new_nn_samples(self, N=None):
        """
        take the NN policy
        Args:
            N:

        Returns:
            samples, costs, ee_points

        """
        """
            Take samples from the policy to see how it's doing.
            Args:
                N  : number of policy samples to take per condition
            Returns: None
        """

        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        # TODO: Take at all conditions for GUI?
        costs = list()
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.mimic_algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False)
            policy_cost = self.mimic_algorithm.cost[0].eval(pol_samples[cond][0])[0]
            policy_cost = np.sum(policy_cost)
            print "cost: %d" % policy_cost  # wait to plot in gui in gps_training_gui.py
            costs.append(policy_cost)

            ee_points = self.agent.get_ee_point(cond)

        return [SampleList(samples) for samples in pol_samples], costs, ee_points

    def test_current_policy(self):
        """
        test the current NN policy in the current position
        Returns:

        """
        verbose = self._hyperparams['verbose_policy_trials']
        for cond in self._train_idx:
            samples = self.agent.sample(
                self.algorithm.policy_opt.policy, cond,
                verbose=verbose, save=False, noisy=False
            )

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
            if self._quit_on_end:
                # Quit automatically (for running sequential expts)
                os._exit(1)

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    elif test_policy_N:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config, args.quit)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(hyperparams.config, itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
