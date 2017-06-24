""" This file defines the main object that runs experiments. """

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
import numpy as np

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList


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

    def run(self, time_experiment, exper_condition, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        itr_start = self._initialize(itr_load)

        # test_position = self.data_logger.unpickle('./position/%d/%d/test_position.pkl'
        #                                           % (time_experiment, exper_condition))
        self.target_ee_point = self.agent._hyperparams['target_ee_points'][:3]

        for itr in range(itr_start, self._hyperparams['iterations']):
            print('itr******:  %d   **********' % itr)
            for cond in self._train_idx:
                for i in range(self._hyperparams['num_samples']):
                    self._take_sample(itr, cond, i)

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                for cond in self._train_idx
            ]

            # Clear agent samples.
            self.agent.clear_samples()
            if itr >= 10:
                print('123')

            self._take_iteration(itr, traj_sample_lists)
            pol_sample_lists = self._take_policy_samples()

            self._log_data(itr, traj_sample_lists, pol_sample_lists)

        """ test policy and collect costs"""
        """
        gradually add the distance of agent position
        """
        center_position = 0.02
        radius = 0.02
        max_error_bound = 0.02
        directory = 9
        for test_condition in range(7):
            # test_position = self.generate_position(center_position, radius, 30, max_error_bound)
            test_position = self.data_logger.unpickle('./position/test_position_%d.pkl'
                                                      % (test_condition + 1))
            costs, position_suc_count, distance = self.test_cost(test_position, len(pol_sample_lists))
            print('distance:', distance)
            # add the position_suc_count
            if test_condition == 0:
                #augement array
                all_pos_suc_count = np.expand_dims(position_suc_count, axis=0)
                all_distance = np.expand_dims(distance, axis=0)
            else:
                all_pos_suc_count = np.vstack((all_pos_suc_count, position_suc_count))
                all_distance = np.vstack((all_distance, distance))

            costs = costs.reshape(costs.shape[0] * costs.shape[1])
            mean_cost = np.array([np.mean(costs)])
            center_position = center_position + radius * 2

        self._end()
        return costs, mean_cost, all_pos_suc_count, all_distance

    def generate_position(self, cposition, radius, conditions, max_error_bound):
        # all_positions = np.zeros(0)

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
        print(all_positions)
        print(all_positions.shape)
        return all_positions

    def test_cost(self, positions, train_cond):
        """
        test policy and collect costs
        Args:
            positions: test position from test_position.pkl

        Returns:
            cost:   mean cost of all test position
            total_suc:  successful pegging trial count  1:successful    0:fail

        """
        iteration = positions.shape[0] / train_cond
        total_costs = list()
        total_ee_points = list()
        total_suc = np.zeros(0)
        total_distance = np.zeros(0)
        for itr in range(iteration):
            for cond in self._train_idx:
                self._hyperparams['agent']['pos_body_offset'][cond] = positions[itr + cond]
            self.agent.reset_model(self._hyperparams)
            _, cost, ee_points = self._test_policy_samples()
            for cond in self._train_idx:
                total_ee_points.append(ee_points[cond])
            total_costs.append(cost)
        print("total_costs:", total_costs)
        for i in range(len(total_ee_points)):
            ee_error = total_ee_points[i][:3] - self.target_ee_point
            distance = ee_error.dot(ee_error)**0.5
            if( distance < 0.06 ):
                total_suc = np.concatenate((total_suc, np.array([1])))
            else:
                total_suc = np.concatenate((total_suc, np.array([0])))
            total_distance = np.concatenate((total_distance, np.array([distance])))
        return np.array(total_costs), total_suc, total_distance

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

    def _test_policy_samples(self, N=None):
        """
        test sample from the policy and collect the costs
        Args:
            N:

        Returns:
            samples
            costs:      list of cost for each condition
            ee_point:   list of ee_point for each condition

        """
        if 'verbose_policy_trials' not in self._hyperparams:
            return None
        verbose = self._hyperparams['verbose_policy_trials']
        pol_samples = [[None] for _ in range(len(self._test_idx))]
        costs = list()
        ee_points = list()
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False
            )
            # in algorithm.py: _eval_cost
            policy_cost = self.algorithm.cost[0].eval(pol_samples[cond][0])[0]
            policy_cost = np.sum(policy_cost)   #100 step
            costs.append(policy_cost)
            ee_points.append(self.agent.get_ee_point(cond))
        return [SampleList(samples) for samples in pol_samples], costs, ee_points

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
    parser.add_argument('-c', '--condition', metavar='N', type=int,
                        help='consider N position')
    parser.add_argument('-m', '--num', metavar='N', type=int,
                        help='test\' N nums of experiment')
    parser.add_argument('-exper', '--exper', metavar='N', type=int,
                        help='time of test experiment')
    parser.add_argument('-set', '--set_cond', metavar='N', type=int,
                        help='train on special position setting')
    parser.add_argument('-algi', '--alg_itr', metavar='N', type=int,
                        help='control the time of train NN')

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
        if args.condition:
            """ if specify the N training position"""
            num_position = args.condition
            data_logger = DataLogger()
            positions = data_logger.unpickle('./position/position_train.pkl')
            # positions = data_logger.unpickle('./position/suc_train_position.pkl')
            hyperparams.agent['conditions'] = num_position
            hyperparams.common['conditions'] = num_position
            hyperparams.algorithm['conditions'] = num_position
            pos_body_offset = list()
            for i in range(num_position):
                pos_body_offset.append(positions[i])
            hyperparams.agent['pos_body_offset'] = pos_body_offset

        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        # set the time of training NN
        if args.alg_itr:
            hyperparams.config['iterations'] = args.alg_itr

        """
        set extend setting
        """
        data_logger = DataLogger()
        # train_position = data_logger.unpickle('./position/position_train.pkl')
        train_position = data_logger.unpickle('./position/all_train_position.pkl')
        hyperparams.agent['pos_body_offset'] = list(train_position)
        hyperparams.agent['conditions'] = len(train_position)
        hyperparams.common['conditions'] = len(train_position)
        hyperparams.algorithm['conditions'] = len(train_position)

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
            costs, mean_cost, position_suc_count, all_distance = gps.run(args.num,
                                                                         exper_condition=args.set_cond,
                                                                         itr_load=resume_training_itr)
            # gps.data_logger.pickle('./position/%d/experiment_%d/md_all_distance.pkl'
            #                        % (args.num, args.exper), all_distance)
            gps.data_logger.pickle('./position/md_all_distance.pkl', all_distance)
            gps.data_logger.pickle('./position/md_all_cost.pkl', costs)

            """
                        if args.condition == 1:
                costs = np.expand_dims(costs, axis=0)
                position_suc_count = np.expand_dims(position_suc_count, axis=0)
                gps.data_logger.pickle('./position/%d/md_all_test_costs_%d.pkl'
                                       % (args.num, args.condition), costs)
                gps.data_logger.pickle('./position/%d/md_test_costs.pkl'
                                       % args.num, mean_cost)
                gps.data_logger.pickle('./position/%d/position_md_%d.pkl'
                                       % (args.num, args.condition), position_suc_count)
                print mean_cost.shape
            elif args.condition > 1:
                # there should be check existing pkl
                all_costs = gps.data_logger.unpickle('./position/%d/md_all_test_costs_%d.pkl'
                                                     % (args.num, args.condition - 1))
                min_len = min(all_costs.shape[1], costs.shape[0])
                all_costs = all_costs[:, :min_len]
                costs = costs[:min_len]
                all_costs = np.vstack((all_costs, costs))

                all_mean_costs = gps.data_logger.unpickle('./position/%d/md_test_costs.pkl'
                                                          % args.num)
                all_mean_costs = np.concatenate((all_mean_costs, mean_cost))

                all_position_suc_count = gps.data_logger.unpickle('./position/%d/position_md_%d.pkl'
                                                                  % (args.num, args.condition - 1))
                min_len = min(all_position_suc_count.shape[1], costs.shape[0])
                all_position_suc_count = all_position_suc_count[:, :min_len]
                position_suc_count = position_suc_count[:min_len]
                all_position_suc_count = np.vstack((all_position_suc_count, position_suc_count))
                print all_costs.shape
                gps.data_logger.pickle('./position/%d/md_test_costs.pkl' % args.num, all_mean_costs)
                gps.data_logger.pickle('./position/%d/md_all_test_costs_%d.pkl' % (args.num, args.condition), all_costs)
                gps.data_logger.pickle('./position/%d/position_md_%d.pkl' % (args.num, args.condition), all_position_suc_count)
            """



if __name__ == "__main__":
    main()
