""" This file defines the iLQG-based trajectory optimization method. """
import logging

import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition
from gps.algorithm.config import ALG_OLGPS

from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_utils import PolicyInfo
import copy
import scipy as sp


LOGGER = logging.getLogger(__name__)


class AlgorithmOLGPS(Algorithm):
    """ Sample-based trajectory optimization. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_OLGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)
        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )
        self.flag_reset = False

        policy_prior = self._hyperparams['policy_prior']
        for m in range(self.M):
            self.cur[m].last_pol = PolicyInfo(self._hyperparams)
            self.cur[m].last_pol.policy_prior = \
                    policy_prior['type'](policy_prior)

    def iteration(self, sample_lists):
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        print('###############I am OLGPS################')
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        self._update_dynamics()

        self._update_step_size()  # KL Divergence step size.

        # Run inner loop to compute new policies.
        for _ in range(self._hyperparams['inner_iterations']):
            self._update_trajectories()
            if self.traj_opt.flag_reset:
                break

        self._advance_iteration_variables()

    def fit_global_linear_policy(self, sample_list):
        """
        fit global policy in specify condition
        fit global policy into linear form
        """
        for m in range(self.M):
            dX, dU, T = self.dX, self.dU, self.T
            # data prepare
            samples = sample_list[m]
            N = len(samples)
            X = samples.get_X()
            obs = samples.get_obs().copy()
            pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]

            pol_info = copy.deepcopy(self.cur[m].last_pol)
            policy_prior = pol_info.policy_prior
            mode = self._hyperparams['policy_sample_mode']
            # there should check that previous will affect this fitting
            policy_prior.global_update(samples, self.policy_opt, mode)

            pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                policy_prior.fit(X, pol_mu, pol_sig)

    def reset_alg(self):
        """
        reset the algorithm to initial state at the beginning.
        reset the traj_distr and traj_info but keep policy_opt
        """
        last_pol = list()
        for m in range(self.M):
            last_pol.append(copy.deepcopy(self.cur[m].last_pol))
        self.cur = [IterationData() for _ in range(self.M)]

        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        for m in range(self.M):
            self.cur[m].last_pol = copy.deepcopy(last_pol[m])
            self.cur[m].traj_info = TrajectoryInfo()
            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)
        self.traj_opt = self._hyperparams['traj_opt']['type'](
            self._hyperparams['traj_opt']
        )
        if type(self._hyperparams['cost']) == list:
            self.cost = [
                self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                for i in range(self.M)
                ]
        else:
            self.cost = [
                self._hyperparams['cost']['type'](self._hyperparams['cost'])
                for _ in range(self.M)
                ]
        self.base_kl_step = self._hyperparams['kl_step']

    def get_previous_sample(self, train_obs):
        """
        ust the state of sample to get the action under the get_previous_sample
        Returns:
            train_mu: the action through the NN policy

        """
        noise = np.zeros(self.dU)
        train_mu = np.zeros((train_obs.shape[0], self.T, self.dU))
        for obs_len in range(train_obs.shape[0]):
            for time_step in range(train_obs.shape[1]):
                obs_data = train_obs[obs_len, time_step, :]
                mu = self.policy_opt.policy.act(None, obs_data, None, noise)
                train_mu[obs_len, time_step, :] = mu

        return train_mu


    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj = self.traj_opt.estimate_cost(
            self.cur[m].traj_distr, self.cur[m].traj_info
        )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_actual_laplace_obj)

        # if np.sum(previous_laplace_obj) > 20000 or \
        #     np.sum(new_predicted_laplace_obj) > 20000 or \
        #     np.sum(new_actual_laplace_obj) > 20000:
        #     self.flag_reset = True
        # else:
        #     self.flag_reset = False

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)
        if not self.flag_reset:
            self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        multiplier = self._hyperparams['max_ent_traj']
        fCm, fcv = traj_info.Cm / (eta + multiplier), traj_info.cv / (eta + multiplier)
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + multiplier) * np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return fCm, fcv
