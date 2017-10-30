""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import mjcpy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_GYM
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET

from gps.sample.sample import Sample

import gym

class AgentGym(Agent):
    """
    All communication between the algorithms and gym is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_GYM)
        config.update(hyperparams)
        self.conds = config['conditions']
        Agent.__init__(self, config)
        self.setUp()

    def setUp(self):
        """
        set up the environment
        :return:
        """
        self._world = [gym.make(self._hyperparams['env'][i]) for i in range(self.conds)]
        self.action_dim = self._world[0].action_space.shape[0]
        self.state_dim = self._world[0].observation_space.shape[0] - 1
        self.x0 = self._hyperparams['x0']

    def reset(self, condition):
        """
        reset the environment
        :return:
        """
        return self._world[condition].reset()

    def reset_world(self, idx=0):
        self._world = [gym.make(self._hyperparams['env'][(i+idx)]) for i in range(self.conds)]

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        sample from gym
        :param policy:
        :param condition:
        :return:
        """
        obs = self.reset(condition)
        new_sample = self._init_sample(obs)
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t+1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    obs, _, _, _ =self._world[condition].step(U[t, :])
                self._set_sample(new_sample, obs, t)
            if verbose:
                self.view(condition)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, x):
        sample = Sample(self)
        self._set_sample(sample, x, -1)
        return sample

    def _set_sample(self, sample, x, t):
        # theta = x[0]
        # theta_vel = x[-1]

        x[0] = np.arccos(x[0])
        # theta = self.angle_normalize(x[0])
        theta = x[0]
        theta_vel = x[-1]

        sample.set(JOINT_ANGLES, np.array([theta]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array([theta_vel]), t=t+1)

    def view(self, cond):
        """
        view the environment
        :param cond:
        :return:
        """
        self._world[cond].render()

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

