""" Hyperparameters for gym."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.gym_env.agent_gym import AgentGym
# from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_olgps import AlgorithmOLGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
# from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 1,
    JOINT_VELOCITIES: 1,
    ACTION: 1,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/box2d_arm_badmm_example/'


common = {
    'experiment_name': 'box2d_arm_badmm_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentGym,
    'env': ['Pendulum-v1', 'Pendulum-v0', 'Pendulum-v2', 'Pendulum-v3'],
    'x0': np.array([0.289, 0.3287]),
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'target_state': np.array([0]),
    'target_vel': np.array([0.0])
}


algorithm = {
    'type': AlgorithmOLGPS,
    'conditions': common['conditions'],
    'iterations': 10,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.1,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1.0])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.array([1, 1]),
            'target_state': agent["target_state"],
        },
        JOINT_VELOCITIES: {
            'wp': np.ones(SENSOR_DIMS[JOINT_VELOCITIES]) * 1e-1,
            'target_state': agent["target_vel"],
        }
    },
}

final_cost = {
    'type': CostState,
    'data_types': {
        JOINT_VELOCITIES:{
            'wp':np.ones(SENSOR_DIMS[JOINT_VELOCITIES]),
            'target_state': agent["target_vel"],
        },
        JOINT_ANGLES: {
            'wp': np.array([10, 10]),
            'target_state': agent["target_state"],
        },
    },
    'l1': 1.0,
    'l2': 0.0,
    'ramp_option': RAMP_FINAL_ONLY,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost, final_cost],
    'weights': [1e-5, 1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

# algorithm['policy_opt'] = {
#     'type': PolicyOptCaffe,
#     'weights_file_prefix': EXP_DIR + 'policy',
# }
algorithm['policy_opt'] = {
        'type': PolicyOptTf,
        'network_params':{
            'obs_include':[JOINT_ANGLES, JOINT_VELOCITIES],
            'obs_vector_data':[JOINT_ANGLES, JOINT_VELOCITIES],
            'sensor_dims':SENSOR_DIMS,
            },
        'network_model':tf_network,
        'iterations': 4000,
        'weights_file_prefix': EXP_DIR + 'policy',
}


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': 100,
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
