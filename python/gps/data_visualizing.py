import sys
import copy
import numpy as np
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.utility.data_logger import DataLogger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_weight(ax, weights):
    """
    plot a figure where x-axis is weight and the y-axis is the value of weight
    Args:
        ax: one of the subplot
        weights:

    Returns:
        drawn ax
    """
    assert len(weights.shape) == 2
    num_weight = weights.shape[1]
    num_value = weights.shape[0]    # the number of plot line
    x_data = np.zeros(0)
    for i in range(num_weight):
        x_data = np.append(x_data, i)

    maker_dataset = Line2D.filled_markers
    # color_dataset = []
    # color_base = 1. / num_value
    # for color in range(num_value):
    #     color_dataset.append(str(color_base * (color+1)))
    color_dataset = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    for line in range(num_value):
        ax.plot(x_data, weights[line],
                linestyle='-',
                marker=maker_dataset[line],
                markersize=5,
                lw=2,
                color=color_dataset[line],
                label='act_%d' % line)

    plt.legend(loc='upper right', frameon=True)


data_logger = DataLogger()


def gradient_visualization(fisher_info):
    """ process fisher information"""
    # mean of parameters to the first sample's list
    for num_sum in range(1, len(fisher_info)):
         for num_act in range(len(fisher_info[0])):
            for num_weight in range(len(fisher_info[0][0])):
                fisher_info[0][num_act][num_weight] += fisher_info[num_sum][num_act][num_weight]

    # copy the data
    sum_fisher = copy.deepcopy(fisher_info[0])

    """
    squeeze, and merge to [w1, b1, w2, b2, ... ]
    each w has [act_op1, act_op2, ...]
    """
    average_weights = list()

    for num_weight in range(len(sum_fisher[0])):
        weight_expand = np.zeros(0)
        for num_act in range(len(sum_fisher)):
            # squeeze_w = np.squeeze(sum_fisher[num_act][num_weight])
            # squeeze_w = np.reshape(sum_fisher[num_act][num_weight], (1, -1))
            flatten_w = sum_fisher[num_act][num_weight].flatten()
            flatten_w = np.expand_dims(flatten_w, axis=0)
            if len(weight_expand.shape) != len(flatten_w.shape):
                weight_expand = flatten_w
            else:
                weight_expand = np.concatenate((weight_expand, flatten_w), axis=0)
        average_weights.append(weight_expand)



    # for num_weight in range(len(average_weights)):
    """ for single weight"""
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # for num_weight in range(1):
    #     ax.set_xlabel('number')
    #     ax.set_xlabel('value')
    #     ax.set_title('act_op_%d' % num_weight)
    #     plot_weight(ax, average_weights[num_weight])
    #     plt.show()

    """ for multiple weights"""
    for num_weight in range(len(average_weights)):
        plt.figure(num_weight)
        ax = plt.gca()
        ax.set_xlabel('number')
        ax.set_xlabel('value')
        ax.set_title('weight_%d' % num_weight)
        plot_weight(ax, average_weights[num_weight])

    plt.show()

def plot_fisher(ax, weights):
    num_act = len(weights)

    """ flatten the weight of each action"""
    weights_flatten = list()
    for act_idx in range(num_act):
        weights_flatten.append(weights[act_idx].flatten())

    x_data = np.zeros(0)
    for num_idx in range(weights_flatten[0].shape[0]):
        x_data = np.append(x_data, num_idx)

    maker_dataset = Line2D.filled_markers
    color_dataset = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

    for act_idx in range(num_act):
        ax.plot(x_data, weights_flatten[act_idx],
                linestyle='-',
                marker=maker_dataset[act_idx],
                markersize=5,
                lw=2,
                color=color_dataset[act_idx],
                label='act_%d' % act_idx
                )
    plt.legend(loc='upper right', frameon=True)


def fisher_visualization(fisher_info):
    """
    fisher that

    """
    fisher_tran = []
    num_act = len(fisher_info)
    num_weight = len(fisher_info[0])
    for weight_idx in range(num_weight):
        weight_single = []
        for act_idx in range(num_act):
            weight_single.append(fisher_info[act_idx][weight_idx])
        fisher_tran.append(weight_single)

    for weight_idx in range(num_weight):
        plt.figure(weight_idx)
        ax = plt.gca()
        ax.set_xlabel('number')
        ax.set_xlabel('value')
        ax.set_title('weight_%d' % weight_idx)
        # plot_weight(ax, fisher_tran[weight_idx])
        plot_fisher(ax, fisher_tran[weight_idx])
    plt.show()




def action_visualization(action):
    """
    visulizing action
    """
    action = np.transpose(action)
    num_act = action.shape[0]
    time_step = action.shape[1]
    x_data = np.zeros(0)
    for i in range(time_step):
        x_data = np.append(x_data, i)

    plt.figure(1)
    ax = plt.gca()
    ax.set_xlabel('number')
    ax.set_xlabel('value')
    ax.set_title('action')
    plot_weight(ax, action)
    plt.show()

data_dir = '/home/sun/work/ocfgps/position/'
action = data_logger.unpickle(data_dir + 'action.pkl')
fisher_info = data_logger.unpickle(data_dir + 'fisher.pkl')
prob = data_logger.unpickle(data_dir + 'prob.pkl')

# action_visualization(action)
# action_visualization(prob)
# gradient_visualization(fisher_info)
fisher_visualization(fisher_info)
