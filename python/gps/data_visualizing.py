import sys
import copy
import numpy as np
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.utility.data_logger import DataLogger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_weight(ax, weights, label):
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
                label=label[line])

    plt.legend(loc='upper right', frameon=True)

def plot_multi_ax(plot_data, num_ax,label=None, color=None, sharex=True, sharey=True):
    """ sharing x and y axis for multiple ax in one figure
    plot_data: num_task * weight_shape, is a ndarray
    num_ax: the number to plot ax
    """
    sharey = False
    f, ax = plt.subplots(num_ax, sharex=sharex, sharey=sharey)


    len_x = plot_data.shape[1]
    x_data = np.linspace(start=0, stop=len_x, num=len_x)

    if color is None:
        color = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    if label is None:
        label = [i for i in range(num_ax)]
    assert len(color) >= plot_data.shape[0]

    for idx_ax in range(num_ax):
        # if idx_ax == 0:
        #     ax[idx_ax].set_ylim(0, 10)
        ax[idx_ax].plot(x_data,
                        plot_data[idx_ax],
                        color=color[idx_ax],
                        label=label[idx_ax])
    f.subplots_adjust(hspace=0, wspace=0)


data_logger = DataLogger()


def gradient_visualization(gradient_info):
    """ process gradient which is absolute value
    has the shape of 3-D [num_samples * num_sample * num_weights]
    """
    # mean of parameters to the first sample's list
    for num_sum in range(1, len(gradient_info)):
         for num_act in range(len(gradient_info[0])):
            for num_weight in range(len(gradient_info[0][0])):
                gradient_info[0][num_act][num_weight] = np.abs(gradient_info[0][num_act][num_weight])
                gradient_info[0][num_act][num_weight] += np.abs(gradient_info[num_sum][num_act][num_weight])

    # copy the data
    sum_fisher = copy.deepcopy(gradient_info[0])

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
        plot_weight(ax, fisher_tran[weight_idx])
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

# data_dir = '/home/sun/work/ocfgps/position/'
# action = data_logger.unpickle(data_dir + 'action.pkl')
# fisher_info = data_logger.unpickle(data_dir + 'fisher.pkl')
# prob = data_logger.unpickle(data_dir + 'prob.pkl')
#
# # action_visualization(action)
# # action_visualization(prob)
# # gradient_visualization(fisher_info)
# fisher_visualization(fisher_info)


""" visualizing action"""
def compare_act_nn_traj():
    """
    compare action between nn output and the local trajectory
    """
    """ load data"""
    # nn_actions = np.load('./position/nn_trajectory_action_0.npy')
    # # traj_mus = np.load('./position/good_trajectory_mu_0.npy')
    # traj_mus = np.load('./position/step_mu_6.npy')
    # traj_mus = np.mean(traj_mus, axis=0)
    # traj_obs = np.load('./position/good_trajectory_obs_0.npy')
    # traj_K = np.load('./position/good_trajectory_K_0.npy')
    # traj_k = np.load('./position/good_trajectory_k_0.npy')
    # traj_covar = np.load('./position/good_trajectory_covar_0.npy')
    #
    # nn_actions = np.load('./position/ori_save_nn_mu.npy')
    # traj_mus = np.load('./position/ori_save_mu.npy')
    #
    # import os.path
    # step_idx = 0
    # while True:
    #     step_idx = step_idx + 1
    #     action_file = './position/step_mu_%d.npy' % step_idx
    #     if os.path.exists(action_file):
    #         traj_mu = np.load(action_file)
    #         traj_mu = traj_mu[0, :, :]
    #         traj_mu = np.expand_dims(traj_mu, axis=0)
    #         if step_idx == 1:
    #             traj_mus_mean = traj_mu
    #         else:
    #             traj_mus_mean = np.concatenate((traj_mus_mean, traj_mu), axis=0)
    #     else:
    #         break
    # traj_mus_mean = np.mean(traj_mus_mean, axis=0)
    #
    # """ calculate u = K*x + k"""
    # traj_actions = np.zeros(0)
    # for i in range(traj_k.shape[0]):
    #     action = traj_K[i].dot(traj_obs[i]) + traj_k[i]
    #     action = np.expand_dims(action, axis=0)
    #     if i == 0:
    #         traj_actions = action
    #     else:
    #         traj_actions = np.concatenate((traj_actions, action), axis=0)
    #
    # """ concatenate the action in [action, step]"""
    # plot_action = []
    # assert traj_actions.shape == nn_actions.shape
    # for action_idx in range(traj_actions.shape[1]):
    #
    #     nn_action = nn_actions[:, action_idx]
    #     nn_action = np.expand_dims(nn_action, axis=0)
    #
    #     traj_action = traj_actions[:, action_idx]
    #     traj_action = np.expand_dims(traj_action, axis=0)
    #
    #     traj_mu = traj_mus[:, action_idx]
    #     traj_mu = np.expand_dims(traj_mu, axis=0)
    #
    #     traj_mu_mean = traj_mus_mean[:, action_idx]
    #     traj_mu_mean = np.expand_dims(traj_mu_mean, axis=0)
    #
    #     plot_action.append(np.concatenate((nn_action, traj_action, traj_mu, traj_mu_mean), axis=0))
    #
    # for action_idx in range(traj_actions.shape[1]):
    #     plt.figure(action_idx)
    #     ax = plt.gca()
    #     ax.set_xlabel('number')
    #     ax.set_xlabel('value')
    #     ax.set_title('action_%d' % action_idx)
    #     plot_weight(ax, plot_action[action_idx])
    # plt.show()

    nn_actions = np.load('./position/ori_save_nn_mu.npy')
    traj_mus = np.load('./position/ori_save_mu.npy')
    traj_mus = np.mean(traj_mus, axis=0)
    nn_actions = np.mean(nn_actions, axis=0)

    plot_action = []
    for action_idx in range(traj_mus.shape[1]):

        nn_action = nn_actions[:, action_idx]
        nn_action = np.expand_dims(nn_action, axis=0)


        traj_mu = traj_mus[:, action_idx]
        traj_mu = np.expand_dims(traj_mu, axis=0)


        plot_action.append(np.concatenate((nn_action, traj_mu), axis=0))

    for action_idx in range(traj_mus.shape[1]):
        plt.figure(action_idx)
        ax = plt.gca()
        ax.set_xlabel('number')
        ax.set_xlabel('value')
        ax.set_title('action_%d' % action_idx)
        plot_weight(ax, plot_action[action_idx])
    plt.show()




def cal_prob_nn():
    """
    calculate the probability of nn on the standard of local trajectory
    """
    """ load data"""
    nn_actions = np.load('./position/nn_trajectory_action_0.npy')
    traj_mus = np.load('./position/good_trajectory_mu_0.npy')
    traj_obs = np.load('./position/good_trajectory_obs_0.npy')
    traj_K = np.load('./position/good_trajectory_K_0.npy')
    traj_k = np.load('./position/good_trajectory_k_0.npy')
    traj_covar = np.load('./position/good_trajectory_covar_0.npy')

    """ construct normal distribution """
    time_step = traj_K.shape[0]

def gradient_plot():
    """ plot gradient using gradient_visualizing
    """
    gradient_data = data_logger.unpickle('./position/gradient.pkl')
    gradient_visualization(gradient_data)

def diff_fisher_plot():
    """
    plot different fisher of l2, l1, non
    """
    """ load data """
    fisher_data = []
    fisher_data.append(data_logger.unpickle('./position/fisher/fisher_none.pkl'))
    fisher_data.append(data_logger.unpickle('./position/fisher/fisher_l1.pkl'))
    fisher_data.append(data_logger.unpickle('./position/fisher/fisher_l2.pkl'))

    """ prepare data for plot"""
    weights_data = []
    for num_weight in range(len(fisher_data[0])):
        weight_data = np.zeros(0)
        for num_fisher in range(len(fisher_data)):
            weight_temp = fisher_data[num_fisher][num_weight].flatten()
            weight_temp = np.expand_dims(weight_temp, axis=0)
            length = len(weight_data.shape)
            if len(weight_data.shape) > 1:
                weight_data = np.concatenate((weight_data, weight_temp), axis=0)
            else:
                weight_data = weight_temp
        weights_data.append(weight_data)

    for num_weight in range(len(weights_data)):
        plt.figure(num_weight)
        ax = plt.gca()
        ax.set_xlabel('number')
        ax.set_xlabel('value')
        ax.set_title('fisher_weight_%d' % num_weight)
        plot_weight(ax, weights_data[num_weight])

    plt.show()

def plot_ocf():
    """ load data"""
    path_name = './position/result/'


    def data_prepare(data1, data2):
        """ reshape the data1 and data2"""
        data1_re = data1.reshape([data1.shape[0] / 30, 30])
        data2_re = data2.reshape([data2.shape[0] / 30, 30])
        assert data1_re.shape == data2_re.shape
        data1_mean = np.mean(data1_re, axis=1)
        data2_mean = np.mean(data2_re, axis=1)
        data_zero = np.array([0])
        data1_mean = np.concatenate((data_zero, data1_mean))
        data2_mean = np.concatenate((data_zero, data2_mean))

        data1_mean = np.expand_dims(data1_mean, axis=0)
        data2_mean = np.expand_dims(data2_mean, axis=0)

        total_data = np.concatenate((data1_mean, data2_mean), axis=0)
        return total_data

    def plot_data(total_data, label, figure_num=0, xlabel=None, ylabel=None, title=None):
        """ plot data"""
        plt.figure(figure_num)
        plt.xticks([0, 1, 2, 3,4])
        ax = plt.gca()
        ax.set_xlabel(xlabel, fontsize=40)
        ax.set_ylabel(ylabel, fontsize=80)
        # ax.set_title(title)
        ax.set_xlim(1, 4)
        ax.set_ylim(0, 0.12)
        # ax.yaxis.label.set_fontsize(50)
        plot_weight(ax, total_data, label)
        return ax

    ocf_cost = data_logger.unpickle(path_name + 'ocf_cost_3.pkl')
    noocf_cost = data_logger.unpickle(path_name + 'noocf_cost_3.pkl')
    total_cost = data_prepare(ocf_cost, noocf_cost)

    ocf_distance = data_logger.unpickle(path_name + 'ocf_distance_3.pkl')
    # while True:
    #     a = np.argmax(ocf_distance)
    #     if ocf_distance[a] > 0.06:
    #         ocf_distance[a] = 0.004
    #     else:
    #         break
    noocf_distance = data_logger.unpickle(path_name + 'noocf_distance_3.pkl')
    total_distance = data_prepare(ocf_distance, noocf_distance)
    print(ocf_distance)
    label = ['OLGPS', 'Policy1']
    ax = plot_data(total_distance, figure_num=1,
              xlabel='position', ylabel='distance to target', title='distance measurement', label=label)
    ax.axhline(0.06, linestyle='--', color='black', label='base line')
    plt.legend(loc='upper right', frameon=True)

    ocf_suc = data_logger.unpickle(path_name + 'ocf_suc_3.pkl')
    # while True:
    #     a = np.argmin(ocf_suc)
    #     if ocf_suc[a] < 0.06:
    #         ocf_suc[a] = 1
    #     else:
    #         break
    noocf_suc = data_logger.unpickle(path_name + 'noocf_suc_3.pkl')
    total_suc = data_prepare(ocf_suc, noocf_suc)

    ax = plot_data(total_suc, figure_num=2,
              xlabel='position', ylabel='sucessful precision', title='precision measurement', label=label)
    ax.axhline(1.0, linestyle='--', color='black', label='100% precision')
    plt.legend(loc='lower right', frameon=True, fontsize=50)



    plt.show()

from matplotlib.animation import FuncAnimation
def fisher_update(weight_idx):
    """ dynamic plot fisher infor for each obs
    weight_idx: the idx of weight that to plot
    """
    """ load data and prepare
    data formate: num_obs * num_weight * weight_dimension
    """
    total_fisher = data_logger.unpickle('./position/fisher/total_fisher_0.pkl')
    len_obs = len(total_fisher)

    total_fisher_re = []
    for num_obs in xrange(len_obs):
        obs_fisher = total_fisher[num_obs]
        weight_data = []
        ##### flatten the weight
        for num_weight in range(len(obs_fisher)):
            weight_temp = obs_fisher[num_weight].flatten()
            weight_data.append(weight_temp)
        total_fisher_re.append(weight_data)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'k-')

    len_plot_data = len(total_fisher_re[0][weight_idx])
    ax.set_xlim(0, len_plot_data)
    ax.set_ylim(0, 5)
    x_data = np.zeros(0)
    for num_idx in range(len_plot_data):
        x_data = np.append(x_data, num_idx)

    def line_update(i):
        """ using for update line date , the parameter is iteration"""
        if i == 0:
            line.set_data([], [])
            return line
        y_data = total_fisher_re[i][weight_idx]
        line.set_data(x_data, y_data)
        return line

    anim = FuncAnimation(fig, line_update, frames=100)
    plt.show()

def compare_ol_md():
    """ compare between olgps and mdgps through distance, cost, suc"""
    path_name = './position/result/'
    md_cost = data_logger.unpickle(path_name + 'md_all_cost.pkl')
    md_distance = data_logger.unpickle(path_name + 'md_all_distance.pkl')
    md_distance = md_distance[0]
    md_suc = data_logger.unpickle(path_name + 'md_all_suc.pkl')
    md_suc = md_suc[0]

    ol_cost = data_logger.unpickle(path_name + 'total_cost_3.pkl')
    ol_distance = data_logger.unpickle(path_name + 'total_distance_3.pkl')
    ol_suc = data_logger.unpickle(path_name + 'total_suc_3.pkl')

    print('md_cost:', np.mean(md_cost))
    print('ol_cost:', np.mean(ol_cost))
    print('md_distance:', np.mean(md_distance))
    print('ol_distance:', np.mean(ol_distance))
    print('md_suc:', np.mean(md_suc))
    print('ol_suc:', np.mean(ol_suc))

    # md_cost_re = md_cost.reshape([30, md_cost.shape[0] / 30])
    # md_distance_re = md_distance.reshape(md_cost_re.shape)
    # md_suc_re = md_suc.reshape(md_cost_re.shape)
    #
    # ol_cost_re = ol_cost.reshape(md_cost_re.shape)
    # ol_distance_re = ol_distance.reshape(md_cost_re.shape)
    # ol_suc_re = ol_suc.reshape(md_cost_re.shape)
    #
    # md_cost_mean = np.mean(md_cost_re, axis=0)
    # md_cost_mean = np.expand_dims(md_cost_mean, axis=0)
    # md_distance_mean = np.mean(md_distance_re, axis=0)
    # md_distance_mean = np.expand_dims(md_distance_mean, axis=0)
    # md_suc_mean = np.mean(md_suc_re, axis=0)
    # md_suc_mean = np.expand_dims(md_suc_mean, axis=0)
    #
    # ol_cost_mean = np.mean(ol_cost_re, axis=0)
    # ol_cost_mean = np.expand_dims(ol_cost_mean, axis=0)
    # ol_distance_mean = np.mean(ol_distance_re, axis=0)
    # ol_distance_mean = np.expand_dims(ol_distance_mean, axis=0)
    # ol_suc_mean = np.mean(ol_suc_re, axis=0)
    # ol_suc_mean = np.expand_dims(ol_suc_mean, axis=0)
    #
    # data_plot = []
    # data_plot.append(np.concatenate((ol_cost_mean, md_cost_mean), axis=0))
    # data_plot.append(np.concatenate((ol_distance_mean, md_distance_mean), axis=0))
    # data_plot.append(np.concatenate((ol_suc_mean, md_suc_mean), axis=0))
    #
    # def plot_data(total_data, label, figure_num=0, xlabel=None, ylabel=None, title=None):
    #     """ plot data"""
    #     plt.figure(figure_num)
    #     ax = plt.gca()
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(title)
    #     plot_weight(ax, total_data, label)
    #
    # label = ['OLGPS', 'MDGPS']
    # y_label = ['cost', 'distance to target', 'pegging precision']
    # title = 'comparison of OLGPS and MDGPS'
    # for i in range(len(data_plot)):
    #     plot_data(data_plot[i], label,
    #               figure_num=i,
    #               xlabel='position',
    #               ylabel=y_label[i],
    #               title=title)
    # plt.show()

def compare_fisher_visualization():
    """
    compare different fisher info of different task
    fisher_info: num_task * num_weight
    Returns:

    """

    """ reshape the total fisher information"""
    total_task = 2
    # total_fisher = data_logger.unpickle('./position/fisher/fisher_info_%d.pkl' % total_task)
    total_fisher = data_logger.unpickle('./position/fisher/fisher_for_compare_%d.pkl' % total_task)

    def compute_frechet_distance(fisher_info):
        """ compute frechet distance for fisher_info
        fisher_info has one for bas vector and other comparative vector
        """
        from sklearn.preprocessing import normalize as normalize
        assert len(fisher_info) > 1
        fisher_info_re = []
        for idx_fisher in range(len(fisher_info)):
            fisher_array = np.zeros(0)
            for v in range(len(fisher_info[0])):
                #TODO should add bias?
                array_temp = fisher_info[idx_fisher][v].flatten()
                fisher_array = np.concatenate((fisher_array, array_temp), axis=0)
            fisher_info_re.append(fisher_array)

        fisher_base = fisher_info_re[0]
        F1 = normalize(fisher_base) ** 2
        F1 = F1[0]
        print('F1 mean:', np.sum(F1))
        F_distance_total = []
        for i in range(1, len(fisher_info_re)):
            fisher_com = fisher_info_re[i]
            F2 = normalize(fisher_com) ** 2
            F2 = F2[0]
            F_distance = 0.5 * np.sum(F1 + F2 - 2*np.sqrt(F1*F2))
            # F_distance = 0
            # for v in range(len(fisher_base)):
            #     F1 = fisher_info[i][v].flatten()
            #     F2 = fisher_base[v].flatten()
            #     F_sqrt = 2 * np.sqrt(F1*F2)
            #     F_distance += 0.5 * np.sum((F1 + F2 - F_sqrt))
            F_distance_total.append(F_distance)
        return F_distance_total

    F_distance = compute_frechet_distance(total_fisher)
    print('F_distance:', F_distance)



    total_fisher_re = [] # [num_weight] * [num_task]
    for idx_weight in range(len(total_fisher[0])):
        weight_fisher = np.zeros(0)
        for idx_task in range(len(total_fisher)):
            weight = total_fisher[idx_task][idx_weight].flatten()
            weight = np.expand_dims(weight, axis=0)
            if idx_task == 0:
                weight_fisher = weight
            else:
                weight_fisher = np.concatenate((weight_fisher, weight), axis=0)
        total_fisher_re.append(weight_fisher)

    label = []
    for idx_num in range(len(total_fisher)):
        name = 'task_%d' % idx_num
        label.append(name)

    for idx_weight in range(len(total_fisher_re)):
        plt.figure(idx_weight)
        plot_multi_ax(total_fisher_re[idx_weight],
                      total_fisher_re[idx_weight].shape[0],
                      label=label)

        # ax = plt.gca()
        # ax.set_xlabel('number')
        # ax.set_ylabel('fisher value')
        # ax.set_title('fisher of weight_%d' % idx_weight)
        # plot_weight(ax, total_fisher_re[idx_weight], label)

    plt.show()

def compare_fisher(with_ewc=True):
    """
    compare fisher between ewc and no ewc
    :return:
    """
    def data_prepare(fisher):
        """
        :param fisher: num_task * layers * np.array(layer_weights)
        :return: layers * np.array(num_task * layer_weights)
        """
        total_weights = []
        for layer in range(len(fisher[0])):
            weights = None
            for num_task in range(len(fisher)):
                weight = fisher[num_task][layer].flatten()
                weight = np.expand_dims(weight, axis=0)
                if num_task == 0:
                    weights = weight
                else:
                    weights = np.concatenate((weights, weight), axis=0)
            total_weights.append(weights)
        return total_weights

    if with_ewc:
        path_name = '/home/sun/work/ocfgps/position/compare/fisher_info_ewc'
    else:
        path_name = '/home/sun/work/ocfgps/position/compare/fisher_info'
    fisher_ewc = data_logger.unpickle(path_name + '_3.pkl')
    # fisher_no = data_logger.unpickle(path_name + 'total_fisher_3.pkl')

    fisher_ewc = data_prepare(fisher_ewc)
    # fisher_no = data_prepare(fisher_no)

    label = []
    for idx_num in range(len(fisher_ewc)):
        name = 'task_%d' % idx_num
        label.append(name)

    for idx_weight in range(len(fisher_ewc)):
        plt.figure(idx_weight)
        plot_multi_ax(fisher_ewc[idx_weight],
                      fisher_ewc[idx_weight].shape[0],
                      label=label)

    plt.show()

def compare_fisher_sub(with_ewc):
    """
    compare fisher between ewc and no ewc
    :return:
    """
    def data_prepare(fisher):
        """
        :param fisher: num_task * layers * np.array(layer_weights)
        :return: layers * np.array(num_task-1 * layer_weights)
        """
        total_diffs = []
        for layer in range(len(fisher[0])):
            diffs = None
            base = fisher[-1][layer].flatten()
            diff = None
            for num_task in range(len(fisher)):
                weight = fisher[num_task][layer].flatten()
                if num_task == len(fisher)-1:
                    base = weight
                    continue
                else:
                    diff = (weight - base)
                    # base = weight
                diff = np.expand_dims(diff, axis=0)
                if num_task == 0:
                    diffs = diff
                else:
                    diffs = np.concatenate((diffs, diff), axis=0)
            total_diffs.append(diffs)
        return total_diffs

    if with_ewc:
        path_name = '/home/sun/work/ocfgps/position/compare/fisher_info_ewc'
    else:
        path_name = '/home/sun/work/ocfgps/position/compare/fisher_info'
    # fisher_no = data_logger.unpickle(path_name + 'total_fisher_3.pkl')

    fisher_ewc = data_logger.unpickle(path_name + '_3.pkl')
    fisher_ewc = data_prepare(fisher_ewc)
    # fisher_no = data_prepare(fisher_no)

    label = []
    for idx_num in range(len(fisher_ewc)):
        name = 'task_%d' % idx_num
        label.append(name)

    for idx_weight in range(len(fisher_ewc)):
        plt.figure(idx_weight)
        plot_multi_ax(fisher_ewc[idx_weight],
                      fisher_ewc[idx_weight].shape[0],
                      label=label)

    plt.show()

def compare_weight_sub(with_ewc=True):
    """
    comare weight between ewc and no ewc
    :param with_ewc:
    :return:
    """
    if with_ewc:
        path_name = '/home/sun/work/ocfgps/position/compare/var_lists_ewc'
    else:
        path_name = '/home/sun/work/ocfgps/position/compare/var_lists'
    all_weibhts = []
    for idx in range(0, 4):
        all_weibhts.append(data_logger.unpickle(path_name + '_%d.pkl' % idx))

    def data_prepare(fisher):
        """
        :param fisher: num_task * layers * np.array(layer_weights)
        :return: layers * np.array(num_task-1 * layer_weights)
        """
        total_diffs = []
        for layer in range(len(fisher[0])):
            diffs = None
            # base = None
            base = fisher[-1][layer].flatten()
            diff = None
            for num_task in range(len(fisher)):
                weight = fisher[num_task][layer].flatten()
                if num_task == len(fisher)-1:
                    # base = weight
                    continue
                else:
                    diff = (weight - base)
                    # base = weight
                diff = np.expand_dims(diff, axis=0)
                if num_task == 0:
                    diffs = diff
                else:
                    diffs = np.concatenate((diffs, diff), axis=0)
            total_diffs.append(diffs)
        return total_diffs

    fisher_ewc = data_prepare(all_weibhts)

    label = []
    for idx_num in range(len(fisher_ewc)):
        name = 'task_%d' % idx_num
        label.append(name)

    for idx_weight in range(len(fisher_ewc)):
        plt.figure(idx_weight)
        plot_multi_ax(fisher_ewc[idx_weight],
                      fisher_ewc[idx_weight].shape[0],
                      label=label)

    plt.show()

def compare_weight(with_ewc=True):
    """
    comare weight between ewc and no ewc
    :param with_ewc:
    :return:
    """
    if with_ewc:
        path_name = '/home/sun/work/ocfgps/position/compare/var_lists_ewc'
    else:
        path_name = '/home/sun/work/ocfgps/position/compare/var_lists'
    all_weibhts = []
    for idx in range(0, 4):
        all_weibhts.append(data_logger.unpickle(path_name + '_%d.pkl' % idx))

    def data_prepare(fisher):
        """
        :param fisher: num_task * layers * np.array(layer_weights)
        :return: layers * np.array(num_task * layer_weights)
        """
        total_weights = []
        for layer in range(len(fisher[0])):
            weights = None
            for num_task in range(len(fisher)):
                weight = fisher[num_task][layer].flatten()
                weight = np.expand_dims(weight, axis=0)
                if num_task == 0:
                    weights = weight
                else:
                    weights = np.concatenate((weights, weight), axis=0)
            total_weights.append(weights)
        return total_weights

    fisher_ewc = data_prepare(all_weibhts)

    label = []
    for idx_num in range(len(fisher_ewc)):
        name = 'task_%d' % idx_num
        label.append(name)

    for idx_weight in range(len(fisher_ewc)):
        plt.figure(idx_weight)
        plot_multi_ax(fisher_ewc[idx_weight],
                      fisher_ewc[idx_weight].shape[0],
                      label=label)

    plt.show()




# diff_fisher_plot()
# fisher_update(2)
import matplotlib
matplotlib.rcParams.update({'font.size': 20})


# plot_ocf()
# compare_ol_md()

# compare_fisher_visualization()

# compare_fisher(with_ewc=False)
# compare_fisher_sub(with_ewc=True)

# compare_weight(with_ewc=False)
compare_weight_sub(with_ewc=False)






