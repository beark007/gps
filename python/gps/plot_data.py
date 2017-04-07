from __future__ import division, print_function
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from gps.utility.data_logger import DataLogger


mpl.rc('figure', figsize=(14, 7))
mpl.rc('font', size=14)
mpl.rc('axes', grid=False)
mpl.rc('axes', facecolor='white')

"""
================    ===============================
        character           description
        ================    ===============================
        ``'-'``             solid line style
        ``'--'``            dashed line style
        ``'-.'``            dash-dot line style
        ``':'``             dotted line style
        ``'.'``             point marker
        ``','``             pixel marker
        ``'o'``             circle marker
        ``'v'``             triangle_down marker
        ``'^'``             triangle_up marker
        ``'<'``             triangle_left marker
        ``'>'``             triangle_right marker
        ``'1'``             tri_down marker
        ``'2'``             tri_up marker
        ``'3'``             tri_left marker
        ``'4'``             tri_right marker
        ``'s'``             square marker
        ``'p'``             pentagon marker
        ``'*'``             star marker
        ``'h'``             hexagon1 marker
        ``'H'``             hexagon2 marker
        ``'+'``             plus marker
        ``'x'``             x marker
        ``'D'``             diamond marker
        ``'d'``             thin_diamond marker
        ``'|'``             vline marker
        ``'_'``             hline marker
        ================    ===============================


        The following color abbreviations are supported:

        ==========  ========
        character   color
        ==========  ========
        'b'         blue
        'g'         green
        'r'         red
        'c'         cyan
        'm'         magenta
        'y'         yellow
        'k'         black
        'w'         white
        ==========  ========
"""
def plotline(x_data, x_label, y1_data, y_label, y2_data=None, y3_data=None, y4_data=None, y5_data=None,
             y6_data=None, y7_data=None, title=None):
    _, ax = plt.subplots()
    #ax.scatter(x_data, y_data, s=30, color='#539caf', alpha=0.75)
    """
    ax.plot(x_data, y1_data, '-o', lw=2, color='blue', alpha=0.75, label='itr 6')
    ax.plot(x_data, y2_data, '-<', lw=2, color='green', alpha=0.75, label='itr 8')
    ax.plot(x_data, y3_data, '-^', lw=2, color='red', alpha=0.75, label='itr 9')
    ax.plot(x_data, y4_data, '-s', lw=2, color='black', alpha=0.75, label='itr 10')
    """
    """MD and OL 4,5,6"""
    # ax.plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='md')
    # ax.plot(x_data, y2_data, linestyle='-', marker='<', markersize=15, lw=2, color='green', alpha=0.75, label='ol_pos4')
    # ax.plot(x_data, y3_data, linestyle='-', marker='s', markersize=15, lw=2, color='red', alpha=0.75, label='ol_pos5')
    # ax.plot(x_data, y4_data, linestyle='-', marker='p', markersize=15, lw=2, color='black', alpha=0.75, label='ol_pos6')
    # ax.plot(x_data, y5_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')

    """OL sequence"""
    ax.plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='ol_pos4')
    ax.plot(x_data, y2_data, linestyle='-', marker='<', markersize=15, lw=2, color='green', alpha=0.75, label='ol_pos5')
    ax.plot(x_data, y3_data, linestyle='-', marker='s', markersize=15, lw=2, color='red', alpha=0.75, label='ol_pos6')
    # ax.plot(x_data, y4_data, linestyle='-', marker='p', markersize=15, lw=2, color='black', alpha=0.75, label='ol_pos6')
    ax.plot(x_data, y4_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')
    # ax.plot(x_data, y5_data, linestyle='--', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='ol_pos4_without')
    # ax.plot(x_data, y6_data, linestyle='--', marker='<', markersize=15, lw=2, color='green', alpha=0.75, label='ol_pos5_without')
    # ax.plot(x_data, y7_data, linestyle='--', marker='s', markersize=15, lw=2, color='red', alpha=0.75, label='ol_pos6_without')

    """ MD vs OL"""
    # ax.plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='MDGPS')
    # ax.plot(x_data, y2_data, linestyle='-', marker='<', markersize=15, lw=2, color='green', alpha=0.75, label='OLGPS')

    """md and extend to 5"""
    # ax.plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='train_area0')
    # ax.plot(x_data, y2_data, linestyle='-', marker='<', markersize=15, lw=2, color='green', alpha=0.75, label='train_extra1')
    # ax.plot(x_data, y3_data, linestyle='-', marker='s', markersize=15, lw=2, color='red', alpha=0.75, label='train_extra2')
    # ax.plot(x_data, y5_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')

    """ol incremental"""
    # ax.plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='with_alpha')
    # ax.plot(x_data, y2_data, linestyle='-', marker='o', markersize=15, lw=2, color='green', alpha=0.75, label='without_alpha')
    # ax.plot(x_data, y3_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')

    plt.legend(loc='upper left', frameon=True)
    # ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plotCompareLine(x_data, x_label, y1_data, y_label, y2_data=None, y3_data=None, y4_data=None, y5_data=None,
             y6_data=None, y7_data=None, title=None):
    """ compare line in the same condition"""
    _, ax = plt.subplots(3)
    ax[0].plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, lw=2, color='blue', alpha=0.75, label='ol_pos4')
    ax[0].plot(x_data, y4_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')
    ax[0].plot(x_data, y5_data, linestyle='--', marker='o', markersize=15, lw=2, color='blue', alpha=0.75,
            label='ol_pos4_without')
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)

    # _, ax = plt.subplots(2)
    ax[1].plot(x_data, y2_data, linestyle='-', marker='o', markersize=15, lw=2, color='green', alpha=0.75, label='ol_pos5')
    ax[1].plot(x_data, y4_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')
    ax[1].plot(x_data, y6_data, linestyle='--', marker='o', markersize=15, lw=2, color='green', alpha=0.75,
            label='ol_pos5_without')
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)

    # _, ax = plt.subplots(3)
    ax[2].plot(x_data, y3_data, linestyle='-', marker='o', markersize=15, lw=2, color='red', alpha=0.75, label='ol_pos6')
    ax[2].plot(x_data, y4_data, linestyle='--', lw=2, color='black', alpha=0.75, label='base line')
    ax[2].plot(x_data, y7_data, linestyle='--', marker='o', markersize=15, lw=2, color='red', alpha=0.75,
            label='ol_pos6_without')
    ax[2].set_xlabel(x_label)
    ax[2].set_ylabel(y_label)

    # plt.legend(loc='upper left', frameon=True)



def plotDoubleLine(x_data, y1_data, y2_data, y3_data, z1_data, z2_data, z3_data, base_line, xlabel, ylabel, zlabel):
    import mpl_toolkits.axisartist as AA
    from mpl_toolkits.axes_grid1 import host_subplot
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()

    # host.set_xlim(0, 4)
    # host.set_ylim(0, 8)

    host.set_xlabel(xlabel)
    host.set_ylabel(ylabel)
    par1.set_ylabel(zlabel)

    # p1, = host.plot(x_data, y1_data,label=xlabel)
    # p2, = host.plot([1, 2, 3, 4], [1, 2, 3, 4], label="DB")
    # p3, = par1.plot([1, 2, 3, 4], [100, 200, 300, 400], label="DC")
    host.plot(x_data, y1_data, linestyle='-', marker='o', markersize=15, color='blue', label='distance_train0')
    host.plot(x_data, y2_data, linestyle='-', marker='<', markersize=15, color='green', label='distance_train1')
    host.plot(x_data, y3_data, linestyle='-', marker='s', markersize=15, color='red', label='distance_train2')
    host.plot(x_data, base_line, linestyle='--', lw=2, color='black', label='base_line')

    par1.plot(x_data, z1_data, linestyle='--', lw=4, label='cost_train0')
    par1.plot(x_data, z2_data, linestyle='--', lw=4, label='cost_train1')
    par1.plot(x_data, z3_data, linestyle='--', lw=4, label='cost_train2')


    host.legend()
    plt.legend(loc='upper left', frameon=True)

    plt.draw()
    plt.show()

def plotpoint(x_data, y1_data, y2_data, x_label=None, y_label=None, title=None, color='blue'):
    _, ax = plt.subplots()
    #ax.scatter(x_data, y_data, s=30, color='#539caf', alpha=0.75)
    ax.scatter(x_data, y1_data, marker='o', s=30, color=color, alpha=0.75)
    ax.scatter(x_data, y2_data, marker='s', s=30, color=color, alpha=0.75)
    #ax.plot(x_data, y_data, lw=1, color='blue', alpha=0.75, linestyle='-', label='init')
    # ax.plot(x_data, yy_data, lw=1, color='green', alpha=0.75, linestyle='-', label='change')
    plt.legend(loc='upper left', frameon=False)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plotAgentPostiion():
    data_logger = DataLogger()
    position = data_logger.unpickle('./position/7/9/train_position.pkl')
    print('position:', position)
    _, ax = plt.subplots()


    # plot circle
    from matplotlib.patches import Ellipse
    center_position = 0.02
    radius = 0.02
    all_num_pos = 6
    circle_label = 'center of testing area'
    color = 'black'
    for pos_count in range(all_num_pos):
        if pos_count == 2:
            color = 'green'
        elif pos_count == 4:
            color = 'red'
        else:
            color = 'black'
        circle_position = np.array([center_position, -center_position])
        circle = Ellipse(xy=circle_position, width=2*radius, height=2*radius, angle=0, fill=False, label='train circle')

        circle.set_clip_box(ax.bbox)
        # circle.set_facecolor(color=color)
        circle.set_linewidth(5)
        circle.set_edgecolor(color=color)
        circle.set_alpha(0.5)
        circle.set_label('train circle')
        circle.set_alpha(1)
        # circle.set_linestyle('-')
        ax.add_artist(circle)
        if pos_count == 0:
            ax.scatter(circle_position[0], circle_position[1], marker='1', color='black', s=200, label=circle_label)
            circle_label = None
        else:
            ax.scatter(circle_position[0], circle_position[1], marker='1', color='black', s=200, label=circle_label)
        center_position = center_position + radius * 2

    marker = 'D'
    color = 'blue'
    label = 'initial four position'
    for i in range(position.shape[0]):
        if i == 0:
            label = 'initial four positions'
        elif i == 4:
            label = 'extra training positions'
        else:
            label = None
        if i <= position.shape[0] - 5:
            color = 'blue'
            marker = 's'
        # if i == position.shape[0] - 2:
        else:
            color = 'm'
            marker = 'p'
        ax.scatter(position[i][0], position[i][1], marker=marker, color=color, s=50, label=label)
    ax.scatter(0, 0, marker='o', color='black', s=200, label='origin')



    plt.legend(loc='upper left', frameon=True)
    # ax.set_title('agent ')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-0.2, 0.24)
    ax.set_ylim(-0.24, 0.2)
    plt.show()

def plotMDTest():
    data_logger = DataLogger()
    position = data_logger.unpickle('./position/9/train_position_1.pkl')
    print(position)
    _, ax = plt.subplots()
    label = 'initial train positions'
    for i in range(position.shape[0]):
        if i == 0:
            ax.scatter(position[i][0], position[i][1], marker='s', color='blue', s=100, label=label)
            label = None
        else:
            ax.scatter(position[i][0], position[i][1], marker='s', color='blue', s=100, label=label)

    from matplotlib.patches import Ellipse
    center_position = np.mean(position, axis=0)
    radius = 0.05
    radius_grow = 0.025
    for i in range(7):
        ell = Ellipse(xy=center_position, width=radius*2, height=radius*2, angle=0, fill=False)
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_facecolor(color='black')
        radius = radius + radius_grow
        print(radius)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(-0.4, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.legend(loc='upper left', frameon=True)
    plt.show()

def plotcost():
    data_logger = DataLogger()
    cost_ol = data_logger.unpickle('./position/5/cost_ol.pkl')
    #with_alpha_costs = np.delete(with_alpha_costs, [10, 11, 12])
    cost_ol_alpha = data_logger.unpickle('./position/5/cost_ol_alpha.pkl')
    cost_ol_alpha_step = data_logger.unpickle('./position/5/cost_ol_alpha_step.pkl')
    cost_md = data_logger.unpickle('./position/5/md_test_costs.pkl')
    #with_alpha_step_costs = data_logger.unpickle('./position/ol_with_alpha_step_costs.pkl')
    #with_alpha_step_costs = np.delete(with_alpha_step_costs, 4)
    print(cost_ol.shape[0])
    for i in range(0, cost_ol.shape[0]):
        if cost_ol[i] > -200:
            cost_ol[i] = -200
            #cost_ol = np.delete(cost_ol, i)
    for i in range(0, cost_ol_alpha.shape[0]):
        if cost_ol_alpha[i] > -200:
            #cost_ol_alpha = np.delete(cost_ol_alpha, i)
            cost_ol_alpha[i] = -200
    for i in range(0, cost_ol_alpha_step.shape[0]):
        if cost_ol_alpha_step[i] > -200:
            #cost_ol_alpha_step = np.delete(cost_ol_alpha_step, i)
            cost_ol_alpha_step[i] = -200
    for i in range(0, cost_md.shape[0]):
        if cost_md[i] > -200:
            #cost_md = np.delete(cost_md, i)
            cost_md[i] = -200

    """ construct x axis"""
    num_positions = np.zeros(0)
    #max_len = min(with_alpha_costs.shape[0], without_alpha_costs.shape[0], md_costs.shape[0], with_alpha_step_costs.shape[0])
    min_len = min(cost_ol.shape[0], cost_ol_alpha.shape[0], cost_ol_alpha_step.shape[0], cost_md.shape[0])
    print('len: %d' % min_len)
    for i in range(min_len):
        num_positions = np.append(num_positions, np.array(i))
    cost_ol = cost_ol[:min_len]
    cost_ol_alpha = cost_ol_alpha[:min_len]
    cost_ol_alpha_step = cost_ol_alpha_step[:min_len]
    cost_md = cost_md[:min_len]

    plotline(x_data=num_positions,
             y1_data=cost_ol,
             y2_data=cost_ol_alpha,
             y3_data=cost_ol_alpha_step,
             y4_data=cost_md,
             x_label='num of position',
             y_label='cost',
             title='compare')

    plt.show()

def plotposition():
    data_logger = DataLogger()
    position = data_logger.unpickle('./position/train_position.pkl')
    print(position)
    plotpoint(x_data=position[:, 0],
             y_data=position[:, 1],
             x_label='position x',
             y_label='position y',
             title='position')
    plt.show()

def plotPositionStep():
    data_logger = DataLogger()
    all_position = data_logger.unpickle('./position/6/train_position.pkl')
    color = 'blue'
    for i in range(all_position.shape[0]):
        cur_position_x = all_position[0:i+1, 0]
        cur_position_y = all_position[0:i+1, 1]
        print(cur_position_x)
        print(cur_position_y)
        if color=='blue':
            color = 'red'
            plotpoint(x_data=cur_position_x,
                      y_data=cur_position_y,
                      x_label='position x',
                      y_label='position y',
                      title='position',
                      color=color)

        else:
            color = 'blue'
            plotpoint(x_data=cur_position_x,
                      y_data=cur_position_y,
                      x_label='position x',
                      y_label='position y',
                      title='position',
                      color=color)
        plt.show()
        temp_char = raw_input()
        plt.close(1)

def plotCountSuc():
    data_logger = DataLogger()
    #count_suc = data_logger.unpickle('./position/1/position_ol_alpha_count_step_5.pkl')
    count_suc = data_logger.unpickle('./position/1/position_md.pkl')
    rate_suc = np.sum(count_suc, axis=1)/count_suc.shape[1]
    print(rate_suc)
    x_date = np.zeros(0)
    for i in range(rate_suc.shape[0]):
        x_data = np.concatenate((x_data, np.array([i])))
    plotline(x_data=x_data,
             y1_data=rate_suc,
             y2_data=rate_suc,
             y3_data=rate_suc,
             y4_data=rate_suc,
             x_label='rate',
             y_label='condition',
             title='successful rate')
    plt.show()

def plotCountSucAll():
    """
    plot varies of successful rate
    """
    data_logger = DataLogger()
    count_suc1 = data_logger.unpickle('./position/1/position_ol_alpha_count_5.pkl')
    count_suc1 = data_logger.unpickle('./position/2/position_ol_alpha_count_5.pkl')
    count_suc1 = data_logger.unpickle('./position/3/position_ol_alpha_count_5.pkl')
    rate_suc1 = np.sum(count_suc1, axis=1)/count_suc1.shape[1]
    print(rate_suc1)

    count_suc2 = data_logger.unpickle('./position/2/position_ol_alpha_count_7.pkl')
    rate_suc2 = np.sum(count_suc2, axis=1)/count_suc2.shape[1]
    print(rate_suc2)

    count_suc3 = data_logger.unpickle('./position/2/position_ol_alpha_count_8.pkl')
    rate_suc3 = np.sum(count_suc3, axis=1)/count_suc3.shape[1]
    print(rate_suc3)

    count_suc4 = data_logger.unpickle('./position/2/position_ol_alpha_count_9.pkl')
    rate_suc4 = np.sum(count_suc4, axis=1)/count_suc4.shape[1]
    print(rate_suc4)

    min_len = min(count_suc1.shape[0], count_suc2.shape[0], count_suc3.shape[0], count_suc4.shape[0])
    x_data = np.zeros(0)
    for i in range(min_len):
        x_data = np.concatenate((x_data, np.array([i])))

    rate_suc1 = rate_suc1[:min_len]
    rate_suc2 = rate_suc2[:min_len]
    rate_suc3 = rate_suc3[:min_len]
    rate_suc4 = rate_suc4[:min_len]

    plotline(x_data=x_data,
             y1_data=rate_suc1,
             y2_data=rate_suc2,
             y3_data=rate_suc3,
             y4_data=rate_suc4,
             x_label='condition',
             y_label='rate',
             title='successful rate')
    plt.show()

def plotSucDistance():
    data_logger = DataLogger()
    position = data_logger.unpickle('./position/6/train_position.pkl')
    print('position:', position)
    # exper = 2
    # cond = 4
    # director = 7
    # distance = data_logger.unpickle('./position/%d/experiment_%d/all_distance_%d.pkl'
    #                                 % (director-1, exper-1, cond))
    # mean_distance1 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/experiment_%d/all_distance_%d.pkl'
    #                                 % (director, exper, cond))
    # mean_distance2 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/experiment_%d/all_distance_%d.pkl'
    #                                 % (director, exper, cond+2))
    # mean_distance3 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/experiment_%d/all_distance_%d.pkl'
    #                                 % (director, exper, cond+4))
    # mean_distance4 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/experiment_%d/all_distance_%d.pkl'
    #                                 % (director, exper, cond))
    # mean_distance5 = np.mean(distance, axis=1)
    #
    # distance = data_logger.unpickle('./position/%d/experiment_%d/all_distance_%d.pkl'
    #                                 % (director - 1, exper - 1, cond))
    # """
    # plot compare ol and md at 4 5 6
    # """
    # exper = 9
    # cond = 6
    # director = 7
    # distance = data_logger.unpickle('./position/6/experiment_1/all_distance_4.pkl')
    # mean_distance1 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/%d/all_cost_%d.pkl'
    #                                 % (director, exper, cond))
    # mean_distance2 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/%d/all_cost_%d.pkl'
    #                                 % (director, exper, cond + 1))
    # mean_distance3 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/%d/%d/all_cost_%d.pkl'
    #                                 % (director, exper, cond + 2))
    # mean_distance4 = np.mean(distance, axis=1)
    #
    # x_data = np.zeros(0)
    # base_data = np.zeros(0)
    # test_pos_step = 0.02
    # test_pos = test_pos_step
    # for i in range(mean_distance2.shape[0]):
    #     x_data = np.concatenate((x_data, np.array([test_pos])))
    #     base_data = np.concatenate((base_data, np.array([0.06])))
    #     test_pos = test_pos + test_pos_step * 2
    # #print(distance)
    # print(x_data.shape)
    # print(distance.shape)
    # mean_distance2[mean_distance2.shape[0]-2] = 0
    # mean_distance2[mean_distance2.shape[0]-1] = 0
    # plotline(x_data=x_data,
    #          y1_data=mean_distance1,
    #          y2_data=mean_distance2,
    #          y3_data=mean_distance3,
    #          y4_data=mean_distance4,
    #          y5_data=base_data,
    #          x_label='Test Distance',
    #          y_label='Distance to Target')
    # plt.show()

    """
    plot md 4 5 7
    """
    # exper = 9
    # cond = 4
    # director = 7
    # distance = data_logger.unpickle('./position/6/experiment_1/all_distance_4.pkl')
    # mean_distance1 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/6/experiment_1/all_distance_5.pkl')
    # mean_distance2 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/6/experiment_1/all_distance_7.pkl')
    # mean_distance3 = np.mean(distance, axis=1)

    """plot ol sequence"""
    # distance = data_logger.unpickle('./position/3/all_distance_1.pkl')
    # mean_distance1 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/3/all_distance_2.pkl')
    # mean_distance2 = np.mean(distance, axis=1)
    # distance = data_logger.unpickle('./position/3/all_distance_3.pkl')
    # mean_distance3 = np.mean(distance, axis=1)

    # train_position = data_logger.unpickle("./position/test_4_position/3/all_train_position.pkl")

    for i in range(1):
        file_name1 = "./position/ol_ability_%d/all_distance_%d.pkl" % (i+1, 1)
        file_name2 = "./position/ol_ability_%d/all_distance_%d.pkl" % (i+1, 2)
        file_name3 = "./position/ol_ability_%d/all_distance_%d.pkl" % (i+1, 3)
        distance1 = data_logger.unpickle(file_name1)
        distance2 = data_logger.unpickle(file_name2)
        distance3 = data_logger.unpickle(file_name3)
        if i == 0:
            all_distance1 = distance1
            all_distance2 = distance2
            all_distance3 = distance3
        else:
            all_distance1 = np.concatenate((all_distance1, distance1), axis=1)
            all_distance2 = np.concatenate((all_distance2, distance2), axis=1)
            all_distance3 = np.concatenate((all_distance3, distance3), axis=1)
    for i in range(all_distance3.shape[0]):
        print("----------------------------")
        print(all_distance3[i])
    mean_distance1 = np.mean(all_distance1, axis=1)
    mean_distance2 = np.mean(all_distance2, axis=1)
    mean_distance3 = np.mean(all_distance3, axis=1)

    # without alpha
    # for i in range(3):
    #     file_name1 = "./position/ol_ability_without_%d/all_distance_%d.pkl" % (i + 1, 1)
    #     file_name2 = "./position/ol_ability_without_%d/all_distance_%d.pkl" % (i + 1, 2)
    #     file_name3 = "./position/ol_ability_without_%d/all_distance_%d.pkl" % (i + 1, 3)
    #     distance1 = data_logger.unpickle(file_name1)
    #     distance2 = data_logger.unpickle(file_name2)
    #     distance3 = data_logger.unpickle(file_name3)
    #     if i == 0:
    #         all_distance1 = distance1
    #         all_distance2 = distance2
    #         all_distance3 = distance3
    #     else:
    #         all_distance1 = np.concatenate((all_distance1, distance1), axis=1)
    #         all_distance2 = np.concatenate((all_distance2, distance2), axis=1)
    #         all_distance3 = np.concatenate((all_distance3, distance3), axis=1)
    # mean_distance11 = np.mean(all_distance1, axis=1)
    # mean_distance22 = np.mean(all_distance2, axis=1)
    # mean_distance33 = np.mean(all_distance3, axis=1)

    # file_name = './position/all_distance_1.pkl'
    # distance = data_logger.unpickle(file_name)
    # print(distance)
    # distance = np.reshape(distance, (6, distance.shape[0]/6))
    # mean_distance2 = np.mean(distance, axis=1)



    x_data = np.zeros(0)
    base_data = np.zeros(0)
    test_pos_step = 0.02
    test_pos = test_pos_step
    for i in range(mean_distance2.shape[0]):
        x_data = np.concatenate((x_data, np.array([test_pos])))
        base_data = np.concatenate((base_data, np.array([0.06])))
        test_pos = test_pos + test_pos_step * 2
    #print(distance)
    print(x_data.shape)
    # plotline(x_data=x_data,
    #          y1_data=mean_distance1,
    #          y2_data=mean_distance2,
    #          y3_data=mean_distance3,
    #          y4_data=base_data,
    #          y5_data=mean_distance11,
    #          y6_data=mean_distance22,
    #          y7_data=mean_distance33,
    #          x_label='Testing Distance',
    #          y_label='Distance to Target')

    # plotCompareLine(x_data=x_data,
    #          y1_data=mean_distance1,
    #          y2_data=mean_distance2,
    #          y3_data=mean_distance3,
    #          y4_data=base_data,
    #          y5_data=mean_distance11,
    #          y6_data=mean_distance22,
    #          y7_data=mean_distance33,
    #          x_label='Testing Distance',
    #          y_label='Distance to Target')

    plotline(x_data=x_data,
             y1_data=mean_distance1,
             y2_data=mean_distance2,
             y3_data=mean_distance3,
             y4_data=base_data,
             y5_data=mean_distance2,
             y6_data=mean_distance2,
             y7_data=mean_distance2,
             x_label='Testing Distance',
             y_label='Distance to Target')

    plt.show()

def plotTestArea():
    """
    test the area to illustrate the generalization
    Returns:

    """
    """MD vs OL"""
    # data_logger = DataLogger()
    # exper = 1
    # cond = 1
    # director = 9
    # distance1 = data_logger.unpickle('./position/%d/experiment_%d/ol_all_distance_1.pkl'
    #                                 % (director, exper))
    # ol_mean_distance_1 = np.mean(distance1, axis=1)
    # distance2 = data_logger.unpickle('./position/%d/experiment_%d/ol_all_distance_2.pkl'
    #                                  % (director, exper))
    # ol_mean_distance_2 = np.mean(distance2, axis=1)
    # distance2 = np.concatenate((distance2, distance1), axis=1)
    # distance3 = data_logger.unpickle('./position/%d/experiment_%d/ol_all_distance_3.pkl'
    #                                  % (director, exper))
    # ol_mean_distance_3 = np.mean(distance3, axis=1)
    # distance3 = np.concatenate((distance3, distance2), axis=1)
    # ol_mean_distance = np.mean(distance3, axis=1)
    # distance1 = data_logger.unpickle('./position/%d/experiment_%d/md_all_distance_1.pkl'
    #                                 % (director, exper))
    # md_mean_distance_1 = np.mean(distance1, axis=1)
    # distance2 = data_logger.unpickle('./position/%d/experiment_%d/md_all_distance_2.pkl'
    #                                  % (director, exper))
    # md_mean_distance_2 = np.mean(distance2, axis=1)
    # distance2 = np.concatenate((distance2, distance1), axis=1)
    # distance3 = data_logger.unpickle('./position/%d/experiment_%d/md_all_distance_3.pkl'
    #                                  % (director, exper))
    # md_mean_distance_3 = np.mean(distance3, axis=1)
    # distance3 = np.concatenate((distance3, distance2), axis=1)
    # md_mean_distance = np.mean(distance3, axis=1)
    # x_data = np.zeros(0)
    # base_data = np.zeros(0)
    # test_pos_step = 0.025
    # test_pos = 0.05
    # for i in range(ol_mean_distance.shape[0]):
    #     x_data = np.concatenate((x_data, np.array([test_pos])))
    #     base_data = np.concatenate((base_data, np.array([0.06])))
    #     test_pos = test_pos + test_pos_step
    # # print(distance)
    # print(x_data.shape)
    # plotline(x_data=x_data,
    #          y1_data=md_mean_distance,
    #          y2_data=ol_mean_distance,
    #          y3_data=ol_mean_distance,
    #          y4_data=ol_mean_distance,
    #          y5_data=ol_mean_distance,
    #          x_label='Testing Radius',
    #          y_label='Distance to Target')
    # plt.show()

    """
    compare mdgps and olgps
    """
    data_logger = DataLogger()
    exper = 1
    cond = 1
    director = 9
    # distance1 = data_logger.unpickle('./position/%d/experiment_%d/ol_all_distance_1.pkl'
    #                                 % (director, exper))
    # ol_mean_distance_1 = np.mean(distance1, axis=1)
    # distance2 = data_logger.unpickle('./position/%d/experiment_%d/ol_all_distance_2.pkl'
    #                                  % (director, exper))
    # ol_mean_distance_2 = np.mean(distance2, axis=1)
    # distance2 = np.concatenate((distance2, distance1), axis=1)
    # distance3 = data_logger.unpickle('./position/%d/experiment_%d/ol_all_distance_3.pkl'
    #                                  % (director, exper))
    # ol_mean_distance_3 = np.mean(distance3, axis=1)
    # distance3 = np.concatenate((distance3, distance2), axis=1)
    # ol_mean_distance = np.mean(distance3, axis=1)
    # distance1 = data_logger.unpickle('./position/%d/experiment_%d/md_all_distance_1.pkl'
    #                                 % (director, exper))
    # md_mean_distance_1 = np.mean(distance1, axis=1)
    # distance2 = data_logger.unpickle('./position/%d/experiment_%d/md_all_distance_2.pkl'
    #                                  % (director, exper))
    # md_mean_distance_2 = np.mean(distance2, axis=1)
    # distance2 = np.concatenate((distance2, distance1), axis=1)
    # distance3 = data_logger.unpickle('./position/%d/experiment_%d/md_all_distance_3.pkl'
    #                                  % (director, exper))
    # md_mean_distance_3 = np.mean(distance3, axis=1)
    # distance3 = np.concatenate((distance3, distance2), axis=1)
    # md_mean_distance = np.mean(distance3, axis=1)

    # ol_all_distance = data_logger.unpickle('./position/ol_all_distance.pkl')
    # ol_mean_distance = np.mean(ol_all_distance, axis=1)
    # md_all_distance = data_logger.unpickle('./position/md_all_distance.pkl')
    # md_mean_distance = np.mean(md_all_distance,axis=1)
    ol_distance1 = data_logger.unpickle('./position/md_ol_1/ol_all_distance.pkl')
    ol_distance2 = data_logger.unpickle('./position/md_ol_2/ol_all_distance.pkl')
    ol_distance3 = data_logger.unpickle('./position/md_ol_3/ol_all_distance.pkl')
    ol_distance4 = data_logger.unpickle('./position/md_ol_4/ol_all_distance.pkl')
    ol_distance5 = data_logger.unpickle('./position/md_ol_5/ol_all_distance.pkl')
    ol_all_distance = np.concatenate((ol_distance1, ol_distance2, ol_distance3, ol_distance4, ol_distance5), axis=1)
    ol_mean_distance = np.mean(ol_all_distance, axis=1)

    md_distance1 = data_logger.unpickle('./position/md_ol_1/md_all_distance.pkl')
    md_distance2 = data_logger.unpickle('./position/md_ol_2/md_all_distance.pkl')
    md_distance3 = data_logger.unpickle('./position/md_ol_3/md_all_distance.pkl')
    md_distance4 = data_logger.unpickle('./position/md_ol_4/md_all_distance.pkl')
    md_distance5 = data_logger.unpickle('./position/md_ol_5/md_all_distance.pkl')
    md_all_distance = np.concatenate((md_distance1, md_distance2, md_distance3, md_distance4, md_distance5), axis=1)
    md_mean_distance = np.mean(md_all_distance, axis=1)

    x_data = np.zeros(0)
    base_data = np.zeros(0)
    test_pos_step = 0.025
    test_pos = 0.05
    for i in range(ol_mean_distance.shape[0]):
        x_data = np.concatenate((x_data, np.array([test_pos])))
        test_pos = test_pos + test_pos_step
    # print(distance)
    print(x_data.shape)
    plotline(x_data=x_data,
             y1_data=md_mean_distance,
             y2_data=ol_mean_distance,
             y3_data=ol_mean_distance,
             y4_data=ol_mean_distance,
             y5_data=ol_mean_distance,
             x_label='position',
             y_label='distance')
    plt.show()

def plotSucDistanceCost():
    """
    plot distance and cost comparing of ol and md at 4 5 6
    """
    data_logger = DataLogger()
    exper = 9
    cond = 6
    director = 7
    unit_len = 6
    # cost
    suc1 = data_logger.unpickle('./position/all_suc_1.pkl')
    suc1 = np.reshape(suc1, (unit_len, suc1.shape[0]/unit_len))
    suc2 = data_logger.unpickle('./position/all_suc_2.pkl')
    suc2 = np.reshape(suc2, (unit_len, suc2.shape[0] / unit_len))
    suc3 = data_logger.unpickle('./position/all_suc_3.pkl')
    suc3 = np.reshape(suc3, (unit_len, suc3.shape[0] / unit_len))
    cost = data_logger.unpickle('./position/all_cost_1.pkl')
    cost = np.reshape(cost, (unit_len, cost.shape[0]/unit_len))
    mean_cost1 = np.mean(cost, axis=1)
    cost = data_logger.unpickle('./position/all_cost_2.pkl')
    cost = np.reshape(cost, (unit_len, cost.shape[0] / unit_len))
    mean_cost2 = np.mean(cost, axis=1)
    cost = data_logger.unpickle('./position/all_cost_3.pkl')
    cost = np.reshape(cost, (unit_len, cost.shape[0] / unit_len))
    mean_cost3 = np.mean(cost, axis=1)
    #distance
    distance = data_logger.unpickle('./position/all_distance_1.pkl')
    print(distance)
    mean_distance1 = np.mean(distance, axis=1)
    print(mean_distance1)
    distance = data_logger.unpickle('./position/all_distance_2.pkl')
    mean_distance2 = np.mean(distance, axis=1)
    distance = data_logger.unpickle('./position/all_distance_3.pkl')
    mean_distance3 = np.mean(distance, axis=1)

    x_data = np.zeros(0)
    base_data = np.zeros(0)
    test_pos_step = 0.02
    test_pos = test_pos_step
    for i in range(mean_distance2.shape[0]):
        x_data = np.concatenate((x_data, np.array([test_pos])))
        base_data = np.concatenate((base_data, np.array([0.06])))
        test_pos = test_pos + test_pos_step * 2
    # print(distance)
    print(x_data.shape)
    print(distance.shape)
    mean_cost1[mean_cost1.shape[0] - 2] = 0
    mean_cost1[mean_cost1.shape[0] - 1] = 0
    mean_distance1[mean_distance1.shape[0] - 2] = 0.6
    mean_distance1[mean_distance1.shape[0] - 1] = 0.6
    # plotline(x_data=x_data,
    #          y1_data=mean_distance1,
    #          y2_data=mean_distance2,
    #          y3_data=mean_distance3,
    #          y5_data=base_data,
    #          x_label='Test Distance',
    #          y_label='Distance to Target')

    # mean_distance1[0] = mean_distance3[0]
    # x_data = x_data * np.sqrt(2)
    plotDoubleLine(x_data=x_data[0: 6],
                   y1_data=mean_distance1[0: 6],
                   y2_data=mean_distance2[0: 6],
                   y3_data=mean_distance3[0: 6],
                   z1_data=mean_cost1[0: 6],
                   z2_data=mean_cost2[0: 6],
                   z3_data=mean_cost3[0: 6],
                   base_line=base_data[0: 6],
                   xlabel='Testing Distance',
                   ylabel='Distance to Target',
                   zlabel='Cost')
    plt.show()

def plotCompareCostAlpha():
    """
    compare two distance with alpha and without alpha
    Returns:

    """
    data_logger = DataLogger()
    for i in range(1, 2):
        root_dir = './position/compare_alpha_%d/' % i
        for j in range(6):
            file_name = root_dir + 'alpha_distance_%d.pkl' % j
            distance = data_logger.unpickle(file_name)
            distance = np.expand_dims(distance, axis=0)
            if j == 0:
                distances = distance
            else:
                distances = np.concatenate((distances, distance), axis=0)
        if i == 2:
            all_distances_alpha = distances
        else:
            all_distances_alpha = np.concatenate((all_distances_alpha, distances), axis=1)
    # print(all_distances_alpha[1])
    all_distances_alpha[all_distances_alpha > 0.6] = 0.6
    mean_distances_alpha = np.mean(all_distances_alpha, axis=1)

    for i in range(1, 2):
        root_dir = './position/compare_alpha_%d/' % i
        for j in range(6):
            file_name = root_dir + 'without_alpha_distance_%d.pkl' % j
            distance = data_logger.unpickle(file_name)
            distance = np.expand_dims(distance, axis=0)
            if j == 0:
                distances = distance
            else:
                distances = np.concatenate((distances, distance), axis=0)
        if i == 1:
            all_distances_alpha = distances
        else:
            all_distances_alpha = np.concatenate((all_distances_alpha, distances), axis=1)
    # print(all_distances_alpha[1])
    all_distances_alpha[all_distances_alpha > 0.6] = 0.6
    mean_distances_alpha_without = np.mean(all_distances_alpha, axis=1)

    x_data = list()
    base_line = list()
    for i in range(mean_distances_alpha.shape[0]):
        x_data.append(i)
        base_line.append(0.06)
    x_data = np.array(x_data)
    base_line = np.array(base_line)

    plotline(x_data=x_data,
             y1_data=mean_distances_alpha,
             y2_data=mean_distances_alpha_without,
             y3_data=base_line,
             x_label="num of positions",
             y_label="distance to target")
    plt.show()



# plotCompareCostAlpha()
# plotMDTest()
# plotAgentPostiion()
plotSucDistance()
# plotSucDistanceCost()
# plotTestArea()
#plotCountSuc()
#plotCountSucAll()
# plotposition()
# plotcost()
# plotPositionStep()

# data_logger = DataLogger()
# #print(data_logger.unpickle('./position/3/position_md.pkl'))
# direcotr = './position/7/9/train_position.pkl'
# # direcotr = './position/9/suc_pos/1.pkl'
# print('directory:', direcotr)
# positions = data_logger.unpickle(direcotr)
# print(positions)
# direcotr = './position/7/9/all_train_position.pkl'
# positions = data_logger.unpickle(direcotr)
# print(positions)
