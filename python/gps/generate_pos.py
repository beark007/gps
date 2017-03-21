from __future__ import division, print_function
import argparse
import logging
try:
    import cPickle as pickle
except:
    import pickle

LOGGER = logging.getLogger(__name__)


from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

# from gps.utility.data_logger import DataLogger

mpl.rc('figure', figsize=(14, 7))
mpl.rc('font', size=14)
mpl.rc('axes', grid=False)
mpl.rc('axes', facecolor='white')

def plotpoint(x_data, y_data, x_label, y_label, title):
    _, ax = plt.subplots()
    ax.scatter(x_data, y_data, s=30, color='#539caf', alpha=0.75)
    #ax.plot(x_data, y_data, lw=1, color='blue', alpha=0.75, linestyle='-', label='init')
    # ax.plot(x_data, yy_data, lw=1, color='green', alpha=0.75, linestyle='-', label='change')
    plt.legend(loc='upper left', frameon=False)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    circle = plt.Circle((0.16, -0.16), 0.02, color='r', fill=False)
    ax.add_artist(circle)

def plotposition(position):
    plotpoint(x_data=position[:, 0],
             y_data=position[:, 1],
             x_label='position x',
             y_label='position y',
             title='position')
    plt.show()

def file_pickle(filename, data):
    """pickle data into file specified by filename"""
    pickle.dump(data, open(filename, 'wb'))

def file_unpickle(filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except IOError:
        LOGGER.debug('unpickle error, cannot find file:%s', filename)
        return None

def generate_position(c_x, c_y, radius, conditions, max_error_bound):
    """

    Args:
        c_x:        the x axis of center position
        c_y:        the y axis of center position
        radius:     area's radius
        conditions: the quantity of generating positions
        max_error_bound: the mean of generated positions' error around cposition

    Returns:

    """
    while True:
        all_positions = np.zeros(0)
        center_position = np.array([c_x, -c_y, 0])
        for i in range(conditions):
            position = np.random.uniform(-radius, radius, 3)
            while True:
                position[2] = 0
                position[1] = -(position[1] + c_y)
                position[0] = position[0] + c_x
                area = (position - center_position).dot(position - center_position)
                if area <= (np.pi * radius**2)/4.0:
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
    print('all_position:', all_positions)
    return all_positions

def main():
    """main function to be run"""
    parser = argparse.ArgumentParser(description='run the GPS cost')
    parser.add_argument('-t', '--test_time', metavar='N', type=int,
                       help='test trained policy N time')
    parser.add_argument('-p', '--train_time', metavar='N', type=int,
                       help='train policy in N positon')
    parser.add_argument('-c', '--center_position', metavar='N', type=float,
                       help='center position')
    parser.add_argument('-r', '--radius', metavar='N', type=float,
                       help='radius of area')
    parser.add_argument('-m', '--num', metavar='N', type=float,
                       help='N num of experiment')
    args = parser.parse_args()
    if args.test_time == None:
       args.test_time = 50
    if args.train_time == None:
       args.train_time = 25
    if args.center_position == None:
       args.center_position = 0.12
    if args.radius == None:
        args.radius = 0.08
    if args.num == None:
       print('Should add the -m num')
       while True:
           a = 1

    max_error_bound = 0.01
    # train_positions
    #train_positions = generate_position(args.center_position, args.radius, args.train_time, max_error_bound)
    bas_pos = 0.04
    # np.array([0.11, -0.11, 0]),np.array([0.181, -0.181, 0])


    """
    use for linear increment positions
    """
    train_positions = np.array([np.array([-bas_pos, -bas_pos, 0]), np.array([bas_pos, -bas_pos, 0]),
                               # np.array([-bas_pos, bas_pos, 0]),
                                np.array([bas_pos, bas_pos, 0]),
                                np.array([0.07, -0.07, 0]),  np.array([0.10, -0.10, 0]),
                                np.array([0.15, -0.15, 0]), np.array([0.18, -0.18, 0])])
    # train_positions = np.array([np.array([0.03, -0.03, 0]), np.array([-0.03, 0.03, 0]), np.array([0.181, -0.181, 0]),
    #                       np.array([0.101, -0.101, 0]), np.array([0.241, -0.241, 0])])
    print(train_positions)
    file_pickle('./position/%d/position_train.pkl' % args.num, train_positions)

    # test_positions
    test_positions = generate_position(args.center_position, -args.center_position, args.radius, args.test_time, max_error_bound)
    # plotposition(test_positions)
    file_pickle('./position/%d/test_position.pkl' % args.num, test_positions)

    # """
    # generate train and test for four areas
    # """
    # director = 8
    # idx_pos = 1
    # # first area [[-0.05, 0.05], [-0.05, 0.05]]
    # train_position = generate_position(0.05, 0.05, 0.05, 3, 0.02)
    # test_position = generate_position(0.05, 0.05, 0.05, 10, 0.02)
    # file_pickle('./position/%d/train_position_%d.pkl' % (director, idx_pos), train_position)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos), test_position)
    #
    # # second area [[0.1, 0.2], [-0.05, 0.05]]
    # train_position = generate_position(0.05, 0.2, 0.05, 3, 0.02)
    # test_position = generate_position(0.05, 0.2, 0.05, 10, 0.02)
    # file_pickle('./position/%d/train_position_%d.pkl' % (director, idx_pos+1), train_position)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+1), test_position)
    #
    # # second area [[0.1, 0.2], [-0.05, 0.05]]
    # train_position = generate_position(0.4, 0.05, 0.05, 3, 0.02)
    # test_position = generate_position(0.4, 0.05, 0.05, 10, 0.02)
    # file_pickle('./position/%d/train_position_%d.pkl' % (director, idx_pos+2), train_position)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+2), test_position)
    #
    # # second area [[0.1, 0.2], [-0.05, 0.05]]
    # train_position = generate_position(0.2, 0.2, 0.05, 3, 0.02)
    # test_position = generate_position(0.2, 0.2, 0.05, 10, 0.02)
    # file_pickle('./position/%d/train_position_%d.pkl' % (director, idx_pos+3), train_position)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+3), test_position)

    """
    test the OLGPS generalization
    """
    # director = args.num
    # idx_pos = 1
    # # train_position = generate_position(0.2, 0.2, 0.050, 4, 0.05)
    # train_position = np.array([[ 0.178, -0.222, 0.   ], [ 0.237, -0.178, 0.   ], [ 0.179, -0.172, 0.   ],
    #                            [0.2, -0.2, 0]])
    # file_pickle('./position/%d/train_position_%d.pkl' % (director, idx_pos), train_position)
    # test_position = generate_position(0.2, 0.2, 0.050, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos), test_position)
    # test_position = generate_position(0.2, 0.2, 0.075, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+1), test_position)
    # test_position = generate_position(0.2, 0.2, 0.100, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+2), test_position)
    # test_position = generate_position(0.2, 0.2, 0.125, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+3), test_position)
    # test_position = generate_position(0.2, 0.2, 0.150, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+4), test_position)
    # test_position = generate_position(0.2, 0.2, 0.175, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+5), test_position)
    # test_position = generate_position(0.2, 0.2, 0.200, 100, 0.05)
    # file_pickle('./position/%d/test_position_%d.pkl' % (director, idx_pos+6), test_position)
    # print('train: ', train_position)


if __name__ == '__main__':
   main()