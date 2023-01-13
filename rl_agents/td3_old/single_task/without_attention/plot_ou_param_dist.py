"""
Plot distribution of traffic flow parameters generated by OU process.

Most of the methods are inherited form "gym_carla/modules/trafficflow/ou_noise.py"

"""

import os

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
import pickle

import matplotlib.pylab as pylab

params = {
    'legend.fontsize': '20',
    # 'figure.figsize': (15, 5),
    # 'axes.labelsize': 'x-large',
    # 'axes.titlesize': 'x-large',
    # 'xtick.labelsize': 'x-large',
    # 'ytick.labelsize': 'x-large',
}

# pylab.rcParams.update(params)


def stat_tf_distribution(
        result_dict: dict,
        save_result: bool = False,
        plot_fig: bool = True,
        save_path: str = None,
):
    """
    Statistics and plot of traffic flow param distribution.
    """

    # todo add to args
    fontsize = 20
    tick_size = 15

    for key, item in result_dict.items():
        speed_list = []
        distance_list = []
        #
        total_vehicle_num = len(item)
        x = list(range(total_vehicle_num))
        #
        for tup in item:
            speed_list.append(tup[0])
            distance_list.append(tup[1])

        plt.figure()
        # plt.figure(figsize=(50, 10))

        plt.suptitle('Traffic Flow Distribution of {}. \n total veh num={}'.format(key, total_vehicle_num))  # , y=0.98)

        axe = plt.subplot(3, 2, 1)
        axe.set_title('speed', fontsize=fontsize)
        axe.set_xlabel('km/h', fontsize=fontsize)
        axe.set_ylabel('pdf', fontsize=fontsize)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.hist(np.array(speed_list), density=True, bins=150)

        axe = plt.subplot(3, 2, 2)
        axe.set_title('distance gap', fontsize=fontsize)
        axe.set_xlabel('m', fontsize=fontsize)
        axe.set_ylabel('pdf', fontsize=fontsize)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.hist(np.array(distance_list), density=True, bins=150)

        axe = plt.subplot(3, 1, 2)
        axe.set_title('speed exploration', fontsize=fontsize)
        axe.set_xlabel('vehicle num index', fontsize=fontsize)
        axe.set_ylabel('speed \n km/h', fontsize=fontsize)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.plot(x, speed_list)

        axe = plt.subplot(3, 1, 3)
        axe.set_title('distance exploration', fontsize=fontsize)
        axe.set_xlabel('vehicle num index', fontsize=fontsize)
        axe.set_ylabel('distance \n m', fontsize=fontsize)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.plot(x, speed_list)

        # plt.subplots_adjust(top=0.8)
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        if save_result:
            if not save_path:
                TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())
                save_path = os.path.join('./traffic_flow_distribution/', TIMESTAMP)

            os.makedirs(save_path, exist_ok=True)
            # save fig
            plt.savefig(os.path.join(save_path, key + '.png'))

        if plot_fig:
            plt.show()

        print('')


def get_tf_stat_file(file_path, save_path):
    """
    Get file and plot tf distribution figures.
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    pylab.rcParams.update(params)

    stat_tf_distribution(
        result_dict=data_dict,
        save_result=True,
        plot_fig=True,
        save_path=save_path,
    )


def main():
    """"""

    path = '/home/lyq/PycharmProjects/gym-carla/outputs/right/rl_results'

    file_path = os.path.join(path, 'tf_distribution.pkl')
    get_tf_stat_file(
        file_path=file_path,
        save_path=os.path.join(path, 'fig_outputs'),
    )


def search_plot():
    """
    Search and plot all figures.

    On lenovo laptop
    """

    path = '/home/liuyuqi/PycharmProjects/gym-carla/outputs/paper_plot/ou_tf_results'

    for route in ['left', 'right', 'straight_0', 'straight_1']:
        file_path = os.path.join(path, route, 'rl_results', 'tf_distribution.pkl')
        get_tf_stat_file(
            file_path=file_path,
            save_path=os.path.join(path, route, 'fig_outputs'),
        )

        print('')


if __name__ == '__main__':

    # main()

    search_plot()

