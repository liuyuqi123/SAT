"""
todo move operation tools to a separate script

OU noise for traffic flow parameters generation.

Test custom noise to explore complete param space.
"""

import os

import numpy as np
import scipy.stats as stats
import math
import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


class TruncatedOUNoise:
    """
    Developing a new noise for traffic flow params generation.

    To explore a wide range of velocity.

    OU process is a unlimited process, we clip it manually.
    """

    def __init__(self,
                 a: float,  # lower bound of the stochastic variate
                 b: float,  # upper bound
                 n_sigma=2.,
                 theta=0.15,
                 damping_ratio=0.01,  # replace original dt
                 x0=None,
                 ):

        #
        self.a = a
        self.b = b
        self.n_sigma = n_sigma
        self.theta = theta

        # mean of the noise
        self.mu = np.array([0.5 * (a + b)])
        self.sigma = np.array([(b - a) / (2 * self.n_sigma)])

        self.damping_ratio = damping_ratio
        self.x0 = x0

        self.distribution = stats.truncnorm(
            a=(self.a - self.mu[0]) / self.sigma[0],
            b=(self.b - self.mu[0]) / self.sigma[0],
            loc=self.mu[0],
            scale=self.sigma[0],
        )

        self.reset()

    def sample(self):
        """
        Run a sample step
        :return:
        """
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.damping_ratio + \
            self.sigma * np.sqrt(self.damping_ratio) * np.random.normal(size=self.mu.shape)

        return x

    def __call__(self):
        """
        Return a new truncated sample.
        :return:
        """
        x = self.sample()

        # check if in the range
        while x > self.b or x < self.a:
            x = self.sample()

        self.x_prev = x

        return x

    def reset(self):

        # original
        # self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        # init x_prev using mu
        self.x_prev = self.x0 if self.x0 is not None else self.mu

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class CoordinateNoise:
    """
    This class is responsible to generate a coordinate noise for distance exploration.

    Use a truncated Normal distribution to get coordinate value.
    """

    def __init__(self,
                 source_range: tuple,
                 target_range: tuple,
                 n_sigma=2.,
                 ):

        #
        self.source_a = source_range[0]
        self.source_b = source_range[1]

        #
        self.myclip_a = target_range[0]
        self.myclip_b = target_range[1]

        self.sigma = np.array([(self.myclip_b - self.myclip_a) / (2 * n_sigma)])

        # linear mapping
        self.k = (self.myclip_b - self.myclip_a) / (self.source_b - self.source_a)
        self.b = self.myclip_a - self.myclip_a * self.k

    def __call__(self, ref_value, debug=False):
        """
        Get rvs from a truncated gaussian distribution.

        todo try different mean value, add a manual shift

        :param ref_value: reference mean of current value
        :return: sample variate
        """
        # linear mapping to get ref_mean
        ref_mean = self.k * ref_value + self.b
        ref_mean = np.clip(ref_mean, self.myclip_a, self.myclip_b)

        # truncnorm range
        a, b = (self.myclip_a - ref_mean) / self.sigma, (self.myclip_b - ref_mean) / self.sigma

        if debug:
            # debug
            x = stats.truncnorm.rvs(
                a, b,
                loc=ref_mean,
                scale=self.sigma,
                size=100000,
            )

            plt.figure()
            plt.hist(x, density=True, bins=100)
            axe = plt.gca()
            axe.set_title('期望间隔距离：{}'.format(ref_value))
            axe.set_xlabel('车辆间隔距离 m')
            axe.set_ylabel('概率密度分布')

            plt.tight_layout()
            plt.show()
        else:
            x = stats.truncnorm.rvs(
                a, b,
                loc=ref_mean,
                scale=self.sigma,
            )

        return x

    def reset(self):
        pass

    def plot_group(self):
        pass


def tuning_truncOU():
    """
    Parameters tuning for the target speed generation.

    This function is copied from exploration noise generation
    """

    font = {
        # 'family': 'normal',
        # 'weight': 'bold',
        'size': 14
    }
    matplotlib.rc('font', **font)

    # if using the chinese label
    matplotlib.rcParams['axes.unicode_minus'] = False  # 坐标轴负号
    plt.rcParams[u'font.sans-serif'] = ["simhei"]  # 设置字体

    # params for distribution
    dist_params = {
        'range': (10., 40.),  # clip range
        'mu': 0.,  # mean of the noise
        'n_sigma': 1.9,  # sigma number of the clip range
        'theta': 0.24,
    }

    trunc_ou_noise = TruncatedOUNoise(
        a=dist_params['range'][0],
        b=dist_params['range'][1],
        n_sigma=dist_params['n_sigma'],
        theta=dist_params['theta'],
        damping_ratio=0.01,
    )

    # total episode number
    N_step = 10000
    index = []
    value = []

    for i in range(N_step):
        index.append(i)
        value.append(trunc_ou_noise())

    # plot
    plt.figure()
    # plt.figure(figsize=(50, 10))

    # plt.suptitle(
    #     'Params: n_sigma={}, theta={}'
    #     .format(dist_params['n_sigma'], dist_params['theta'])
    # )

    # chinese title
    # plt.suptitle('截断OU过程的采样')

    plt.subplot(2, 1, 1)
    axe = plt.gca()
    axe.set_title('截断OU过程的采样')
    axe.set_xlabel('采样步数')
    axe.set_ylabel('采样值')

    plt.plot(index, value, '-.')

    plt.subplot(2, 1, 2)
    axe = plt.gca()
    axe.set_title('截断OU过程采样的分布')
    axe.set_xlabel('采样数值')
    axe.set_ylabel('概率密度')

    plt.hist(np.array(value), density=True, bins=150)

    plt.tight_layout()
    plt.show()


def tuning_coord_noise():
    """
    Tuning coordinate noise.

    This method is used to generate distance of traffic flow vehicle.
    """

    # if using the chinese label
    matplotlib.rcParams['axes.unicode_minus'] = False  # 坐标轴负号
    plt.rcParams[u'font.sans-serif'] = ["simhei"]  # 设置字体

    font = {
        # 'family': 'normal',
        # 'weight': 'bold',
        'size': 16
    }
    matplotlib.rc('font', **font)

    coord_noise = CoordinateNoise(
        source_range=(10., 50.),
        target_range=(10., 50.),
        n_sigma=1.2,
    )

    x_list = np.arange(10, 50, 10)
    # x_list = [25]

    for x in x_list:
        y = coord_noise(x, debug=True)


def stat_tf_distribution(result_dict: dict,
                         save_result: bool = False,
                         plot_fig: bool = True,
                         save_path: str = None):
    """
    Statistics and plot of traffic flow param distribution.
    """
    # # plt.get_cachedir()
    # plt.rc('font', family='Times New Roman')

    # if using the chinese label
    matplotlib.rcParams['axes.unicode_minus'] = False  # 坐标轴负号
    plt.rcParams[u'font.sans-serif'] = ["simhei"]  # 设置字体

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

        # original title
        # plt.suptitle('Traffic flow parameter distribution of {} \n Total vehicle number={}'.format(key, total_vehicle_num))  # , y=0.98)

        # Chinese title
        plt.suptitle('交通流参数分布，起始方向 {} \n 车辆生成总数={}'.format(key, total_vehicle_num))  # , y=0.98)

        axe = plt.subplot(3, 2, 1)

        # axe.set_title('Target speed')
        # axe.set_xlabel('km/h')
        # axe.set_ylabel('PDF')

        axe.set_title('目标速度')
        axe.set_xlabel('速度 km/h')
        axe.set_ylabel('概率分布密度')

        plt.hist(np.array(speed_list), density=True, bins=150)

        axe = plt.subplot(3, 2, 2)

        # axe.set_title('Gap distance')
        # axe.set_xlabel('m')
        # axe.set_ylabel('PDF')

        axe.set_title('车辆间距')
        axe.set_xlabel('间距 m')
        axe.set_ylabel('概率分布密度')

        plt.hist(np.array(distance_list), density=True, bins=150)

        axe = plt.subplot(3, 1, 2)
        # axe.set_title('Speed exploration')
        # axe.set_xlabel('Vehicle number index')
        # axe.set_ylabel('Speed \n km/h')

        axe.set_title('目标速度采样过程')
        axe.set_xlabel('车辆生成序号')
        axe.set_ylabel('速度 km/h')

        plt.plot(x, speed_list)

        axe = plt.subplot(3, 1, 3)
        # axe.set_title('Distance exploration')
        # axe.set_xlabel('Vehicle number index')
        # axe.set_ylabel('Distance \n m')

        axe.set_title('车辆间隔距离的取值')
        axe.set_xlabel('车辆生成序号')
        axe.set_ylabel('间隔距离 \n m')

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

        # print('')


def get_tf_stat_file(file_path, save_path):
    """
    Get file and plot tf distribution figures.
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    stat_tf_distribution(
        result_dict=data_dict,
        save_result=True,
        plot_fig=True,
        save_path=save_path,
    )


if __name__ == '__main__':

    # # ================================
    # # visualize results statistics of traffic flow params
    # path = '/home/lyq/PycharmProjects/gym-carla/outputs/tf_dist/straight'  # left, right, straight
    # file_path = os.path.join(path, 'tf_distribution.pkl')
    #
    # output_path = os.path.join(path, 'chinese_label')
    #
    # get_tf_stat_file(file_path, save_path=output_path)

    # # ================  distance param generation  ================
    # tuning_coord_noise()

    # # ================================
    tuning_truncOU()
