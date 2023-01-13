"""
Plot multiple training curves in single image.

Average each point and plot the std range.s
"""

import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.5)

output_path = './plot_outputs/'


def load_pkl_data(pkl_path):
    """
    todo fix this

    Load pkl file into numpy

    :param pkl_path:
    :return:
    """
    # #
    file = os.path.join(pkl_path, 'reward_data.pkl')
    with open(file, "rb") as f:
        data = pickle.load(f)
    x1 = data["avarage_rewards"]  # do not fix the spelling error
    rewards = x1.T

    file = os.path.join(pkl_path, 'success_rate_data.pkl')
    with open(file, "rb") as f:
        data = pickle.load(f)
    x1 = data["recent_success_rates"]  # do not fix the spelling error
    success_rate = x1.T

    return rewards, success_rate


# process each category data and plot
def get_plot_data(data_array, threshold=0.):
    """

    :param threshold:
    :param data_array:
    :return:
    """
    mean = np.nanmean(np.where(data_array >= threshold, data_array, np.nan), axis=0)
    std = np.nanstd(np.where(data_array >= threshold, data_array, np.nan), axis=0)

    return mean, std


def process_single_category(category_folder):
    """
    Calculate mean and std value for each category
    """
    #
    date_list = os.listdir(category_folder)

    # todo set len as a tunable parameter
    # target length of the plot data
    target_len = 5000

    # how many sets of data
    data_set_num = len(date_list)  # 5

    # reward value range is different from success rate
    reward_array = -1e5 * np.ones((data_set_num, target_len))
    success_rate_array = -1 * np.ones((data_set_num, target_len))

    for date_index, date in enumerate(date_list):

        rl_result_folder = os.path.join(category_folder, date, 'rl_results')
        rewards, success_rate = load_pkl_data(rl_result_folder)

        padding_len = min([np.shape(rewards)[0], target_len])

        for data_index in range(padding_len):
            if data_index <= 50:
                # reward_array[date_index, data_index] = -100.
                reward_array[date_index, data_index] = rewards[data_index]

                success_rate_array[date_index, data_index] = 0.
            else:
                reward_array[date_index, data_index] = rewards[data_index]
                success_rate_array[date_index, data_index] = success_rate[data_index]

    # todo check this line
    rewards_mean, rewards_std = get_plot_data(reward_array, -1e4)
    success_rate_mean, success_rate_std = get_plot_data(success_rate_array)

    return rewards_mean, rewards_std, success_rate_mean, success_rate_std


def run(result_folder, title=None):
    """"""

    # # put all result folder into same folder
    # result_folder = ''
    category_list = os.listdir(result_folder)
    category_list.sort()

    # category_list[0], category_list[1], category_list[2], category_list[3] \
    #     = category_list[2], category_list[3], category_list[0], category_list[1]

    # data_list format: category,
    data_list = []

    for item in category_list:
        category_folder = os.path.join(result_folder, item)
        rewards_mean, rewards_std, success_rate_mean, success_rate_std = process_single_category(category_folder)

        data_list.append([
            item,
            [rewards_mean, rewards_std],
            [success_rate_mean, success_rate_std],
        ])

    # plot rewards
    fig = plt.figure()
    plt.tight_layout()

    for item in data_list:
        category, [rewards_mean, rewards_std] = item[0], item[1]

        # plot figures
        x = range(1, rewards_mean.shape[0] + 1)
        # plt.plot(x, mean_array, 'b-', label='mean_1')
        plt.plot(x, rewards_mean, label=category)

        # plt.fill_between(x, mean_array - std_array, mean_array + std_array, color='b', alpha=0.2)
        plt.fill_between(x, rewards_mean - rewards_std, rewards_mean + rewards_std, alpha=0.2)

        plt.legend(loc=4)

    plt.title('Training Rewards({})'.format(title))
    plt.xlabel('Episode Number')
    plt.ylabel('Rewards')

    # # todo add save option and result name
    # output_path = ''
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, title + "_rewards.png"), bbox_inches='tight')

    plt.show()

    # plot success rate
    fig = plt.figure()
    plt.tight_layout()

    for item in data_list:
        category, [success_rate_mean, success_rate_std] = item[0], item[2]

        # plot figures
        x = range(1, success_rate_mean.shape[0] + 1)
        # plt.plot(x, mean_array, 'b-', label='mean_1')
        plt.plot(x, success_rate_mean, label=category)
        # plt.fill_between(x, mean_array - std_array, mean_array + std_array, color='b', alpha=0.2)
        plt.fill_between(x, success_rate_mean - success_rate_std, success_rate_mean + success_rate_std, alpha=0.2)

        # plt.legend(title=category)
        plt.legend(loc=4)

    plt.title('Success Rate in Training Phase({})'.format(title))
    plt.xlabel('Episode Number')
    plt.ylabel('Success Rate(%)')

    # # todo add save option and result name
    # output_path = ''
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, title + "_success_rate.png"), bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    # # debug
    # result_folder = '/home/lyq/PycharmProjects/gym-carla/rl_agents/td3_old/multi_task/result_analysis/debug'

    # # ===================================
    # result_folder = ''
    # run(result_folder)

    # ===================================
    task_list = [
        ['Left',
         '/home/lyq/PycharmProjects/gym-carla/rl_agents/td3_old/multi_task/result_analysis/train_ablation_outputs/single-task', ],
        ['Multi-task',
         '/home/lyq/PycharmProjects/gym-carla/rl_agents/td3_old/multi_task/result_analysis/train_ablation_outputs/multi-task', ]
    ]

    for item in task_list:
        run(item[1], item[0])
