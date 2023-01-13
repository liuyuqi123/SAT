"""
Fix and merge the plot method into single function.
"""

import os
import seaborn as sns
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle


def plot_reward(folder, title: str, show):
    """
    Plot and save the reward curve from single rl_result folder.
    """

    fig = plt.figure()
    plt.tight_layout()
    # f, ax = plt.subplots(1, 1)

    sns.set(style="darkgrid", font_scale=1.5)

    file = os.path.join(folder, 'reward_data.pkl')

    with open(file, "rb") as f:
        data = pickle.load(f)
    x1 = data["avarage_rewards"]  # do not fix the spelling error
    x1 = x1.T

    time = []
    for i in range(x1.shape[0]):
        time.append(i)

    # # todo add route info to curves
    # # label refers to the curve name
    # sns.lineplot(x=time, y=x1, label=route)

    sns.lineplot(x=time, y=x1)

    plt.ylabel("Rewards")
    plt.xlabel("Episode")
    plt.title(title)

    # save figure
    plt.savefig(os.path.join(folder, "rewards.png"), bbox_inches='tight')
    if show:
        plt.show()


def plot_success(folder, title: str, show):
    """"""

    fig = plt.figure()
    plt.tight_layout()
    # f, ax = plt.subplots(1, 1)

    sns.set(style="darkgrid", font_scale=1.5)

    file = os.path.join(folder, 'success_rate_data.pkl')

    with open(file, "rb") as f:
        data = pickle.load(f)
    x1 = data["recent_success_rates"]
    x1 = x1.T

    time = []
    for i in range(x1.shape[0]):
        time.append(i)

    # # todo fix
    # sns.lineplot(x=time, y=x1, label=route)

    sns.lineplot(x=time, y=x1)

    plt.ylim(-0.05, 1.05)
    plt.yticks(np.arange(0, 1.1, step=0.1))

    plt.ylabel("Success rates")
    plt.xlabel("Episode")
    plt.title(title)

    # save figure
    plt.savefig(os.path.join(folder, "success_rate.png"), bbox_inches='tight')
    if show:
        plt.show()


def plot_curves(result_folder, title='TD3', show=False):
    """"""

    plot_success(result_folder, title, show)
    plot_reward(result_folder, title, show)

    print('Training curves are saved.')


if __name__ == '__main__':

    result_folder = ''

    plot_curves(result_folder, show=True)


