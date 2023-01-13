"""
Some methods related to memory collection and manipulation.


Try to fix the memory capacity after a certain length of memory is collected.
"""

import os
import sys

from rl_agents.td3_old.multi_task.networks.prioritized_replay import Memory


def load_memory():
    """
    Load a collected memory instance using given path.
    """

    # # if given the parent folder
    # os.path.join(memory_path, 'memory.pkl')

    memory_path = '/home/liuyuqi/PycharmProjects/gym-carla/rl_agents/td3_old/multi_task/utils/outputs/memory/fix_env/baseline/left/memory.pkl'

    # memory_size = int(1e5)
    # pretrain_length = int(500)
    # memory = Memory(memory_size, pretrain_length)

    memory_cls = Memory

    memory_instance = memory_cls.load_memory(memory_path)

    print('')


if __name__ == '__main__':

    load_memory()

