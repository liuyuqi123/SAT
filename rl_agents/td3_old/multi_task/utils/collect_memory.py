"""
This script is for memory collection of all ablation study.

"""
import glob
import os
import sys

# ================   Append Project Path   ================
path = os.getcwd()
index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1
project_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'
sys.path.append(project_path)

# ================   Append CARLA Path   ================
from gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

carla_root = os.path.join(root_path, 'CARLA_' + carla_version)
carla_path = os.path.join(carla_root, 'PythonAPI')
sys.path.append(carla_path)
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla'))
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/agents'))

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import numpy as np
import json
import argparse
import logging
import pickle
from datetime import datetime

from gym_carla.envs.carla_env_multi_task import CarlaEnvMultiTask
from rl_agents.td3_old.multi_task.developing.train_ablation import init_carla_env
from rl_agents.td3_old.multi_task.networks.prioritized_replay import Memory


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


def run(args):
    """"""

    env = init_carla_env(env_cls=CarlaEnvMultiTask, args=args)

    # ablation settings
    if args.attention:
        if not args.multi_task:
            memory_option = 'attention'
        else:  # multi-task and attention
            memory_option = 'mt_attention'
    else:
        if args.multi_task:  # multi-task without attention
            memory_option = 'multi_task'
        else:  # baseline
            memory_option = 'baseline'

    if args.tag:
        tag = args.tag
    else:
        tag = TIMESTAMP

    memory_path = os.path.join('./outputs', 'memory', tag, memory_option, args.task_option)
    os.makedirs(memory_path, exist_ok=True)

    # # default size
    # memory_size = 500000
    # pretrain_length = 10000

    if args.debug:
        memory_size = 1000
        pretrain_length = 100
    else:
        memory_size = 300000
        pretrain_length = 100000

    # todo add params to args
    memory = Memory(memory_size, pretrain_length)
    memory.fill_memory(env)

    memory.save_memory(os.path.join(memory_path, 'memory.pkl'), memory)

    print('Memory collection is finished.')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='Memory collection for ablation experiments.')
    argparser.add_argument(
        '-p', '--port',
        default=int(2000),
        type=int,
        dest='carla_port',
        help='carla simulator port number')
    argparser.add_argument(
        '--tm-port',
        default=int(8100),
        type=int,
        dest='tm_port',
        help='traffic manager port number')
    argparser.add_argument(
        '--seed',
        default=int(0),
        type=int,
        dest='tm_seed',
        help='traffic manager seed number')
    argparser.add_argument(
        '-r', '--route',
        default="left",  # multi-task is added into env class
        dest='task_option',
        help='Task option for env initialization. Available options: left, right, right, straight, multi_task')
    argparser.add_argument(
        '--init-speed',
        default=None,
        type=float,
        dest='initial_speed',
        help='initial speed of ego vehicle')
    argparser.add_argument(
        '--debug',
        action='store_true',
        dest='debug',
        help='Run with debug mode.')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='En-able all visualization function.')
    argparser.add_argument(
        '--tag',
        default=None,
        dest='tag',
        help='additional tag for current result path')
    argparser.add_argument(
        '-a', '--attention',
        action='store_true',
        dest='attention',
        help='Ablation experiments options on attention setting.')
    argparser.add_argument(
        '-m', '--multi-task',
        action='store_true',
        dest='multi_task',
        help='Ablation experiments options on the multi-task option.')
    argparser.add_argument(
        '-d', '--render',
        action='store_true',
        dest='render_mode',
        help='Whether use no-render mode on CARLA server, default is no-render mode.'
    )

    args = argparser.parse_args()

    # automatically fix tm_port according to carla port number
    if args.carla_port != int(2000):
        d = args.carla_port - 2000
        args.tm_port = int(8100+d)

    # arg multi-task has higher priority than route
    if args.multi_task:
        args.task_option = 'multi_task'

    if args.debug:
        args.tag = 'debug'

    run(args)
