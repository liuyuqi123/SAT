"""
This script is developed based on the env_run_loop safety.

A easy version to tune modules of carla env.

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

# ==================================================
# -------------  import carla module  -------------
# ==================================================
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
# ==================================================

import carla

import math
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from gym_carla.envs.carla_env_multi_task import CarlaEnvMultiTask

from gym_carla.safety_layer.data_analysis import tan_normalize

from rl_agents.td3_old.multi_task.developing.train_ablation import restore_safety_layer


safety_model_path = os.path.join(project_path, 'gym_carla/safety_layer/safetylayer_outputs/runnable_models')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_env(args):
    """
    Test env and all modules' methods

    :return:
    """

    # # available task options
    # task_options = ['left', 'right', 'straight', 'multi_task']
    task_option = 'left'

    env = CarlaEnvMultiTask(
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        tm_seed=args.tm_seed,  # seed for autopilot controlled NPC vehicles

        train=True,  # training mode or evaluation mode
        task_option=args.task_option,
        attention=False,

        initial_speed=None,  # if set a initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        no_render_mode=args.no_render_mode,
        debug=False,
        verbose=False,
    )

    # restore the safety layer
    # models are stored in safet_layer folder
    model_path = os.path.join(safety_model_path, task_option)
    # default safety layer class is included in restore_safety_layer method
    action_modifier, safety_layer, config_dict = restore_safety_layer(output_path=model_path)

    # =========================
    env.traffic_flow_manager.clear_traffic_flow_info()
    state = env.reset()

    # ====================  run the roll-out loop  =========================

    # total test episode number
    # test_num = args.test_number
    # total_test_number = 20  # 100, 1000

    total_test_number = args.test_number

    elapsed_episode_number = 0

    # elapsed time step number in current episode
    step = 0
    success = 0
    failure = 0
    collision = 0
    time_exceed = 0
    episode_time_record = []

    episode_reward = 0
    done = False

    while elapsed_episode_number < total_test_number:

        # todo add roll-out policy selection
        #  merge rule-based policy into class
        # random policy
        action = np.random.rand(2)
        action = list(action)

        # # acceleration policy with safety layer
        # action = np.array([1., 0.])

        # retrieve safety value
        # todo fix the env API for getting constraint value
        state_data_dict, full_data_dict = env.get_single_frame_data()

        # use the 'ttc_2'
        c_key = 'ttc_2'
        c = state_data_dict[c_key]  # original c value

        # todo add args on determining norm method
        norm_c = tan_normalize(c, 4.)

        # todo constraints are supposed to be in list
        # debug_info is added into the return of action modification
        action_new, debug_info = action_modifier(state, action, [norm_c])

        # debug
        action = action_new

        # debug the ttc value correction
        alpha = 6.
        beta = .9
        fixed_c = alpha * env.simulator_timestep_length - beta * c

        # debug info
        print(
            '\n',
            '-*-' * 20, '\n',
            'ttc_2 value: ', c, '\n',
            'fixed ttc value: {}'.format(fixed_c), '\n',

            # 'constraint value: ', debug_info['constraint'], '\n',
            # 'original action: ', debug_info['original_action'], '\n',
            # 'g value: ', debug_info['g'], '\n',
            # 'multipliers: ', debug_info['multipliers'], '\n',
            # 'correction: ', debug_info['correction'], '\n',
            # 'new action: ', debug_info['new_action'], '\n',
            # 'c_next_predicted: ', debug_info['c_next_predicted'], '\n',
            '-*-' * 20, '\n',
        )

        print('')

        if fixed_c >= 0.:
            print('action modification.')

        state, reward, done, info = env.step(action)

        # test data collection
        state_data_dict, full_data_dict = env.get_single_frame_data()

        ttc_1 = state_data_dict['ttc_1']
        ttc_2 = state_data_dict['ttc_2']

        ttc_1_norm = env.state_manager.normalize_ttc(ttc_1)
        ttc_2_norm = env.state_manager.normalize_ttc(ttc_2)

        episode_reward += reward
        step += 1

        if done:
            # retrieve episode time from env
            episode_duration_time = env.episode_time

            # record result
            if info['exp_state'] == 'collision':
                collision += 1
                failure += 1
            elif info['exp_state'] == 'time_exceed':
                time_exceed += 1
                failure += 1
            else:
                episode_time_record.append(episode_duration_time)
                success += 1

            # print result of single episode
            print(
                '\n',
                '---' * 20, '\n',
                "Episode No.{} ended. ".format(elapsed_episode_number+1), '\n',
                "Result is {}".format(info['exp_state']), '\n',
                'Total time step number is: {}, duration time is: {:.1f} s'.format(step, episode_duration_time), '\n',

                '-*-*' * 10, '\n',
                'Run with safety layer: {}'.format(args.safety), '\n',
                'Elapsed test episode number is: {}.'.format(elapsed_episode_number+1), '\n',
                "Success Counts: {:.2%}".format(success, (elapsed_episode_number+1)), '\n',
                "Average Duration Time is {:.2f}".format(np.mean(episode_time_record)), '\n',
                '---' * 20, '\n',
            )

            episode_reward = 0
            step = 0
            done = False
            elapsed_episode_number += 1

            state = env.reset()  # init state

            print(
                '\n',
                '==' * 30, '\n',
                '==' * 30, '\n',
                '==' * 30, '\n',
            )

    # print total test result
    print(
        '\n',
        '='*50, '\n',
        '-' * 15, ' Final Test Result ', '-' * 15, '\n',
        'Run with safety layer: {}'.format(args.safety), '\n',
        'Total test episode number is: {}.'.format(total_test_number), '\n',
        "Total Success Rate: {:.2%}".format(success / total_test_number), '\n',
        "Total Average Duration Time is {:.2f}".format(np.mean(episode_time_record)), '\n',

        '-' * 15, 'Test Result Counts', '-' * 15, '\n',
        'Success counts: {} / {}'.format(success, total_test_number), '\n',
        'Collision counts: {} / {}'.format(collision, total_test_number), '\n',
        'Time exceeding counts: {} / {}'.format(time_exceed, total_test_number), '\n',
        '=' * 50, '\n',
    )


if __name__ == '__main__':

    # # original line
    # run_env(safety=True)

    argparser = argparse.ArgumentParser(
        description='Test random policy in env_run_loop.')
    argparser.add_argument(
        '-s', '--safety',
        action='store_true',
        dest='safety',
        help='Run with safety layer correction.')
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
        '-e', '--seed',
        default=int(0),
        type=int,
        dest='tm_seed',
        help='traffic manager seed number')
    argparser.add_argument(
        '-r', '--route',
        default="left",  # multi-task is added into env class
        dest='task_option',
        help='Task option for env initialization. Available options: left, right, right, straight, multi-task')
    argparser.add_argument(
        '-d', '--no-render',
        action='store_true',
        dest='no_render_mode',
        help='Whether use the no render mode.')
    argparser.add_argument(
        '-n', '--test_number',
        default=int(100),
        type=int,
        dest='test_number',
        help='Total test episode number.')

    args = argparser.parse_args()

    # automatically fix tm_port according to carla port number
    if args.carla_port != int(2000):
        d = args.carla_port - 2000
        args.tm_port = int(8100+d)

    run_env(args)

