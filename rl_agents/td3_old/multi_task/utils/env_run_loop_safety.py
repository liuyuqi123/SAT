"""
Run carla env run loop methods to rollout.

Modified from original env_run_loop_safety

Major function:

 - test the calculation of safety constraint value
 - test Front Propagation of the safety layer
 - test the modification on random action

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
import json

from gym_carla.envs.carla_env_multi_task import CarlaEnvMultiTask
from gym_carla.util_development.util_visualization import plot_actor_bbox, draw_2D_velocity, visualize_actor

from gym_carla.safety_layer.data_analysis import tan_normalize
from gym_carla.safety_layer.dataset_manipulation import atan_norm

from rl_agents.td3_old.multi_task.developing.train_ablation import restore_safety_layer


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

safety_model_path = os.path.join(project_path, 'gym_carla/safety_layer/safetylayer_outputs/runnable_models')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0, 1, 2, 3


def plot_hist(data: list, info_dict: dict, show=False):
    """
    todo merge this method into util script

    Plot hist figure of 1-dim array data.
    """

    # plt.figure()
    # plt.figure(figsize=(50, 10))

    fig, ax = plt.subplots()

    plt.hist(np.array(data), density=True, bins=100)

    plt.tight_layout()

    if info_dict['title']:
        ax.set_title(info_dict['title'])
    if 'xlabel' in info_dict.keys():
        ax.set_xlabel(info_dict['xlabel'])
    if 'ylabel' in info_dict.keys():
        ax.set_ylabel(info_dict['ylabel'])

    # if save:
    #     save_path = os.path.join('./ttc_distribution/', TIMESTAMP)
    #     os.makedirs(save_path, exist_ok=True)
    #     plt.savefig(os.path.join(save_path, k + '.png'))

    save_path = os.path.join('./rollout_result/', TIMESTAMP, 'ttc_distribution')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, info_dict['title'] + '.png'))

    if show:
        plt.show()

    print('')


def run_env(args):
    """
    Test env and all modules' methods

    :return:
    """

    # # available task options
    # 'left', 'right', 'straight', 'multi_task'

    # todo fix multi-task
    task_option = args.task_option

    env = CarlaEnvMultiTask(
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        tm_seed=args.tm_seed,  # seed for autopilot controlled NPC vehicles

        train=False,  # training mode or evaluation mode
        task_option=task_option,
        attention=False,

        initial_speed=None,  # if set a initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        no_render_mode=not args.render_mode,
        debug=False,
        verbose=False,
    )

    # restore the safety layer
    if args.safety:
        # models are stored in safet_layer folder
        model_path = os.path.join(safety_model_path, task_option)

        # # model to be tested
        # model_path = os.path.join(safety_model_path, task_option, 'clip_linear_failure')

        # default safety layer class is included in restore_safety_layer method
        action_modifier, safety_layer, config_dict = restore_safety_layer(output_path=model_path)

        # record constraint value
        ttc_count = []
        norm_c_count = []

    # =========================
    env.traffic_flow_manager.clear_traffic_flow_info()

    # ====================  run the roll-out loop  =========================

    # total test episode number
    # test_num = args.test_number
    # total_test_number = 20  # 100, 1000

    total_test_number = args.test_number

    # counters
    elapsed_episode_number = 0
    success = 0
    failure = 0
    collision = 0
    time_exceed = 0
    episode_time_record = []

    while elapsed_episode_number < total_test_number:

        # reset counters for single episode
        # elapsed time step number in current episode
        step = 0
        episode_reward = 0

        state = env.reset()

        while True:

            # todo add roll-out policy selection
            #  merge rule-based policy into class
            # random policy
            action = np.random.rand(2)
            action = list(action)

            # # acceleration policy with safety layer
            # action = np.array([1., 0.])

            if args.safety:
                # retrieve safety value
                # todo fix the env API for getting constraint value
                state_data_dict, full_data_dict = env.get_single_frame_data()

                # use the 'ttc_2'
                c_key = 'ttc_2'
                c = state_data_dict[c_key]  # original c value

                # # original line
                # norm_c = tan_normalize(c, 4.)

                # fixed c calculation
                # A, dx = 10., 1.
                # norm_c = atan_norm(c, A=A, dx=dx)

                # only consider the condition which c <= 5.
                if c <= 4.:
                    # clip with linear transformation
                    alpha = 6.
                    beta = .9
                    # fixed_c = alpha * env.simulator_timestep_length - beta * c

                    norm_c = alpha * env.simulator_timestep_length - beta * c

                    ttc_count.append(c)
                    norm_c_count.append(norm_c)

                    # todo constraints are supposed to be in list
                    # debug_info is added into the return of action modification
                    action_new, debug_info = action_modifier(state, action, [norm_c])

                    # debug
                    action = action_new

                    # debug
                    if any(abs(debug_info['correction']) > 0.):
                        print(
                            '\n',
                            '-.-' * 20, '\n',
                            'Episode step number: {}'.format(step), '\n',
                            'ttc value: ', c, '\n',
                            'constraint value: ', debug_info['constraint'], '\n',
                            'c next predicted: ', debug_info['c_next_predicted'], '\n',
                            '-'*5, 'modifications', '-'*5, '\n',
                            'g value: ', debug_info['g'], '\n',
                            'multipliers: ', debug_info['multipliers'], '\n',
                            'correction: ', debug_info['correction'], '\n',
                            'original action: ', debug_info['original_action'], '\n',
                            'new action: ', debug_info['new_action'], '\n',
                            '-.-' * 20, '\n',
                        )

                        # todo add a heat map for action modification visualization
                        # additional method to visualize action modification
                        # env.visualize_action_modification()
                        if env.ego_vehicle:
                            ego_vehicle = env.ego_vehicle

                            red = carla.Color(r=255, g=0, b=0)
                            color = red

                            visualize_actor(
                                carla_debug_helper=env.debug_helper,
                                vehicle=ego_vehicle,
                                color=color,
                                thickness=0.5,
                                duration_time=1 * env.simulator_timestep_length
                            )

                            print('')

                    # print('')

            state, reward, done, info = env.step(action)

            # test data collection
            state_data_dict, full_data_dict = env.get_single_frame_data()

            episode_reward += reward
            step += 1

            # # debug reward
            # # print info of each time step
            # print(
            #     '\n',
            #     '=='*20, '\n',
            #     'step: {}'.format(step), '\n',
            #     'step reward: {}'.format(reward), '\n',
            #     'episode_reward: {}'.format(episode_reward), '\n',
            #     '=='*20, '\n',
            # )
            #
            # print('')

            if done:

                elapsed_episode_number += 1

                # retrieve episode time from env
                episode_duration_time = env.episode_time

                # record result
                if info['exp_state'] == 'collision':
                    collision += 1
                    failure += 1

                    # # ===================================
                    # # print final step state
                    # print('state array: ', state)
                    # print('state data: ', state_data_dict)
                    # print('=*'*20)
                    # print('Full data: ', full_data_dict)
                    # print('')

                elif info['exp_state'] == 'time_exceed':
                    time_exceed += 1
                    failure += 1
                else:
                    episode_time_record.append(episode_duration_time)
                    success += 1

                # print result of single episode
                print(
                    '\n',
                    '-*-' * 20, '\n',
                    "Current time: {0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now()), '\n',
                    'Run with safety layer: {}'.format(args.safety), '\n',
                    "Episode No.{} ended. ".format(elapsed_episode_number), '\n',
                    "Result is {}".format(info['exp_state']), '\n',
                    'Episodic time step number is: {}, duration time is: {:.1f} s'.format(step, episode_duration_time), '\n',
                    '---' * 10, '\n',
                    'Elapsed episode number is: {}.'.format(elapsed_episode_number), '\n',
                    "Success Counts: {}/{}".format(success, elapsed_episode_number), '\n',
                    "Success Rate: {:.2%}".format(success / elapsed_episode_number), '\n',
                    "Average Duration Time is {:.2f}".format(np.mean(episode_time_record)), '\n',
                    '-*-' * 20, '\n',
                )

                break

    # print total test result
    print(
        '\n',
        '='*50, '\n',
        '-' * 15, ' Final Test Result ', '-' * 15, '\n',
        'Run with safety layer: {}'.format(args.safety), '\n',
        'Total test episode number is: {}.'.format(total_test_number), '\n',
        "Total Success Rate: {:.2%}".format(success / total_test_number), '\n',
        "Total Average Duration Time is {:.2f}".format(np.mean(episode_time_record)), '\n',

        # '-' * 15, 'Test Result Counts', '-' * 15, '\n',
        'Success counts: {} / {}'.format(success, total_test_number), '\n',
        # 'Collision counts: {} / {}'.format(collision, total_test_number), '\n',
        # 'Time exceeding counts: {} / {}'.format(time_exceed, total_test_number), '\n',
        '=' * 50, '\n',
    )

    # plot distribution of constraint value
    plot_hist(ttc_count, {'title': 'TTC_2 value'})
    plot_hist(norm_c_count, {'title': 'Normed ttc_2 value'})

    # test results
    test_result = {
        'total_test_number': total_test_number,
        'success_counts': success,
        'success_rate': (success / total_test_number),
        'average_duration_time': np.mean(episode_time_record),
    }

    # save config and result to json file
    safety_layer_model_path = model_path if args.safety else None

    rollout_config_dict = {
        'safety': args.safety,
        'safety_layer': safety_layer_model_path,
        'task_option': args.task_option,
        'seed': args.tm_seed,
        'test_result': test_result,
    }

    save_path = os.path.join('./rollout_result/', TIMESTAMP)
    with open(os.path.join(save_path, 'result.json'), 'w') as fp:
        json.dump(rollout_config_dict, fp, indent=2)


if __name__ == '__main__':

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
        help='Task option for env initialization. Available options: left, right, right, straight, multi-task',
    )
    argparser.add_argument(
        '-m', '--multi-task',
        action='store_true',
        dest='multi_task',
        help='Ablation experiments options on the multi-task option.',
    )
    argparser.add_argument(
        '-d', '--render',
        action='store_true',
        dest='render_mode',
        help='Whether use no-render mode on CARLA server, default is no-render mode.'
    )
    argparser.add_argument(
        '-n', '--test_number',
        default=int(1000),
        type=int,
        dest='test_number',
        help='Total test episode number.')

    args = argparser.parse_args()

    # automatically fix tm_port according to carla port number
    if args.carla_port != int(2000):
        d = args.carla_port - 2000
        args.tm_port = int(8100+d)

    # arg multi-task has higher priority than route
    if args.multi_task:
        args.task_option = 'multi_task'

    run_env(args)

