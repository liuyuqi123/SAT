"""
Run stochastic test for the trained model.

The models for tests are trained via OU process-based traffic flow.
"""

import glob
import os
import sys

# ================   Append Project Path   ================
path = os.getcwd()
index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1
proj_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'
sys.path.append(proj_path)

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
import tensorflow as tf
import json
from datetime import datetime

import argparse
import logging
import pickle

# todo safety layer
# from torch.optim.lr_scheduler import MultiStepLR
# from experiments.safe_td3.safety_layer import SafetyLayer

# from rl_agents.td3_old.single_task.without_attention.rl_utils import ActorNetwork, CriticNetwork, Memory, \
#     get_split_batch
# from rl_agents.td3_old.single_task.without_attention.rl_config import DebugConfig, hyperParameters

from rl_utils import ActorNetwork, CriticNetwork, Memory, get_split_batch
from rl_config import DebugConfig, hyperParameters

# from gym_carla.envs.carla_env2 import CarlaEnv2
# from gym_carla.envs.carla_env3 import CarlaEnv3
from gym_carla.envs.carla_env4 import CarlaEnv4
from gym_carla.envs.carla_env4_fixed import CarlaEnv4Fixed

from rl_agents.td3_old.single_task.without_attention.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def init_tensorflow():
    # tensorflow init
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


def test_loop(args, rl_config, model_path):

    # todo get model dict through route option
    # model_path = ''

    # check route, use straight not straight_0 and straight_1
    # if args.route_option not in ['left', 'right', 'straight']:
    #     raise ValueError('Route option is wrong.')

    # parent_folder = ''
    # model_path = os.path.join(parent_folder, args.route_option)

    # todo safety layer
    # use_action_modifier = True
    # if use_action_modifier:
    #     # load safety layer
    #     action_modifier = get_action_modifier()

    configProto = init_tensorflow()
    actor = ActorNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lra, 'actor')
    # critic = CriticNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lrc, 'critic')
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=configProto) as sess:
        # load network
        # saver = tf.compat.v1.train.import_meta_graph(rl_config.model_save_path + '.meta')
        # saver.restore(sess, rl_config.model_save_path)

        # saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        if saver is None:
            print("did not load")

        # env = CarlaEnv4(carla_port=args.carla_port,
        #                 tm_port=args.tm_port,
        #                 route_option=args.route_option,
        #                 state_option=args.state_option,
        #                 tm_seed=args.tm_seed,
        #                 initial_speed=args.initial_speed,
        #                 use_tls_control=args.use_tls_control,
        #                 switch_traffic_flow=args.switch_traffic_flow,
        #                 multi_task=args.multi_task,
        #                 attention=False,
        #                 debug=True,
        #                 # training=True,
        #                 )

        env = CarlaEnv4Fixed(
            carla_port=args.carla_port,
            tm_port=args.tm_port,
            route_option=args.route_option,
            state_option=args.state_option,
            tm_seed=args.tm_seed,
            initial_speed=args.initial_speed,
            use_tls_control=args.use_tls_control,
            switch_traffic_flow=args.switch_traffic_flow,
            multi_task=args.multi_task,
            attention=False,
            debug=False,
            # training=True,
            tf_randomize=True,  # True for training, False to run with fixed traffic flow
            collision_prob_decay=False,  # if True, enable collision prob decay
            tf_params_decay=False,  # traffic params decay, shrink to a compact range with training proceeding
        )

        env.traffic_flow.clear_traffic_flow_info()
        state = env.reset()

        # ====================   loop   =========================

        # total test episode number
        test_num = args.test_number

        test_ep = 0
        step = 0  # step in one episode
        success = 0
        failure = 0
        collision = 0
        time_exceed = 0
        episode_time_record = []

        episode_reward = 0
        done = False

        while test_ep < test_num:
            action = actor.get_action(sess, state)
            # target_speed = map_action(action)
            # state, reward, done, info = env.step(target_speed)

            # if use_action_modifier:
            #     c = env.query_constraint_value(state)
            #     action_new = action_modifier(state, action, c)
            # else:
            #     # not use action modifier
            #     action_new = action

            action_new = action
            state, reward, done, info = env.step(action_new)

            episode_reward += reward
            step += 1

            if done:
                # record result
                if info['exp_state'] == 'collision':
                    collision += 1
                    failure += 1
                elif info['exp_state'] == 'time_exceed':
                    time_exceed += 1
                    failure += 1
                else:
                    # get episode time
                    episode_time_record.append(env.episode_time)
                    success += 1
                # print
                print(test_ep, "EPISODE ended", "TOTAL REWARD {:.4f}".format(episode_reward), 'Result:',
                      info['exp_state'])
                print('total step of this episode: ', step)
                state = env.reset()  # init state
                episode_reward = 0
                step = 0
                done = False
                test_ep += 1

        print('-*' * 15, ' result ', '-*' * 15)
        print('success: ', success, '/', test_num)
        print('collision: ', collision, '/', test_num)
        print('time_exceed: ', time_exceed, '/', test_num)
        print('average time: ', np.mean(episode_time_record))

        # ===============  store result  ===============
        result_dict = {
            'task': args.route_option,
            'Success rate': (success / test_num),
            'average time: ': np.mean(episode_time_record),
            'success': success,
            'collision': collision,
            'time_exceed': time_exceed,
            'total_episodes': test_num,
        }

        result_path = os.path.join('./rl_model_stochastic_test', TIMESTAMP)
        os.makedirs(result_path, exist_ok=True)

        with open(os.path.join(result_path, 'result.jsonl'), 'a') as f:
            f.write(json.dumps(result_dict) + '\n')


def render_loop(args):
    if args.debug:
        rl_config = DebugConfig
    else:
        rl_config = hyperParameters

    # todo get state dimension according to state option
    # if args.state_option not in state_options.keys():
    #     raise ValueError('Wrong state option, please check')
    # elif args.state_option == 'sumo':
    #     rl_config.ego_feature_num = int(4)
    # elif args.state_option == 'kinetics':
    #     rl_config.ego_feature_num = int(3)

    # if args.state_option == 'absolute_all':

    # rl_config.tag = rl_config.tag + '_state_' + args.state_option

    # new 2021.04.13, append route option into output path
    rl_config.tag = 'state_' + args.state_option

    try:
        print('---------Begin TESTing---------')

        # general model dict path
        # on 2080ti server
        model_folder = '/home1/lyq/PycharmProjects/gym-carla/outputs/best_models_ou_process'

        # test models of all 3 tasks
        for route in ['straight', 'left', 'right']:  # 'left', 'right', 'straight']

            if route == 'straight':
                args.route_option = 'straight_0'
            else:
                args.route_option = route

            model_base = os.path.join(model_folder, route)

            model_name = None
            files = os.listdir(model_base)
            for file in files:
                if file.split('.')[-1] == 'meta':
                    model_name = file.split('.')[0] + '.ckpt'
                    break
                else:
                    continue
            if not model_name:
                raise ValueError('model dict not found, please check.')
            model_path = os.path.join(model_base, model_name)

            test_loop(args, rl_config, model_path)

    finally:
        pass


def main():

    argparser = argparse.ArgumentParser(
        description='CARLA Intersection Scenario')
    argparser.add_argument(
        '--test_number',
        default=int(99),
        dest='test_number',
        help='Number of the test episodes.')
    argparser.add_argument(
        '--port',
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
        '--route',
        default="left",
        dest='route_option',
        help='route option for single task')
    argparser.add_argument(
        '--state',
        default="sumo_1",  # ["sumo", 'absolute_all', "sumo_1"],
        dest='state_option',
        help='state representation option')
    argparser.add_argument(
        '--init-speed',
        default=None,
        type=float,
        dest='initial_speed',
        help='initial speed of ego vehicle')
    argparser.add_argument(
        '--seed',
        default=int(0),
        type=int,
        dest='tm_seed',
        help='traffic manager seed number')
    argparser.add_argument(
        '--tag',
        default=None,
        dest='tag',
        help='additional tag for current result path')
    argparser.add_argument(
        '--switch-tf',
        action='store_true',
        dest='switch_traffic_flow',
        help='whether switch traffic flow in single task training')
    argparser.add_argument(
        '--tls-control',
        action='store_true',
        dest='use_tls_control',
        help='whether use traffic lights control')
    argparser.add_argument(
        '--multi-task',
        action="store_true",
        dest='multi_task',
        help='whether run multi task training')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    # todo add other args: attention, safety, model path for evaluation

    args = argparser.parse_args()

    try:
        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
