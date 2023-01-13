"""
Run env loop of stochastic test for the rules based agent.

The rules-based agent should inherit agent class.

"""

# ==================================================
# -------------  import carla module  -------------
# ==================================================
import glob
import os
import sys

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

# ==================================================

# ================   Append Project Path   ================
path = os.getcwd()
index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1
proj_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'
sys.path.append(proj_path)

# set the scenario runner root
scenario_runner_path = os.path.abspath(os.path.join(proj_path, '..', 'scenario_runner'))
sys.path.append(scenario_runner_path)


import numpy as np
import tensorflow as tf
import argparse
import logging
import pickle
import json
from datetime import datetime

from rl_utils import ActorNetwork, CriticNetwork, Memory, get_split_batch
from rl_config import DebugConfig, hyperParameters

from gym_carla.envs.carla_env4 import CarlaEnv4
from gym_carla.envs.carla_env4_fixed import CarlaEnv4Fixed
from gym_carla.envs.carla_env4_fixed3 import CarlaEnv4Fixed3

# rules-based agents
from rl_agents.challenge_agents.rules_based_agents.test_agent_AEB import TestAgentAeb
from rl_agents.challenge_agents.rules_based_agents.test_agent_IDM import TestAgentIdm


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def test_loop(args, rl_config, agent_cls):

    model_path = ''

    # todo test rules based methods with safety layer
    # use_action_modifier = True
    # if use_action_modifier:
    #     # load safety layer
    #     action_modifier = get_action_modifier()


    # env = CarlaEnv4(
    #     carla_port=args.carla_port,
    #     tm_port=args.tm_port,
    #     route_option=args.route_option,
    #     state_option=args.state_option,
    #     tm_seed=args.tm_seed,
    #     initial_speed=args.initial_speed,
    #     use_tls_control=args.use_tls_control,
    #     switch_traffic_flow=args.switch_traffic_flow,
    #     multi_task=args.multi_task,
    #     attention=False,
    #     debug=True,
    #     # training=True,
    # )

    # fixed version
    env = CarlaEnv4Fixed3(
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
        # ==========   new args for traffic flow   ==========
        tf_randomize=True,  # True for training, False to run with fixed traffic flow
        collision_prob_decay=False,  # if True, enable collision prob decay
        tf_params_decay=False,  # traffic params decay, shrink to a compact range with training proceeding
    )

    env.traffic_flow.clear_traffic_flow_info()
    # reset env to init some key info
    _ = env.reset()

    # get critical info from env
    carla_api = env.carla_api
    junction = env.junction
    ego_route = env.ego_route

    # AEB
    # agent_cls = TestAgentAeb

    # IDM
    # agent_cls = TestAgentIdm

    agent = agent_cls(None)

    # set critical parameters for agent instance
    agent.set_client(
        client=carla_api['client'],
        tm_port=args.tm_port,
    )
    agent.set_junction(junction)
    agent.set_route(ego_route)
    agent.set_ego_vehicles(env.ego_vehicle)
    agent.reset_buffer()

    # ====================   loop   =========================

    # total test episode number
    test_num = args.test_number

    test_ep = 0  # elapsed epispde
    step = 0  # step in one episode
    success = 0
    failure = 0
    collision = 0
    time_exceed = 0
    episode_time_record = []

    episode_reward = 0
    # state = env.reset()
    done = False

    while test_ep < test_num:

        # get action from the agent
        action = agent.run_step(None, None)

        # fixme
        # if use_action_modifier:
        #     c = env.query_constraint_value(state)
        #     action_new = action_modifier(state, action, c)
        # else:
        #     # not use action modifier
        #     action_new = action

        state, reward, done, info = env.step(action)

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

            # reset the env
            state = env.reset()
            # init info of agent
            agent.set_ego_vehicles(env.ego_vehicle)
            agent.reset_buffer()

            episode_reward = 0
            step = 0
            done = False
            test_ep += 1

    print('-*' * 15, ' result ', '-*' * 15)
    print('current route task: {}'.format(args.route_option))
    print('current model type: {}'.format(agent_cls.__name__))
    print('total test episode number: {}'.format(test_num))
    print('Success rate: {:.2%}'.format(success / test_num))
    print('success: ', success, '/', test_num)
    print('collision: ', collision, '/', test_num)
    print('time_exceed: ', time_exceed, '/', test_num)
    print('average time: ', np.mean(episode_time_record))

    result_dict = {
        'task': args.route_option,
        'model': agent_cls.__name__,
        'Success rate': (success / test_num),
        'average time: ': np.mean(episode_time_record),
        'success': success,
        'collision': collision,
        'time_exceed': time_exceed,
    }

    # ===============  store result  ===============
    result_path = os.path.join('./rules_based_stochastic_test', TIMESTAMP)
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, 'result.jsonl'), 'a') as f:
        f.write(json.dumps(result_dict) + '\n')

    try:
        env.client.reload_world()
        print('Reload carla world successfully.')
    except:
        raise RuntimeError('Fail to reload carla world.')


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

        for route in ['straight']:  # ['left', 'right', 'straight']

            for agent_cls in [TestAgentAeb, TestAgentIdm]:  # [TestAgentAeb, TestAgentIdm]

                if route in ['left', 'right'] and agent_cls is TestAgentIdm:
                    continue

                args.route_option = route

                test_loop(args, rl_config, agent_cls)

    finally:
        pass


def main():

    argparser = argparse.ArgumentParser(
        description='CARLA Intersection Scenario',
    )
    argparser.add_argument(
        '--test_number',
        default=int(300),
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

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # print(__doc__)

    # add model path for test phase

    try:
        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
