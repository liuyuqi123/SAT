"""
todo some methods are shared with resume_training.
 deploy argparser to run from cmd line

Running the test loop.

We deploy argparser
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
import tensorflow as tf

import json
import argparse
import logging
import pickle
from datetime import datetime

# carla env
from gym_carla.envs.carla_env_multi_task import CarlaEnvMultiTask

from rl_agents.td3_old.multi_task.developing.train_ablation import restore_safety_layer
from gym_carla.safety_layer.data_analysis import tan_normalize

# network classes
from rl_agents.td3_old.multi_task.networks.pure_td3_networks import ActorNetwork, CriticNetwork
from rl_agents.td3_old.multi_task.networks.mt_td3_networks import MtActorNetwork, MtCriticNetwork
from rl_agents.td3_old.multi_task.networks.soc_td3_networks import SocActorNetwork, SocCriticNetwork
from rl_agents.td3_old.multi_task.networks.soc_mt_td3_networks import SocMtActorNetwork, SocMtCriticNetwork

from rl_agents.td3_old.multi_task.developing.train_ablation import init_carla_env

# hyper parameters
from rl_agents.td3_old.multi_task.developing.rl_config import hyperParameters


# # assign available GPU in cmd line
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

safety_model_path = os.path.join(project_path, 'gym_carla/safety_layer/safetylayer_outputs/runnable_models')


def init_tensorflow():
    # tensorflow init
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True

    # # todo this line is not needed in test
    # # reset tensorflow graph
    # tf.compat.v1.reset_default_graph()

    return configProto


def load_experiment_config(result_folder):
    """
    Get config dict file from the result folder.

    :param result_folder:
    :return:
    """
    config_path = os.path.join(result_folder, 'config.json')
    with open(config_path, 'r') as fp:
        config_dict = json.load(fp)

    return config_dict


def run_test(args):
    """
    Run the test loop using current trained model.
    """
    # not necessary to init in test
    rl_config = hyperParameters

    # Load the model and config file
    config_dict = load_experiment_config(args.model_path)

    # restore info from the dict
    task_option = config_dict['task']
    attention = config_dict['attention']
    multi_task = config_dict['multi_task']
    safety = config_dict['safety']

    # fix task option for env init
    if multi_task:
        task_option = 'multi_task'

    # ====================================================================
    # determine the Network class and ablation category through args
    if attention:
        if not multi_task:
            actor_cls = SocActorNetwork
            critic_cls = SocCriticNetwork
            memory_option = 'attention'
        else:  # multi-task and attention
            actor_cls = SocMtActorNetwork
            critic_cls = SocMtCriticNetwork
            memory_option = 'mt_attention'
    else:
        if multi_task:  # multi-task without attention
            actor_cls = MtActorNetwork
            critic_cls = MtCriticNetwork
            memory_option = 'multi_task'
        else:  # baseline
            actor_cls = ActorNetwork
            critic_cls = CriticNetwork
            memory_option = 'baseline'

    if safety:
        ablation_option = memory_option + '_safety'
    else:
        ablation_option = memory_option

    # ==================================================
    # restore the safety layer
    if safety:
        # models are stored in safet_layer folder
        model_path = os.path.join(safety_model_path, task_option)

        # run with safety layer trained using hybrid data
        if args.hybrid_safety:
            if task_option is 'left':
                model_path = os.path.join(safety_model_path, task_option, 'clip_linear_hybrid')

        # default safety layer class is included in the restore method
        action_modifier, safety_layer, config_dict = restore_safety_layer(output_path=model_path)

        print('Safety layer is restored successfully.')

    # init AC NN
    actor = actor_cls(rl_config, 'actor')
    # # todo critics are not needed, remove critics
    # critic_1 = critic_cls(rl_config, 'critic_1')
    # critic_2 = critic_cls(rl_config, 'critic_2')

    # # tensorflow summary for tensorboard visualization
    # log_path = os.path.join(rl_config.output_path, 'log')
    # writer = tf.compat.v1.summary.FileWriter(log_path)

    # get model dict through route option
    file_list = os.listdir(args.model_path)
    model_name = None
    for file in file_list:
        if file.split('.')[-1] == 'meta':
            model_name = file.split('.')[0] + '.ckpt'
            break
    if not model_name:
        raise ValueError('model dict not found, please check.')
    model_path = os.path.join(args.model_path, model_name)

    # # this line is not required in test phase
    configProto = init_tensorflow()
    saver = tf.compat.v1.train.Saver()

    # init a carla env
    env_cls = CarlaEnvMultiTask
    env = env_cls(
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        tm_seed=args.tm_seed,  # seed for autopilot controlled NPC vehicles

        task_option=task_option,
        train=False,  # collision rate is set to 99% in none-train mode
        attention=attention,

        initial_speed=None,  # if set a initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        no_render_mode=not args.render_mode,
        debug=False,
        verbose=False,
    )

    with tf.compat.v1.Session(config=configProto) as sess:
        # load network
        # saver = tf.compat.v1.train.import_meta_graph(rl_config.model_save_path + '.meta')
        # saver.restore(sess, rl_config.model_save_path)

        # saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        if saver is None:
            print("Fail to load model, please check.")

        # ===============  run the test loop  ===============

        # total test number
        total_test_number = args.test_number

        elapsed_episode_number = 0
        # elapsed time step number in current episode
        step = 0

        # counters
        success = 0
        failure = 0
        collision = 0
        time_exceed = 0
        episode_time_record = []

        episode_reward = 0

        # reset to start
        state = env.reset()
        # done = False

        while elapsed_episode_number < total_test_number:

            # get action
            action = actor.get_action(sess, state)
            action = np.squeeze(action)

            if safety:
                # retrieve safety value
                # todo fix the env API for getting constraint value
                state_data_dict, full_data_dict = env.get_single_frame_data()

                # use the 'ttc_2'
                c_key = 'ttc_2'
                c = state_data_dict[c_key]  # original c value

                # only use action modification under limited condition
                if c <= 4.:
                    # clip with linear transformation
                    alpha = 6.
                    beta = .9
                    # fixed_c = alpha * env.simulator_timestep_length - beta * c

                    norm_c = alpha * env.simulator_timestep_length - beta * c

                    # ========== original lines ==========

                    # todo constraints are supposed to be in list
                    # debug_info is added into the return of action modification
                    action_new, debug_info = action_modifier(state, action, [norm_c])

                    # update action
                    action = action_new

                    # print to debug
                    if any(abs(debug_info['correction']) > 0.):
                        print(
                            '\n',
                            '-.-' * 20, '\n',
                            'Episode step number: {}'.format(env.episode_step_number), '\n',
                            'ttc value: ', c, '\n',
                            'constraint value: ', debug_info['constraint'], '\n',
                            'c next predicted: ', debug_info['c_next_predicted'], '\n',
                            '-' * 5, 'modifications', '-' * 5, '\n',
                            'g value: ', debug_info['g'], '\n',
                            'multipliers: ', debug_info['multipliers'], '\n',
                            'correction: ', debug_info['correction'], '\n',
                            'original action: ', debug_info['original_action'], '\n',
                            'new action: ', debug_info['new_action'], '\n',
                            '-.-' * 20, '\n',
                        )

                        print('')

            # tick single step of RL
            state, reward, done, info = env.step(action)

            episode_reward += reward
            step += 1

            # record single episode result
            if done:

                elapsed_episode_number += 1

                # retrieve episode time from env
                episode_duration_time = env.episode_time

                if info['exp_state'] == 'collision':
                    collision += 1
                    failure += 1
                elif info['exp_state'] == 'time_exceed':
                    time_exceed += 1
                    failure += 1
                else:
                    episode_time_record.append(episode_duration_time)
                    success += 1

                # print result
                print(
                    '\n',
                    '---' * 20, '\n',
                    "Episode No.{} ended. Result is {}".format(elapsed_episode_number, info['exp_state']), '\n',
                    'Total time step number is: {}, duration time is: {:.1f} s'.format(step, episode_duration_time), '\n',
                    '-*-' * 15, '\n',
                    "Success Counts: {}/{}".format(success, elapsed_episode_number), '\n',
                    "Success Rate: {:.2%}".format(success / elapsed_episode_number), '\n',
                    "Average Duration Time is {:.2f}".format(np.mean(episode_time_record)), '\n',
                    '---' * 20, '\n',
                )

                # reset env
                state = env.reset()
                episode_reward = 0
                step = 0

        # print total test result
        print(
            '\n',
            '='*50, '\n',
            '-' * 15, ' Final Test Result ', '-' * 15, '\n',
            'Total test episode number is: {}.'.format(total_test_number), '\n',
            "Total Success Rate: {:.2%}".format(success / total_test_number), '\n',
            "Total Average Duration Time is {:.2f}".format(np.mean(episode_time_record)), '\n',
            '=' * 50, '\n',
        )

        # save the result file
        config_dict = {
            'model_path': args.model_path,
            'task_option': task_option,
            'attention': attention,
            'multi_task': multi_task,
            'safety': safety,
            'tm_seed': args.tm_seed,
            'safety_type': args.hybrid_safety,
        }

        result_info = {
            'total_test_number': total_test_number,
            'success': success,
            'success_rate': (success / total_test_number),
            'collision': collision,
            'time_exceed': time_exceed,
            'average time': np.mean(episode_time_record),
        }

        test_result_dict = {
            'config': config_dict,
            'result': result_info,
        }

        save_path = os.path.join('./test_outputs/', ablation_option, TIMESTAMP)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'result.json'), 'w') as fp:
            json.dump(test_result_dict, fp, indent=2)

        print('Test result is saved.')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='Test the trained model.')
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
        '--debug',
        action='store_true',
        dest='debug',
        help='Run with debug mode.')
    argparser.add_argument(
        '-m', '--model-path',
        default=None,
        type=str,
        dest='model_path',
        help='The datetime folder of the training result is supposed to be given.')
    argparser.add_argument(
        '-d', '--render',
        action='store_true',
        dest='render_mode',
        help='Whether use no-render mode on CARLA server, default is no-render mode.'
    )
    argparser.add_argument(
        '-n', '--test-number',
        default=int(1000),
        type=int,
        dest='test_number',
        help='Total test episode number.',
    )
    argparser.add_argument(
        '-r', '--hybrid',
        action='store_true',
        dest='hybrid_safety',
        help='In single task, use safety layer trained with hybrid data.'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # print(__doc__)

    # automatically fix tm_port according to carla port number
    if args.carla_port != int(2000):
        d = args.carla_port - 2000
        args.tm_port = int(8100+d)

    if args.debug:

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args.test_number = int(20)

        print('Run test in debug mode.')

    # check model path
    if not args.model_path:
        raise ValueError('The model path must be assigned.')

    try:
        run_test(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


