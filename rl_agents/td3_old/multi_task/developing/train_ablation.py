"""
A developing version of train loop for ablation experiments.
"""

import glob
import os
import sys

# ================   Append Project Path   ================
# path = os.getcwd()
# index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1
# project_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'

project_path = '/home1/lyq/PycharmProjects/gym-carla/'
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

# network classes
from rl_agents.td3_old.multi_task.networks.pure_td3_networks import ActorNetwork, CriticNetwork
from rl_agents.td3_old.multi_task.networks.mt_td3_networks import MtActorNetwork, MtCriticNetwork
from rl_agents.td3_old.multi_task.networks.soc_td3_networks import SocActorNetwork, SocCriticNetwork
from rl_agents.td3_old.multi_task.networks.soc_mt_td3_networks import SocMtActorNetwork, SocMtCriticNetwork

from rl_agents.td3_old.multi_task.networks.prioritized_replay import Memory
from rl_agents.td3_old.multi_task.networks.utils import get_split_batch

from rl_agents.td3_old.single_task.without_attention.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from gym_carla.safety_layer.safety_layer_devised import SafetyLayerDevised
from gym_carla.safety_layer.data_analysis import tan_normalize
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# hyper parameters
from rl_agents.td3_old.multi_task.developing.rl_config import hyperParameters

from rl_agents.td3_old.multi_task.result_analysis.plot_results import plot_curves


# # assign available GPU in cmd line
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

# append safety layer model path
safety_model_path = os.path.join(project_path, 'gym_carla/safety_layer/safetylayer_outputs/runnable_models')


def init_tensorflow():
    # tensorflow init
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()

    return configProto


def retrieve_config_dict(config_instance):
    """
    Get the config dict from rl_config instance.
    """
    config_dict = {}
    for attr in dir(config_instance):
        if attr.startswith('_'):
            continue
        if not callable(getattr(config_instance, attr)):
            config_dict[attr] = getattr(config_instance, attr)

    return config_dict


def restore_safety_layer(output_path):
    """
    Load model dict to the safety layer NN, and prepare for the evaluation.

    :return: the runnable action modifier from safety layer instance
    """

    config_path = os.path.join(output_path, 'config.json')
    model_path = os.path.join(output_path, 'model_0_final.pth')

    with open(config_path, 'r') as fp:
        config_dict = json.load(fp)

    # todo fix this if using other lr scheduler
    # append lr scheduler
    if config_dict['lr_scheduler'] == 'MultiStepLR':
        config_dict['lr_scheduler'] = MultiStepLR
    else:
        raise ValueError('lr scheduler not defined.')

    safety_layer_cls = SafetyLayerDevised
    # init safety layer class
    safety_layer = safety_layer_cls(config_dict)
    # load trained safety layer model
    safety_layer.load_models([model_path])

    # the action modifier is determined by the config file, eg. state clip method
    action_modifier = safety_layer.get_safe_action

    return action_modifier, safety_layer, config_dict


def train_loop(args, rl_config):
    """
    Run the train loop.
    """

    # save the config file into result folder
    training_config_dict = {
        'task': args.task_option,
        'attention': args.attention,
        'safety': args.safety,
        'multi_task': args.multi_task,
        'tag': args.tag,
        # get the rl hyper parameters
        'rl_settings': retrieve_config_dict(rl_config),
    }

    # save to json file
    with open(os.path.join(rl_config.output_path, 'config.json'), 'w') as fp:
        json.dump(training_config_dict, fp, indent=2)

    # init a carla env
    env = init_carla_env(
        env_cls=CarlaEnvMultiTask,
        args=args,
    )

    # make comparison on reward settings
    reward_dict_indices = {
        'IV': {
            'collision': -500.,
            'time_exceed': -100.,  # not introduced in paper
            'success': 50.,
            'step': -0.15,
        }
    }
    # check the tag and actual reward setting
    for key, item in reward_dict_indices.items():
        if args.tag == key:
            print('Reward setting of {} is deployed.'.format(key))
            env.set_reward_dict(item)

    # load safety layer network if required
    if args.safety:
        # models are stored in safet_layer folder
        model_path = os.path.join(safety_model_path, args.task_option)

        # # CAUTION!!! use following path if testing new model
        # model_path = os.path.join(safety_model_path, args.task_option, 'clip_linear_hybrid')

        # default safety layer class is included in the restore method
        action_modifier, safety_layer, config_dict = restore_safety_layer(output_path=model_path)

        # # todo test setter
        # safety_layer.set_attention_flag(args.attention)

        print('Safety layer is restored successfully.')

    configProto = init_tensorflow()

    # OU noise for action exploration
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.array([0.5, 0.2]),  # 0, 0
        sigma=np.array([0.3, 0.05]),  # 0.3, 0.3
        theta=np.array([0.15, 0.15]),
        dt=env.rl_timestep_length,  # set to env timestep, default is 0.05
        x0=np.array([0.5, 0.2]),
    )

    # todo compare normal noise for action exploration
    # normal_noise = NormalActionNoise()

    # ==================================================
    # determine the NN through args
    if args.attention:
        if not args.multi_task:
            actor_cls = SocActorNetwork
            critic_cls = SocCriticNetwork
        else:  # multi-task and attention
            actor_cls = SocMtActorNetwork
            critic_cls = SocMtCriticNetwork
    else:
        if args.multi_task:  # multi-task without attention
            actor_cls = MtActorNetwork
            critic_cls = MtCriticNetwork
        else:  # baseline
            actor_cls = ActorNetwork
            critic_cls = CriticNetwork

    # init AC NN
    actor = actor_cls(rl_config, 'actor')
    critic_1 = critic_cls(rl_config, 'critic_1')
    critic_2 = critic_cls(rl_config, 'critic_2')

    # tensorflow summary for tensorboard visualization
    log_path = os.path.join(rl_config.output_path, 'log')
    writer = tf.compat.v1.summary.FileWriter(log_path)

    # todo save the info by the tensorboard
    # losses
    # tf.compat.v1.summary.scalar("Loss", critic.loss)
    # tf.compat.v1.summary.histogram("ISWeights", critic.ISWeights)
    # write_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    # ===============  Memory initialization  ===============
    # todo fix the load memory option, use a relative fixed path
    if rl_config.load_memory:
        # try to load the stored memory
        try:
            # todo add memory dimension check
            print('Try to load memory from path:\n{}'.format(os.path.join(rl_config.memory_path, 'memory.pkl')))
            memory = Memory.load_memory(os.path.join(rl_config.memory_path, 'memory.pkl'))
            print("Memory is loaded.")
        except:
            # init the memory instance
            memory = Memory(rl_config.memory_size, rl_config.pretrain_length)
            print('Fail to load memory, new memory will be collected.')
            print('Start collecting memory...')
            memory.fill_memory(env)

            # # deprecated
            # memory.save_memory(os.path.join(rl_config.memory_path, 'memory.pkl'), memory)

            print("Memory is filled.")
        finally:
            print('Memory for RL training is prepared.')
    else:
        # init the memory instance
        memory = Memory(rl_config.memory_size, rl_config.pretrain_length)
        print('Start collecting memory...')
        memory.fill_memory(env)

        # # deprecated
        # Memory.save_memory(os.path.join(rl_config.memory_path, 'memory.pkl'), memory)

        print("Memory is filled.")
        print('Memory for RL training is prepared.')

    # init the rl_results folder for success rate and
    result_path = os.path.join(rl_config.output_path, 'rl_results')
    os.makedirs(result_path, exist_ok=True)

    # Reinforcement Learning loop
    with tf.compat.v1.Session(config=configProto) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # update param of target network
        writer.add_graph(sess.graph)
        actor.update_target(sess)
        critic_1.update_target(sess)
        critic_2.update_target(sess)

        recent_rewards = []  # rewards from recent 100 episodes
        avarage_rewards = []  # average reward of recent 100 episodes
        recent_success = []
        recent_success_rate = []

        # reset internal episode number counter
        env.reset_episode_count()

        for episode in range(1, rl_config.total_episodes + 1):

            # reload_world_period = int(1000)
            # # manually reload the env periodically
            # if episode % reload_world_period == 0:
            #     print('start reloading the carla env...')
            #     env = init_carla_env(
            #         env_cls=CarlaEnvMultiTask,
            #         args=args,
            #     )
            #     print('carla env is reloaded.')

            # move the vehicle to a spawn_point and return state
            state = env.reset()
            episode_reward = 0
            done = False

            # epsilon refers to the decay rate of the noise
            min_factor = 0.15
            noised_episodes = rl_config.noised_episodes
            if episode > noised_episodes:
                epsilon = 0.
            else:
                epsilon = (min_factor - 1) / noised_episodes * episode + 1

            # rollout single episode
            while True:

                # old version of OU-noised action
                # action_noise = actor.get_action_noise(sess, state, rate=epsilon)

                # original action
                action = actor.get_action(sess, state)
                action = np.squeeze(action)

                # todo check merge exploration with safety constraints
                # do not use safety correction until noised episodes end
                if episode >= noised_episodes:
                    # use safety
                    if args.safety:
                        # retrieve safety value
                        # todo fix the env API for getting constraint value
                        state_data_dict, full_data_dict = env.get_single_frame_data()

                        # use the 'ttc_2'
                        c_key = 'ttc_2'
                        c = state_data_dict[c_key]  # original c value

                        # # todo add args on determining norm method
                        # norm_c = tan_normalize(c, 4.)

                        # New ttc settings
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
                                    '-'*5, 'modifications', '-'*5, '\n',
                                    'g value: ', debug_info['g'], '\n',
                                    'multipliers: ', debug_info['multipliers'], '\n',
                                    'correction: ', debug_info['correction'], '\n',
                                    'original action: ', debug_info['original_action'], '\n',
                                    'new action: ', debug_info['new_action'], '\n',
                                    '-.-' * 20, '\n',
                                )

                # add noise in exploration period
                action_noise = action + ou_noise() * epsilon

                action_new = action_noise
                next_state, reward, done, info = env.step(action_new)

                episode_reward += reward
                experience = state, action_noise, reward, next_state, done
                memory.store(experience)

                # todo set the learning rate for the NN through a external API
                # todo add decay schedule on train_frequency
                # Lets learn
                if env.episode_step_number % rl_config.train_frequency == 0:
                    # "Delayed" Policy Updates
                    policy_delayed = rl_config.policy_delayed
                    for _ in range(int(policy_delayed)):
                        # First we need a mini-batch with experiences (s, a, r, s', done)
                        tree_idx, batch, ISWeights_mb = memory.sample(rl_config.batch_size)
                        # print(ISWeights_mb)
                        s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(batch)
                        # print(a_mb.shape)

                        # for multi-task
                        task_code_mb = s_mb[:, -rl_config.task_code_size:]
                        next_task_code_mb = next_s_mb[:, -rl_config.task_code_size:]

                        # Get q_target values for next_state from the critic_target
                        a_target_next_state = actor.get_action_target(sess, next_s_mb)  # with Target Policy Smoothing

                        if args.multi_task:
                            # # multi task q value
                            q_target_next_state_1 = critic_1.get_q_value_target(sess, next_s_mb,
                                                                                a_target_next_state) * next_task_code_mb
                            q_target_next_state_2 = critic_2.get_q_value_target(sess, next_s_mb,
                                                                                a_target_next_state) * next_task_code_mb
                        else:
                            # original
                            q_target_next_state_1 = critic_1.get_q_value_target(sess, next_s_mb, a_target_next_state)
                            q_target_next_state_2 = critic_2.get_q_value_target(sess, next_s_mb, a_target_next_state)

                        q_target_next_state = np.minimum(q_target_next_state_1, q_target_next_state_2)

                        # Set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Qtarget(s',a')
                        target_Qs_batch = []
                        for i in range(0, len(dones_mb)):
                            terminal = dones_mb[i]
                            # if we are in a terminal state. only equals reward
                            if terminal:
                                if args.multi_task:
                                    target_Qs_batch.append((r_mb[i]*task_code_mb[i]))
                                else:
                                    target_Qs_batch.append((r_mb[i]))
                            else:
                                # take the Q taregt for action a'
                                if args.multi_task:
                                    # todo test this for the multi-task
                                    target = r_mb[i] * task_code_mb[i] + rl_config.gamma * q_target_next_state[i]
                                else:
                                    target = r_mb[i] + rl_config.gamma * q_target_next_state[i]

                                target_Qs_batch.append(target)

                        targets_mb = np.array([each for each in target_Qs_batch])

                        # critic train
                        if len(a_mb.shape) > 2:
                            a_mb = np.squeeze(a_mb, axis=1)
                        loss, absolute_errors = critic_1.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)
                        loss_2, absolute_errors_2 = critic_2.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)

                        # todo add to tensorboard to log
                        # print('loss:',loss)

                    # actor train
                    a_for_grad = actor.get_action(sess, s_mb)
                    a_gradients = critic_1.get_gradients(sess, s_mb, a_for_grad)
                    # print(a_gradients)
                    actor.train(sess, s_mb, a_gradients[0])
                    # target train
                    actor.update_target(sess)
                    critic_1.update_target(sess)
                    critic_2.update_target(sess)

                    # update replay memory priorities
                    if args.multi_task:
                        absolute_errors = np.sum(absolute_errors, axis=1)
                    memory.batch_update(tree_idx, absolute_errors)

                # store episode data or continue
                if done:
                    # save the memory
                    if episode >= noised_episodes:
                        if episode % rl_config.memory_save_frequency == 0:
                           Memory.save_memory(os.path.join(rl_config.memory_save_path, 'memory.pkl'), memory)

                    # todo traffic flow parameters decay
                    # # get and save traffic flow distribution
                    # dist_dict = env.traffic_flow.tf_distribution
                    # with open(os.path.join(result_path, "tf_distribution" + ".pkl"), "wb") as f:
                    #     pickle.dump(dist_dict, f, pickle.HIGHEST_PROTOCOL)

                    # todo plot and save the figure when saving the ckpt
                    # save checkpoint model
                    if episode <= 3000:
                        model_save_frequency = 500
                    else:
                        model_save_frequency = rl_config.model_save_frequency

                    if episode % model_save_frequency == 0:
                        ckpt_path = os.path.join(rl_config.model_ckpt_path, 'checkpoint_' + str(episode) + '.ckpt')
                        saver.save(sess, ckpt_path)
                        print('checkpoint model {} is saved.'.format(episode))

                    # visualize reward data
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    avarage_rewards.append(np.mean(recent_rewards))
                    avarage_rewards_data = np.array(avarage_rewards)
                    d = {"avarage_rewards": avarage_rewards_data}

                    with open(os.path.join(result_path, "reward_data" + ".pkl"), "wb") as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

                    # visualize success rate data
                    if info['exp_state'] == 'success':
                        recent_success.append(1)
                    else:
                        recent_success.append(0)
                    if len(recent_success) > 100:
                        recent_success.pop(0)

                    # calculate success rate and save best models
                    if len(recent_success) >= 50:
                        avarage_success_rate = recent_success.count(1) / len(recent_success)
                    else:
                        avarage_success_rate = 0.

                    # ===============   save best model   ===============
                    # save model after 4k episodes
                    if episode > 5000:
                        # save the best model
                        if episode % rl_config.model_save_frequency_high_success == 0:
                            if rl_config.model_save_frequency and avarage_success_rate >= 0.99:  # ref success rate
                                best_model_path = os.path.join(rl_config.best_model_path,
                                                               'best_model_' + str(episode) + ".ckpt")
                                saver.save(sess, best_model_path)
                                print('best model saved')

                        # save the latest model
                        if episode % rl_config.latest_model_save_frequency == 0:
                            latest_model_path = os.path.join(rl_config.output_path, args.task_option+'_latest_model.ckpt')
                            saver.save(sess, latest_model_path)
                            print('latest model saved')

                    recent_success_rate.append(avarage_success_rate)
                    recent_success_rate_data = np.array(recent_success_rate)
                    d = {"recent_success_rates": recent_success_rate_data}
                    with open(os.path.join(result_path, "success_rate_data" + ".pkl"), "wb") as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

                    # save training curves
                    save_plot_frequency = int(100)  # 100
                    if episode % save_plot_frequency == 0:
                        # save the training curves
                        title = rl_config.ablation_option
                        plot_curves(result_path, title)

                    # print results in terminal
                    print(
                        '\n',
                        '---' * 20, '\n',
                        'Episode No.{} is finished'.format(episode), '\n',
                        'Episodic total reward is: {}'.format(episode_reward), '\n',
                        'Episode total steps: {}, total duraion time: {:.2f}'.format(
                            env.episode_step_number,
                            env.episode_step_number*env.simulator_timestep_length
                        ), '\n',
                        "Average reward is {:.2f}".format(np.mean(recent_rewards)), '\n',
                        "Success rate: {:.1%}".format(avarage_success_rate), '\n',
                        '---' * 20, '\n',
                    )

                    # end single episode
                    break

                else:
                    state = next_state


def init_carla_env(env_cls, args):
    """
    todo debug using args to init a env instance

    Init the carla env instance with env cls.

    :return: env instance
    """
    # todo add use attention option

    # CAUTION! this cls is CarlaEnvMultiTask
    env = env_cls(
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        tm_seed=args.tm_seed,  # seed for autopilot controlled NPC vehicles

        task_option=args.task_option,
        train=True,  # training mode or evaluation mode
        attention=args.attention,

        initial_speed=None,  # if set a initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        no_render_mode=not args.render_mode,
        debug=False,
        verbose=False,
    )

    return env


def render_loop(args):
    """"""

    rl_config = hyperParameters

    # fix args in debug mode
    if args.debug:
        #
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        args.tag = 'debug'

        rl_config.total_episodes = 20
        rl_config.noised_episodes = 10
        rl_config.model_save_frequency = 5
        rl_config.pretrain_length = 1

        rl_config.memory_save_frequency = int(1)

        rl_config.lra = 1.
        rl_config.decay_steps = 1
        rl_config.decay_rates = 0.5

        rl_config.load_memory = True

        # # todo fix the memory collection for debug mode
        # rl_config.memory_path = os.path.join('./outputs', 'memory/debug', args.task_option)
        # os.makedirs(rl_config.memory_path, exist_ok=True)

        print('-*'*10, 'Debug mode is running', '-*'*10)
        print('Parameters for debug mode is fixed.')

    # # todo fix try except method, reload carla after running
    # try:
    #     # training or testing
    #     if args.test:
    #         print('---------Begin Testing---------')
    #         test_loop(args, rl_config(args))
    #     else:
    #         print('---------Begin Training---------')
    #         train_loop(args=args, rl_config=rl_config(args))
    # finally:
    #     pass

    print('--'*10, 'Begin training', '--'*10)
    train_loop(args=args, rl_config=rl_config(args))


def main():

    argparser = argparse.ArgumentParser(
        description='CARLA Intersection Scenario')
    # argparser.add_argument(
    #     '--test',
    #     action='store_true',
    #     dest='test',
    #     help='test a trained model')
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
        help='Task option for env initialization. Available options: left, right, right, straight, multi-task')
    argparser.add_argument(
        '--init-speed',
        default=None,
        type=float,
        dest='initial_speed',
        help='initial speed of ego vehicle')
    argparser.add_argument(
        '--tag',
        default=None,
        dest='tag',
        help='additional tag for current result path')
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
        '-s', '--safety',
        action='store_true',
        dest='safety',
        help='Whether use the safety layer.')
    argparser.add_argument(
        '-d', '--render',
        action='store_true',
        dest='render_mode',
        help='Whether use no-render mode on CARLA server, default is no-render mode.'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # print(__doc__)

    # automatically fix tm_port according to carla port number
    if args.carla_port != int(2000):
        d = args.carla_port - 2000
        args.tm_port = int(8100+d)

    # arg multi-task has higher priority than route
    if args.multi_task:
        args.task_option = 'multi_task'

    try:
        render_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
