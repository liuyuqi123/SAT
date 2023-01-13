"""
todo how to know that the model is resumed correctly??

Develop the restore model method and test it.
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

# network classes
from rl_agents.td3_old.multi_task.networks.pure_td3_networks import ActorNetwork, CriticNetwork
from rl_agents.td3_old.multi_task.networks.mt_td3_networks import MtActorNetwork, MtCriticNetwork
from rl_agents.td3_old.multi_task.networks.soc_td3_networks import SocActorNetwork, SocCriticNetwork
from rl_agents.td3_old.multi_task.networks.soc_mt_td3_networks import SocMtActorNetwork, SocMtCriticNetwork

from rl_agents.td3_old.multi_task.networks.prioritized_replay import Memory
from rl_agents.td3_old.multi_task.networks.utils import get_split_batch

from rl_agents.td3_old.single_task.without_attention.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from gym_carla.safety_layer.data_analysis import tan_normalize

# hyper parameters
from rl_agents.td3_old.multi_task.developing.rl_config import hyperParameters

from rl_agents.td3_old.multi_task.developing.train_ablation import init_tensorflow, restore_safety_layer

from rl_agents.td3_old.multi_task.developing.run_test import load_experiment_config

# append safety layer model path
safety_model_path = os.path.join(project_path, 'safety_layer/safetylayer_outputs/runnable_models')

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

# # assign available GPU in cmd line
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def resume_training(args, rl_config):
    """

    """

    # todo fix rl_config with loaded config dict
    rl_config = hyperParameters()

    # Load the model and config file
    config_dict = load_experiment_config(args.model_path)

    # restore info from the dict
    task_option = config_dict['task_option']
    attention = config_dict['attention']
    multi_task = config_dict['multi_task']
    safety = config_dict['safety']

    # todo test this line
    if multi_task:
        task_option = 'multi_task'
    else:
        task_option = task_option

    # init a carla env
    env_cls = CarlaEnvMultiTask
    env = env_cls(
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        tm_seed=args.tm_seed,  # seed for autopilot controlled NPC vehicles

        task_option=task_option,
        train=True,  # training mode or evaluation mode
        attention=attention,

        initial_speed=None,  # if set a initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        no_render_mode=args.no_render_mode,
        debug=False,
        verbose=False,
    )

    # ==================================================
    # determine the Network class through args
    if attention:
        if not multi_task:
            actor_cls = SocActorNetwork
            critic_cls = SocCriticNetwork
        else:  # multi-task and attention
            actor_cls = SocMtActorNetwork
            critic_cls = SocMtCriticNetwork
    else:
        if multi_task:  # multi-task without attention
            actor_cls = MtActorNetwork
            critic_cls = MtCriticNetwork
        else:  # baseline
            actor_cls = ActorNetwork
            critic_cls = CriticNetwork

    # init AC NN
    actor = actor_cls(rl_config, 'actor')
    critic_1 = critic_cls(rl_config, 'critic_1')
    critic_2 = critic_cls(rl_config, 'critic_2')

    # restore the safety layer
    if safety:
        # models are stored in safet_layer folder
        model_path = os.path.join(safety_model_path, args.task_option)
        # default safety layer class is included in restore_safety_layer method
        action_modifier, safety_layer, config_dict = restore_safety_layer(args, output_path=model_path)

    configProto = init_tensorflow()

    # OU noise for action exploration
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.array([0., 0.]),
        sigma=np.array([0.3, 0.3]),
        theta=np.array([0.15, 0.15]),
        dt=env.simulator_timestep_length,  # set to env timestep, default is 0.05
        x0=np.array([0., 0.]),
    )

    # tensorflow summary for tensorboard visualization
    log_path = os.path.join(rl_config.output_path, 'log')
    writer = tf.compat.v1.summary.FileWriter(log_path)

    # todo save the info by the tensorboard
    # losses
    # tf.compat.v1.summary.scalar("Loss", critic.loss)
    # tf.compat.v1.summary.histogram("ISWeights", critic.ISWeights)
    # write_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    # ===============  fix memory initialization  ===============
    # init the memory instance
    memory = Memory(rl_config.memory_size, rl_config.pretrain_length)

    # try to load the stored memory
    try:
        # todo add memory dimension check
        memory = memory.load_memory(os.path.join(rl_config.memory_path, 'memory.pkl'))
        print("Memory is loaded.")
    except:
        print('Fail to load memory, will directly continue training.')

    # todo add codes to merge success rate and reward result

    # # ==========  original lines  ==========
    # # init the rl_results folder for success rate and
    # result_path = os.path.join(rl_config.output_path, 'rl_results')
    # os.makedirs(result_path, exist_ok=True)




    # RL training loop
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

        # todo fix the remaining training episodes
        for episode in range(1, rl_config.total_episodes + 1):

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
            min_factor = 0.25
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
                    # whether use safety
                    if args.safety:
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

                    # todo add a method to better save best models
                    # calculate success rate and save best models
                    avarage_success_rate = recent_success.count(1) / len(recent_success)

                    # ===============   save best model   ===============
                    # save model after 4k episodes
                    if episode > 4000:
                        # save best model
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
                    # print results on terminal
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
                    break
                else:
                    state = next_state


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='Test trained model.')
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
        '--no-render',
        action='store_true',
        dest='no_render_mode',
        help='Whether use the no render mode.')
    argparser.add_argument(
        '--model-path',
        default=None,
        type=str,
        dest='model_path',
        help='Output path of training')
    argparser.add_argument(
        '-n', '--test-number',
        default=int(100),
        type=int,
        dest='test_number',
        help='Total test episode number.')

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
        resume_training(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


