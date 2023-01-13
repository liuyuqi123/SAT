"""
This is the main entrance of rl agent training and testing.

In this script, we will deploy TD3 agent solely and combined with:
 - social attention module
 - multi-task
 - safety layer
 - all above with ablation



Notice:
 - This script inherits from previous rl_agent2.py file.

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

# todo add safety layer
# from torch.optim.lr_scheduler import MultiStepLR
# from experiments.safe_td3.safety_layer import SafetyLayer

import argparse
import logging
import pickle
from datetime import datetime

# carla env
from gym_carla.envs.carla_env_multi_task import CarlaEnvMultiTask

# todo add all ablation experiments modules to import
from rl_utils import ActorNetwork, CriticNetwork, Memory, get_split_batch
from rl_config import hyperParameters

from rl_agents.td3_old.single_task.without_attention.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

reload_world_period = int(1000)


def init_tensorflow():
    # tensorflow init
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


def train_loop(args, rl_config):

    # init a carla env
    env = init_carla_env(
        env_cls=CarlaEnvMultiTask,
        args=args,
    )

    # todo fix safety layer methods, move it to external loop
    # whether to use action modifier
    # use_action_modifier = True

    # if use_action_modifier:
    #     # load safety layer
    #     action_modifier = get_action_modifier()

    configProto = init_tensorflow()

    # OU noise for action exploration
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.array([0., 0.]),
        sigma=np.array([0.3, 0.3]),
        theta=np.array([0.15, 0.15]),
        dt=env.simulator_timestep_length,  # set to env timestep, default is 0.05
        x0=np.array([0., 0.]),
    )

    # todo compare normal noise for action exploration
    # normal_noise = NormalActionNoise()

    actor = ActorNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lra, 'actor')
    critic_1 = CriticNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lrc, 'critic_1')
    critic_2 = CriticNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lrc, 'critic_2')

    # tensorflow summary for tensorboard visualization
    log_path = os.path.join(rl_config.output_path, 'log')
    writer = tf.compat.v1.summary.FileWriter(log_path)
    # losses
    # tf.compat.v1.summary.scalar("Loss", critic.loss)
    # tf.compat.v1.summary.histogram("ISWeights", critic.ISWeights)
    # write_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    # initialize memory and fill it with examples, for prioritized replay
    os.makedirs(rl_config.memory_path, exist_ok=True)

    memory = Memory(rl_config.memory_size, rl_config.pretrain_length, rl_config.action_size)
    if rl_config.load_memory:
        memory = memory.load_memory(os.path.join(rl_config.memory_path, 'memory.pkl'))
        print("MEMORY: Memory Loaded")
    else:
        # initialize memory and fill it with examples, for prioritized replay
        os.makedirs(rl_config.memory_path, exist_ok=True)

        memory.fill_memory(env)
        memory.save_memory(os.path.join(rl_config.memory_path, 'memory.pkl'), memory)
        print("MEMORY: Memory Filled")

    # make result folder
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

        # reset the episode number
        env.reset_episode_count()

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
                # add noise on original action
                # todo compare with and without noise decay
                action_noise = action + ou_noise() * epsilon

                # todo add safety layer method
                # ================   use action modifier   ================
                # if use_action_modifier:
                #     c = env.query_constraint_value(state)
                #     action_new = action_modifier(state, action_noise, c)
                # else:
                #     action_new = action_noise

                action_new = action_noise
                next_state, reward, done, info = env.step(action_new)

                episode_reward += reward
                experience = state, action_noise, reward, next_state, done
                memory.store(experience)

                # Lets learn
                if env.episode_step_number % rl_config.train_frequency == 0:
                    # "Delayed" Policy Updates
                    policy_delayed = rl_config.policy_delayed
                    for _ in range(policy_delayed):
                        # First we need a mini-batch with experiences (s, a, r, s', done)
                        tree_idx, batch, ISWeights_mb = memory.sample(rl_config.batch_size)
                        # print(ISWeights_mb)
                        s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(batch)
                        # print(a_mb.shape)

                        # Get q_target values for next_state from the critic_target
                        a_target_next_state = actor.get_action_target(sess, next_s_mb)  # with Target Policy Smoothing
                        q_target_next_state_1 = critic_1.get_q_value_target(sess, next_s_mb, a_target_next_state)
                        q_target_next_state_2 = critic_2.get_q_value_target(sess, next_s_mb, a_target_next_state)
                        q_target_next_state = np.minimum(q_target_next_state_1, q_target_next_state_2)

                        # Set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Qtarget(s',a')
                        target_Qs_batch = []
                        for i in range(0, len(dones_mb)):
                            terminal = dones_mb[i]
                            # if we are in a terminal state. only equals reward
                            if terminal:
                                target_Qs_batch.append((r_mb[i]))
                            else:
                                # take the Q taregt for action a'
                                target = r_mb[i] + rl_config.gamma * q_target_next_state[i]
                                target_Qs_batch.append(target)
                        targets_mb = np.array([each for each in target_Qs_batch])

                        # critic train
                        if len(a_mb.shape) > 2:
                            a_mb = np.squeeze(a_mb, axis=1)
                        loss, absolute_errors = critic_1.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)
                        loss_2, absolute_errors_2 = critic_2.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)
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
                    memory.batch_update(tree_idx, absolute_errors)

                # store episode data or continue
                if done:

                    # save checkpoint model
                    if episode <= 3000:
                        model_save_frequency = 500
                    else:
                        model_save_frequency = rl_config.model_save_frequency
                    if (episode) % model_save_frequency == 0:
                        ckpt_path = os.path.join(rl_config.model_ckpt_path, 'checkpoint_' + str(episode) + '.ckpt')
                        saver.save(sess, ckpt_path)
                        print('model saved')

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
                    # todo add this to rl_config
                    if episode > 4000:
                        if episode % rl_config.model_save_frequency_high_success == 0:
                            if rl_config.model_save_frequency and avarage_success_rate >= 0.985:  # ref success rate
                                best_model_path = os.path.join(rl_config.best_model_path,
                                                               'best_model_' + str(episode) + ".ckpt")
                                saver.save(sess, best_model_path)
                                print('best model saved')

                        # save the latest model
                        if episode % rl_config.latest_model_save_frequency == 0:
                            latest_model_path = os.path.join(rl_config.output_path, args.route+'_latest_model.ckpt')
                            saver.save(sess, latest_model_path)
                            print('latest model saved')

                    recent_success_rate.append(avarage_success_rate)
                    recent_success_rate_data = np.array(recent_success_rate)
                    d = {"recent_success_rates": recent_success_rate_data}
                    with open(os.path.join(result_path, "success_rate_data" + ".pkl"), "wb") as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
                    # print results on terminal
                    print('---' * 15)
                    print("avarage_rewards:", np.mean(recent_rewards))
                    print("recent_success_rate:", avarage_success_rate)
                    print(episode, 'episode finished. Episode total reward:', episode_reward)
                    print(episode, 'episode finished. Episode total steps:', env.episode_step_number, 'Episode time:',
                          env.episode_step_number*env.simulator_timestep_length)
                    print('---' * 15)
                    break
                else:
                    state = next_state


def test_loop(args, rl_config):

    #
    env = init_carla_env(
        env_cls=CarlaEnvMultiTask,
        args=args,
    )
    # todo get model dict through route option
    # model_path = ''

    # check route, use straight not straight_0 and straight_1
    if args.route not in ['left', 'right', 'straight']:
        raise ValueError('Route option is wrong.')

    # todo fix this
    # best ou process model
    parent_folder = '/home1/lyq/PycharmProjects/gym-carla/outputs/best_models_ou_process'
    model_path = os.path.join(parent_folder, args.route)

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

        # init parameters for training
        test_num = 100

        test_ep = 0
        step = 0  # step in one episode
        success = 0
        failure = 0
        collision = 0
        time_exceed = 0
        episode_time_record = []

        episode_reward = 0
        state = env.reset()
        # done = False

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
        attention=False,

        initial_speed=None,  # if set a initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        debug=False,
        verbose=False,
    )

    return env


def render_loop(args):
    """"""

    rl_config = hyperParameters

    # fix args in debug mode
    if args.debug:
        rl_config.total_episodes = 30
        rl_config.noised_episodes = 5
        rl_config.model_save_frequency = 33

        rl_config.pretrain_length = 20
        args.tag = 'debug'

        rl_config.lra = 1.
        rl_config.decay_steps = 1
        rl_config.decay_rates = 0.5

        rl_config.memory_path = os.path.join('./outputs', 'memory/debug', args.task_option)
        os.makedirs(rl_config.memory_path, exist_ok=True)

        # # deprecated
        # global reload_world_period
        # reload_world_period = int(3)

        print('Some parameters for debug mode is fixed.')


    # todo get state dimension according to state option
    # if args.state_option not in state_options.keys():
    #     raise ValueError('Wrong state option, please check')
    # elif args.state_option == 'sumo':
    #     rl_config.ego_feature_num = int(4)
    # elif args.state_option == 'kinetics':
    #     rl_config.ego_feature_num = int(3)

    # if args.state_option == 'absolute_all':

    # # init a carla env
    # env = init_carla_env(
    #     env_cls=CarlaEnvMultiTask,
    #     args=args,
    # )

    # todo fix try method with restore the carla client
    try:
        # training or testing
        if args.test:
            print('---------Begin TESTing---------')
            test_loop(args, rl_config(args))
        else:
            print('---------Begin TRAINing---------')
            train_loop(args=args, rl_config=rl_config(args))
    finally:
        pass


def main():

    argparser = argparse.ArgumentParser(
        description='CARLA Intersection Scenario')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
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
        '--route',
        default="left",  # multi-task is added into env class
        dest='task_option',
        help='Task option for env initialization. Available options: left, right, right, straight, multi-task')
    argparser.add_argument(
        '--multi-task',
        action="store_true",
        dest='multi_task',
        help='whether run multi task training')
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
