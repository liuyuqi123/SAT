"""
Entrance for training and testing the RL agent.

"""

import sys
sys.path.append('/home/gdg/CarlaRL/rl_cross_gyf_multi_task/')

import glob
import os
import time
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import numpy as np
import tensorflow as tf

import argparse
import pickle
from pure_td3_networks import ActorNetwork, CriticNetwork
from soc_td3_networks import SocActorNetwork, SocCriticNetwork
from soc_mt_td3_networks import SocMtActorNetwork, SocMtCriticNetwork
from prioritized_replay import Memory
from utils import get_split_batch
from config import hyperParameters

from gym_carla.envs.carla_env import CarlaEnv

def init_tensorflow():
    # tensorflow init
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto

def train_loop(args, config, fine_tune=False):
    configProto = init_tensorflow()
    fine_tune = fine_tune
    attention = True
    if not args.route:
        print('Please assign a route for train!!!')
        return
    else:
        route_option = args.route
    if route_option == 'left':
        port = 2000
    elif route_option == 'straight':
        port = 3000
    elif route_option == 'right':
        port = 4000
    elif route_option == 'multi':
        port = 5000
        multi_task = True
    env = CarlaEnv(port=port,
                    attention=attention,
                    route_option=route_option,
                    train=True,
                    debug=False)
    if multi_task:
        actor = SocMtActorNetwork(config, 'actor')
        critic_1 = SocMtCriticNetwork(config, 'critic_1')
        critic_2 = SocMtCriticNetwork(config, 'critic_2') 
    elif attention:
        actor = SocActorNetwork(config, 'actor')
        critic_1 = SocCriticNetwork(config, 'critic_1')
        critic_2 = SocCriticNetwork(config, 'critic_2')
    else:
        actor = ActorNetwork(config, 'actor')
        critic_1 = CriticNetwork(config, 'critic_1')
        critic_2 = CriticNetwork(config, 'critic_2')
    
    # tensorflow summary for tensorboard visualization
    writer = tf.compat.v1.summary.FileWriter("log")
    # losses
    # tf.compat.v1.summary.scalar("Loss_c1", critic_1.loss)
    # tf.compat.v1.summary.scalar("Loss_c2", critic_2.loss)
    # tf.compat.v1.summary.histogram("ISWeights", critic_1.ISWeights)
    # write_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    memory = Memory(config.memory_size, config.pretrain_length, config.action_size)
    # load memory if fine tune histroy agent
    if config.load_memory or fine_tune:
        memory = memory.load_memory(os.path.join("memory/", str(route_option), "memory"+".pkl"))
        print("MEMORY: Memory Loaded")
    else:
        memory.fill_memory(env)
        memory.save_memory(os.path.join("memory/", str(route_option), "memory"+".pkl"), memory)
        print("MEMORY: Memory Filled")

    # Reinforcement Learning loop
    with tf.compat.v1.Session(config=configProto) as sess:
        if fine_tune:
            # find correct episode num
            fine_tune_episode = 0
            for file_name in os.listdir(os.path.join("memory/", str(route_option))):
                if 'episode' in file_name:
                    if fine_tune_episode < int(file_name[8:]):
                        fine_tune_episode = int(file_name[8:])
                print('fine tune episode:', fine_tune_episode)
            # load corresponding net params
            model_path = '/home/gdg/CarlaRL/rl_cross_gyf_multi_task/rl_agents/models/' + str(route_option) + '/td3_' + str(fine_tune_episode) +'.ckpt'
            saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)
        else:
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
        EPSILON = 1

        # reset the episode number
        env.episode_number = 0

        # Game loop
        for episode in range(1, config.total_episodes+1):
            # make current episode num equal to correct num
            if fine_tune:
                if episode < fine_tune_episode:
                    continue
                # env.episode_number = fine_tune_episode - 1
            # move the vehicle to a spawn_point and return state
            state = env.reset()
            episode_reward = 0
            done = False
            EPSILON = (config.noised_episodes-episode)/config.noised_episodes

            # rollout each episode
            while True:
                # start_time = time.time()
                # interaction with environment
                action_noise = actor.get_action_noise(sess, state, rate=EPSILON)
                next_state, reward, done, aux_info = env.step(action_noise)
                # print(state[-(config.vehicle_mask_size + config.task_code_size) : -config.task_code_size])
                # print(state[-config.task_code_size:])

                episode_reward += np.sum(reward)
                experience = state, action_noise, reward, next_state, done
                memory.store(experience)

                # Lets learn
                if env.episode_step_number % config.train_frequency == 0:
                    # "Delayed" Policy Updates
                    policy_delayed = 2
                    for _ in range(policy_delayed):
                        # First we need a mini-batch with experiences (s, a, r, s', done)
                        tree_idx, batch, ISWeights_mb = memory.sample(config.batch_size)
                        s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(batch)
                        task_code_mb = s_mb[:, -config.task_code_size:]
                        next_task_code_mb = next_s_mb[:, -config.task_code_size:]

                        # Get q_target values for next_state from the critic_target
                        a_target_next_state = actor.get_action_target(sess, next_s_mb) # with Target Policy Smoothing
                        # multi task q value
                        q_target_next_state_1 = critic_1.get_q_value_target(sess, next_s_mb, a_target_next_state) * next_task_code_mb 
                        q_target_next_state_2 = critic_2.get_q_value_target(sess, next_s_mb, a_target_next_state) * next_task_code_mb
                        q_target_next_state = np.minimum(q_target_next_state_1, q_target_next_state_2)

                        # Set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Qtarget(s',a')
                        target_Qs_batch = []
                        for i in range(0, len(dones_mb)):
                            terminal = dones_mb[i]
                            # if we are in a terminal state. only equals reward
                            if terminal:
                                target_Qs_batch.append((r_mb[i]*task_code_mb[i]))
                            else:
                                # take the Q taregt for action a'
                                target = r_mb[i]*task_code_mb[i] + config.gamma * q_target_next_state[i]
                                target_Qs_batch.append(target)
                        targets_mb = np.array([each for each in target_Qs_batch])

                        # critic train
                        if len(a_mb.shape) > 2:
                            a_mb = np.squeeze(a_mb, axis=1)
                        loss, absolute_errors = critic_1.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)
                        loss_2, absolute_errors_2 = critic_2.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)

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
                    absolute_errors = np.sum(absolute_errors, axis=1)
                    memory.batch_update(tree_idx, absolute_errors)

                # store episode data or continue
                if done:
                    # save checkpoint model
                    if episode % config.latest_model_save_frequency == 0:
                        saver.save(sess, "models/" + str(route_option) + '/td3' + ".ckpt")
                        print('latest model saved')
                    if episode % config.regular_model_save_frequency == 0:
                        saver.save(sess, "models/" + str(route_option) + '/td3_' + str(episode) + ".ckpt")
                        print('regular model saved')

                    # save memory data, explore rate and current episode num in case training is ended unexpectedly
                    if episode % config.regular_model_save_frequency == 0:
                        memory.save_memory(os.path.join("memory/", str(route_option), "memory"+".pkl"), memory)
                        if not os.path.join("memory/", str(route_option), 'episode:' + str(episode)):
                            os.mknod(os.path.join("memory/", str(route_option), 'episode:' + str(episode)))

                    # visualize reward data
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    avarage_rewards.append(np.mean(recent_rewards))
                    avarage_rewards_data = np.array(avarage_rewards)

                    # visualize success rate data
                    if aux_info == 'success':
                        recent_success.append(1)
                    else:
                        recent_success.append(0)
                    if len(recent_success) > 100:
                        recent_success.pop(0)
                    # calculate success rate
                    avarage_success_rate = recent_success.count(1)/len(recent_success)
                    recent_success_rate.append(avarage_success_rate)
                    recent_success_rate_data = np.array(recent_success_rate)

                    # store results data
                    if episode % 50 == 0:
                        d = {"recent_success_rates": recent_success_rate_data}
                        with open(os.path.join("results/", str(route_option), "success_rate_data"+".pkl"), "wb") as f:
                            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

                        d = {"avarage_rewards": avarage_rewards_data}
                        with open(os.path.join("results/", str(route_option), "reward_data"+".pkl"), "wb") as f:
                            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

                    # print results on terminal
                    print("recent success rate:", avarage_success_rate)
                    print(episode, 'episode finished.')
                    print('[total steps:', env.episode_step_number, ',total rewards:', episode_reward, ',passing time:', env.episode_time, ']')
                    print('---'*15)
                    break
                else:
                    state = next_state
                # end_time = time.time()
                # print('step time:', end_time-start_time)


def test_loop(args, config):

    configProto = init_tensorflow()
    model_path = '/home/gdg/CarlaRL/rl_cross_gyf_multi_task/rl_agents/models/multi/td3.ckpt'
    multi_task = True
    attention = True
    # set carla ports based on route
    if not args.route:
        print('Please assign a route!!!')
        return
    else:
        route_option = args.route
    if route_option == 'left':
        port = 6000
    elif route_option == 'straight':
        port = 7000
    elif route_option == 'right':
        port = 8000

    env = CarlaEnv(port=port,
                    attention=attention,
                    route_option=route_option,
                    train=False,
                    debug=False)
    # load networks params
    if multi_task:
        actor = SocMtActorNetwork(config, 'actor')
    elif attention:
        actor = SocActorNetwork(config, 'actor')
    else:
        actor = ActorNetwork(config, 'actor')
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=configProto) as sess:
        saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)
        if saver is None:
            print("did not load")

        # save results data
        success = 0
        failure = 0
        collision = 0
        collision_traffic = []
        time_exceed = 0
        time_exceed_traffic = []
        episode_time_record = []


        # initialize traffic params
        speed_step = 5
        speed_range = (10, 30)
        distance_step = 5
        distance_range = (15, 50)

        traffic_params_list = []
        for i in range(int((speed_range[1] - speed_range[0])/speed_step + 1)):
            for j in range(int((distance_range[1] - distance_range[0])/distance_step + 1)):
                traffic_params_list.append((speed_range[0] + i*speed_step, distance_range[0] + j*distance_step))

        scenario_num = len(traffic_params_list)
        # print(scenario_num, traffic_params_list)
        traffic_params = traffic_params_list[0] # traffic_params_list.pop(0)
        traffic_all_done = False

        # initialize episode params
        state = env.reset(traffic_params=traffic_params)
        episode_reward = 0
        done = False
        while not traffic_all_done:
            action = actor.get_action(sess, state)
            state, reward, done, aux_info = env.step(action)
            episode_reward += np.sum(reward)

            if done:
                # record this episode's result
                if aux_info == 'collision':
                    collision += 1
                    failure += 1
                    collision_traffic.append(traffic_params)
                elif aux_info == 'time_exceed':
                    time_exceed += 1
                    failure += 1
                    time_exceed_traffic.append(traffic_params)
                else:
                    # get success episode time
                    success += 1
                    episode_time_record.append(env.episode_time)
                # print on terminal
                print('___'*15)
                print("EPISODE speed:", traffic_params[0], "distance:", traffic_params[1], 'Result:', aux_info)
                print('passing time of this episode(s): ', env.episode_time)
                # start next episode
                # set new traffic params
                if traffic_params[0] == speed_range[1] and traffic_params[1] == distance_range[1]:
                    traffic_all_done = True
                else:
                    traffic_params = traffic_params_list.pop(0)
                # init state
                state = env.reset(traffic_params)
                episode_reward = 0
                done = False

        print('-*'*15, ' result ', '-*'*15)
        print('success: ', success, '/', scenario_num)
        print('collision: ', collision, '/', scenario_num)
        print('time_exceed: ', time_exceed, '/', scenario_num)
        print('average time: ', np.mean(episode_time_record))
        print('collision traffics: ',collision_traffic)
        print('time exceed traffics: ',time_exceed_traffic)


def render_loop(args):
    try:
        # init env and start training or testing
        config = hyperParameters()  # algorithm hyperparameters
        if args.test:
            print('---------Begin TESTing---------')
            test_loop(args, config)
        else:
            print('---------Begin TRAINing---------')
            train_loop(args, config, fine_tune=False)
    finally:
        pass


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA_CROSS_RL')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
    argparser.add_argument(
        '--route',
        type=str,
        default=None,
        help='route option')

    args = argparser.parse_args()

    try:
        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
