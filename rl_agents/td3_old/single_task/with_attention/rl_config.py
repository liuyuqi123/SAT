import os
import numpy as np
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class DebugConfig:
    """
    This is parameters for experiment debug.
    """
    # Training parameters
    total_episodes = 20
    noised_episodes = 10
    max_steps = 300
    batch_size = 256  # 256
    train_frequency = 2  # 2

    # NN architecture
    ego_feature_num = 4
    npc_num = 5
    npc_feature_num = 4

    mask_size = npc_num + 1
    state_size = ego_feature_num + npc_num * npc_feature_num
    state_mask_size = state_size + mask_size

    action_size = 2
    lra = 2e-5
    lrc = 1e-4

    # Fixed Q target hyper parameters
    tau = 1e-3

    # exploration hyperparamters for ep. greedy. startegy
    explore_start = 0.75  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    explore_step = 40000  # 40k steps
    decay_rate = (explore_start - explore_stop) / explore_step  # exponential decay rate for exploration prob

    # Q LEARNING hyperparameters
    gamma = 0.99  # Discounting rate
    pretrain_length = 500  # Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
    memory_size = 200000  # Number of experiences the Memory can keep  --INTIALLY 100k
    load_memory = False  # If True load memory, otherwise fill the memory with new data

    # ==================================================
    # output paths
    tag = 'debug'
    output_path = os.path.join('./outputs', tag, TIMESTAMP)

    memory_path = os.path.join(output_path, 'rl_replay_memory')
    # os.makedirs(memory_path, exist_ok=True)

    memory_load_path = os.path.join(memory_path, 'memory.pkl')
    memory_save_path = os.path.join(memory_path, 'memory.pkl')

    # model saving
    model_save_frequency = 2  # frequency to save the model. 0 means not to save
    model_save_frequency_no_paste = 500  # ???

    # frequency to check best models
    model_save_frequency_high_success = 10

    model_test_frequency = 10
    model_test_eps = 10  # ???

    # final model save path
    model_save_path = os.path.join(output_path, 'final_model', 'final_model.ckpt')
    # checkpoint save path
    model_ckpt_path = os.path.join(output_path, 'checkpoints')
    # best model
    best_model_path = os.path.join(output_path, 'best_models')


class hyperParameters(object):
    """
    Hyperparameters for RL agent
    """

    # Training parameters
    total_episodes = 4000
    noised_episodes = 1500
    max_steps = 300
    batch_size = 256  # 256
    train_frequency = 2  # 2

    # NN architecture
    ego_feature_num = 4
    npc_num = 5
    npc_feature_num = 4

    mask_size = npc_num + 1
    state_size = ego_feature_num + npc_num * npc_feature_num
    state_mask_size = state_size + mask_size

    action_size = 2
    lra = 2e-5
    lrc = 1e-4

    # Fixed Q target hyper parameters
    tau = 1e-3

    # exploration hyperparamters for ep. greedy. startegy
    explore_start = 0.75  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    explore_step = 40000  # 40k steps
    decay_rate = (explore_start - explore_stop) / explore_step  # exponential decay rate for exploration prob

    # Q LEARNING hyperparameters
    gamma = 0.99  # Discounting rate
    pretrain_length = 500  # Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
    memory_size = 200000  # Number of experiences the Memory can keep  --INTIALLY 100k
    load_memory = False  # If True load memory, otherwise fill the memory with new data

    # ==================================================
    # output paths
    tag = 'base_run'
    # tag = 'run'
    # tag = 'run_new_c'

    output_path = os.path.join('./td3_outputs', tag, TIMESTAMP)

    memory_path = os.path.join(output_path, 'rl_replay_memory')
    # os.makedirs(memory_path, exist_ok=True)

    memory_load_path = os.path.join(memory_path, 'memory.pkl')
    memory_save_path = os.path.join(memory_path, 'memory.pkl')

    # model saving
    model_save_frequency = 100  # frequency to save the model. 0 means not to save
    model_save_frequency_no_paste = 500  # ???

    # frequency to check best models
    model_save_frequency_high_success = 10

    model_test_frequency = 10
    model_test_eps = 10  # ???

    # final model save path
    model_save_path = os.path.join(output_path, 'final_model', 'final_model.ckpt')
    # checkpoint save path
    model_ckpt_path = os.path.join(output_path, 'checkpoints')
    # best model
    best_model_path = os.path.join(output_path, 'best_models')

