import os
import numpy as np
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class hyperParameters:
    """
    Hyper parameters for RL agent
    """

    # Training parameters
    total_episodes = 30000  # original is 10000
    noised_episodes = 1000

    # todo add max_steps setter for env, if needed
    max_steps = 450
    # batch_size = 256  # 256, 512, 1024

    # test smaller batch_size for attention
    batch_size = 64  # original is 128

    train_frequency = 2  # 2

    # td3 hyperparameter
    policy_delayed = 2  # original is 2

    action_size = 2

    # NN architecture
    ego_feature_num = 4  # 9 for abs_all, 4 for sumo and sumo_1
    npc_num = 5
    npc_feature_num = 6
    # todo fix this for the multi-task training
    state_size = ego_feature_num + npc_num * npc_feature_num

    vehicle_mask_size = npc_num + 1  # attention
    task_code_size = 4

    # state size for other experiments
    vehicle_state_size = state_size

    # multi-task TD3
    state_size_multi_task = vehicle_state_size + task_code_size
    # sole attention
    state_size_with_att_mask = vehicle_state_size + vehicle_mask_size
    # attention + multi-task
    state_size_with_att_mask_and_task_code = vehicle_state_size + vehicle_mask_size + task_code_size
    # sole multi-task
    state_size_with_task_code = vehicle_state_size + task_code_size

    # Fixed Q target hyper parameters
    tau = 1e-3

    # exploration hyper-parameters for epsilon-greedy strategy
    explore_start = 0.5  # exploration probability at start
    explore_stop = 0.05  # minimum exploration probability
    explore_step = 20000  # 40k, 40000
    # decay_rate = (explore_start - explore_stop) / explore_step  # exponential decay rate for exploration prob

    # Q-LEARNING hyper-parameter: Discounting Rate
    gamma = 0.99

    # RL memory parameters
    memory_size = 300000  # Number of experiences the Memory can keep
    pretrain_length = 10000  # 20000  # Number of experiences stored in the Memory when initialized for the first time

    # todo improve the usage of this arg
    # If True load memory, otherwise fill the memory with new data
    # load_memory = True
    load_memory = False

    # # tag for memory
    # memory_tag = 'fix_env2'  # None, 'short', fix_env

    memory_tag = None

    # always save newly collected memory
    # save_memory = False
    # save collected memory periodically
    memory_save_frequency = int(100)

    # model saving
    model_save_frequency = 500  # frequency to save the model. 0 means not to save

    # model saving
    latest_model_save_frequency = 50

    # frequency to check best models
    model_save_frequency_high_success = 50

    # todo remove if not used
    model_test_frequency = 10
    model_test_eps = 10  # ???

    # ================   learning rate   ================
    # # -------  original lines  -------
    # # whether use the learning rate decay for actor and critic
    # use_lra_decay = True  # False
    # use_lrc_decay = False
    #
    # # original lr settings
    # lra = 2e-5  # 2e-5
    # lrc = 5e-5  # 1e-4
    #
    # # consider frame skipping factor is 2
    # guessing_episode_length = 200
    # # decay after certain number of episodes
    # decay_episodes = 1500
    # decay_steps = guessing_episode_length / train_frequency * decay_episodes
    # decay_rate = 1 / 2.15  # 2.15 = 10^(1/3)

    # todo fix lr decay API
    # todo add API to set lr decay from external loop
    # todo add staircase option
    # # todo use summary to log lr in tensorboard
    # tf.summary.scalar("learning_rate", learning_rate)

    # add lr decay mode
    use_lr_decay = False

    if use_lr_decay:
        use_lra_decay = True
        use_lrc_decay = True

        lra = 1e-3  # 2e-5
        lrc = 1e-2  # 1e-4

        guessing_episode_length = 200
        # decay after certain number of episodes
        decay_episodes = 1000
        decay_steps = guessing_episode_length / train_frequency * decay_episodes
        decay_rate = 1 / 2.15  # 2.15 = 10^(1/3), 3.16 = 10^(1/2),  = 10^(1/2)
    else:
        use_lra_decay = False
        use_lrc_decay = False

        lra = 2e-5
        lrc = 1e-4

    # # =========================
    # # todo fix the API of params decay if necessary
    # # traffic parameters decay
    # params_decay = True
    #
    # # details of the params decay
    # collision_rate_start = 0.5
    # collision_rate_end = 0.95
    #
    # # # todo add decay schedule
    # # linear_decay, exp decay...
    #
    # # example of traffic param decay
    # # collision detect decay
    # param = {
    #     'initial_value': 0.5,
    #     'target_value': 1.,
    #     'episode_number': int(2000),
    #     'scheduler': 'linear',
    # }
    #
    # # =========================

    def __init__(self, args):

        # # todo add args to a class attribute
        # self.args = args

        self.tag = args.tag if args.tag else 'not_named'

        # determine the ablation_option
        if args.attention:
            if not args.multi_task:
                memory_option = 'attention'
            else:  # multi-task and attention
                memory_option = 'mt_attention'
        else:
            if args.multi_task:  # multi-task without attention
                memory_option = 'multi_task'
            else:  # baseline
                memory_option = 'baseline'

        if args.safety:
            ablation_option = memory_option + '_safety'
        else:
            ablation_option = memory_option

        self.ablation_option = ablation_option

        self.output_path = os.path.join('./outputs', ablation_option, args.task_option, self.tag, TIMESTAMP)
        os.makedirs(self.output_path, exist_ok=True)

        # memory load path
        if self.memory_tag:
            self.memory_path = os.path.join('./outputs', 'memory', self.memory_tag, memory_option, args.task_option)
        else:
            self.memory_path = os.path.join('./outputs', 'memory', memory_option, args.task_option)

        os.makedirs(self.memory_path, exist_ok=True)

        # ====================  other output paths  ====================
        # checkpoints
        self.model_ckpt_path = os.path.join(self.output_path, 'checkpoints')
        # best model
        self.best_model_path = os.path.join(self.output_path, 'best_models')
        # final model save path
        self.model_save_path = os.path.join(self.output_path, 'final_model', 'final_model.ckpt')

        # memory save path
        self.memory_save_path = os.path.join(self.output_path, 'memory')
        os.makedirs(self.memory_save_path, exist_ok=True)

