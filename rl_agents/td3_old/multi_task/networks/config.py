"""
todo fix parameter name
 merge with baseline td3 API

Config parameters for the ablation experiments.
"""


class hyperParameters(object):

    # Environment interaction
    total_episodes = 10000
    noised_episodes = 2000
    max_steps = 300

    # State & action parameters
    ego_feature_num = 4
    npc_num = 5
    npc_feature_num = 6  # 6
    vehicle_state_size = ego_feature_num + npc_num*npc_feature_num

    vehicle_mask_size = npc_num + 1
    task_code_size = 4

    state_size_with_att_mask = vehicle_state_size + vehicle_mask_size
    state_size_with_att_mask_and_task_code = vehicle_state_size + vehicle_mask_size + task_code_size

    action_size = 2

    # Network update parameters
    lra = 2e-5
    lrc = 5e-5  # 1e-4
    tau = 1e-3
    batch_size = 1024
    train_frequency = 2  # todo

    # Value calculation parameter
    gamma = 0.99  # Discounting rate

    # Memory Buffer parameters
    pretrain_length = 2500  # Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
    memory_size = 200000  # Number of experiences the Memory can keep  --INTIALLY 100k
    load_memory = True # If True load memory, otherwise fill the memory with new data

    # model saving
    latest_model_save_frequency = 5
    regular_model_save_frequency = 250


