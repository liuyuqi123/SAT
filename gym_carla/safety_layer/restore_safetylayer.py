"""
This script contains some methods to restore safety layer NN for action modification.

"""

import os
import json

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from gym_carla.safety_layer.safety_layer_devised import SafetyLayerDevised, baseline_config


def restore_safety_layer(safety_layer_cls, output_path):
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

    # init safety layer class
    safety_layer = safety_layer_cls(config_dict)
    # load trained safety layer model
    safety_layer.load_models([model_path])

    # the action modifier is determined by the config file, eg. state clip method
    action_modifier = safety_layer.get_safe_action

    return action_modifier, safety_layer, config_dict


def init_safety_layer(output_path):
    """
    Provide a usable entrance for the safety layer.

    :return: the action modifier
    """
    # # for debug
    # output_path = './safetylayer_outputs/runnable/exp_task-left_state-clip-False_TTC-c_2_TTC-type-q_norm_dataset-failure'

    cls = SafetyLayerDevised
    action_modifier, safety_layer, config_dict = restore_safety_layer(cls, output_path)

    # todo print debug info

    return action_modifier, safety_layer, config_dict


def test_restore_safetylayer():
    """
    todo move this method to train loop script
        Merge this method into env run loop safety.

    :return:
    """

    # load config and the from the training output folder
    output_path = './safetylayer_outputs/runnable/exp_task-left_state-clip-False_TTC-c_2_TTC-type-q_norm_dataset-failure'

    cls = SafetyLayerDevised
    action_modifier, safety_layer, config_dict = restore_safety_layer(cls, output_path)

    # =========================
    # test evaluation through a dataset
    dataset_path = './data_collection/output/merge_normed/left/failure/dataset_0.npz'

    eval_batch = safety_layer.load_dataset_as_batch(dataset_path)

    # todo determine which constraint value to use through config dict
    observation = eval_batch['observation'][0]
    action = eval_batch['action'][0]

    # constraint key
    c_key = config_dict['ttc_version'] + '_' + config_dict['ttc_type']
    c = eval_batch[c_key][0]

    action_new = action_modifier(observation, action, c)

    print('')


def test_restore_multi_task():
    """
    Test get a safe action from the multi-task action modifier.
    """

    model_path = '/home/liuyuqi/PycharmProjects/gym-carla/gym_carla/safety_layer/safetylayer_outputs/runnable_models/multi-task'

    #
    action_modifier, safety_layer, config_dict = restore_safety_layer(SafetyLayerDevised, model_path)

    dataset_path = '/home/liuyuqi/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/merge_normed_multitask/dataset_0.npz'

    eval_batch = safety_layer.load_dataset_as_batch(dataset_path)

    # todo determine which constraint value to use through config dict
    observation = eval_batch['observation'][0]
    action = eval_batch['action'][0]

    # constraint key
    c_key = config_dict['ttc_version'] + '_' + config_dict['ttc_type']
    c = eval_batch[c_key][0]

    action_new = action_modifier(observation, action, c)

    print('')


if __name__ == '__main__':

    # test_restore_safetylayer()

    test_restore_multi_task()
