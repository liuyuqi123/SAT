"""
Training for the safet layer.

In this script, some different settings are supposed to be compared.

Current data setting is that:

 - the length of each dataset is 10000
 - merge datasets into larger one

"""

import os
import sys

# ================   Append Project Path   ================
path = os.getcwd()
index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1
project_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'
sys.path.append(project_path)

import argparse
import logging
import numpy as np
from datetime import datetime

# import torch
# from torch.optim.lr_scheduler import StepLR, MultiStepLR
# from gym_carla.safety_layer.safety_layer import SafetyLayer
from safety_layer_devised import SafetyLayerDevised, baseline_config


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class SafetyLayerTraining:
    """
    Management of main loop for safety layer training.
    """
    def __init__(self):

        self.time_stamp = TIMESTAMP

        self.safety_layer_cls = SafetyLayerDevised
        self.safety_layer_list = []

        # self.init_safety_layers()

    def init_safety_layers(self, setting_dict):
        """
        Initialization for group of safety layer instances.
        """
        # ttc comparison
        config_list = []

        # # original lines
        # for task_option in ['left']:  # ['left', 'straight', 'right']
        #     for state_clip in [True, False]:
        #         for data_type in ['failure']:  # 'hybrid', 'failure'
        #             for ttc_version in ['c_1', 'c_2']:  # 'c_1', 'c_2'
        #                 for ttc_type in ['norm', 'q_norm', 'clipped']:  # 'clipped', 'norm', 'q_norm'

        for state_clip in setting_dict['state_clip']:
            for task_option in setting_dict['task_option']:
                for data_type in setting_dict['data_type']:
                    for ttc_version in setting_dict['ttc_version']:
                        for ttc_type in setting_dict['ttc_type']:

                            new_config = {}
                            for k, v in baseline_config.items():
                                new_config[k] = v

                            new_config['task_option'] = task_option
                            new_config['state_clip'] = state_clip
                            new_config['ttc_version'] = ttc_version
                            new_config['ttc_type'] = ttc_type
                            new_config['data_type'] = data_type

                            # add timestamp to config
                            new_config['time_stamp'] = self.time_stamp

                            config_list.append(new_config)

        # put all config dicts and safety layer instance into a dict
        safety_layer_list = []
        for config in config_list:

            # todo fix the multi_task key
            config['multi_task'] = False

            safety_layer_instance = self.safety_layer_cls(config)
            safety_layer_list.append(
                (config, safety_layer_instance)
            )

        self.safety_layer_list = safety_layer_list

        print('')

    def train(self, dataset_path):
        """
        todo debug training curves

        Main entrance of training loop.
        """

        for config, safety_layer in self.safety_layer_list:
            # todo add multi-task into task_option
            # current task_option: ['left', 'straight', 'right']
            route = config['task_option']
            # dataset category, ['success', 'failure', 'hybrid']
            category = config['data_type']

            # dataset for loading
            _dataset_path = os.path.join(
                dataset_path,
                route,
                category,
            )

            # tag a name of current NN
            tag = 'exp_task-{}_state-clip-{}_TTC-{}_TTC-type-{}_dataset-{}'.format(
                config['task_option'],
                str(config['state_clip']),
                config['ttc_version'],
                config['ttc_type'],
                config['data_type'],
            )

            safety_layer.train_with_datasets(
                dataset_path=_dataset_path,
                tag=tag,
            )

            print('')

        print('')

    def debug_train(self):
        """
        Debug version of train method.
        """
        # ===== 1. config file for debug =====
        # ttc comparison
        config_list = []

        for task_option in ['left']:  # ['left', 'straight', 'right']
            for state_clip in [True, False]:
                for data_type in ['failure']:  # 'failure', 'hybrid'
                    for ttc_type in ['norm']:  # 'clipped', 'norm', 'q_norm'
                        for ttc_version in ['c_1']:  # 'c_1', 'c_2'

                            new_config = {}
                            for k, v in baseline_config.items():
                                new_config[k] = v

                            new_config['task_option'] = task_option
                            new_config['state_clip'] = state_clip
                            new_config['ttc_version'] = ttc_version
                            new_config['ttc_type'] = ttc_type
                            new_config['data_type'] = data_type

                            new_config['time_stamp'] = self.time_stamp

                            # quick
                            new_config['epochs'] = 1
                            new_config['save_freq'] = 1

                            config_list.append(new_config)

        # put all config dicts and safety layer instance into a dict
        safety_layer_list = []
        for config in config_list:
            safety_layer_instance = self.safety_layer_cls(config)
            safety_layer_list.append(
                (config, safety_layer_instance)
            )

        print('')

        # # ===== 2. train procedure =====
        # dataset_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/debug_train'
        # 2080ti
        dataset_path = '/home1/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug_train'
        # ===== 2. train procedure =====
        dataset_path = '/home/lyq/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/output/debug_train'
        # 1660ti
        dataset_path = '/home/liuyuqi/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug_train'

        for config, safety_layer in safety_layer_list:
            # todo add multi-task into task_option
            # current task_option: ['left', 'straight', 'right']
            route = config['task_option']
            # dataset category, ['success', 'failure', 'hybrid']
            category = config['data_type']

            # dataset for loading
            _dataset_path = os.path.join(
                dataset_path,
                route,
                category,
            )

            # tag a name of current NN
            tag = 'exp_task-{}_state-clip-{}_TTC-{}_TTC-type-{}_dataset-{}'.format(
                config['task_option'],
                str(config['state_clip']),
                config['ttc_version'],
                config['ttc_type'],
                config['data_type'],
            )

            safety_layer.train_with_datasets(_dataset_path, tag)

            print('')

        print('')


def run():
    """"""
    training = SafetyLayerTraining()

    # training.debug_train()

    # fix dataset path according to local path
    training.train(dataset_path='./data_collection/merge_normed')


def main():
    """"""
    argparser = argparse.ArgumentParser(
        description='Safety layer training for CARLA intersection experiments.')
    argparser.add_argument(
        '--task_option',
        type=str, nargs='+',
        default=['left'],
        dest='task_option',
        help='Task option of trained model.')  # ['left', 'straight', 'right'] multi-task in isolated script
    # argparser.add_argument(
    #     '--state_clip',
    #     action='store_true',
    #     dest='state_clip',
    #     help='Whether use state clip trick.')
    argparser.add_argument(
        '--ttc_version',
        type=str, nargs='+',
        default=['c_1', 'c_2'],
        dest='ttc_version',
        help='available options: c_1 or c_2.')
    argparser.add_argument(
        '--ttc_type',
        type=str, nargs='+',
        default=['clipped', 'norm', 'q_norm'],  # 'clipped', 'norm', 'q_norm'
        dest='ttc_type',
        help='clipped, norm, q_norm')
    argparser.add_argument(
        '--data_type',
        type=str, nargs='+',
        default=['failure', 'hybrid'],  # 'failure', 'hybrid'
        dest='data_type',
        help='dataset type for training.')

    args = argparser.parse_args()

    # # todo figure out usage of logging usage
    # log_level = logging.DEBUG if args.debug else logging.INFO
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # todo fix this
    # if args.test:  # evaluation loop
    #     eval_safetylayer()
    # else:  # train loop
    #     train_safety_layer()

    # todo 1. add args for training epoch and minibatch size
    #  2. add evaluation entrance

    # # parse args to training settings
    # training_setting = {
    #     # todo check if need this
    #     'state_clip': [True, False],
    #
    #     'task_option': args.task_option,
    #     'ttc_version': args.ttc_version,
    #     'ttc_type': args.ttc_type,
    #     'data_type': args.data_type,
    # }

    # we don't use argparser anymore!!!
    # final version
    training_setting = {
        # todo check if need this
        'state_clip': [False],
        'task_option': ['left'],  # ['left', 'straight', 'right']
        'ttc_version': ['c_2'],  # c_2, c_2_norm
        'ttc_type': ['clipped'],
        'data_type': ['hybrid'],  # ['success', 'failure', 'hybrid']

        'multi_task': False,

        # param epochs refers to how many times to reuse a single dataset
        'epochs': 100,
    }

    training = SafetyLayerTraining()
    training.init_safety_layers(setting_dict=training_setting)

    # # use relative path to load datasets
    relative_path = './data_collection/output_filtered'

    training.train(relative_path)


if __name__ == '__main__':

    # # for debug and a quick run
    # run()

    # start script with args
    main()
