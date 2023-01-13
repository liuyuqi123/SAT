"""
Training loop for the multi-task version safety layer.
"""

import glob
import os
import sys

# ================   Append Project Path   ================
path = os.getcwd()
index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1
project_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'
sys.path.append(project_path)

from datetime import datetime
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from safety_layer_devised import SafetyLayerDevised, baseline_config


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


multi_task_config = {

    'time_stamp': TIMESTAMP,

    # multi-task training
    'multi_task': True,

    # NN structure, for both the task code is not included
    'state_clip': False,
    # if using whole state array
    # 'input_dim': int(34),
    # # using only the nearest vehicle state
    # 'input_dim': int(10),

    # ==========  Settings for experiments  ==========
    # dataset type option
    'data_type': 'failure',  # ['hybrid', 'failure', 'success']
    # ttc modification
    'ttc_type': 'q_norm',  # ['clipped', 'norm', 'q_norm']
    'ttc_version': 'c_2',  # ['c_1', 'c_2']

    # todo extend to a public config dict
    # ==========  public hyper-parameters  ==========
    'output_dim': int(2),  # action
    'lr': 1e-2,  # initial lr value
    'lr_scheduler': MultiStepLR,  # or StepLR  # todo use str to init, easy to save config
    'minibatch_num': 10,
    'epochs': 20,
    'save_freq': 500,  # total checkpoints number
    'use_gpu': True,  # don't delete!!!
}


def simple_run():
    """"""

    # # debug
    # dataset_path = './data_collection/debug/tar/multi-task/hybrid'

    # datasets for multi-task training
    dataset_path = './data_collection/output/multi-task/hybrid'

    safety_layer = SafetyLayerDevised(multi_task_config)

    safety_layer.train_with_datasets(
        dataset_path=dataset_path,
        tag='multi-task',
    )


if __name__ == '__main__':

    simple_run()
