"""
Train a safety layer with collected data.

"""
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from gym_carla.safety_layer.safety_layer import SafetyLayer


class SafetyLayer2(SafetyLayer):
    """
    Fix the observation clipping method
    """

    def __init__(self, config):
        super().__init__(config)

    # ==========  original method  ==========
    # def clip_obs(self, observation):
    #     """
    #     This method should be modified in different experiment setting.
    #     :param observation: original observation ndarray
    #     :return: clipped observation
    #     """
    #     # this method shoulf be fixed if usage of observation changed
    #     if isinstance(observation, np.ndarray) and len(observation.shape) > 1:
    #         clipped_state = np.array(observation[:, 0:9])
    #         task_code = np.array(observation[:, -4:])
    #         clipped_observation = np.hstack((clipped_state, task_code))
    #
    #     else:
    #         clipped_state = np.array(observation[0:9])
    #         task_code = np.array(observation[-4:])
    #         clipped_observation = np.hstack((clipped_state, task_code))
    #
    #     return clipped_observation

    def clip_obs(self, observation):
        """

        This method should be modified in different experiment setting.

        :param observation: original observation ndarray
        :return: clipped observation
        """

        # data from dataset, ndarray format
        if isinstance(observation, np.ndarray) and len(observation.shape) > 1:
            # todo add args for dimension of ego state and NPC state
            clipped_state = np.array(observation[:, 0:9])  # ego 4 + npc 6

            # todo add args and API to process multi-task data
            # task_code = np.array(observation[:, -4:])

            # clipped_observation = np.hstack((clipped_state, task_code))
            clipped_observation = clipped_state

        else:  # data in List format
            clipped_state = np.array(observation[0:9])
            # task_code = np.array(observation[-4:])
            # clipped_observation = np.hstack((clipped_state, task_code))

            clipped_observation = clipped_state

        return clipped_observation

    def _evaluate_batch(self, batch):
        """
        Evaluate a batch data.
        :param batch: batch data in dict
        :return: losses <list>
        """
        if self.config['use_gpu']:
            observation = self._as_tensor(batch["observation"]).cuda()
            action = self._as_tensor(batch["action"]).cuda()

            # c = self._as_tensor(batch["c"]).cuda()
            # c_next = self._as_tensor(batch["c_next"]).cuda()

            # todo add API to select c_1 or c_2
            c = self._as_tensor(batch["c_1"]).cuda()
            c_next = self._as_tensor(batch["c_1_next"]).cuda()
        else:
            observation = self._as_tensor(batch["observation"])
            action = self._as_tensor(batch["action"])

            # c = self._as_tensor(batch["c"])
            # c_next = self._as_tensor(batch["c_next"])

            c = self._as_tensor(batch["c_1"])
            c_next = self._as_tensor(batch["c_1_next"])

        gs = [x(observation) for x in self._models]

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(gs)]

        # parse loss value
        for i in range(self._num_constraints):
            loss = (c_next[:, i] - c_next_predicted[i]) ** 2
            if self.config['use_gpu']:
                loss = loss.detach().cpu().numpy()
            else:
                loss = loss.detach().numpy()
            # print('')

        losses = [torch.mean((c_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self._num_constraints)]

        return losses


def eval_safetylayer():
    """
    Test safety layer correctly loaded and trained.
    """

    # safety layer config
    safetylayer_config = {
        'input_dim': int(9),
        'output_dim': int(2),
        'lr': 1e-4,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 100,
        'save_freq': 100,  # total checkpoints number
        'use_gpu': True,
        'dataset_path': './safety_layer_datasets/random_policy/2020-10-07-Time12-38-27',
        # fix the tag for different settings
        'tag': 'random',
    }

    # init safety layer class
    safetylayer = SafetyLayer(config=safetylayer_config)

    # ==================================================
    # load trained safety layer model
    model_path = '/experiments/safe_ddpg/without_attention/safetylayer_outputs/random/2020-10-07-Time18-19-54/model/model_0_final.pth'
    safetylayer.load_models([model_path])

    # debug dataset
    # dataset_path = './safety_layer_datasets/debug/2020-10-07-Time12-21-56/dataset_0.npz'

    # eval dataset
    dataset_path = './safety_layer_datasets/random_policy/2020-10-07-Time12-38-27/dataset_0.npz'

    eval_batch = safetylayer.load_dataset_as_batch(dataset_path)
    losses = safetylayer._evaluate_batch(eval_batch)

    print('losses： ', losses)


def test_get_safe_action():
    """
    Test method of get safe action
    """
    # safety layer config
    safetylayer_config = {
        'input_dim': int(9),
        'output_dim': int(2),
        'lr': 1e-4,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 100,
        'save_freq': 100,  # total checkpoints number
        'use_gpu': True,
        'dataset_path': './safety_layer_datasets/random_policy/2020-10-07-Time12-38-27',
        # fix the tag for different settings
        'tag': 'random',
    }

    # init safety layer class
    safetylayer = SafetyLayer(config=safetylayer_config)

    # ==================================================
    # load trained safety layer model
    model_path = '/experiments/safe_ddpg/without_attention/safetylayer_outputs/random/2020-10-07-Time18-19-54/model/model_0_final.pth'
    safetylayer.load_models([model_path])

    # ==================================================
    # load trained safety layer model
    model_path = '/experiments/safe_ddpg/without_attention/safetylayer_outputs/random/2020-10-07-Time18-19-54/model/model_0_final.pth'
    safetylayer.load_models([model_path])

    # debug dataset
    # dataset_path = './safety_layer_datasets/debug/2020-10-07-Time12-21-56/dataset_0.npz'

    # eval dataset
    dataset_path = './safety_layer_datasets/random_policy/2020-10-07-Time12-38-27/dataset_0.npz'

    eval_batch = safetylayer.load_dataset_as_batch(dataset_path)

    observation = eval_batch['observation'][0]
    action = eval_batch['action'][0]
    c = eval_batch['c'][0]

    # module
    action_modifier = safetylayer.get_safe_action

    # test
    action_new = action_modifier(observation, action, c)

    print('d')


def train_safety_layer():
    """
    Train safety layer with datasets
    """
    # debug
    debug_safetylayer_config = {
        'input_dim': int(9),
        'output_dim': int(2),
        'lr': 1e-4,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 10,
        'save_freq': 5,  # total checkpoints number
        'use_gpu': True,  # False
        'tag': 'debug',  # todo add this to train method args
    }

    tags = [
        'accel_slack_1_5',
        'random',
    ]

    # safety layer config
    safetylayer_config = {
        'input_dim': int(13),
        'output_dim': int(2),
        'lr': 1e-2,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 100,
        'save_freq': 100,  # total checkpoints number
        'use_gpu': True,
        'dataset_path': './safety_layer_datasets/random_policy/2020-10-17-Time17-36-05',
        # fix the tag for different settings
        'tag': 'random',
    }

    # init safety layer class
    # safetylayer = SafetyLayer(config=safetylayer_config)

    # safetylayer = SafetyLayer(config=safetylayer_config)

    safety_layer = SafetyLayer2(
        config=debug_safetylayer_config,
    )

    # ==================================================
    # train safety layer with datasets

    safety_layer.train_with_datasets(
        dataset_path='/home/liuyuqi/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/test/original_state/left/hybrid',
    )

    print('Finish training safety layer model.')


def test_load_data():
    """

    """

    # debug
    debug_safetylayer_config = {
        'input_dim': int(9),
        'output_dim': int(2),
        'lr': 1e-4,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 10,
        'save_freq': 5,  # total checkpoints number
        'use_gpu': True,  # False
        'tag': 'debug',  # todo add this to train method args
    }

    # init safety layer class
    safety_layer = SafetyLayer2(
        config=debug_safetylayer_config
    )

    # test clip_obs method
    dataset_path = '/home/liuyuqi/PycharmProjects/gym-carla/gym_carla/safety_layer/data_collection/debug/test/original_state/left/hybrid/dataset_0.npz'

    safety_layer.load_dataset_as_batch(dataset_path)

    print('d')


if __name__ == '__main__':

    # todo add argparser
    # ==================================================
    # train loop
    train_safety_layer()

    # ==================================================
    # eval
    # eval_safetylayer()

    # ==================================================
    # test get safe action
    # test_get_safe_action()

    # test_load_data()




