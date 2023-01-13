""""""

import os

import json
import random
import numpy as np
import time
from datetime import datetime
from functional import seq
from tensorboardX import SummaryWriter

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR


from gym_carla.safety_layer.safety_layer import SafetyLayer, ConstraintModel, for_each

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


baseline_config = {

    # NN structure, for both the task code is not included
    'state_clip': False,
    # if using whole state array
    # 'input_dim': int(34),
    # # using only the nearest vehicle state
    # 'input_dim': int(10),

    # ==========  Settings for experiments  ==========
    # dataset type option
    'data_type': 'hybrid',  # ['hybrid', 'failure', 'success']
    # ttc modification
    'ttc_type': 'clipped',  # ['clipped', 'norm', 'q_norm']
    'ttc_version': 'c_1',  # ['c_1', 'c_2']

    # todo extend to a public config dict
    # ==========  public hyper-parameters  ==========
    'output_dim': int(2),  # action
    'lr': 1e-2,  # initial lr value
    'lr_scheduler': MultiStepLR,  # or StepLR  # todo use str to init, easy to save config
    'minibatch_num': 10,
    'epochs': 100,  # 500
    'save_freq': 500,  # total checkpoints number
    'use_gpu': True,  # don't delete!!!
}


class SafetyLayerDevised(SafetyLayer):
    """
    Fix the observation clipping method
    """

    # output_path = '../../outputs/safetylayer_outputs'
    output_path = './safetylayer_outputs'

    ttc_upper_bound = 99.

    def __init__(self, config):
        """

        config refers to a dict with init info
        """

        self.config = config
        self.time_stamp = self.config['time_stamp']

        # keep previous experiments on state clip
        # add arg for state clip option
        self.state_clip = True if self.config['state_clip'] else False

        if self.state_clip:
            self.input_dim = int(10)
        else:  # using whole state array
            self.input_dim = int(34)

        # whether use the multi-task method
        self.multi_task = self.config['multi_task']

        if self.multi_task:
            self.input_dim = int(self.input_dim + 4)

        self.output_dim = self.config['output_dim']

        # todo for various RL env
        # self._num_constraints = env.get_num_constraints()
        # number of env constraints
        self._num_constraints = int(1)  # default constraint number is 1

        # batch size of each dataset
        self.batch_size = None
        # actually class of lr scheduler
        self.lr_scheduler = self.config['lr_scheduler']

        self._initialize_constraint_models()

        self._train_global_step = 0
        self._eval_global_step = 0

        # tensorboardX logger
        self._writer = None

        # # if use attention mechanism, we merge attention flag to init config file
        # if self.config.has_key:
        #     self.attention = self.config['attention']
        # else:
        #     print('Attention setting is not included in this config file, please check.')

    # ==========  original method as a reference  ==========
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

    def load_dataset_as_batch(self, dataset_path):
        """
        Load a dataset as a batch.

        :param dataset_path: dataset path
        :return: batch data in <dict>
        """

        # load dataset
        dataset = np.load(dataset_path, allow_pickle=True)
        keys = dataset.files
        # print('available keys: ', keys)

        #
        # store data in a dict
        batch = dict()
        batch_size_list = []
        for key in keys:
            value = dataset[key]
            batch[key] = value

            # check and get batch size
            # batch_size = value.shape[0]
            batch_size_list.append(value.shape[0])

        # check batch size equal
        batch_size = batch_size_list[0]
        for batchsize in batch_size_list[1:]:
            assert batch_size == batchsize, 'Wrong dimension in batch data.'
        self.batch_size = batch_size
        # print('dataset is loaded.')

        # todo add comparison of original ttc???
        # process
        # clip ttc value by a bound
        self.process_constraint(batch)

        # ==================================================
        # old version, manually clip the obs
        # clip observation
        # clipped_observation = batch['observation'][:, 0:self.input_dim]
        # batch['observation'] = clipped_observation

        # use a method to clip the obs
        # rewrite method for different safety layer
        clipped_observation = self.clip_obs(batch)
        batch['observation'] = clipped_observation

        return batch

    def process_constraint(self, batch):
        """
        Clip ttc value according to the bound value.

        This method is deprecated for the filtered data.
        """
        # # =========================
        # # original lines
        # copy_batch = {}
        # for k, v in batch.items():
        #     # clip the buffer until filled position
        #     copy_batch[k] = v
        #
        # if self.config['ttc_type'] == 'clipped':
        #     for key in ['c_1', 'c_1_next', 'c_2', 'c_2_next']:
        #         batch[key] = np.clip(batch[key], 0., self.ttc_upper_bound)
        #
        # return batch

        pass

    def clip_obs(self, batch):
        """
        This method is suitable for different experiment setting.

        :param batch: original batch data from the dataset file
        :return: clipped observation
        """
        observation = batch['observation']

        # add multi-task option
        if self.multi_task:
            # append task code to state array
            task_code = np.squeeze(batch['task_code'])
            state = np.concatenate((observation, task_code), axis=1)

            return state

        # if not clip the state
        if not self.state_clip:
            state = np.array(observation[:, 0:self.input_dim])

            return state
        else:
            # data from dataset, ndarray format
            if isinstance(observation, np.ndarray) and len(observation.shape) > 1:
                clipped_state = np.array(observation[:, 0:self.input_dim])  # ego 4 + npc 6

                # todo add args and API to process multi-task data
                # task_code = np.array(observation[:, -4:])

                # clipped_observation = np.hstack((clipped_state, task_code))
                clipped_observation = clipped_state

            else:  # data in List format
                clipped_state = np.array(observation[0:self.input_dim])
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
        # # original lines
        # if self.config['ttc_version'] == 'c_1':
        #     if self.config['ttc_type'] == 'norm':
        #         c_key = 'c_1_norm'
        #     elif self.config['ttc_type'] == 'q_norm':
        #         c_key = 'c_1_q_norm'
        #     else:
        #         c_key = 'c_1'
        # else:
        #     if self.config['ttc_type'] == 'norm':
        #         c_key = 'c_2_norm'
        #     elif self.config['ttc_type'] == 'q_norm':
        #         c_key = 'c_2_q_norm'
        #     else:
        #         c_key = 'c_2'

        # use the clipped ttc with linear transformation
        c_key = 'c'

        # use the new constraint value and directly retrieve data array from batch
        c_array = batch[c_key]
        c_next_array = batch[c_key+'_next']

        if self.config['use_gpu']:
            observation = self._as_tensor(batch["observation"]).cuda()
            action = self._as_tensor(batch["action"]).cuda()

            # ===============  get batched c value  ===============
            # original lines
            # c = self._as_tensor(batch["c"]).cuda()
            # c_next = self._as_tensor(batch["c_next"]).cuda()

            # # previous version
            # c = self._as_tensor(batch[c_key]).cuda()
            # c_next = self._as_tensor(batch[c_key+"_next"]).cuda()

            # clipped ttc and linear transformation
            c = self._as_tensor(c_array).cuda()
            c_next = self._as_tensor(c_next_array).cuda()
        else:
            observation = self._as_tensor(batch["observation"])
            action = self._as_tensor(batch["action"])

            # ===============  get batched c value  ===============
            # original lines
            # c = self._as_tensor(batch["c"])
            # c_next = self._as_tensor(batch["c_next"])

            # # previous version
            # c = self._as_tensor(batch[c_key])
            # c_next = self._as_tensor(batch[c_key+"_next"])

            # clipped ttc and linear transformation
            c = self._as_tensor(c_array)
            c_next = self._as_tensor(c_next_array)

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

    def train_with_datasets(self, dataset_path, tag):
        """
        Train the safety model with datasets.

        With hierarchical structure to fit multiple datasets file

        :param dataset_path: parent folder of datasets
        :param tag: refer to the config info of current model
        :return:
        """
        # todo add setter to change class attribute output_path
        # create outputs path at the beginning of the training phase
        # output_path = os.path.join(self.output_path, self.time_stamp, tag)
        log_path = os.path.join(self.output_path, 'log', self.time_stamp, tag)
        model_path = os.path.join(self.output_path, 'model', self.time_stamp, tag)
        output_path = model_path

        checkpoints_path = os.path.join(model_path, 'checkpoints')

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)

        self._writer = SummaryWriter(log_path)

        # save config dict to file
        config_dict = {}
        for k, v in self.config.items():
            config_dict[k] = v

        config_dict['lr_scheduler'] = self.config['lr_scheduler'].__name__

        # append dataset source
        config_dict['dataset_path'] = dataset_path

        with open(os.path.join(output_path, 'config.json'), 'w') as fp:
            json.dump(config_dict, fp, indent=2)

        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        # get dataset file name and store in a list
        name_list = os.listdir(dataset_path)
        name_list.sort()
        dataset_list = [os.path.join(dataset_path, name) for name in name_list]

        # todo split training and validation datasets
        # split_factor = 0.9
        # epoch = split_factor * len(dataset_list)

        # save checkpoint frequency
        save_freq = self.config['save_freq']

        # todo fix this
        # total training epochs calculation
        epochs = self.config['epochs']

        #
        _index = [*range(len(dataset_list))]
        epoch_seq = _index * epochs
        random.shuffle(epoch_seq)

        # # all epochs in sequences
        # epoch_num = epoch_amplification * len(dataset_list)  # total epoch number
        # epoch_seq = _index
        # while len(epoch_seq) < epoch_num:
        #     random.shuffle(_index)
        #     epoch_seq += _index

        # ==================================================
        # assign a lr scheduler at beginning of training phase
        # store lr schedulers as list in case there are multiple constraints
        # default lr scheduler is MultiStepLR

        # original staircase
        # milestone_list = [int(len(epoch_seq)*x) for x in [0.25, 0.5, 0.75, 0.9]]

        # for straight task traing
        milestone_list = [int(len(epoch_seq)*x) for x in [0.3, 0.6, 0.75, 0.9]]

        self._lr_schedulers = [
            self.lr_scheduler(x, milestones=milestone_list, gamma=0.1, last_epoch=-1) for x in self._optimizers
        ]
        # ==================================================

        batch_losses = []
        for epoch, batch_id in enumerate(epoch_seq):
            # get batch data from dataset file path
            dataset_path = dataset_list[batch_id]
            batch = self.load_dataset_as_batch(dataset_path)

            minibatch_num = self.config['minibatch_num']
            minibatch_size = int(self.batch_size / minibatch_num)  # should be 1e3

            losses_list = []  # loss of each minibatch
            for i in range(minibatch_num):
                # update on each minibatch
                # todo need to check this
                minibatch = {k: v[i * minibatch_size:i * minibatch_size + (minibatch_size)] for k, v in batch.items()}
                losses_list.append(self._update_batch(minibatch))  # record loss of each minibatch

                # ==================================================
                # # record losses and histograms to tensorboard of each update
                # for_each(
                #     lambda x: self._writer.add_scalar(f"constraint {x[0]} training loss", x[1],
                #                                       self._train_global_step),
                #     enumerate(losses))
                #
                # (seq(self._models)
                #  .zip_with_index()  # (model, index)
                #  .map(lambda x: (f"constraint_model_{x[1]}", x[0]))  # (model_name, model)
                #  .flat_map(
                #     lambda x: [(x[0], y) for y in x[1].named_parameters()])  # (model_name, (param_name, param_data))
                #  .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1]))  # (modified_param_name, param_data)
                #  .for_each(lambda x: self._writer.add_histogram(x[0], x[1].data.numpy(), self._train_global_step)))
                #
                # self._train_global_step += 1
                #
                # print(f"Finished epoch {epoch} with losses: {losses}. Running validation ...")
                # ==================================================

            # loss of this epoch(dataset batch)
            losses = np.mean(np.concatenate(losses_list).reshape(-1, self._num_constraints), axis=0)
            batch_losses.append(losses)

            # ==================================================
            # Write losses and histograms to tensorboard
            for_each(
                lambda x: self._writer.add_scalar(f"constraint {x[0]} training loss", x[1], self._train_global_step),
                enumerate(losses))

            (seq(self._models)
             .zip_with_index()  # (model, index)
             .map(lambda x: (f"constraint_model_{x[1]}", x[0]))  # (model_name, model)
             .flat_map(lambda x: [(x[0], y) for y in x[1].named_parameters()])  # (model_name, (param_name, param_data))
             .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1]))  # (modified_param_name, param_data)
             .for_each(lambda x: self._writer.add_histogram(x[0], x[1].data.cpu().numpy(), self._train_global_step)))

            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {losses}. Running validation ...")
            # ==================================================

            # evaluate at the end of an epoch
            # get a evaluation batch
            eval_batch_path = random.sample(dataset_list, 1)[0]
            eval_batch = self.load_dataset_as_batch(eval_batch_path)
            self.evaluate_on_batch(eval_batch)

            # save checkpoint model
            if (epoch + 1) % save_freq == 0:
                for_each(lambda x: torch.save(x[1].state_dict(),
                                              os.path.join(model_path, 'checkpoints',
                                                           'model_'+str(x[0])+'_checkpoint_'+str(epoch + 1)+'.pth')),
                                              enumerate(self._models))
                print('checkpoint is saved.')

            # update lr scheduler
            for_each(lambda x: print('scheduler lr: ', x.get_lr()[0]), self._lr_schedulers)  # print current lr
            for_each(lambda x: x.step(), self._lr_schedulers)

            print("----------------------------------------------------------")

        # all epochs finish
        self._writer.close()
        print("==========================================================")
        print(f"Finished training constraint model. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")

        # save final model
        try:
            for_each(lambda x: torch.save(x[1].state_dict(),
                                          os.path.join(model_path, 'model_' + str(x[0]) + '_final.pth')),
                     enumerate(self._models))
        except:
            print('Fail to save final model.')

    # def set_attention_flag(self, attention):
    #     """
    #     Setter method for the attention usage.
    #     """
    #
    #     self.attention = attention

    def clip_observation(self, observation):
        """
        This method is for get_safe_action, only clip the observation from the env.

        Default observation is (, dim)
        """
        if self.multi_task:
            task_code = observation[-4:]
            clipped_observation = np.hstack((
                observation[0:34],
                observation[-4:],
            ))
        else:
            clipped_observation = observation[0:34]

        # add dim
        clipped_observation = clipped_observation[np.newaxis, :]

        return clipped_observation

    def get_safe_action(self, observation, action, c):
        """
        This method is referred as the action modifier.
        This method must be initialized with the safety layer instance.

        :param observation:
        :param action:
        :param c:
        :return action_new: modified action
        """
        clipped_observation = self.clip_observation(observation)

        if self.config['use_gpu']:
            clipped_observation = self._as_tensor(clipped_observation).view(1, -1).cuda()
        else:
            clipped_observation = self._as_tensor(clipped_observation).view(1, -1)

        # Find the values of G
        self._eval_mode()
        # g = [x(self._as_tensor(clipped_observation).view(1, -1)) for x in self._models]
        g = [x(clipped_observation) for x in self._models]
        self._train_mode()  # remove??? I don't think this is necessary..

        # Find the lagrange multipliers
        g = [x.data.cpu().numpy().reshape(-1) for x in g]
        # print('g value: ', g)

        # # todo we fix g here because of the error in constraint definition
        # g = [-1 * gs for gs in g]

        # # ====================  CAUTION!!!  ====================
        # # multipliers refer to the lambda in original paper, which are the lagrange multipliers
        # # original lines is from sumo experiment, the constraint is defined using the -ttc

        # original lines
        multipliers = [(np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]

        # # fixed line
        # multipliers = [(np.dot(g_i, action) - c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        # =======================================================

        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
        # print('multipliers: ', multipliers)
        # calculate correction on action
        correction = np.max(multipliers) * g[np.argmax(multipliers)]
        # print('correction: ', correction)

        action_new = action - correction

        # # original calculation of next step constraint value
        # c_next_predicted = [c[:, i] + \
        #                     torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
        #                     for i, x in enumerate(gs)]

        # simplified constraint value calculation
        c_next_predicted = c + np.matmul(g, np.array(action))

        # todo fix the debug info
        debug_info = {
            'constraint': c,
            'original_action': action,
            'g': g,
            'multipliers': multipliers,
            'correction': correction,
            'new_action': action_new,
            'c_next_predicted': c_next_predicted,
        }

        # # debug
        # print(
        #     '\n',
        #     '-*-'*20, '\n',
        #     'constraint value: {}'.format(c), '\n',
        #     'original action: {}'.format(action), '\n',
        #     'g value: ', g, '\n',
        #     'multipliers: ', multipliers, '\n',
        #     'correction: ', correction, '\n',
        #     'new action: ', action_new, '\n',
        #     'c_next_predicted: ', c_next_predicted, '\n',
        #     '-*-'*20, '\n',
        # )

        return action_new, debug_info
