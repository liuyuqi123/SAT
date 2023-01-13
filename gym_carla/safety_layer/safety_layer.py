"""
Copied from multi-task safety layer.

Use 2 dimension action for ddpg agent

Take task code as NN inputs to fit multi-task structure.

"""

# from safe_explorer.safety_layer.constraint_model import ConstraintModel
# from safe_explorer.utils.list import for_each

from torch.nn.init import uniform_
# from safe_explorer.core.net import Net

import torch
# import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from tensorboardX import SummaryWriter

import os
from datetime import datetime
from functional import seq
import numpy as np
import time
import random
import argparse

import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.init import uniform_, normal_
import torch.nn.functional as F


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


def for_each(f, l):
    for x in l:
        f(x)


class Net(Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_dims,
                 init_bound,
                 initializer,
                 last_activation):
        super(Net, self).__init__()

        self._initializer = initializer
        self._last_activation = last_activation

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        self._layers = ModuleList(seq(_layer_dims[:-1])
                                  .zip(_layer_dims[1:])
                                  .map(lambda x: Linear(x[0], x[1]))
                                  .to_list())

        self._init_weights(init_bound)

    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (seq(self._layers[:-1])
         .map(lambda x: x.weight)
         .for_each(self._initializer))
        # Init last layer with uniform initializer
        uniform_(self._layers[-1].weight, -bound, bound)

    def forward(self, inp):
        out = inp

        for layer in self._layers[:-1]:
            # parse output
            output_1 = layer(out)
            out = F.relu(output_1)

        if self._last_activation:
            out = self._last_activation(self._layers[-1](out))
        else:
            out = self._layers[-1](out)

        return out


class ConstraintModel(Net):
    def __init__(self, observation_dim, action_dim):
        # config = Config.get().safety_layer.constraint_model

        super(ConstraintModel, self) \
            .__init__(observation_dim,
                      action_dim,
                      [256, 512, 512, 256],  # layers
                      0.1,  # init bound of layer weight initialization
                      uniform_,
                      last_activation=None)


class ReplayBuffer:
    """A FIFO buffer implemented with fixed size numpy array"""

    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._buffers = {}
        self._current_index = 0
        self._filled_till = 0

    def _increment(self):
        self._current_index = (self._current_index + 1) % self._buffer_size
        self._filled_till = min(self._filled_till + 1, self._buffer_size)

    def _initialize_buffers(self, elements):
        self._buffers = {k: np.zeros([self._buffer_size, *v.shape], dtype=np.float32) \
                         for k, v in elements.items()}

    def add(self, elements):
        if len(self._buffers.keys()) == 0:
            self._initialize_buffers(elements)

        for k, v in elements.items():
            self._buffers[k][self._current_index] = v

        self._increment()

    def sample(self, batch_size):
        random_indices = np.random.randint(0, self._filled_till, batch_size)
        return {k: v[random_indices] for k, v in self._buffers.items()}

    def get_sequential(self, batch_size):
        for i in range(self._filled_till // batch_size):
            yield {k: v[i * batch_size: (i + 1) * batch_size] for k, v in self._buffers.items()}

    def clear(self):
        self._buffers = {}
        self._current_index = 0
        self._filled_till = 0


class SafetyLayer:

    # output_path = '../../outputs/safetylayer_outputs'
    output_path = './safetylayer_outputs'

    def __init__(self, config):

        self.config = config

        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']

        # get constraint number
        # self._num_constraints = env.get_num_constraints()
        self._num_constraints = int(1)  # default constraint number is 1

        # actually class of lr scheduler
        self.lr_scheduler = self.config['lr_scheduler']

        self._initialize_constraint_models()

        self._train_global_step = 0
        self._eval_global_step = 0

        # model path
        self.timestamp = TIMESTAMP
        self.tag = None
        self.model_path = None
        self.log_path = None
        self._writer = None  # tensorboardX logger

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models)

    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models)

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models)

    def _initialize_constraint_models(self):
        """
        Dimension for init NN is fixed.

        Init constraints models

        consider constraints as distance between ego and npc vehicles
        """
        # create constraint model
        self._models = [ConstraintModel(self.input_dim, self.output_dim) \
                        for _ in range(self._num_constraints)]
        # assign optimizer
        self._optimizers = [Adam(x.parameters(), lr=self.config['lr']) for x in self._models]
        # cuda usage
        # todo fix cuda error
        if self.config['use_gpu']:
            self._cuda()

    def load_models(self, model_path_list):
        """
        Load trained model dict from pth file.11

        :param model_path: <list> model dict file path
        """
        # todo check len
        # notice that usage of for each: for_each(f, l)
        for_each(lambda x: x[1].load_state_dict(torch.load(model_path_list[x[0]])), enumerate(self._models))
        self._eval_mode()
        # print('safety layer models are loaded.')

    def clip_obs(self, observation):
        """
        This method should be modified in different experiment setting.

        Fix this method to design input of safety layer model.

        :param observation: original observation ndarray
        :return: clipped observation
        """
        # this method shoulf be fixed if usage of observation changed
        if isinstance(observation, np.ndarray) and len(observation.shape) > 1:
            # if add task code into input
            clipped_state = np.array(observation[:, 0:9])
            task_code = np.array(observation[:, -4:-1])
            clipped_observation = np.hstack((clipped_state, task_code))
        else:
            clipped_state = np.array(observation[0:9])
            task_code = np.array(observation[-4:-1])
            clipped_observation = np.hstack((clipped_state, task_code))

        return clipped_observation

    def _update_batch(self, batch):
        """
        Update the model by a given batch of data(minibatch)
        :param batch: batch data in dict
        :return: training loss
        """
        # batch = self._replay_buffer.sample(self._config.batch_size)
        # batch = self._replay_buffer.sample(config['batch_size'])

        # Update critic
        for_each(lambda x: x.zero_grad(), self._optimizers)
        losses = self._evaluate_batch(batch)
        for_each(lambda x: x.backward(), losses)

        # print current optimizer lr value
        for_each(lambda x: print('optimizer lr: ', x.param_groups[0]['lr']), self._optimizers)  # print current lr

        for_each(lambda x: x.step(), self._optimizers)

        return np.asarray([x.item() for x in losses])

    def _evaluate_batch(self, batch):
        """
        Evaluate a batch data.
        :param batch: batch data in dict
        :return: losses <list>
        """
        if self.config['use_gpu']:
            observation = self._as_tensor(batch["observation"]).cuda()
            action = self._as_tensor(batch["action"]).cuda()
            c = self._as_tensor(batch["c"]).cuda()
            c_next = self._as_tensor(batch["c_next"]).cuda()
        else:
            observation = self._as_tensor(batch["observation"])
            action = self._as_tensor(batch["action"])
            c = self._as_tensor(batch["c"])
            c_next = self._as_tensor(batch["c_next"])

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

        # ==================================================
        # old version, manually clip the obs
        # clip observation
        # clipped_observation = batch['observation'][:, 0:self.input_dim]
        # batch['observation'] = clipped_observation

        # use a method to clip the obs
        # rewrite method for different safety layer
        clipped_observation = self.clip_obs(batch['observation'])
        batch['observation'] = clipped_observation

        return batch

    def evaluate_on_batch(self, batch):
        """
        Use a given batch to evaluate.
        :param batch:
        :return:
        """

        self._eval_mode()

        # ==================================================
        # compute losses on minibatches
        minibatch_num = self.config['minibatch_num']
        minibatch_size = int(self.batch_size / minibatch_num)  # should be 1e3

        loss_dict = []
        for i in range(minibatch_num):  # iterate on each minibatch
            """
            # ==================================================
            # original method, shuffle minibatch of a batch
            random_indices = np.random.randint(0, batch_size, minibatch_size)
            minibatch = {k: v[random_indices] for k, v in batch.items()}
            """
            minibatch = {k: v[i * minibatch_size:i * minibatch_size + (minibatch_size)] for k, v in batch.items()}
            loss_dict.append(self._evaluate_batch(minibatch))

        losses = np.mean(np.concatenate(loss_dict).reshape(-1, self._num_constraints), axis=0)

        # Log to tensorboard
        for_each(lambda x: self._writer.add_scalar(f"constraint {x[0]} eval loss", x[1], self._eval_global_step),
                 enumerate(losses))
        self._eval_global_step += 1

        self._train_mode()

        print(f"Validation completed, average loss {losses}")

    def train_with_datasets(self, dataset_path=None):
        """
        Train the safety model with datasets.

        With hierarchical structure to fit multiple datasets file

        :param dataset_path: parent folder of datasets
        :param tag: tag of this model
        :return:
        """
        # if not dataset_path:
        #     dataset_path = self.config['dataset_path']

        # tag the trained model
        self.tag = os.path.join(self.config['tag'], self.timestamp)

        # create outputs path only in training phase
        self.log_path = os.path.join(self.output_path, self.tag, 'log')
        self.model_path = os.path.join(self.output_path, self.tag, 'model')
        self._writer = SummaryWriter(self.log_path)

        os.makedirs(self.model_path, exist_ok=True)
        checkpoints_path = os.path.join(self.model_path, 'checkpoints')
        os.makedirs(checkpoints_path, exist_ok=True)

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
        self.save_freq = self.config['save_freq']
        # total training epochs
        epochs = self.config['epochs']

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

        milestone_list = [int(len(epoch_seq)*x) for x in [0.25, 0.5, 0.75, 0.9]]
        self._lr_schedulers = [self.lr_scheduler(x, milestones=milestone_list, gamma=0.1, last_epoch=-1) \
                               for x in self._optimizers]
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

            # to save the params of networks
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
            if (epoch + 1) % self.save_freq == 0:
                for_each(lambda x: torch.save(x[1].state_dict(),
                                              os.path.join(self.model_path, 'checkpoints',
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
                                          os.path.join(self.model_path, 'model_' + str(x[0]) + '_final.pth')),
                     enumerate(self._models))
        except:
            print('Fail to save final model.')

    def get_safe_action(self, observation, action, c):
        """
        Get a safe action
        :param observation:
        :param action:
        :param c:
        :return action_new: modified action
        """

        # check if observation is clipped
        # todo check dimension
        clipped_observation = self.clip_obs(observation)#.view(1, -1)

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

        multipliers = [(np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
        # print('multipliers: ', multipliers)
        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]
        # print('correction: ', correction)

        action_new = action - correction

        # print correction
        # if any(correction) > 0:
        #     # print('-'*10)
        #     print('correction: ', correction)
        #     print('old action: ', action)
        #     print('new action: ', action_new)
        #     print('g value: ', g)
        #     print('multipliers: ', multipliers)

        return action_new

    # todo fix online sample and evaluate
    # def evaluate(self):
    #     # Sample steps
    #     self._sample_steps(config['evaluation_steps'])
    #
    #     self._eval_mode()
    #     # compute losses
    #     losses = [list(map(lambda x: x.item(), self._evaluate_batch(batch))) for batch in \
    #               # self._replay_buffer.get_sequential(self._config.batch_size)]
    #               self._replay_buffer.get_sequential(config['batch_size'])]
    #
    #     losses = np.mean(np.concatenate(losses).reshape(-1, self._num_constraints), axis=0)
    #
    #     self._replay_buffer.clear()
    #     # Log to tensorboard
    #     for_each(lambda x: self._writer.add_scalar(f"constraint {x[0]} eval loss", x[1], self._eval_global_step),
    #              enumerate(losses))
    #     self._eval_global_step += 1
    #
    #     self._train_mode()
    #
    #     print(f"Validation completed, average loss {losses}")
    #
    # def _sample_steps(self, num_steps):
    #     """
    #     num_steps is the desired env step number.
    #
    #     :param num_steps:
    #     :return:
    #     """
    #
    #     episode_length = 0
    #
    #     observation = self._env.reset()
    #     clipped_observation = self.clip_obs(observation)
    #
    #     for step in range(num_steps):
    #         action = self._env.action_space.sample()
    #         # todo this method may require fixing
    #         c = self._env.get_constraint_values()
    #         observation_next, _, done, _ = self._env.step(action)
    #
    #         c_next = self._env.get_constraint_values()
    #
    #         self._replay_buffer.add({
    #             "action": action,
    #             "observation": clipped_observation,
    #             "c": c,
    #             "c_next": c_next
    #         })
    #
    #         observation = observation_next
    #         clipped_observation = self.clip_obs(observation)
    #         episode_length += 1
    #
    #         # if done or (episode_length == self._config.max_episode_length):
    #         # remove max_episode_length limit
    #         if done:
    #             observation = self._env.reset()
    #             clipped_observation = self.clip_obs(observation)
    #             episode_length = 0


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
        'input_dim': int(12),
        'output_dim': int(2),
        'lr': 1e-2,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 10,
        'save_freq': 5,  # total checkpoints number
        'use_gpu': True,  # False
        'dataset_path': '/home/liuyuqi/PycharmProjects/gym-sumo/experiments/multi_task/with_attention/safety_layer_datasets/debug/new_c/2020-10-20-Time09-42-47',
        'tag': 'debug',
    }

    tags = ['accel_slack_1_5',
            'random',
            'multi_task',
            'multi_task_task_code',
            'multi_task_without_task_code',
            ]

    # safety layer config
    safetylayer_config = {
        'input_dim': int(12),  # add task code
        'output_dim': int(2),
        'lr': 1e-2,  # initial lr value
        'lr_scheduler': MultiStepLR,
        'minibatch_num': 10,
        'epochs': 100,
        'save_freq': 500,  # total checkpoints number
        'use_gpu': True,
        'dataset_path': './with_attention/safety_layer_datasets/random_policy/multi_task_new_c/datasets',
        # fix the tag for different settings
        'tag': 'multi_task_with_task_code',
    }

    # init safety layer class
    # debug
    # safetylayer = SafetyLayer(config=debug_safetylayer_config)
    # train
    safetylayer = SafetyLayer(config=safetylayer_config)

    # ==================================================
    # train safety layer with datasets

    safetylayer.train_with_datasets()

    print('Finish training safety layer model.')


def test_load_data():
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
        'dataset_path': '/home/lyq/PycharmProjects/gym-sumo/experiments/multi_task/without_attention/safety_layer_datasets/debug/2020-10-17-Time16-00-00',
        'tag': 'debug',
    }

    # init safety layer class
    safetylayer = SafetyLayer(config=debug_safetylayer_config)

    # test usage of clip obs
    dataset_path = '/home/lyq/PycharmProjects/gym-sumo/experiments/multi_task/without_attention/safety_layer_datasets/debug/2020-10-17-Time17-59-02/dataset_0.npz'

    safetylayer.load_dataset_as_batch(dataset_path)

    print('d')


if __name__ == '__main__':

    # ==================================================
    # train loop
    train_safety_layer()

    # ==================================================
    # eval
    # eval_safetylayer()

    # ==================================================
    # test get safe action
    # test_get_safe_action()

    # test load dataset
    # test_load_data()
