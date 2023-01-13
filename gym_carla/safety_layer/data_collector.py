"""
This script is supposed to run a data collection method for safety layer training.

Data is collected through roll-out data using gym_carla env.

todo list
 - merge safety layer training process into RL training
    this requires to merge related methods into the main training loop
 - Compare several different settings: single-task and multi-task as input
 - use RNN to predict safety constraints
 - record data format and respective NN input dim
 - add a meta method for data collection, add options on setting the sensor data type

"""

import os
import glob
import sys

gym_carla_path = '/home1/lyq/PycharmProjects/gym-carla/'
sys.path.append(gym_carla_path)

from gym_carla.config.carla_config import version_config

# from ..config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

carla_root = os.path.join(root_path, 'CARLA_' + carla_version)
carla_path = os.path.join(carla_root, 'PythonAPI')
sys.path.append(carla_path)
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla'))
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/agents'))

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import numpy as np
import pickle
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter
from interval import Interval

# =====  carla scripts  =====
from gym_carla.envs.carla_env_multi_task import CarlaEnvMultiTask

from development.replay_buffers import FullDataBuffer


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class ReplayBuffer:
    """
    A FIFO buffer implemented with fixed size numpy array.

    todo add load method to retrieve dataset

    """

    def __init__(self, buffer_size):
        self._buffer_size = int(buffer_size)
        self._buffers = {}  # where data stored
        self._current_index = 0
        self._filled_till = 0

    def _increment(self):
        self._current_index = (self._current_index + 1) % self._buffer_size
        self._filled_till = min(self._filled_till + 1, self._buffer_size)

    def _initialize_buffers(self, elements):
        # self._buffers = {k: np.zeros([self._buffer_size, *v.shape], dtype=np.float32) \
        #                  for k, v in elements.items()}

        # parse original line
        for k, v in elements.items():
            shape = v.shape  # shape can be multi-dimensional
            self._buffers[k] = np.zeros([self._buffer_size, *v.shape], dtype=np.float32)

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

    def save(self, path):
        """
        Note that path is a .npz path
        :param path:
        :return:
        """
        save_dict = dict()
        for k, v in self._buffers.items():
            # clip the buffer until filled position
            save_dict[k] = v[0: self._filled_till, :]  # 1st dim is step length

        # save the dataset
        # np.savez("small_dataset.npz", **save_dict)

        np.savez(path, **save_dict)
        print('dataset is saved.')

        # try:
        #     np.savez(path, **save_dict)
        #     print('dataset is saved.')
        # except:
        #     print('fail to save the dataset.')

    def filtered_dataset(self):
        """
        Adjust dataset distribution.

        todo unfinished, add a method to collect data of specific condition
        """
        proportion = 0.1  # proportion of done condition
        save_dict = dict()
        for k, v in self._buffers.items():
            if k == 'done':
                # v[i, ] for i in range(self._filled_till)
                print('d')
            print('d')


class AccelPolicy:
    """Full acceleration policy"""

    policy_name = 'Accel'

    @staticmethod
    def predict(obs):
        acc = np.array([1.0])
        return acc, None


class RandomPolicy:
    """Random policy"""

    policy_name = 'Random'

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space

    def predict(self, obs):
        action = self.action_space.sample()
        return action, None


class DataCollector:
    """
    Collect roll out data by given policy.
    """

    env_cls = CarlaEnvMultiTask
    # format: state array in dict
    replay_buffer_cls = ReplayBuffer

    def __init__(
            self,
            args,
            policy_name: str,
            buffer_size=10000,
            dataset_number=100,
            output_folder='./data_collection',  # default output folder path
    ):
        self.args = args
        self.env = self.init_carla_env(self.args.route[0])

        self.policy_name = policy_name
        self.policy = self.register_policy(policy_name)  # random policy requires init

        self.output_folder = output_folder

        self.buffer_size = int(buffer_size)
        self.dataset_number = int(dataset_number)

        # we always collect data in separate buffer
        # data type to collect
        self.collect_failure_data = True if 'failure' in self.args.data_type else False
        self.collect_success_data = True if 'success' in self.args.data_type else False

        # # original replay buffer for safety layer training
        # self.replay_buffer = self.replay_buffer_cls(buffer_size=self.buffer_size)
        # print('Replay buffer size is {}'.format(self.buffer_size))

        # buffers for success and failure data
        if self.collect_failure_data:
            self.replay_buffer_failure = self.replay_buffer_cls(buffer_size=self.buffer_size)
            print('Replay buffer for Failure data is initialized, the size is {}'.format(self.buffer_size))

        if self.collect_success_data:
            self.replay_buffer_success = self.replay_buffer_cls(buffer_size=self.buffer_size)
            print('Replay buffer for Success data is initialized, the size is {}'.format(self.buffer_size))

        # todo fix this part
        #  deprecated, self-defined data stored in dict
        # for contrastive data
        self.full_data_buffer_cls = FullDataBuffer

        # todo store key params in dict and save it, fix the API
        self.params = {}

    def init_carla_env(self, task_option='left'):
        """
        Set the route option with setter API: set_task_option()

        Default route option is left.
        """

        # todo add args to set carla port for multi-client
        env = self.env_cls(
            carla_port=self.args.port,
            tm_port=self.args.trafficManagerPort,
            tm_seed=self.args.trafficManagerSeed,  # seed for autopilot controlled NPC vehicles

            train=False,  # training mode or evaluation mode
            task_option=task_option,
            attention=False,

            initial_speed=None,  # if set a initial speed to ego vehicle
            state_noise=False,  # if addings noise on state vector

            no_render_mode=not self.args.render_mode,
            debug=False,
            verbose=False,
        )

        return env

    def clear_data_buffer(self):
        """"""
        pass

    def get_config_info(self):
        """
        todo fix this method, save data collection config info
        """

        # info_dict = {}
        #
        # for key, item in self.env_cls.modules:
        #     info_dict[key] = item.__name__
        #
        # return info_dict

        pass

    def register_policy(self, policy):
        """
        todo register full accel policy.
        todo add RL model for the data collection

        """
        if self.policy_name == 'random':
            policy = RandomPolicy(env=self.env)
            # print('Current Policy: Random policy')

        # if isinstance(policy, RandomPolicy):  # check if using random policy
        #     policy = RandomPolicy(env=self.env)
        #     print('Random policy will be used for data collection.')
        # else:
        #     # self.policy = policy
        #     print('Use a trained policy collect rollout data.')

        print('Current Policy is: {}'.format(self.policy_name))

        return policy

    def create_output_folder(self, tag):
        """
        Generate the output folder and dataset name.

        :param tag:
        :return:
        """

        save_path = os.path.join(self.output_folder, tag)
        os.makedirs(save_path, exist_ok=True)

        return save_path

    def collect_all_data(self):
        """
        Collect roll-out data and save into datasets.

        The success and failure data are saved separately at the end of each episode.

        :return:
        """
        # todo fix dataset size and number
        # replay buffers for success and failure data
        success_buffer = self.full_data_buffer_cls(self.buffer_size/10)
        collision_buffer = self.full_data_buffer_cls(self.buffer_size/10)

        # loop for roll-out all routes
        # ['straight_0', 'left', 'right']
        for route_option in self.args.route:

            # check and reset the route option
            if route_option is not self.env.route_option:
                self.env.set_task_option(route_option)

            # episodic data buffer
            episode_buffer = []

            # clear all buffer when the route changes
            self.replay_buffer.clear()
            self.replay_buffer_success.clear()
            self.replay_buffer_failure.clear()
            # full data buffers
            success_buffer.clear()
            collision_buffer.clear()

            # original state data buffer
            original_state_data_buffer = []
            original_success_state_dataset_index = int(0)
            original_failure_state_dataset_index = int(0)

            # external loop for multiple dataset collection
            for dataset_index in range(self.dataset_number):

                # clear_period = 3
                # if (dataset_index + 1) % clear_period == 0:
                #     self.env = self.init_carla_env(task_option=route_option)
                #     if route_option is not self.env.route_option:
                #         self.env.set_task_option(route_option)

                # tags for dataset classification
                # tag_list = [
                #     'original_state',
                #     'original_state_success',
                #     'original_state_failure',
                #
                #     'full_data_success',
                #     'full_data_failure',
                # ]

                save_path_dict = {
                    'original_state': os.path.join(self.output_folder, 'original_state', route_option, 'hybrid'),
                    'original_state_success': os.path.join(self.output_folder, 'original_state', route_option, 'success'),
                    'original_state_failure': os.path.join(self.output_folder, 'original_state', route_option, 'failure'),

                    'full_data_success': os.path.join(self.output_folder, 'full_data', route_option, 'success'),
                    'full_data_failure': os.path.join(self.output_folder, 'full_data', route_option, 'failure'),
                }

                for key, path in save_path_dict.items():
                    os.makedirs(path, exist_ok=True)

                success_buffer.set_save_path(save_path_dict['full_data_success'])
                collision_buffer.set_save_path(save_path_dict['full_data_failure'])

                # initial step
                observation = self.env.reset()
                # get constraint value from state manager
                state_data_dict, full_data_dict = self.env.get_single_frame_data()
                c_1 = state_data_dict['ttc_1']
                c_2 = state_data_dict['ttc_2']

                single_dataset_full = False

                # keep rolling out untill single dataset is collected
                while not single_dataset_full:

                    reward = None
                    done = False
                    aux = None

                    # timestep
                    elapsed_timestep = 0

                    # roll out by episodes
                    # append buffer data into the dataset
                    while not done:

                        # todo use a external policy class
                        # random action
                        action = np.random.rand(2)

                        # state, reward, done, aux
                        observation_next, reward, done, aux = self.env.step(list(action))

                        # data collection API
                        state_data_dict, full_data_dict = self.env.get_single_frame_data()

                        # task code of current route option
                        task_code = self.env.task_code

                        # get constraint value from new API
                        c_1_next = state_data_dict['ttc_1']
                        c_2_next = state_data_dict['ttc_2']

                        # for state data collection, use 0,1 to state success or failure
                        done_flag = 1 if done else 0

                        # save into replay buffer
                        step_state_data = {
                            "action": action,
                            "observation": np.array(observation),
                            "c_1": np.array([c_1]),
                            "c_1_next": np.array([c_1_next]),
                            # "c_1_norm": np.array([self.env.state_manager.normalize_ttc(c_1)]),
                            # "c_1_next_norm": np.array([self.env.state_manager.normalize_ttc(c_1_next)]),

                            "c_2": np.array([c_2]),
                            "c_2_next": np.array([c_2_next]),
                            # "c_2_norm": np.array([self.env.state_manager.normalize_ttc(c_2)]),
                            # "c_2_next_norm": np.array([self.env.state_manager.normalize_ttc(c_2_next)]),

                            'done': np.array([done_flag]),
                            'task_code': np.array([task_code]),
                        }

                        self.replay_buffer.add(step_state_data)
                        original_state_data_buffer.append(step_state_data)

                        # save the dataset
                        if self.replay_buffer._filled_till == self.replay_buffer._buffer_size:
                            self.replay_buffer.save(
                                os.path.join(
                                    save_path_dict['original_state'],
                                    'dataset_' + str(dataset_index) + '.npz',
                                )
                            )
                            self.replay_buffer.clear()

                            single_dataset_full = True

                            continue

                        observation = observation_next
                        c_1 = c_1_next
                        c_2 = c_2_next
                        # episode_length += 1

                        episode_buffer.append(full_data_dict)

                        elapsed_timestep += 1

                        if done:

                            observation = self.env.reset()
                            state_data_dict, full_data_dict = self.env.get_single_frame_data()
                            c_1 = state_data_dict['ttc_1']
                            c_2 = state_data_dict['ttc_2']

                            # ===============  save episodic data to success or failure  ===============
                            #
                            if aux['exp_state'] == 'collision':
                                todo_buffer = collision_buffer
                                state_data_buffer = self.replay_buffer_failure
                                save_path = save_path_dict['original_state_failure']
                            elif aux['exp_state'] == 'success':
                                todo_buffer = success_buffer
                                state_data_buffer = self.replay_buffer_success
                                save_path = save_path_dict['original_state_success']
                            else:
                                # todo fix this
                                raise ValueError('Wrong buffer instance.')

                            os.makedirs(save_path, exist_ok=True)

                            for item in episode_buffer:
                                # todo merge auto save method into buffer classmethod
                                todo_buffer.add(item)

                            for item in original_state_data_buffer:
                                state_data_buffer.add(item)

                                # check and save the dataset
                                if state_data_buffer._filled_till == state_data_buffer._buffer_size:
                                    if aux['exp_state'] == 'collision':
                                        _index = original_failure_state_dataset_index
                                    elif aux['exp_state'] == 'success':
                                        _index = original_success_state_dataset_index
                                    else:
                                        raise ValueError('No time-exceeding dataset.')

                                    dataset_save_path = os.path.join(
                                        save_path,
                                        'dataset_' + str(_index) + '.npz',
                                    )

                                    state_data_buffer.save(dataset_save_path)
                                    state_data_buffer.clear()

                                    if aux['exp_state'] == 'collision':
                                        original_failure_state_dataset_index += 1
                                    if aux['exp_state'] == 'success':
                                        original_success_state_dataset_index += 1

                            # reset buffer
                            episode_buffer = []
                            original_state_data_buffer = []

                            # todo add termination conditions based on dataset number
                            # success_buffer.dataset_num
                            # collision_buffer.dataset_num

                            elapsed_timestep = 0

    def run_collect_data(self):
        """
        Collect roll-out data and save into datasets.

        The success and failure data are saved separately at the end of each episode.

        :return:
        """
        # roll-out and collect data over all specified routes
        # ['straight_0', 'left', 'right']
        for route_option in self.args.route:

            # check and reset the route option in env
            if route_option is not self.env.route_option:
                self.env.set_task_option(route_option)

            # generate the output path for the datasets
            failure_dataset_path = os.path.join(self.output_folder, 'original_state', route_option, 'failure')
            success_dataset_path = os.path.join(self.output_folder, 'original_state', route_option, 'success')

            os.makedirs(failure_dataset_path, exist_ok=True)
            os.makedirs(success_dataset_path, exist_ok=True)

            # clear all buffer in case the route changes
            if self.collect_success_data:
                self.replay_buffer_success.clear()
                success_dataset_index = int(0)
            if self.collect_failure_data:
                self.replay_buffer_failure.clear()
                failure_dataset_index = int(0)

            # original state data buffer
            data_buffer = []

            success_data_done = False
            failure_data_done = False
            collection_done = False

            # loop for multiple dataset collection
            while not collection_done:

                # ==========  initial step  ==========
                observation = self.env.reset()
                # get constraint value from state manager
                state_data_dict, full_data_dict = self.env.get_single_frame_data()
                c_2 = state_data_dict['ttc_2']

                reward = None
                done = False  # episode done flag
                aux = None

                # timestep of single episode
                elapsed_timestep = 0

                # roll out single episode dataset
                while not done:

                    # todo use a external policy class
                    # random action
                    action = np.random.rand(2)

                    # state, reward, done, aux
                    observation_next, reward, done, aux = self.env.step(list(action))

                    # data collection API
                    state_data_dict, full_data_dict = self.env.get_single_frame_data()

                    # task code of current route option
                    task_code = self.env.task_code

                    # get constraint value from new API
                    c_2_next = state_data_dict['ttc_2']

                    # for state data collection, use 0,1 to state success or failure
                    done_flag = 1 if done else 0

                    # save into replay buffer
                    step_state_data = {
                        "action": action,
                        "observation": np.array(observation),

                        "ttc": np.array([c_2]),
                        "ttc_next": np.array([c_2_next]),

                        'done': np.array([done_flag]),
                        'task_code': np.array([task_code]),
                    }

                    # store in data buffer
                    data_buffer.append(step_state_data)

                    # update adjacent step data
                    observation = observation_next
                    c_2 = c_2_next

                    elapsed_timestep += 1

                    if done:

                        # ===============  save episodic data to success or failure  ===============
                        if aux['exp_state'] == 'collision':
                            if self.collect_failure_data and not failure_data_done:
                                # filter values of c_2 and c_2_next
                                filter_data_buffer = self.filter_data_buffer(data_buffer)
                                # save buffered data into replay buffer
                                for item in filter_data_buffer:
                                    self.replay_buffer_failure.add(item)
                                    # check and save the dataset
                                    if self.replay_buffer_failure._filled_till == self.replay_buffer_failure._buffer_size:
                                        dataset_save_path = os.path.join(
                                            failure_dataset_path,
                                            'dataset_' + str(failure_dataset_index) + '.npz',
                                        )

                                        self.replay_buffer_failure.save(dataset_save_path)
                                        self.replay_buffer_failure.clear()

                                        failure_dataset_index += 1

                        elif aux['exp_state'] == 'success':
                            if self.collect_success_data and not success_data_done:
                                # filter values of c_2 and c_2_next
                                filter_data_buffer = self.filter_data_buffer(data_buffer)
                                # save buffered data into replay buffer
                                for item in filter_data_buffer:
                                    self.replay_buffer_success.add(item)
                                    # check and save the dataset
                                    if self.replay_buffer_success._filled_till == self.replay_buffer_success._buffer_size:
                                        dataset_save_path = os.path.join(
                                            success_dataset_path,
                                            'dataset_' + str(success_dataset_index) + '.npz',
                                        )
                                        self.replay_buffer_success.save(dataset_save_path)
                                        self.replay_buffer_success.clear()

                                        success_dataset_index += 1

                        else:
                            # todo fix this
                            raise ValueError('Wrong buffer instance.')

                        # directly clear the buffer
                        data_buffer = []
                        elapsed_timestep = 0

                        # check if collection is done
                        if self.collect_success_data:
                            if success_dataset_index >= self.dataset_number:
                                success_data_done = True
                        else:
                            success_data_done = True

                        if self.collect_failure_data:
                            if failure_dataset_index >= self.dataset_number:
                                failure_data_done = True
                        else:
                            failure_data_done = True

                        if success_data_done and failure_data_done:
                            collection_done = True

                        # if not collection_done:
                        #
                        #     observation = self.env.reset()
                        #     state_data_dict, full_data_dict = self.env.get_single_frame_data()
                        #
                        #     # update value for next episode
                        #     c_2 = state_data_dict['ttc_2']

    @staticmethod
    def filter_data_buffer(data_buffer: list):
        """"""
        # todo add args fot the filter range
        zoom = Interval(0., 5.)

        # index_list = []
        filtered_data_buffer = []

        for index, item in enumerate(data_buffer):

            ttc = item["ttc"][0]
            ttc_next = item["ttc_next"][0]

            if ttc in zoom and ttc_next in zoom:

                # index_list.append(index)

                filtered_data_buffer.append(item)

        return filtered_data_buffer

    def collect_dataset(
            self,
            route_options: list,
            dataset_num,
            dataset_size,
            policy_name='random',  # todo fix this
            tag=None,
    ):
        """
        Collect datasets.
        """

        for route_option in route_options:

            # check and reset the route option
            if route_option is not self.env.route_option:
                self.env.set_task_option(route_option)

            # log file
            # policy for data collection, class name, location
            # env name, location
            # route info



            # collect multiple datasets
            for num in range(dataset_num):

                # collect data
                # self.rollout(batchsize)


                # a rollout with single route option
                self.rollout_single_route(dataset_size=dataset_size)

                # check replay buffer size
                # is equal to _filled_till
                print('==================================================')
                print('Dataset Number: {}'.format(num))
                print('Current batch size: ', self.replay_buffer._filled_till)

                # save dataset
                dataset_name = 'dataset_' + str(num) + '.npz'


                # todo fix this
                # dataset_path = os.path.join(self.parent_folder, route_option, dataset_name)
                dataset_path = './fix_this.npz'

                self.replay_buffer.save(dataset_path)

                # clean the replay buffer
                self.replay_buffer.clear()

    def rollout_single_route(self, dataset_size):
        """
        todo append task code into the dataset

        Rollout with single route option
        """
        # initial state
        observation = self.env.reset()
        c = self.env.get_constraint_values()

        for step in range(dataset_size):

            # ==================================================
            # todo 1 get action through policy
            # if not self.policy:  # use random policy
            #     action = self.env.action_space.sample()
            # else:  # use trained model
            #     # check API of the model
            #     action, _ = self.policy.predict(observation)

            # todo 2 get action from env action space
            # action = self.env.action_space.sample()
            # ==================================================

            # manual randomaction
            action = np.random.rand(2)
            action = list(action)

            observation_next, _, done, _ = self.env.step(action)

            # todo continue from here, add task code into the info dict
            task_code = self.env.task_code

            c_next = self.env.get_constraint_values()

            done_flag = 0
            if done:
                done_flag = 1

            # save into replay buffer
            state_data_dict = {
                "action": action,
                "observation": observation,
                "c": c,  # np.array([c]),
                "c_next": c_next,  # np.array([c_next]),
                'done': np.array([done_flag]),
                'task_code': task_code,
            }
            self.replay_buffer.add(state_data_dict)

            observation = observation_next
            c = c_next
            # episode_length += 1

            if done:
                observation = self.env.reset()
                c = self.env.get_constraint_values()
                # episode_length = 0


def run(args):
    """
    Run collection loop with given args.
    """

    # debug
    if args.debug:
        args.dataset_number = 3
        args.dataset_size = 50

        output_folder = './data_collection/debug/' + TIMESTAMP

    else:
        output_folder = './data_collection/' + args.tag

    # todo move init env into main()
    data_collector = DataCollector(
        args=args,
        policy_name='random',
        buffer_size=args.dataset_size,
        dataset_number=args.dataset_number,
        output_folder=output_folder,
    )

    # # original lines
    # data_collector.collect_dataset(
    #     route_options=['left', 'right', 'straight'],
    #     dataset_num=args.dataset_number,
    #     dataset_size=args.dataset_size,
    #     policy_name='random',
    #     tag=None,
    # )

    # # original lines
    # data_collector.collect_all_data()

    # we only collect failure data now
    data_collector.run_collect_data()


def main():

    description = "Data collection using carla env.\n"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)

    # todo add args to set success and failure dataset
    parser.add_argument('--policy',
                        default='random',
                        dest='policy',
                        help='Agent policy option in collecting data.')
    parser.add_argument(
        '-r', '--route',
        type=str, nargs='+',
        default=['left'],
        dest='route',
        help='Route options for data collection.'
    )
    parser.add_argument(
        '-n', '--number',
        default=int(50),
        dest='dataset_number',
        help='Number of datasets to be collected.'
    )
    parser.add_argument(
        '-s', '--size',
        default=int(10000),
        dest='dataset_size',
        help='Size of each dataset.'
    )
    parser.add_argument(
        '-p', '--port',
        default=2000, type=int,
        dest='port',
        help='CARLA port number(default: 2000).'
    )
    parser.add_argument('--trafficManagerPort', default=int(8100), type=int,
                        help='CARLA TrafficManager port number(default: 8100)')
    parser.add_argument('--trafficManagerSeed', default=int(0), type=int,
                        help='Seed used by the TrafficManager(default is 0)')
    parser.add_argument('--timeout', default="10.0",
                        help='CARLA client timeout value(seconds).')
    parser.add_argument('--debug', action="store_true", help='Run under the debug mode.')
    parser.add_argument('--tag', default=None,
                        help='Tag for current data collection.')
    # todo we fix render mode, set no-render as the default mode, merge this to other scripts
    parser.add_argument(
        '-d', '--render',
        action='store_true',
        dest='render_mode',
        help='Whether use no-render mode on CARLA server, default is no-render mode.'
    )
    parser.add_argument(
        '-y', '--data-type',
        type=str, nargs='+',
        default=['failure', 'success'],  # 'failure', 'success'
        dest='data_type',
        help='Which type of data to collect, select from: failure, success.'
    )

    arguments = parser.parse_args()

    # set dataset tag
    if arguments.debug:
        arguments.tag = 'debug'

    if not arguments.tag:
        arguments.tag = TIMESTAMP
    else:
        arguments.tag = arguments.tag + '/' + TIMESTAMP

    # automatically fix tm_port according to carla port number
    if arguments.port != int(2000):
        d = arguments.port - 2000
        arguments.trafficManagerPort = int(8100+d)

    run(arguments)

    # # todo reload carla world when error occurs
    # try:
    #     run(arguments)
    # except:
    #     pass


if __name__ == '__main__':

    sys.exit(main())
