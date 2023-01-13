"""
carla env multi-task is a new version of carla env for multi-task and social attention experiments.

This will inherit all methods developed in previous version.

This env is developed to deploy safety layer method.

Main features:
 - use autopilot controlled NPC vehicles
 - tls is considered to control the traffic flow

Following methods are added:
 - add new APIs for data collection.
 - action policy option

todo list
 - merge safety related methods with multi-task and attention methods

"""

import os
import sys
import glob

import math

import gym
import numpy as np
import random
from datetime import datetime
from collections import deque

import carla

# ====================   CARLA modules   ====================
from gym_carla.envs.BasicEnv import BasicEnv
from gym_carla.util_development.vehicle_control import Controller
from gym_carla.util_development.sensors import Sensors
from gym_carla.util_development.kinetics import get_transform_matrix

# developing modules
from gym_carla.modules.rl_modules.state_representation.state_manager6 import StateManager6
from gym_carla.modules.trafficflow.traffic_flow_manager_multi_task import TrafficFlowManagerMultiTask
from gym_carla.modules.traffic_lights.traffic_lights_manager2 import TrafficLightsManager
from gym_carla.modules.route_generator.junction_route_manager2 import JunctionRouteManager

# utils
from gym_carla.util_development.util_junction import get_junction_by_location


TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())

# default junction center in Town03
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

# parameters for scenario
start_location = carla.Location(x=53.0, y=128.0, z=1.5)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)

DEBUG_TRAFFICFLOW_MANAGER = False


class CarlaEnvMultiTask(gym.Env):
    """
    This method is responsible to generate a multi-task carla env.
    """

    # todo check if there are any better way to append all carla modules
    # carla modules
    state_manager_cls = StateManager6
    traffic_flow_manager_cls = TrafficFlowManagerMultiTask
    traffic_lights_manager_cls = TrafficLightsManager
    route_manager_cls = JunctionRouteManager

    # todo add API to set by args
    simulator_timestep_length = 0.05
    # max episode time in seconds
    max_episode_time = 60  # default episode length in seconds
    # max target speed of ego vehicle
    ego_max_speed = 15  # in m/s

    traffic_clear_period = int(100)
    reload_world_period = int(100)

    # available route options
    route_options = [
        'left',
        'right',
        'straight',  # refers to straight_0

        # todo fix route option
        # straight_0
        # straight_1 is deprecated in this env
    ]

    # all available routes and spawn points
    route_info = {
        'left': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'right': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'straight': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'straight_0': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
        'straight_1': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
    }

    def __init__(
            self,
            carla_port=2000,
            tm_port=int(8100),
            tm_seed=int(0),  # seed for autopilot controlled NPC vehicles

            train=True,  # training mode or evaluation mode
            task_option='left',  # all available route options stored in the class attributes
            attention=False,

            initial_speed=1,  # if set an initial speed to ego vehicle
            state_noise=False,  # if adding noise on state vector

            no_render_mode=False,
            debug=False,
            verbose=False,
    ):

        # todo add doc to denote the settings of the training mode
        self.carla_port = carla_port
        self.tm_port = tm_port  # port number of the traffic manager
        self.tm_seed = tm_seed  # seed value for the traffic manager

        self.train = train  # if using training mode

        # task option and route option
        self.task_option = task_option
        if self.task_option == 'multi_task':
            self.multi_task = True
            self.route_option = 'left'  # current route in str, one of the route_options
        else:
            self.multi_task = False
            self.route_option = self.task_option

        self.task_code = None

        # params
        self.attention = attention  # if using attention mechanism

        self.initial_speed = initial_speed  # initial speed of ego vehicle when reset
        self.state_noise = state_noise  # noise on state vector

        self.no_render_mode = no_render_mode
        self.debug = debug  # debug mode
        self.verbose = verbose  # visualization

        # # todo fix this
        # if self.train:
        #     frame_skipping_factor = int(2)
        # else:
        #     frame_skipping_factor = int(1)

        """
        use the frame skipping in training mode

        frame_skipping_factor is denoted as: f = N / n = t / T = F/ f
        in which:
             - N, T, F refers to (simulation) system params
             - n, t, f refers to RL module params
        """
        self.frame_skipping_factor = int(2)
        # RL timestep length
        self.rl_timestep_length = self.frame_skipping_factor * self.simulator_timestep_length
        # max episode timestep number for RL
        self.max_episode_timestep = int(self.max_episode_time / self.rl_timestep_length)

        # todo add args to set
        #  - map for different scenarios
        #  - client timeout
        self.carla_env = BasicEnv(
            town='Town03',
            host='localhost',
            port=carla_port,
            client_timeout=20.0,
            timestep=self.simulator_timestep_length,
            tm_port=tm_port,
            sync_mode=True,
            no_render_mode=self.no_render_mode,
        )

        # get carla API
        self.carla_api = self.carla_env.get_env_api()
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        # set the spectator on the top of junction
        self.carla_env.set_spectator_overhead(junction_center, yaw=270, h=70)

        # todo merge waypoints buffer into local planner
        # ==================================================
        # ----------   waypoint buffer begin  ----------
        # a queue to store original route
        self._waypoints_queue = deque(maxlen=100000)  # maximum waypoints to store in current route
        # buffer waypoints from queue, get nearby waypoint
        self._buffer_size = 50
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # todo near waypoints is waypoint popped out from buffer
        # self.near_waypoint_queue = deque(maxlen=50)
        # ----------   waypoint buffer end  ----------
        # ==================================================

        # ================   carla modules  ================
        #

        # todo develop method to retrieve junction from general map
        self.junction = get_junction_by_location(
            carla_map=self.map,
            location=junction_center,
        )

        # ================   route manager   ================
        self.route_manager = self.route_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,
            route_distance=(3., 3.),
        )

        # generate all routes for each route option
        # route format: <list>.<tuple>.(transform, RoadOption)
        for route in ['left', 'right', 'straight', 'straight_0']:
            ego_route, spawn_point, end_point = self.route_manager.get_route(route_option=route)

            self.route_info[route]['ego_route'] = ego_route
            self.route_info[route]['spawn_point'] = spawn_point
            self.route_info[route]['end_point'] = end_point

        self.ego_route = []  # list of route waypoints
        self.spawn_point = None
        self.end_point = None

        # attributes for ego routes management
        self.route_seq = []
        self.route_seq_index = None
        # set a route sequence for different task setting
        self.set_route_sequence()

        # ================   traffic flow manager   ================
        # todo use 2 types of traffic flow, CARLA Autopilot and AEB
        self.traffic_flow_manager = self.traffic_flow_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,

            tm_port=self.tm_port,
            tm_seed=self.tm_seed,

            # tls_red_phase_duration=20.,  # duration time of red phase of traffic lights
            phase_time=self.traffic_lights_manager_cls.phase_time,

            # debug=False,
            debug=DEBUG_TRAFFICFLOW_MANAGER,
            verbose=False,
        )

        # todo improve method for init task and route

        # update route info of ego vehicle
        self.update_route_info()

        # todo add params decay through training procedure
        #  - use some open-source codes?
        # collision detect decay
        param = {
            'initial_value': 0.5,
            'target_value': 1.,
            'episode_number': int(2000),
            'scheduler': 'linear',
        }

        # todo fix the API of traffic flow params decay
        # self.collision_prob_decay = collision_prob_decay
        # if not self.collision_prob_decay:
        #     collision_prob = 1.  # todo add arg to set this value
        #     self.traffic_flow.set_collision_probability(collision_prob)
        #
        # self.tf_params_decay = tf_params_decay

        # ================   traffic_light_manager   ================
        # todo add API to set duration time of traffic lights
        self.traffic_light_manager = self.traffic_lights_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,
            # todo check this arg and training setting
            use_tls_control=True,
        )

        # ================ reward tuning ================
        # CAUTION: the key element must agree
        # todo add a check
        self.reward_dict = {
            'collision': -350.,
            'time_exceed': -100.,
            'success': 150.,
            'step': -0.3,
        }

        # ================   state manager   ================
        # todo fix multi-task in state manager 6
        multi_task = True if self.route_option == 'multi_task' else False

        self.state_manager = self.state_manager_cls(
            carla_api=self.carla_api,
            attention=self.attention,
            multi_task=multi_task,
            noise=self.state_noise,
            debug=self.debug,
            verbose=self.verbose,
        )

        # set junction for state manager
        self.state_manager.set_junction(self.junction)

        # ================   about vehicles   ================
        self.ego_vehicle = None
        self.ego_id = None  # id is set by the carla, cannot be changed by user

        self.ego_location = None
        self.ego_transform = None

        self.ego_collision_sensor = None  # collision sensor for ego vehicle
        self.collision_flag = False  # flag to identify a collision with ego vehicle
        self.controller = None  # ego vehicle controller

        self.npc_vehicles = []  # active NPC vehicles
        self.actor_list = []  # list contains all carla actors

        # ================   episodic info   ================
        # state of current timestep
        self.state_array = None  # state array of current timestep
        self.step_reward = None  # reward of current timestep
        self.episode_reward = 0  # accumulative reward of this episode
        self.action_array = None  # action of current timestep
        self.running_state = None  # running state of current timestep, [running, success, collision, time_exceed]

        self.frame_id = None
        self.elapsed_episode_number = int(0)  # a counter for episodes
        self.elapsed_timestep = int(0)  # elapsed timestep number of current episode
        self.elapsed_time = 0.  # elapsed time number of current episode

        self.episode_step_number = int(0)  # self.elapsed_timestep
        self.episode_time = 0.  # self.elapsed_time

        # help with episode time counting
        self.start_frame, self.end_frame = None, None
        self.start_elapsed_seconds, self.end_elapsed_seconds = None, None

        # todo print additional info, port number, and assign an ID for env
        print('A gym-carla env is initialized.')

    def set_task_option(self, task_option):
        """
        A setter method.

        Set a new task option for env, update related modules immediately.
        """
        # task option and route option
        self.task_option = task_option
        if self.task_option == 'multi_task':
            self.multi_task = True
            self.route_option = 'left'  # current route in str, one of the route_options
        else:
            self.multi_task = False
            self.route_option = self.task_option

        # update modules
        self.update_route_info()

        # todo add arg to determine if clear the traffic
        # reset traffic flow to initial state
        self.clear_traffic_flow()

        self.reset()

    def reset_episode_count(self):
        """
        Reset class attribute elapsed_episode_number to 0.
        :return:
        """
        self.elapsed_episode_number = int(0)

    def set_ego_route(self):
        """
        This method is called in reset.

        Set ego route for agent in reset method, before each episode starts.

        This method must be called before spawning ego vehicle.
        """
        # retrieve route option for current episode
        self.route_option = self.route_seq[self.route_seq_index]
        # update index for next route
        if self.route_seq_index == (len(self.route_seq) - 1):
            self.route_seq_index = 0
        else:
            self.route_seq_index += 1

        # update route info for next route
        self.update_route_info()

        # todo check if need to
        # # update current route option to state manager
        # self.state_manager.set_ego_route(self.ego_route, self.spawn_point, self.end_point)

    def set_route_sequence(self):
        """
        todo add params to set seq length for multi-task training

        This method is supposed to be called in init.

        Generate a sequence of route options for ego vehicle.

        Experiments settings:
         - for multi-task training, we will deploy a sequence consists 10*3 route.
         - for single-task training, we will run the specified route repeatedly
        """
        # multi task condition
        if self.task_option == 'multi_task':
            # todo test different seq length
            #  add arg to set value
            # seq_len = int(10)
            # left_seq = ['left'] * seq_len
            # right_seq = ['right'] * seq_len
            # straight_seq = ['straight'] * seq_len

            seq_len_list = [int(5), int(3), int(2)]
            left_seq = ['left'] * seq_len_list[0]
            right_seq = ['right'] * seq_len_list[1]
            straight_seq = ['straight'] * seq_len_list[2]

            self.route_seq = left_seq + right_seq + straight_seq
        else:  # single task condition
            self.route_seq = [self.route_option]

        # reset route sequence index
        self.route_seq_index = int(0)

    def update_route_info(self):
        """
        Update ego route of current episode.
        """
        # retrieve route from class attribute route info
        self.ego_route = self.route_info[self.route_option]['ego_route']
        self.spawn_point = self.route_info[self.route_option]['spawn_point']
        self.end_point = self.route_info[self.route_option]['end_point']

    def get_obs(self):
        """
        Get observation of current timestep through the state manager.

        In current version, the task code is append in this class method

        todo merge task code into state manager
        """
        state = self.state_manager.get_state()
        # in list
        state = list(state)

        # append task code to the state vector
        if self.route_option == 'left':
            task_code = [1, 0, 0, 1]
        elif self.route_option == 'right':
            task_code = [0, 0, 1, 1]
        elif self.route_option == 'straight':
            task_code = [0, 1, 0, 1]
        else:
            raise ValueError('Route option is not correct.')

        self.task_code = task_code

        # todo check if state is a list
        if self.multi_task:
            state += task_code

        # update state vector
        self.state_array = np.array(state)

        return self.state_array

    def spawn_ego_vehicle(self):
        """
        Spawn ego vehicle.
        """
        # ego vehicle blueprint
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz2017'))
        # set color
        if bp.has_attribute('color'):
            color = '255, 0, 0'  # use string to identify a RGB color
            bp.set_attribute('color', color)

        # attributes of a blueprint
        # print('-'*10, 'blueprint attributes', '-'*10)
        # print(bp)
        # for attr in bp:
        #     print('  - {}'.format(attr))

        # set role name
        # hero is the very specified name to activate physic mode
        bp.set_attribute('role_name', 'hero')

        # attributes of a blueprint is stored as dict
        # ego_attri = self.ego_vehicle.attributes

        # set sticky control
        """
        carla Doc:
        when “sticky_control” is “False”, 
        the control will be reset every frame to 
        its default (throttle=0, steer=0, brake=0) values.
        """
        bp.set_attribute('sticky_control', 'True')

        if self.ego_route:
            start_trans = self.spawn_point.transform
            # spawn transform need to be checked, z value must larger than 0
            start_trans.location.z = 0.2
            try:
                self.ego_vehicle = self.world.spawn_actor(bp, start_trans)

                # time.sleep(0.1)
                self.try_tick_carla()

                # update ego vehicle id
                self.ego_id = self.ego_vehicle.id

                # print('Ego vehicle is spawned.')
            except:
                raise Exception("Fail to spawn ego vehicle!")
        else:
            raise Exception("Ego route is not assigned!")

        # controller must be assigned to vehicle
        self.controller = Controller(
            vehicle=self.ego_vehicle,
            dt=self.rl_timestep_length,
        )

        # add collision sensors
        self.ego_collision_sensor = Sensors(self.world, self.ego_vehicle)
        self.try_tick_carla()

        print('Ego vehicle is ready.')

    def get_min_distance(self):
        """
        Get a minimum distance for waypoint buffer
        """
        if self.ego_vehicle:

            # fixme use current ego speed
            # speed = get_speed(self.ego_vehicle)
            #
            # target_speed = 4.0  # m/s
            # ref_speed = max(speed, target_speed)

            # reference speed, in m/s
            ref_speed = 3

            # min distance threthold of waypoint reaching
            MIN_DISTANCE_PERCENTAGE = 0.75
            sampling_radius = ref_speed * 1.  # maximum distance vehicle move in 1 seconds
            min_distance = sampling_radius * MIN_DISTANCE_PERCENTAGE

            return min_distance
        else:
            raise

    def clear_traffic_flow(self):
        """
        Clear all NPC vehicles and their collision sensors through traffic flow manager.
        """
        if self.traffic_flow_manager:
            self.traffic_flow_manager.clean_up()
            print('Traffic flow is cleared.')

    def init_traffic_flow(self):
        """
        Reset traffic flow and check if traffic flow is ready for start a new episode.
        """
        print('start traffic flow initializing...')

        # # clear traffic periodically
        # if (self.elapsed_episode_number + 1) % self.traffic_clear_period == 0:
        #     self.clear_traffic_flow()

        # todo fix reload carla world, problem is that all modules need to be reset.
        # if (self.elapsed_episode_number + 1) % self.reload_world_period == 0:
        #     pass

        # minimum npc vehicle number to start
        min_npc_number = 5 if self.multi_task else 5

        # =====  traffic lights state  =====
        # get target traffic lights phase according to route_option
        # phase = [0, 2] refers to x and y direction Green phase respectively
        if self.route_option in ['straight', 'left']:
            target_phase = [2, 3]  # the phase index is same as traffic light module
        else:  # right
            target_phase = [0, 1, 2, 3]  # [0, 1, 2, 3]
            # todo in training phase, make it more difficult
            # if self.training:
            #     target_phase = [0, 1, 2, 3]
            # else:
            #     target_phase = [0, 1]

        # todo add condition check for training and evaluation
        # if self.train_phase:
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]
        # else:
        #     # conditions = [vehNum_cond, tls_cond]  # huawei setting for evaluation
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]

        # ==============   init conditions   ==============
        # conditions for init traffic flow finishes
        # vehicle number condition
        self.update_vehicles()
        vehicle_numbers = len(self.npc_vehicles)
        vehNum_cond = vehicle_numbers >= min_npc_number

        # traffic light state condition
        # todo in this version, the traffic flow lights switch is deprecated
        # if self.use_tls_control:
        #     current_phase_index = self.tls_manager.get_phase_index()  # index, int
        #     # if current tls phase coordinates with target phase
        #     tls_cond = current_phase_index in target_phase
        # else:
        #     tls_cond = True

        current_phase_index = self.traffic_light_manager.get_phase_index()  # index, int
        # if current tls phase coordinates with target phase
        tls_cond = current_phase_index in target_phase

        # remain time condition, additional to tls cond
        # remain_time_cond = False

        # conditions in a list
        conditions = [vehNum_cond, tls_cond]

        while not all(conditions):
            # tick the simulator before check conditions
            self.tick_simulation()

            # ==============  vehicle number condition  ==============
            # self.update_vehicles()
            vehicle_numbers = len(self.npc_vehicles)
            vehNum_cond = vehicle_numbers >= min_npc_number

            # if self.debug:
            #     if vehicle_numbers >= min_npc_number:
            #         print('npc vehicle number: ', vehicle_numbers)

            # ==============  traffic light state condition  ==============
            # get tls phase from tls module
            current_phase_index = self.traffic_light_manager.get_phase_index()  # index, int
            # if current tls phase coordinates with target phase
            tls_cond = current_phase_index in target_phase

            # # debug tls cond
            # if not tls_cond:
            #     print('')

            # if self.debug:
            #     if tls_cond:
            #         print('')

            # todo add condition on the remain time of current tls phase
            # # remain time is considered in training phase
            # if tls_cond and vehNum_cond:
            #     remain_time = traci.trafficlight.getNextSwitch('99810003') - traci.simulation.getTime()
            #     remain_time_cond = remain_time >= 36.5
            #
            #     remain_time_cond = remain_time >= 36.9  # original
            #     remain_time_cond = remain_time <= 0.1  # debug waiting before junction
            #
            #     if using interval, interval module is required
            #     zoom = Interval(25., 37.)
            #     remain_time_cond = remain_time in zoom

            # ==============  append all conditions  ==============
            conditions = [vehNum_cond, tls_cond]

        print('traffic flow is ready.')

    def parameters_decay(self):
        """
        todo fix and test this method
        todo use a config file to set the parameters and their range
        todo add args and APIs to tune params' range of traffic flow

        :return:
        """

        # # ==========   set collision probability   ==========
        # """
        # TrafficFlowManager5 and TrafficFlowManager5Fixed has such API.
        #
        # TrafficFlowManager5Fixed is a developing version.
        #
        # Instruction:
        # There are several mode of traffic flow settings:
        #
        #  - fixed traffic flow param
        #  - whether use stochastic process to generate traffic params
        #
        # """
        # if self.traffic_flow.__class__.__name__ in ['TrafficFlowManager5']:
        #     if self.collision_prob_decay:
        #         collision_prob_range = [0.75, 0.95]
        #         # todo add args to set decay range
        #         collision_decay_length = int(1000)  # episode number for collision prob increasing
        #         # default episode number is 2000
        #         collision_prob = collision_prob_range[0] + \
        #                          self.episode_number * (
        #                                      collision_prob_range[1] - collision_prob_range[0]) / collision_decay_length
        #         collision_prob = np.clip(collision_prob, collision_prob_range[0], collision_prob_range[1])
        #         self.traffic_flow.set_collision_probability(collision_prob)
        #
        #     # traffic flow params decay
        #     if self.tf_params_decay:
        #         # this setting is available when param noise is enabled
        #         if self.tf_randomize:
        #             # decay length, episode number, 5000 is current minimum training length
        #             # tf_param_decay_length = int(5000)
        #
        #             # for debug
        #             tf_param_decay_length = int(10)
        #
        #             # todo store params range through external data storage
        #             #  or retrieve traffic flow params form the class
        #             # for now the target params are set manually here
        #             final_speed_range = (25, 40)
        #             final_distance_range = (10, 25)
        #
        #             # retrieve original traffic flow params
        #             for tf in self.traffic_flow.active_tf_directions:
        #                 info_dict = self.traffic_flow.traffic_flow_info[tf]
        #                 # original range
        #                 speed_range = info_dict['target_speed_range']
        #                 distance_range = info_dict['distance_range']
        #
        #                 # new range
        #                 current_speed_range = (
        #                     self.linear_mapping(speed_range[0], final_speed_range[0], tf_param_decay_length,
        #                                         self.episode_number),
        #                     self.linear_mapping(speed_range[1], final_speed_range[1], tf_param_decay_length,
        #                                         self.episode_number),
        #                 )
        #                 current_distance_range = (
        #                     self.linear_mapping(distance_range[0], final_distance_range[0], tf_param_decay_length,
        #                                         self.episode_number),
        #                     self.linear_mapping(distance_range[1], final_distance_range[1], tf_param_decay_length,
        #                                         self.episode_number),
        #                 )
        #
        #                 self.traffic_flow.traffic_flow_info[tf]['target_speed_range'] = current_speed_range
        #                 self.traffic_flow.traffic_flow_info[tf]['distance_range'] = current_distance_range

        # todo add setter methods to set the params
        if self.train:
            collision_prob_range = [0.5, 0.95]
            # episode number for collision prob increasing
            collision_decay_length = int(10000)

            # calculate the param by linear decay
            if self.elapsed_episode_number <= collision_decay_length:
                collision_prob = \
                    collision_prob_range[0] + \
                    (self.elapsed_episode_number - 1) * (collision_prob_range[1] - collision_prob_range[0]) / collision_decay_length
                collision_prob = np.clip(collision_prob, collision_prob_range[0], collision_prob_range[1])

                self.traffic_flow_manager.set_collision_detection_rate(collision_prob)
            else:
                self.traffic_flow_manager.set_collision_detection_rate(collision_prob_range[-1])

            # todo add the traffic flow speed and gap distance/time decay


    def reset(self):
        """
        Reset ego vehicle.
        :return:
        """
        # check and destroy ego vehicle and its sensors
        if self.ego_vehicle:
            self.destroy()
            # make destroy ego vehicle take effect
            self.tick_simulation()

        # clear buffered waypoints
        self._waypoints_queue.clear()
        self._waypoint_buffer.clear()

        # reset ego route
        self.set_ego_route()

        # prepare the traffic flow
        self.init_traffic_flow()

        # spawn ego only after traffic flow is ready
        self.spawn_ego_vehicle()

        # set initial speed
        if self.initial_speed:
            self.set_velocity(self.ego_vehicle, self.initial_speed)

        # todo move to spawn vehicle
        # buffer waypoints from ego route
        for elem in self.ego_route:
            self._waypoints_queue.append(elem)

        # get state vector from manager
        state = self.get_obs()

        # ==========  reset episodic counters  ==========
        # accumulative reward of single episode
        self.episode_reward = 0

        # reset step number at the end of the reset method
        self.elapsed_timestep = 0
        self.elapsed_episode_number += 1  # update episode number

        # this API is still working for some modules
        self.episode_step_number = self.elapsed_timestep

        # update start time
        start_frame, start_elapsed_seconds = self.get_carla_time()
        self.start_frame, self.start_elapsed_seconds = start_frame, start_elapsed_seconds

        # todo add callback API for methods called at the beginning/ending of the episode/timestep
        # update params decay of modules
        self.parameters_decay()

        print('CARLA Env is reset.')

        return state

    def buffer_waypoint(self):
        """
        Buffering waypoints for planner.

        This method should be called each timestep.

        _waypoint_buffer: buffer some wapoints for local trajectory planning
        _waypoints_queue: all waypoints of current route

        :return: 2 nearest waypoint from route list
        """

        # add waypoints into buffer
        least_buffer_num = 10
        if len(self._waypoint_buffer) <= least_buffer_num:
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:  # when there is not enough waypoint in the waypoint queue
                    break

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, tuple in enumerate(self._waypoint_buffer):

            transform = tuple[0]
            # get transform and location
            self.ego_transform = self.ego_vehicle.get_transform()
            self.ego_location = self.ego_transform.location

            # current location of ego vehicle
            # self.ego_location = self.ego_vehicle.get_location()

            # todo check if z coord value effect the dist calculation
            _dist = self.ego_location.distance(transform.location)
            _min_dist = self.get_min_distance()

            # if no.i waypoint is in the radius
            if _dist < _min_dist:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):  # (max_index+1) waypoints to pop out of buffer
                self._waypoint_buffer.popleft()

    def map_action(self, action):
        """
        Map agent action to the vehicle control command.

        todo map action to [-1, 1], by setting decel_ratio = 1.
        """
        decel_ratio = 0.25
        # map action to vehicle target speed
        target_speed = action[0] - decel_ratio * action[1]
        target_speed = np.clip(target_speed, 0., 1.)
        target_speed = 3.6 * self.ego_max_speed * target_speed  # in km/h

        return target_speed

    def tick_controller(self, action):
        """
        Tick controller of ego vehicle.
        """

        self.buffer_waypoint()

        # todo check sticky control usage in sync mode
        if self._waypoint_buffer:
            # constant speed for debug
            constant_speed = False
            if constant_speed:
                target_speed = self.ego_max_speed * 3.6  # m/s
            else:  # conduct control action by algorithm
                target_speed = self.map_action(action)

            # map target speed to VehicleControl for a carla vehicle
            veh_control = self.controller.generate_control(target_speed=target_speed,
                                                           next_waypoint=self._waypoint_buffer[0])
        else:
            target_speed = 0.
            # hold brake
            veh_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

            # todo test this, buffer last waypoint to avoid buffer empty
            # veh_control = self.controller.generate_control(target_speed=30,
            #                                                next_waypoint=self.end_point)  # carla.Waypoint

            print('route is finished, please terminate the simulation manually.')

        if self.debug:
            # print('='*30)
            print('-' * 8, 'timestep ', self.episode_step_number, '-' * 8)

            print('action: ', action)
            print('target speed: ', target_speed)

            print('Ego vehicle control: ')
            print('throttle: ', veh_control.throttle)
            print('steer: ', veh_control.steer)
            print('brake: ', veh_control.brake)

        self.ego_vehicle.apply_control(veh_control)
        # print('controller ticked.')

    def step(self, action: tuple):
        """
        Run a step for RL training.

        This method will:

         - tick simulation multiple times according to frame_skipping_factor
         - tick carla world and all carla modules
         - apply control on ego vehicle
         - compute rewards and check if episode ends
         - update information

        :param action:
        :return:
        """
        # update action array to class attribute
        self.action_array = np.array([
            action[0],
            action[1],
        ])

        # # ==================================================
        # # if using a pid controller
        # # ==================================================
        # self.buffer_waypoint()
        #
        # # todo check sticky control usage in sync mode
        # if self._waypoint_buffer:
        #     # constant speed for debug
        #     constant_speed = False
        #     if constant_speed:
        #         target_speed = self.ego_max_speed * 3.6  # m/s
        #     else:  # conduct control action by algorithm
        #         target_speed = self.map_action(action)
        #
        #     # map target speed to VehicleControl for a carla vehicle
        #     veh_control = self.controller.generate_control(target_speed=target_speed,
        #                                                    next_waypoint=self._waypoint_buffer[0])
        # else:
        #     target_speed = 0.
        #     # hold brake
        #     veh_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        #
        #     # todo test this, buffer last waypoint to avoid buffer empty
        #     # veh_control = self.controller.generate_control(target_speed=30,
        #     #                                                next_waypoint=self.end_point)  # carla.Waypoint
        #
        #     print('route is finished, please terminate the simulation manually.')
        #
        # if self.debug:
        #     # print('='*30)
        #     print('-' * 8, 'timestep ', self.elapsed_timestep, '-' * 8)
        #
        #     print('action: ', action)
        #     print('target speed: ', target_speed)
        #
        #     print('Ego vehicle control: ')
        #     print('throttle: ', veh_control.throttle)
        #     print('steer: ', veh_control.steer)
        #     print('brake: ', veh_control.brake)
        #
        # self.ego_vehicle.apply_control(veh_control)

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # original lines
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        # # ================   Tick simulation   ================
        # # todo ref carla co-sim method with time.time() to sync with real world time
        # # execute multiple timestep according to frame_skipping_factor
        # for i in range(self.frame_skipping_factor):
        #     # tick ego vehicle controller
        #     self.tick_controller(action)
        #     # other modules
        #     self.tick_simulation()
        # # =====================================================
        #
        # # frame, elapsed_seconds = self.get_frame()
        # # print('frame: ', frame)
        # # print('elapsed_seconds: ', elapsed_seconds)
        #
        # # todo add a getter method
        # # update collision flag from collision sensor
        # self.collision_flag = self.ego_collision_sensor.collision_flag
        #
        # # test state manager module
        # state = self.get_obs()
        #
        # # todo get kinetic information from state manager
        # # if self.debug:
        # #     ego_acceleration = self.ego_vehicle.get_acceleration()
        # #     ego_velocity = self.ego_vehicle.get_velocity()
        # #
        # #     accel_norm = math.sqrt(ego_acceleration.x ** 2 + ego_acceleration.y ** 2 + ego_acceleration.z ** 2)
        # #     speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
        # #
        # #     # print('ego acceleration: [', ego_acceleration.x, ego_acceleration.y, ego_acceleration.z, ']')
        # #     print('acceleration norm: ', accel_norm)
        # #     # print('ego velocity: [', ego_velocity.x, ego_velocity.y, ego_velocity.z, ']')
        # #     print('speed: ', speed)
        #
        # # get reward of current step
        # # done print is in this method
        # reward, done, info = self.compute_reward()
        # aux = {'exp_state': info}
        #
        # # state, reward, done, aux_info = None, None, None, None
        #
        # # update the step number
        # if not done:
        #     self.elapsed_timestep += 1
        #     self.episode_step_number = self.elapsed_timestep

        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # new lines
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        # ================   Tick simulation   ================
        # todo ref carla co-sim method with time.time() to sync with real world time
        # execute multiple timestep according to frame_skipping_factor
        for i in range(self.frame_skipping_factor):
            # tick ego vehicle controller
            self.tick_controller(action)

            # ================ tick the all functional modules ================
            # # original lines
            # # other modules
            # self.tick_simulation()

            # ----------------  NOTICE: merge collision check into tick_simulation method  ----------------
            # spawn new NPC vehicles
            self.traffic_flow_manager.run_step_1()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----')
            # print('after traffic_flow_manager.run_step_1()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)

            # tick carla world
            self.try_tick_carla()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----')
            # print('carla tick')
            # print('before traffic_flow_manager.run_step_2()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)

            # =====  get state from state manager module  =====
            state = self.get_obs()

            # get reward of current step
            # done print is within this method
            reward, done, info = self.compute_reward()
            aux = {'exp_state': info}

            # if done:
            #     print('')

            # register vehicles to traffic manager
            # delete collision vehicles
            self.traffic_flow_manager.run_step_2()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----')
            # print('after traffic_flow_manager.run_step_2()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # frame, elapsed_seconds = self.get_frame()
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)

            # traffic light module
            self.traffic_light_manager.run_step()

            # frame, elapsed_seconds = self.get_frame()
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)

            # update active npc vehicles
            self.update_vehicles()

            # frame, elapsed_seconds = self.get_frame()
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)

            # end the frame skipping if violation happens
            if done:
                break

        # update the step number
        self.elapsed_timestep += 1
        self.episode_step_number = self.elapsed_timestep

        # update episodic reward
        self.episode_reward += reward

        if done:
            # result in str
            episode_result = info
            print(
                '\n',
                '=' * 10, 'env internal counts', '=' * 10, '\n',
                'episode result: {}'.format(episode_result), '\n',
                'episode number: {}'.format(self.elapsed_episode_number), '\n',
                'episode step number: {}'.format(self.elapsed_timestep), '\n',
                'episode reward: {:.2f}'.format(self.episode_reward), '\n',
                "Current time: {0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now()), '\n',
                '=' * 41, '\n',
            )

            # elapsed episode frame number
            end_frame, end_elapsed_seconds = self.get_carla_time()
            self.end_frame, self.end_elapsed_seconds = end_frame, end_elapsed_seconds
            episode_frame = self.end_frame - self.start_frame

            # duration time of this episode
            episode_elapsed_seconds = self.end_elapsed_seconds - self.start_elapsed_seconds
            self.episode_time = episode_elapsed_seconds

        return state, reward, done, aux

    def try_tick_carla(self):
        """
        todo fix this method as a static method for all carla modules

        Tick the carla world with try exception method.

        In fixing fail to tick the world in carla
        """

        max_try = int(100)
        tick_success = False
        tick_counter = int(0)
        while not tick_success:
            if tick_counter >= max_try:
                raise RuntimeError('Fail to tick carla for ', max_try, ' times...')
            try:
                # if this step success, a frame id will return
                frame_id = self.world.tick(20.)
                if frame_id:
                    self.frame_id = frame_id
                    tick_success = True
            except:  # for whatever the error is..
                print('*-' * 20)
                print('Fail to tick the world for once...')
                print('Last frame id is ', self.frame_id)
                print('*-' * 20)
                tick_counter += 1

        if tick_counter > 0:
            print('carla client is successfully ticked after ', tick_counter, 'times')

    def get_frame(self):
        """
        Get frame id from world.

        :return: frame_id
        """
        snapshot = self.world.get_snapshot()
        frame = snapshot.timestamp.frame
        elapsed_seconds = snapshot.timestamp.elapsed_seconds

        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        return frame, elapsed_seconds

    def tick_simulation(self):
        """
        todo we don't use this method self.step() method, since the error in NPC collision check.
        todo manage modules with a manager, and tick all registered modules using loop

        Tick the simulation.

        The following modules will be ticked.

        :return:
        """

        # todo fix if controller and simulation has different timestep
        # ===================  co-simulation code  ===================
        # for _ in range(int(self.control_dt/self.server_dt)):
        #     # world tick
        #     start = time.time()
        #     self.world.tick()
        #     end = time.time()
        #     elapsed = end - start
        #     if elapsed < self.server_dt:
        #         time.sleep(self.server_dt - elapsed)

        # todo render in local map
        # ===================   render local map in pygame   ===================
        #
        # vehicle_poly_dict = self.localmap.get_actor_polygons(filter='vehicle.*')
        # self.all_polygons.append(vehicle_poly_dict)
        # while len(self.all_polygons) > 2: # because two types(vehicle & walker) of polygons are needed
        #     self.all_polygons.pop(0)
        # # pygame render
        # self.localmap.display_localmap(self.all_polygons)

        # frame, elapsed_seconds = self.get_frame()
        # print('-----'*5)
        # print('-----')
        # print('before traffic_flow_manager.run_step_1()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # # fix traffic flow manager run_step_1
        # try:
        #     self.traffic_flow_manager.run_step_1()
        # except:
        #     raise RuntimeError('run_step_1')

        self.traffic_flow_manager.run_step_1()

        # frame, elapsed_seconds = self.get_frame()
        # print('-----')
        # print('after traffic_flow_manager.run_step_1()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # tick carla server
        self.try_tick_carla()

        # frame, elapsed_seconds = self.get_frame()
        # print('-----')
        # print('carla tick')
        # print('before traffic_flow_manager.run_step_2()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # try:
        #     self.traffic_flow_manager.run_step_2()
        # except:
        #     raise RuntimeError('run_step_2')

        self.traffic_flow_manager.run_step_2()

        # frame, elapsed_seconds = self.get_frame()
        # print('-----')
        # print('after traffic_flow_manager.run_step_2()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)
        # print('-----'*5)

        # todo register all carla modules, add a standard api
        # ================   tick carla modules   ================

        # # original line
        # # tick the traffic flow module
        # self.traffic_flow_manager.run_step()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # traffic light module
        self.traffic_light_manager.run_step()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # update active npc vehicles
        self.update_vehicles()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

    def update_vehicles(self):
        """
        Modified from BasicEnv.

        Only update NPC vehicles.
        """
        self.npc_vehicles = []

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a ActorList instance, iterable

        if vehicle_list:
            for veh in vehicle_list:
                attr = veh.attributes  # dict
                # filter ego vehicle by role name
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':
                    continue
                else:
                    self.npc_vehicles.append(veh)

    def set_reward_dict(self, reward_dict):
        """
        Set the reward dict through an external API.

        original reference:

        reward_dict = {
            'collision': -350.,
            'time_exceed': -100.,
            'success': 150.,
            'step': -0.3,
        }
        """
        self.reward_dict = reward_dict
        print('Reward setting is reset, current reward dict is: \n{}'.format(self.reward_dict))

    def compute_reward(self):
        """
        Compute reward value of current time step.
        Notice that the time step is specified for RL process.

        :return:
        """
        # todo check reward calculation in multi-task mode
        reward = 0.

        # # original training reward settings
        # reward_dict = {
        #     'collision': -350.,
        #     'time_exceed': -100.,
        #     'success': 150.,
        #     'step': -0.3,
        # }

        # use the class attribute to set the reward dict
        reward_dict = self.reward_dict

        # todo fix aux info for stable baselines
        # collision = False
        # time_exceed = False
        aux_info = 'running'

        done = False

        # todo add a getter method
        # update collision condition from collision sensor attribute collision flag
        self.collision_flag = self.ego_collision_sensor.collision_flag
        # check if collision happens by collision sensor of ego vehicle
        collision = self.collision_flag
        # check time exceed
        time_exceed = self.elapsed_timestep > self.max_episode_timestep
        # check if success
        # check if ego reach goal
        # check distance between ego vehicle and end_point
        dist_threshold = 5.0
        ego_loc = self.ego_vehicle.get_location()
        end_loc = self.end_point.transform.location
        dist_ego2end = ego_loc.distance(end_loc)
        # success indicator
        success = True if dist_ego2end <= dist_threshold else False

        episode_result = None
        # compute reward value based on above
        if collision:
            reward += reward_dict['collision']
            done = True
            aux_info = 'collision'
            # print('Failure: collision!')
            episode_result = 'Failure. Collision!'
        elif time_exceed:
            reward += reward_dict['time_exceed']
            done = True
            aux_info = 'time_exceed'
            # print('Failure: Time exceed!')
            episode_result = 'Failure. Time exceed!'
        elif success:
            done = True
            aux_info = 'success'
            reward += reward_dict['success']
            # print('Success: Ego vehicle reached goal.')
            episode_result = 'Success!'
        else:  # still running
            done = False
            aux_info = 'running'
            # calculate step reward according to elapsed time step
            if self.elapsed_timestep > 0.5 * self.max_episode_timestep:
                reward += 2 * reward_dict['step']
            else:
                reward += reward_dict['step']

        return reward, done, aux_info

    def destroy(self):
        """
        Destroy ego vehicle and its collision sensors

        In this method we use the client command to delete sensors
        """
        delete_list = []
        # delete ego vehicle actor
        if self.ego_vehicle:
            delete_list.append(self.ego_vehicle)

        # retrieve collision sensor from the Sensor API
        if self.ego_collision_sensor:
            for sensor in self.ego_collision_sensor.sensor_list:
                delete_list.append(sensor)

        # todo original version of delete_actors() is from carla_module
        self.traffic_flow_manager.delete_actors(delete_list)

        self.ego_vehicle = None
        self.ego_collision_sensor = None

        # print('Ego vehicle and its sensors are destroyed.')

    def get_carla_time(self):
        """
        Get carla simulation time.
        :return:
        """

        # reset carla
        snapshot = self.world.get_snapshot()
        frame = snapshot.timestamp.frame
        elapsed_seconds = snapshot.timestamp.elapsed_seconds

        return frame, elapsed_seconds

    @staticmethod
    def linear_mapping(original_value: float, target_value: float, total_step: int, current_step: int):
        """
        One-dimensional linear mapping.

        Shrinking of traffic flow params

        :return: current_range
        """
        delta_y = target_value - original_value
        k = delta_y / total_step
        b = original_value

        current_value = k * current_step + b

        return current_value

    @staticmethod
    def set_velocity(vehicle, target_speed: float):
        """
        Set a vehicle to the target velocity.

        params: target_speed: in m/s
        """

        transform = vehicle.get_transform()
        # transform matrix Actor2World
        trans_matrix = get_transform_matrix(transform)

        # target velocity in world coordinate system
        target_vel = np.array([[target_speed], [0.], [0.]])
        target_vel_world = np.dot(trans_matrix, target_vel)
        target_vel_world = np.squeeze(target_vel_world)

        # carla.Vector3D
        target_velocity = carla.Vector3D(
            x=target_vel_world[0],
            y=target_vel_world[1],
            z=target_vel_world[2],
        )
        #
        vehicle.set_target_velocity(target_velocity)

        # tick twice to reach target speed
        # for i in range(2):
        #     self.try_tick_carla()

    # ====================   getters   ====================

    def get_carla_world(self):

        return self.world

    def get_ego_vehicle(self):

        return self.ego_vehicle

    # ====================   Future Features   ====================

    def render_local_map(self):
        """
        todo render a local map using pygame
        """
        pass

    def reset_task_code(self):
        """
        This method is supposed to be called in reset().


        Update task route option of current episode

        todo if multi_task code is given by state manager, fix the state_manager API
        """

        pass

    def clear_world(self):
        """
        todo clear all alive actors at once
        """
        # actor_list = self.world.get_actors()
        # # clear traffic lights
        # if actor_list.filter('traffic.traffic_light'):
        #     # self.client.apply_batch([carla.command.DestroyActor(traffic_light) for traffic_light in actor_list.filter('traffic.traffic_light')])
        #     for traffic_light in actor_list.filter('traffic.traffic_light'):
        #         traffic_light.destroy()
        # # clear vehicles and walkers
        # if actor_list.filter('vehicle.*'):
        #     # self.client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in actor_list.filter('vehicle.*')])
        #     for vehicle in actor_list.filter('vehicle.*'):
        #         vehicle.destroy()
        # if actor_list.filter('walker.*'):
        #     # self.client.apply_batch([carla.command.DestroyActor(walker) for walker in actor_list.filter('walker.*')])
        #     for walker in actor_list.filter('walker.*'):
        #         walker.destroy()

        # todo delete actors by methods provided in carla_module
        # self.world.get_actors().filter('sensor.other.collision')
        # always lead to error, maybe caused by collision sensor state
        # is_listening state is False when using the carla.World.get_actors() or carla.World.get_actor()

        # todo test if vehicle is destroyed, will the sensor be deleted as well??

        # actor_id = [x.id for x in delete_actors]
        # # carla.Client.apply_batch_sync method
        # response_list = self.client.apply_batch_sync(
        #     [carla.command.DestroyActor(x) for x in delete_actors],
        #     False,
        # )
        #
        # # add a manual tick here
        # self.try_tick_carla()
        #
        # failures = []
        # for response in response_list:
        #     if response.has_error():
        #         failures.append(response)
        #
        # if self.debug:
        #     if not failures:
        #         print('Following actors are destroyed.')
        #         print(actor_id)
        #     else:
        #         print('Failure occurs when try to delete: ')
        #         print(failures)

        pass

    # ==========  safety layer methods  ==========
    def get_constraint_value(self):
        """"""

        ttc = self.state_manager.get_precise_ttc()

        return ttc

    def debug_safety_methods(self):
        """
        Methods for safety layer development.
        :return:
        """

        ttc = self.state_manager.get_precise_ttc()

        precise_ttc = self.state_manager.get_precise_ttc()

        print('')

    # =============== methods for get_rollout data ===============
    def get_single_frame_data(self):
        """
        todo route info

        Get data of a single frame data.
        """

        # get single frame data from state manager
        state_data_dict, full_data_dict = self.state_manager.get_single_frame_data()

        # append action data
        for dic in [state_data_dict, full_data_dict]:
            dic['action'] = self.action_array

        return state_data_dict, full_data_dict

    def get_episodic_data(self):
        """
        Call this method at the end of a episode, before the reset method is called.

        :return: total timestep and time duration of last episode
        """

        return self.elapsed_timestep, self.elapsed_time

