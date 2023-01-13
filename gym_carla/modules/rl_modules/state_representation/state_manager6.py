"""
This state manager is a complete version of state manager.

State representation is inherited form state_manager5,

other basic methods are directly from state_manager_2,

as well, methods for safety layer experiments are added to this version.

Major features:

 - State

 - add ttc calculation methods
 -

todo
 - fix module import in env

"""

import numpy as np
import math
import heapq

from gym.spaces import Box

import carla

# visualization module
from gym_carla.util_development.carla_color import *
from gym_carla.util_development.util_visualization import plot_actor_bbox, draw_2D_velocity, visualize_actor

# kinetics method
from gym_carla.util_development.kinetics import get_distance_along_route, get_inverse_transform_matrix, angle_reg

from gym_carla.modules.carla_module import CarlaModule


def draw_velocity():
    """
    todo merge this method into util_visualization

    Draw velocity arrow to debug.
    """
    pass


class StateManager6(CarlaModule):
    """
    We merge all developed methods into this version.
    """

    # inherit form sumo experiments, dimension details of state array
    state_config = {
        'ego_state_len': int(4),
        'npc_number': int(5),
        'npc_state_len': int(6),
    }

    ego_state_len = int(4)
    state_npc_number = int(5)
    npc_state_len = int(6)

    # todo add setter for state range
    # maximum distance between ego and npc vehicle
    range_bound = 100.

    # todo add setter from outer scope
    ego_max_speed = 15  # in m/s

    def __init__(
            self,
            carla_api,
            attention=False,
            multi_task=False,
            noise=True,  # todo add noise params, noise class and param value
            debug=False,
            verbose=True,
    ):

        super(StateManager6, self).__init__(carla_api=carla_api)

        # state setting args
        self.attention = attention
        self.multi_task = multi_task

        # todo add noise on kinetic info retrieving
        # using noise on ground truth value
        self.noise = noise

        # running mode
        self.debug = debug
        self.verbose = verbose

        # ====================   Init the ob space for gym.Env   ====================
        self.state_len = None
        # self.observation_space = self.init_ob_space()

        # ===============   attributes for RL   ===============
        self.ego_vehicle = None
        # a dict to store kinetics of ego vehicle
        self.ego_info = {}
        # transform matrix from ego coord system to world system
        self.T_world2ego = None
        self.T_ego2world = None

        # list of all npc vehicles(carla.Vehicle)
        self.npc_vehicles = []
        # NPC vehicles for state representation, dict: (carla.Vehicle, info_dict)
        self.state_npc_vehicles = {}

        # state of current timestep
        self.state_array = None  # np.array
        self.position_code = None

        # carla.Junction, if in a intersection scenario
        self.junction = None
        # the edge of junction bbox in coordinate
        self.junction_edges = {}

        # route and scenarios
        self.start_waypoint = None
        self.end_waypoint = None
        self.ego_route = None
        self.route_length = None

        # ttc related info stored in dict
        self.ttc_info_list = []

        print('State manager is initialized.')

    def set_junction(self, junction):
        """"""
        self.junction = junction
        # parse junction info
        self.get_junction_edges()

    def get_junction_edges(self):
        """
        Parse junction info.
        """
        bbox = self.junction.bounding_box
        extent = bbox.extent
        location = bbox.location

        # todo fix for junction whose bbox has a rotation
        # rotation = bbox.rotation  # original rotation of the bbox

        # get junction vertices
        x_max = location.x + extent.x
        x_min = location.x - extent.x
        y_max = location.y + extent.y
        y_min = location.y - extent.y

        junction_edges = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
        }

        self.junction_edges = junction_edges

    def init_ob_space(self):
        """
        This class method is responsible for initializing observation space.
        """

        # get state info
        ego_state_len = self.ego_state_len
        state_npc_number = self.state_npc_number  # npc vehicle number for state representation
        npc_state_len = self.npc_state_len  # state of single npc vehicle

        # state length of
        state_len = int(ego_state_len + state_npc_number * npc_state_len)

        ego_low = np.array([0, 0, 0, 0])
        ego_high = np.array([1, 1, 1, self.ego_max_speed])

        single_npc_low = [-1*self.range_bound, -1*self.range_bound, 0, 0, -1, -1]
        single_npc_high = [self.range_bound, self.range_bound, 2*self.ego_max_speed, 2*self.ego_max_speed, 1, 1]  # assuming that max npc speed is 2*max_ego_speed

        state_low = ego_low + single_npc_low * self.state_npc_number
        state_high = ego_high + single_npc_high * self.state_npc_number

        # append attention component
        if self.attention:
            state_len = int(state_len + 1 + state_npc_number)
            # musk value is 0 or 1
            attention_musk_low = [0, 0, 0, 0, 0]
            attention_musk_high = [1, 1, 1, 1, 1]

            state_low = state_low + attention_musk_low
            state_high = state_high + attention_musk_high

        # todo add methods of get task vector
        # append multi-task component
        if self.multi_task:
            state_len = int(state_len + 4)
            task_low = [0, 0, 0, 0]
            task_high = [1, 1, 1, 1]

            state_low = state_low + task_low
            state_high = state_high + task_high

        low = np.array(state_low)
        high = np.array(state_high)

        # total dimension of the total state array
        print('Total length of state array: {}'.format(state_len))

        observation_space = Box(high=high, low=low, dtype=np.float32)

        return observation_space

    def set_route_option(self, route_option):
        """"""

        self.route_option = route_option

    def set_ego_route(self, route, start_waypoint, end_waypoint):
        """
        A setter method.
        Set ego route from env.

        :param end_waypoint:
        :param start_waypoint:
        :param route: list of tuple (transform, RoadOption)
        """
        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint

        # short distance to avoid get_distance_along_route error
        _dist = 2.0
        pre_wp = start_waypoint.previous(_dist)[0]
        after_ap = end_waypoint.next(_dist)[0]

        # set and extend route
        road_option = route[0][1]
        self.ego_route = [(pre_wp.transform, road_option)] + route + [(after_ap.transform, road_option)]

        # get total length of the route
        end_location = self.ego_route[-1][0].location
        # if using route length
        self.route_length, _ = get_distance_along_route(self.map, self.ego_route, end_location)

    def get_ego_state(self):
        """
        Get ego vehicle state.
        """
        # update ego vehicle info
        self.ego_info = self.get_veh_info(self.ego_vehicle)
        # coordinate in global coordinate system
        location = self.ego_info['location']
        x, y = location.x, location.y
        # speed velocity in world coord system
        velocity = self.ego_info['velocity']
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # in m/s
        # use the 2d velocity
        # speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)  # in m/s
        # init state with ego speed
        state = [speed]

        if not self.junction:
            raise RuntimeError('Please assign junction for the state manager, using set_junction() method.')
        # edges of junction
        x_min = self.junction_edges['x_min']
        x_max = self.junction_edges['x_max']
        y_min = self.junction_edges['y_min']
        y_max = self.junction_edges['y_max']
        # position_code refer to which part of route ego is at
        before_junction = y >= y_max
        in_junction = x_min <= x <= x_max and y_min <= y <= y_max
        # get position code
        if in_junction:
            position_code = [0, 1, 0]
        else:
            if before_junction:
                position_code = [1, 0, 0]
            else:  # after leaving junction
                position_code = [0, 0, 1]

        position_code = list(map(float, position_code))
        # update the class attribute
        self.position_code = position_code
        state = position_code + state

        if self.debug:
            print('ego location(2D): ', x, y)
            print('ego speed(km/h): ', 3.6*speed)

            # plot ego vehicle
            plot_actor_bbox(self.ego_vehicle)

        return state

    def get_single_state(self, veh_info):
        """
        Get state of a single npc vehicle.

        todo consider npc vehicle rotation
        """
        # calculate relative velocity
        ego_velo = self.ego_info['velocity']
        ego_yaw = self.ego_info['rotation'].yaw
        veh_velo = veh_info['velocity']
        rel_velo = veh_velo - ego_velo
        # use ndarray
        _rel_velo = np.array([rel_velo.x, rel_velo.y])
        # 2D velocity in ego coord frame
        relative_velocity = np.dot(self.T_world2ego, _rel_velo)  # in m/s

        # get from veh_info dict
        relative_location = veh_info['relative_location']
        npc_state = np.concatenate((relative_location, relative_velocity), axis=0)
        npc_state = list(npc_state)  # in list

        # get relative yaw
        rotation = veh_info['rotation']
        yaw = rotation.yaw
        relative_yaw = yaw - ego_yaw
        relative_yaw = np.deg2rad(relative_yaw)
        heading_direction = [np.cos(relative_yaw), np.sin(relative_yaw)]

        npc_state += heading_direction

        return npc_state

    def get_ego_vehicle(self):
        """
        A Getter to get ego vehicle from instance.

        This method serves the test_agent for scenario test.
        """
        if self.ego_vehicle:
            return self.ego_vehicle
        else:
            raise RuntimeError('Ego vehicle not found!')

    def check_is_near(self, veh_info):
        """
        Method for filter, check if a vehicles is near ego.
        """
        # condition result
        flag = False

        location = veh_info['location']
        ego_location = self.ego_info['location']
        distance = ego_location.distance(location)  # in meters

        # add distance into veh_info dict
        veh_info['distance'] = distance

        # check if this vehicle is within the range bound
        if distance <= self.range_bound:
            flag = True

        return flag

    def check_is_front(self, veh_info):
        """
        Method for filter, check if a npc vehicle
        is in front of ego vehicle.

        This method will change veh_info dict.

        ps: Vehicle heading direction is fixed to x axis
        """
        flag = False

        # todo add API for this value
        # threshold of front discrimination
        dist_threshold = -5.0

        # ego information
        ego_location = self.ego_info['location']
        # ego_x, ego_y = ego_location.x, ego_location.y
        ego_rotation = self.ego_info['rotation']
        ego_yaw = np.radians(ego_rotation.yaw)

        # check if this npc is in front of ego vehicle
        location = veh_info['location']
        rotation = veh_info['rotation']
        yaw = np.radians(rotation.yaw)

        # relative location in global coordinate system
        relative_location = location - ego_location
        relative_loc = np.array([relative_location.x, relative_location.y])

        # todo check rads or degree
        # update transform matrix
        trans_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], [-1*np.sin(ego_yaw), np.cos(ego_yaw)]])
        self.T_world2ego = trans_matrix
        self.T_ego2world = trans_matrix.T

        # relative location vector in ego coord frame
        relative_loc = np.dot(trans_matrix, relative_loc)
        relative_yaw = yaw - ego_yaw  # npc relative to ego

        # regularize angle
        relative_yaw = angle_reg(relative_yaw)

        # add relative location and velocity into veh_info dict
        veh_info['relative_location'] = relative_loc  # 2D relative location in ego coord system
        veh_info['relative_yaw'] = relative_yaw

        if relative_loc[0] >= dist_threshold:
            flag = True

        return flag

    def filter_npc_vehicles(self):
        """
        This method is responsible for filtering suitable
        npc vehicles for state representation.

        Return a dict contains info of candidate npc vehicles.

        Currently, 2 rules are considered:
         - if npc vehicle in range bound
         - if npc vehicle is in front of ego vehicle
        """
        # list for carla.Vehicles
        near_npc_vehicles = []

        for index, veh in enumerate(self.npc_vehicles):
            veh_info = self.get_veh_info(veh)  # return a info dict
            # add distance into veh_info
            is_near = self.check_is_near(veh_info)

            if is_near:
                near_npc_vehicles.append(veh_info)

        # select vehicles in front of ego vehicle
        candidate_npc_vehicles = []
        for veh_info in near_npc_vehicles:
            is_front = self.check_is_front(veh_info)
            if is_front:
                candidate_npc_vehicles.append(veh_info)

        # select npc vehicles and info for state representation
        selected_npc_vehicles = heapq.nsmallest(self.state_npc_number,  # int, number of npc vehicles
                                                candidate_npc_vehicles,  # list in which stores the dict
                                                key=lambda s: s['distance'])

        return selected_npc_vehicles

    def get_state(self):
        """
        Get state vector with attention.

        Get state of current timestep for RL module.

        This method is re-writed for attention mechanism.
        """
        # update alive vehicles
        self.update_vehicles()

        # init state list with ego state
        if self.ego_vehicle:  # check ego vehicle
            state = self.get_ego_state()
        else:
            raise RuntimeError('Ego vehicle is not found!')

        # filter npc vehicles
        selected_npc_vehicles = self.filter_npc_vehicles()  # in dict
        # the real npc vehicle number for state representation
        _state_npc_number = len(selected_npc_vehicles)

        # todo need to test this line
        # store state NPC vehicles
        self.state_npc_vehicles = {}
        for dic in selected_npc_vehicles:
            self.state_npc_vehicles[dic['vehicle']] = dic

        # visualize npc vehicles for state
        if self.debug:
            # get actors from the dict
            plot_veh_list = [info_dict['vehicle'] for info_dict in selected_npc_vehicles]
            self.visualize_npc_vehicles(plot_veh_list)

            # print padding info if exists
            if _state_npc_number < self.state_npc_number:
                print('Only ', _state_npc_number, ' npc vehicles are suitable for state.')
                print(self.state_npc_number - _state_npc_number, 'padding states will be appended.')

        # # ========================
        # # quick debug
        # # get actors from the dict
        # plot_veh_list = [info_dict['vehicle'] for info_dict in selected_npc_vehicles]
        # self.visualize_npc_vehicles(plot_veh_list)
        #
        # # print padding info if exists
        # if _state_npc_number < self.state_npc_number:
        #     print('Only ', _state_npc_number, ' npc vehicles are suitable for state.')
        #     print(self.state_npc_number - _state_npc_number, 'padding states will be appended.')
        #
        # # ========================

        # append npc vehicle state
        for veh_info in selected_npc_vehicles:
            npc_state = self.get_single_state(veh_info)
            state += npc_state

        # check if need to padding state
        if _state_npc_number < self.state_npc_number:
            # todo add padding_state into class attribute
            # padding state according to npc vehicles number
            padding_state = np.array([self.range_bound, self.range_bound, 0, 0, 1, 0])
            padding_state = list(padding_state)
            # todo padding state should be coordinate with single vehicle state
            if len(padding_state) is not self.npc_state_len:
                raise RuntimeError('Padding npc state is different from definition, please check!')

            for _ in range(int(self.state_npc_number - _state_npc_number)):
                state += padding_state

        # todo add attention mechanism, need to fix API of the carla env
        # ================  add attention mask   ================
        if self.attention:
            mask = self.get_attention_mask(selected_npc_vehicles)
            # todo add args to check attention state length
            state = state + mask

            if self.debug:
                print('Using attention mechanism. The state length is: ', len(state))

        # todo fix state dimension check, merge with multi-task code
        # if len(state) is not self.state_len:
        #     raise RuntimeError('state length is not coordinate, please check.')

        # store state vector in ndarray format
        state_array = np.array(state)
        self.state_array = state_array

        return state_array

    def get_attention_mask(self, selected_npc_vehicles):
        """
        Get attention mask according to current selected vehicles number.
        """
        # musk size equals to vehicle number(npc number + ego)
        total_veh_num = self.state_npc_number + 1
        mask_size = int(total_veh_num)

        mask = list(np.ones(mask_size))

        # if not enough npc vehicles, padding state vector
        if len(selected_npc_vehicles) < self.state_npc_number:

            # missing npc number
            missing_npc_num = self.state_npc_number - len(selected_npc_vehicles)

            # pop missing bit
            for i in range(missing_npc_num):
                mask.pop()

            # append 0 to missing bit
            for i in range(missing_npc_num):
                mask.append(0)

        return mask

    # ==========  original ttc method  ==========
    def get_ttc(self):
        """
        Calculate time to collision through state vector(StateManager5).
        :return:
        """
        ttc_list = []

        for i in range(self.state_config['npc_number']):
            index = 4 + i*6
            clipped_state = self.state_array[index:index+4]

            # relative location
            loc_vector = np.array([
                clipped_state[0],
                clipped_state[1],
            ])
            norm_loc_vector = np.linalg.norm(loc_vector)

            # relative distance
            rel_distance = np.linalg.norm(loc_vector) - 2.67 * 2  # 2.67 is collision radius of mkz, in meters
            rel_distance = np.clip(rel_distance, 0.1, 9999)

            # relative velocity
            vel_vector = np.array([
                clipped_state[2],
                clipped_state[3],
            ])

            # velocity vector projected on relative location vector direction
            vel_projection = np.dot(loc_vector, vel_vector) / norm_loc_vector  # minus value
            vel_projection = np.clip(vel_projection, -1 * np.Inf, -1e-8)  # todo manual clip value

            ttc = rel_distance / (-1 * vel_projection)
            ttc_list.append(ttc)

            # # print ttc value
            # print('TTC of vehicle {}: {}'.format(int(i), ttc))

        # get the minimum
        ttc = min(ttc_list)

        return ttc

    def get_single_precise_ttc(self, vehicle, info_dict, ego_info: dict):
        """
        Calculate new precise ttc value of single social vehicle.

        :param vehicle:
        :param info_dict:
        :param ego_info:
        :return:
        """
        # ===============  parse ego info  ===============
        trans_mat_ego = ego_info['trans_mat_ego']
        location_ego = ego_info['location_ego']
        x_ego, y_ego = location_ego.x, location_ego.y
        bbox_ego = ego_info['bbox_ego']
        velocity_ego = ego_info['velocity_ego']

        if self.debug:
            plot_actor_bbox(vehicle, color=green)
            # self.world.tick()

        transform_npc = info_dict['transform']

        # world to npc vehicle
        trans_mat_npc = get_inverse_transform_matrix(transform_npc)

        # coordinate
        location_npc = info_dict['location']
        x_npc, y_npc = location_npc.x, location_npc.y
        # yaw_npc = transform_npc.rotation.yaw

        # relative location is stored in info_dict
        # relative_location = info_dict['relative_location']  # 2D relative location in ego coord system
        relative_yaw = info_dict['relative_yaw']

        # todo check if calculation is correct, compare with info_dict value
        # 3D relative location in world coord system
        # ego related to npc
        relative_location = np.array([x_ego - x_npc, y_ego - y_npc, 0.])

        # todo check this value
        # distance of the centers of the vehicles
        center_distance = np.linalg.norm(relative_location)

        # NPC vehicle may use different bp from ego vehicle
        # bbox projection length
        bbox_npc = vehicle.bounding_box
        # theta_0_npc = np.arctan(bbox_npc.extent.y, bbox_npc.extent.x)  # in rads

        # 3D relative location vector
        relative_loc_vector_npc2ego = np.dot(trans_mat_ego, -1 * relative_location.T)  # in ego coord system
        relative_loc_vector_ego2npc = np.dot(trans_mat_npc, relative_location.T)  # in npc coord system

        # calculate current theta
        # theta_ego = np.arctan(relative_loc_vector_npc2ego[0], relative_loc_vector_npc2ego[1])
        # theta_npc = np.arctan(relative_loc_vector_ego2npc[0], relative_loc_vector_ego2npc[1])

        def get_bbox_dist(location_vector, bbox):
            """
            Get current distance gap inside the bbox.

            :param bbox: bounding box of the target vehicle
            :param location_vector: relative location vector in target vehicle's body coordinate system
            :return: distance inside the bbox
            """
            # theta refers to the angle between relative location vector and x-axis of body system(left-hand rule)
            tan_theta = location_vector[1] / location_vector[0]
            theta = np.arctan([tan_theta])

            tan_theta_0 = bbox.extent.y / bbox.extent.x
            if -1 * tan_theta_0 <= tan_theta <= tan_theta_0:
                l_e = abs(bbox.extent.x / np.cos(theta))
                in_theta_0 = True
            else:
                l_e = abs(bbox.extent.y / np.sin(theta))
                in_theta_0 = False

            # # debug
            # print('tan_theta_0: ', np.rad2deg(tan_theta_0))
            # print('theta(degrees): ', np.rad2deg(theta))
            # print('l_e: ', l_e)
            # print('in_theta_0: ', in_theta_0)

            max_l_e = math.sqrt((bbox.extent.x**2 + bbox.extent.y**2))
            # if location_vector[0] < 0 and location_vector[1] < 0:
            #     print('')

            if l_e > max_l_e or l_e < bbox.extent.y:
                print('wrong')
                raise ValueError('bbox distance error.')

            return l_e

        # distance in ego bbox
        bbox_dist_ego = get_bbox_dist(
            location_vector=relative_loc_vector_npc2ego,
            bbox=bbox_ego,
        )

        # distance in npc bbox
        bbox_dist_npc = get_bbox_dist(
            location_vector=relative_loc_vector_ego2npc,
            bbox=bbox_npc,
        )

        # center distance
        center_distance = location_ego.distance(location_npc)
        # net distance
        net_distance = center_distance - bbox_dist_ego - bbox_dist_npc

        # =====  get velocity projection for ttc calculation  =====
        # get relative velocity vector
        velocity_npc = info_dict['velocity']  # carla.Vector3D

        # carla.Vector3D __sub__ method
        # this velocity vector is in world coord system
        relative_velocity = velocity_npc - velocity_ego
        # to ndarray
        relative_velocity = np.array([relative_velocity.x, relative_velocity.y, relative_velocity.z])

        # transform to ego coordinate system, using 3D transformation
        relative_velocity = np.dot(trans_mat_ego, relative_velocity)

        # norm value of relative velocity
        norm_relative_velocity = np.linalg.norm(relative_velocity)

        # normalized relative location vector
        # todo check this value, should be equal to center_distance
        distance_check = np.linalg.norm(relative_loc_vector_npc2ego)
        _rel_loc_vector = relative_loc_vector_npc2ego / np.linalg.norm(relative_loc_vector_npc2ego)

        # velocity projection value
        vel_projection = np.dot(_rel_loc_vector, relative_velocity)  # minus value

        # clip to a minus value
        vel_projection = np.clip(vel_projection, -1 * np.Inf, -1e-8)

        # calculate ttc value
        ttc = net_distance / (-1 * vel_projection)

        # # debug to print ttc value
        # print('TTC of vehicle {}: {}'.format(int(i), ttc))

        return ttc

    def get_precise_ttc_list(self):
        """
        <Getter method>
        This method is supposed to get a full list of ttc respect to each state social vehicles.

        :return:
        """

        """
        <element format>
        ttc_info = {
            'vehicle': vehicle,
            'distance': distance,
            'ttc': ttc,
        }
        """

        return self.ttc_info_list

    def get_minimum_ttc(self):
        """
        todo add API methods in env to call this method

        Return the minimum ttc value of current timestep
        :return:
        """
        # clear previous values
        self.ttc_info_list.clear()

        # ===============   ego vehicle info   ===============
        # get transform matrix, world2vehicle
        transform_ego = self.ego_info['transform']
        trans_mat_ego = get_inverse_transform_matrix(transform_ego)

        # coordinate
        location_ego = self.ego_info['location']
        x_ego, y_ego = location_ego.x, location_ego.y
        # todo yaw is for 2D transform matrix
        # yaw_ego = transform_ego.rotation.yaw

        # todo improve this part, get the bbox info for once
        #　ego vehicle bbox projection length
        bbox_ego = self.ego_vehicle.bounding_box
        # theta_0_ego = np.arctan(bbox_ego.extent.y, bbox_ego.extent.x)  # in rads

        # 2D relative velocity vector
        velocity_ego = self.ego_info['velocity']

        ego_info_dict = {
            'trans_mat_ego': trans_mat_ego,
            'location_ego': location_ego,
            'bbox_ego': bbox_ego,
            'velocity_ego': velocity_ego,
        }

        # if npc vehicles exist
        if self.state_npc_vehicles:
            for vehicle, info_dict in self.state_npc_vehicles.items():

                ttc = self.get_single_precise_ttc(
                    vehicle=vehicle,
                    info_dict=info_dict,
                    ego_info=ego_info_dict,
                )

                # distance related to ego vehicle
                distance = info_dict['distance']

                ttc_info = {
                    'vehicle': vehicle,
                    'distance': distance,
                    'ttc': ttc,
                }

                # element is the dict
                self.ttc_info_list.append(ttc_info)

            # todo filter the min ttc of all npc vehicles
            # get the minimum ttc value
            ttc_result = heapq.nsmallest(
                1,  # int, number of npc vehicles
                self.ttc_info_list,  # list in which stores the dict
                key=lambda s: s['ttc']
            )

            # todo add verbose to visualize min ttc vehicle
            min_ttc_veh = ttc_result[0]['vehicle']
            ttc = ttc_result[0]['ttc'][0]
        else:  # when None npc vehicle exists
            ttc = 999.

        return ttc

    def get_precise_ttc(self):
        """
        todo remove this method and use new version classmethods

        This is a devised version of ttc calculation.
        The bounding box of vehicle is considered.

        The distance gap will be calculated using distance of center minus vehicle overlap distance.

        :return:
        """

        #
        # get transform matrix, world2vehicle
        transform_ego = self.ego_info['transform']
        trans_mat_ego = get_inverse_transform_matrix(transform_ego)

        # coordinate
        location_ego = self.ego_info['location']
        x_ego, y_ego = location_ego.x, location_ego.y
        # todo yaw is for 2D transform matrix
        # yaw_ego = transform_ego.rotation.yaw

        # todo improve this part, get the bbox info for once
        #　ego vehicle bbox projection length
        bbox_ego = self.ego_vehicle.bounding_box
        # theta_0_ego = np.arctan(bbox_ego.extent.y, bbox_ego.extent.x)  # in rads

        # 2D relative velocity vector
        velocity_ego = self.ego_info['velocity']

        # store ttc value of each vehicle
        ttc_list = []

        # todo check if the dict is empty
        for vehicle, info_dict in self.state_npc_vehicles.items():

            if self.debug:
                plot_actor_bbox(vehicle, color=green)
                # self.world.tick()

            transform_npc = info_dict['transform']

            # world to npc vehicle
            trans_mat_npc = get_inverse_transform_matrix(transform_npc)

            # coordinate
            location_npc = info_dict['location']
            x_npc, y_npc = location_npc.x, location_npc.y
            # yaw_npc = transform_npc.rotation.yaw

            # relative location is stored in info_dict
            # relative_location = info_dict['relative_location']  # 2D relative location in ego coord system
            relative_yaw = info_dict['relative_yaw']

            # todo check if calculation is correct, compare with info_dict value
            # 3D relative location in world coord system
            # ego related to npc
            relative_location = np.array([x_ego - x_npc, y_ego - y_npc, 0.])

            # todo check this value
            # distance of the centers of the vehicles
            center_distance = np.linalg.norm(relative_location)

            # NPC vehicle may use different bp from ego vehicle
            # bbox projection length
            bbox_npc = vehicle.bounding_box
            # theta_0_npc = np.arctan(bbox_npc.extent.y, bbox_npc.extent.x)  # in rads

            # 3D relative location vector
            relative_loc_vector_npc2ego = np.dot(trans_mat_ego, -1 * relative_location.T)  # in ego coord system
            relative_loc_vector_ego2npc = np.dot(trans_mat_npc, relative_location.T)  # in npc coord system

            # calculate current theta
            # theta_ego = np.arctan(relative_loc_vector_npc2ego[0], relative_loc_vector_npc2ego[1])
            # theta_npc = np.arctan(relative_loc_vector_ego2npc[0], relative_loc_vector_ego2npc[1])

            # todo debug this part
            def get_bbox_dist(location_vector, bbox):
                """
                Get current distance gap inside the bbox.

                :param bbox: bounding box of the target vehicle
                :param location_vector: relative location vector in coord system of current vehicle
                :return: distance inside the bbox
                """
                # this is current theta value of relative location vector
                _theta = np.arctan([location_vector[0] / location_vector[1]])

                # # todo fix and check this the vector in 3rd and 4th Quadrant will not effect bbox distance
                # # check if vector is in 3rd and 4th Quadrant
                # if location_vector[0] <= 0.:
                #     if -1*_theta <= theta <= _theta:
                #     _theta = 0.5 * np.pi - _theta
                #     l_e = bbox.x / np.cos(theta_e)
                # else:
                #     l_e = bbox.y / np.sin(theta_e)

                # in rads
                theta_0 = np.arctan([bbox.extent.y / bbox.extent.x])

                if -1*theta_0 <= _theta <= theta_0:
                    l_e = bbox.extent.x / np.cos(_theta)
                else:
                    l_e = bbox.extent.y / np.sin(_theta)

                return l_e

            # distance in ego bbox
            bbox_dist_ego = get_bbox_dist(
                    location_vector=relative_loc_vector_npc2ego,
                    bbox=bbox_ego,
            )

            # distance in npc bbox
            bbox_dist_npc = get_bbox_dist(
                location_vector=relative_loc_vector_ego2npc,
                bbox=bbox_npc,
            )

            # center distance
            center_distance = location_ego.distance(location_npc)
            # net distance
            net_distance = center_distance - bbox_dist_ego - bbox_dist_npc

            # =====  get velocity projection for ttc calculation  =====
            # get relative velocity vector
            velocity_npc = info_dict['velocity']  # carla.Vector3D

            # carla.Vector3D __sub__ method
            # this velocity vector is in world coord system
            relative_velocity = velocity_npc - velocity_ego
            # to ndarray
            relative_velocity = np.array([relative_velocity.x, relative_velocity.y, relative_velocity.z])

            # transform to ego coordinate system, using 3D transformation
            relative_velocity = np.dot(trans_mat_ego, relative_velocity)

            # norm value of relative velocity
            norm_relative_velocity = np.linalg.norm(relative_velocity)

            # normalized relative location vector
            # todo check this value, should be equal to center_distance
            distance_check = np.linalg.norm(relative_loc_vector_npc2ego)
            _rel_loc_vector = relative_loc_vector_npc2ego / np.linalg.norm(relative_loc_vector_npc2ego)

            # velocity projection value
            vel_projection = np.dot(_rel_loc_vector, relative_velocity)  # minus value

            # clip to a minus value
            vel_projection = np.clip(vel_projection, -1 * np.Inf, -1e-8)

            # calculate ttc value
            ttc = net_distance / (-1 * vel_projection)

            # element is the dict
            ttc_list.append(
                {
                    'vehicle': vehicle,
                    'ttc': ttc,
                }
            )

            # # debug to print ttc value
            # print('TTC of vehicle {}: {}'.format(int(i), ttc))

        # todo filter the min ttc of all npc vehicles
        # get the minimum ttc value
        ttc_result = heapq.nsmallest(
            1,  # int, number of npc vehicles
            ttc_list,  # list in which stores the dict
            key=lambda s: s['ttc']
        )

        # todo add verbose for debug
        min_ttc_veh = ttc_result[0]['vehicle']
        ttc = ttc_result[0]['ttc'][0]

        return ttc

    # ==========  Using shapely to calculate ttc  ==========
    def get_ttc_devised(self):
        """
        Using shapely to calculate net distance.
        """

        # # shapely
        # ego_box =
        # target_box =
        #
        # net_distance =
        #
        # ttc =

        pass

    def get_single_frame_data(self):
        """
        Collect rollout data.

        :return:
        """
        # ===============  1. state data  ===============
        # append action into this dict in env
        state_data_dict = {
            'state': self.state_array,
            'ttc_1': self.get_ttc(),
            'ttc_2': self.get_minimum_ttc(),
        }

        # ===============  2. full data  ===============
        ego_data = self.collect_single_vehicle_data(self.ego_vehicle)
        # position code of ego
        ego_data['position_code'] = self.position_code

        # state vehicles
        state_vehicles_data_dict_list = []
        for state_vehicle, info_dict in self.state_npc_vehicles.items():
            state_vehicles_data_dict_list.append(
                self.collect_single_vehicle_data(state_vehicle)
            )

        # all vehicles
        all_vehicles_data_dict_list = []
        for vehicle in self.npc_vehicles:
            all_vehicles_data_dict_list.append(
                self.collect_single_vehicle_data(vehicle)
            )

        full_data_dict = {
            'ego_data': ego_data,
            'state_vehicles_data': state_vehicles_data_dict_list,  # filtered according to current rules
            'all_vehicles_data': all_vehicles_data_dict_list,
        }

        return state_data_dict, full_data_dict

    # =====================  staticmethod  =====================
    @staticmethod
    def collect_single_vehicle_data(vehicle: carla.Vehicle):
        """
        todo develop this method into a isolated class

        Collect data of single vehicle.
        """

        def vector2array(carla_vector):
            """
            todo put this method to util script

            This method transform carla Vector3D to a ndarray
            """

            vec_array = np.array([
                carla_vector.x,
                carla_vector.y,
                carla_vector.z,
            ])

            return vec_array

        # =========  major content  ==========
        data_dict = {}

        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        angular_velocity = vehicle.get_angular_velocity()

        attributes = vehicle.attributes
        vehicle_id = vehicle.id
        type_id = vehicle.type_id

        bounding_box = vehicle.bounding_box
        bbox_extent = bounding_box.extent
        bbox_location = bounding_box.location
        bbox_rotation = bounding_box.rotation

        # transform to ndarray
        location = vector2array(location)
        # rotation is not Vector3D, in degrees
        rotation = np.array([
            rotation.pitch,
            rotation.yaw,
            rotation.roll,
        ])

        velocity = vector2array(velocity)
        acceleration = vector2array(acceleration)
        angular_velocity = vector2array(angular_velocity)

        bbox_extent = vector2array(bbox_extent)
        bbox_location = vector2array(bbox_location)
        bbox_rotation = np.array([
            bbox_rotation.pitch,
            bbox_rotation.yaw,
            bbox_rotation.roll,
        ])

        data_dict = {
            'attributes': attributes,
            'vehicle_id': vehicle_id,
            'type_id': type_id,
            'location': location,
            'rotation': rotation,
            'velocity': velocity,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'bbox_extent': bbox_extent,
            'bbox_location': bbox_location,
            'bbox_rotation': bbox_rotation,

        }

        return data_dict

    @staticmethod
    def visualize_npc_vehicles(vehicle_list):
        """
        Visualize vehicles in vehicle_list
        """
        color_list = [
            red, yellow, blue, green, magenta,
        ]

        for index, veh in enumerate(vehicle_list):
            if index <= 5:
                color = color_list[index]
            else:
                color = white

            # plot bbox of vehicles
            plot_actor_bbox(veh, color=color)

            # # todo check this method
            # # draw velocity vector
            # draw_2D_velocity(veh, life_time=0.1)

            # todo visualize relative location by
            #  drawing relative location vector

    @staticmethod
    def vel2speed(vel):
        """
        Get speed from velocity, in km/s.

        vel is in m/s from carla.
        """
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    @staticmethod
    def get_veh_info(vehicle: carla.Vehicle):
        """
        Get info dict of a vehicle.
        :return:
        """
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        velocity = vehicle.get_velocity()
        bounding_box = vehicle.bounding_box

        info_dict = {
            'vehicle': vehicle,
            'transform': transform,
            'location': location,
            'rotation': rotation,
            'bounding_box': bounding_box,
            'velocity': velocity,
        }

        return info_dict

    @staticmethod
    def get_vehicle_data_simple(vehicle):
        """
        This method is for data collection.
        :param vehicle:
        :return:
        """

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        bounding_box = vehicle.bounding_box

        data_dict = {
            'vehicle': vehicle,
            'transform': transform,
            'bounding_box': bounding_box,
            'velocity': velocity,
        }

        return data_dict

    @staticmethod
    def compare_yaw_angle(vehicle):
        """
        Compare angle difference between vehicle heading and velocity

        Yaw angle refers to heading direction of a vehicle,
        compares with direction of velocity vector.

        :return: angle difference
        """
        #
        transform = vehicle.get_transform()
        rotation = transform.rotation
        yaw = np.radians(rotation.yaw)
        heading_direction = np.array([np.cos(yaw), np.sin(yaw)])

        velocity = vehicle.get_velocity()
        velo_2D = np.array([velocity.x, velocity.y])

        cos_angle = np.dot(heading_direction, velo_2D) / np.linalg.norm(heading_direction) / np.linalg.norm(velo_2D)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)  # in radians
        angle = np.degrees(angle)

        return angle

    @staticmethod
    def normalize_ttc(ttc):
        """
        Normalize ttc value with certain function.
        """

        norm_ttc = np.arctan(ttc) * 2 / np.pi

        return norm_ttc
