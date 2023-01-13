"""
This traffic flow manager is responsible for multi-task experiments

Features from previous versions which deploy an autopilot-based traffic flow are merged.

"""

import numpy as np
import random
import time

import carla

from gym_carla.modules.carla_module import CarlaModule
from gym_carla.util_development.kinetics import get_transform_matrix
from gym_carla.util_development.scenario_helper_modified import generate_target_waypoint, get_waypoint_in_distance

from gym_carla.util_development.carla_color import *
from gym_carla.util_development.sensors import Sensors
from gym_carla.navigation.misc import get_speed
from gym_carla.util_development.util_junction import plot_coordinate_frame, get_junction_by_location
from gym_carla.util_development.util_visualization import draw_waypoint


# default Town03 junction center location
junction_center_location = carla.Location(x=-1.32, y=132.69, z=0.00)


class TrafficFlowManagerMultiTask(CarlaModule):
    """
    Traffic flows from all directions are activated in multi-task experiments.

    Use gap time as vehicle spawning
    """

    # todo check if better way to set density of traffic flow
    # TODO add args to set the distribution type, uniform or Gaussian
    # # traffic flow generation params
    # tf_params = {
    #     'time_interval': [5, 10],
    #     'distance_threshold': 3.,  # refer to the minimum distance gap
    # }

    # an easy version
    # traffic flow generation params
    tf_params = {
        'time_interval': [7, 12],
        'distance_threshold': 5.,  # refer to the minimum distance gap
    }

    # carla.TrafficManager params
    tm_params = {
        # percentage to ignore traffic light
        'ignore_lights_percentage': 0.,  # 100.
        # percentage to ignore traffic sign
        'ignore_signs_percentage': 100.,

        # target speed of current traffic flow, in km/h
        # use random.uniform()
        'target_speed': (50., 5),

        # # probability refers to whether a vehicle enables collisions(with a target vehicle)
        # # the probability to set vehicle collision_detection is True
        # 'collision_probability': 0.99,  # float [0, 1]

        # an easy traffic flow
        'collision_probability': 0.9,  # float [0, 1]

        # lane changing behaviour for a vehicle
        # True is default and enables lane changes. False will disable them.
        'auto_lane_change': False,

        # minimum distance in meters that a vehicle has to keep with the others.
        'distance_to_leading_vehicle': 1.,
    }

    # TODO add args to select different settings of traffic flow: easy/hard
    print('Traffic flow is set, the collision probability is: {}'.format(tm_params['collision_probability']))

    # available traffic flow directions
    tf_directions = ['positive_x', 'negative_x', 'negative_y_0', 'negative_y_1']

    # todo the params for Town 03 junction requires manually tuning, fix method for other random junction
    # this dict stores all traffic flow information
    # max spawn distance: {negative_x: 45, positive_y: 40(ego direction)}
    traffic_flow_info = {
        'positive_x': {
            'spawn_transform': carla.Transform(carla.Location(x=71.354889, y=130.074112, z=0.1),
                                               carla.Rotation(pitch=359.836853, yaw=179.182800, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],  # list of tuple, (vehicle, sensor)
        },

        'negative_x': {
            'spawn_transform': carla.Transform(carla.Location(x=-64.222389, y=135.423065, z=0.100000),
                                               carla.Rotation(pitch=0.000000, yaw=-361.296783, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'negative_y_0': {
            'spawn_transform': carla.Transform(carla.Location(x=-6.411462, y=68.223877, z=0.100000),
                                               carla.Rotation(pitch=0.000000, yaw=89.637459, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        'negative_y_1': {
            'spawn_transform': carla.Transform(carla.Location(x=-9.911532, y=68.223877, z=0.100000),
                                               carla.Rotation(pitch=0.000000, yaw=89.637459, roll=0.000000)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'vehicle_sensor': [],
        },

        # # positive_y directions indicate the start location of ego vehicle
        # 'positive_y': {
        #     'spawn_transform': carla.Transform(carla.Location(x=2.354240, y=189.210159, z=0.000000),
        #                                        carla.Rotation(pitch=0.000000, yaw=-90.362534, roll=0.000000)),
        #     'last_spawn_time': 0,
        #     'target_spawn_time': None,
        #     'vehicle_sensor': [],
        # },

    }

    def __init__(
            self,
            carla_api,
            phase_time,

            junction=None,

            tm_port=int(8100),
            tm_seed=int(0),

            # tls_red_phase_duration=20.,  # duration time of red phase of traffic lights

            debug=False,
            verbose=False,
    ):

        self.carla_api = carla_api

        self.tm_port = tm_port
        self.tm_seed = tm_seed

        # self.tls_red_phase_duration = tls_red_phase_duration

        self.phase_time = phase_time

        """
        Default format:

        phase_time = {
            'x_green_phase': 40.,
            'x_yellow_phase': 10.,
            'y_green_phase': 40.,
            'y_yellow_phase': 10.,
        }

        """

        # todo improve this part, if using different phase duration time on each direction
        self.red_phase_duration = self.phase_time['x_green_phase']
        self.yellow_phase_duration = self.phase_time['x_yellow_phase']


        self.debug = debug
        self.verbose = verbose

        super(TrafficFlowManagerMultiTask, self).__init__(
            carla_api=self.carla_api,
        )

        # assign junction
        if junction:
            self.junction = junction
        else:  # default junction is Town03 junction
            self.junction = get_junction_by_location(self.map, junction_center_location)

        # set traffic manager seed
        self.traffic_manager.set_random_device_seed(self.tm_seed)

        # list of carla.Location
        self.sink_locations = []

        # get timestep length of simulator
        settings = self.world.get_settings()
        self.timestep_length = settings.fixed_delta_seconds

        # todo merge into carla module class
        # private attributes
        self.ego_vehicle = None
        self.npc_vehicles = []
        # list to store vehicle and its info_dict, element: (vehicle, vehicle_info)
        # format of vehicle info dict
        self.vehicle_info_list = []
        """
        info dict format
        
        info_dict = {
            'id': vehicle.id,
            'vehicle': vehicle,  # carla.Vehicle
            'sensor_api': sensor,  # instance of Sensor
            'sensor': sensor.collision,  # carla.Actor
            'block_time': 0.,  # the time this vehicle has been blocked
        }    
        """

        # newly spawned vehicle needs to be registered to the traffic manager
        # this list is fed and cleared in each timestep
        self.tm_register_list = []

        # this may helpful when reload carla world
        self.clean_up()

    def set_traffic_lights_phase_duration(self, phase_time):
        """
        Setter for traffic lights duration time.
        Determine the blocking time check condition.
        """

        self.phase_time = phase_time

    def __call__(self):
        """
        todo add this method to a public method

        if some methods require multiple called
        """
        pass

    def delete_vehicles(self):
        """
        Delete social vehicles managed by the traffic flow module.
        """
        delete_list = []
        for info_tuple in self.vehicle_info_list:
            vehicle = info_tuple[0]
            info_dict = info_tuple[1]
            sensor = info_dict['sensor']

            delete_list.append(vehicle)
            delete_list.append(sensor)

        # todo test batch_delete_actors method

        self.delete_actors(delete_list)

    def clean_up(self):
        """
        Clean the traffic flows.

        Delete all the social vehicles generated by traffic

        :return:
        """
        # =====  STAGE 1: Delete vehicles and their collision sensors  =====
        self.delete_vehicles()

        # =====  STAGE 2: clear info storage
        self.clear_traffic_flow_info()

    def clear_traffic_flow_info(self):
        """"""

        for tf, info_dict in self.traffic_flow_info.items():

            info_dict['last_spawn_time'] = 0
            info_dict['target_spawn_time'] = None

            # todo check if manually add sensors and vehicles to delete list
            info_dict['vehicle_sensor'] = []  # list of tuple, (vehicle, sensor)

            # other info storage
            self.npc_vehicles = []
            self.vehicle_info_list = []  # element: (vehicle, vehicle_info)

    def run_step_1(self):
        """
        Split run_step into 2 methods.
        Call this one before and after carla world tick.
        """
        try:
            # spawn new vehicles
            self.actor_source_1()
        except:
            raise RuntimeError('actor_source_1.')

    def run_step_2(self):
        """
        Call this one after carla world tick.
        """

        self.update_vehicles()

        try:
            # register to traffic manager
            self.actor_source_2()
        except:
            raise RuntimeError('Error in actor_source_2.')

        # check and delete npc vehicles
        try:
            self.delete_npc()
        except:
            raise RuntimeError('Error in delete_npc.')
        # check and add collision detection on ego vehicle
        # make sure existing npc vehicles be aware of ego vehicle
        if self.ego_vehicle:
            for veh in self.npc_vehicles:
                self.set_collision_detection(veh, self.ego_vehicle)

    def run_step(self):
        """
        Tick the traffic flow at each timestep.

        # todo print newly spawned and deleted vehicles.
        """
        # try to spawn new traffic flow vehicles
        self.spawn_new_vehicles()
        # check and delete npc vehicles
        self.delete_npc()
        # make sure existing npc vehicles be aware of ego vehicle
        self.update_vehicles()

        # check and add collision detection on ego vehicle
        if self.ego_vehicle:
            for veh in self.npc_vehicles:
                self.set_collision_detection(veh, self.ego_vehicle)

    def update_vehicles(self):
        """
        Check and update vehicles exist in current world.
        """
        # reset the storage
        npc_vehicles = []
        self.ego_vehicle = None

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a actorlist instance, iterable

        if vehicle_list:
            for veh in vehicle_list:
                attr = veh.attributes  # dict
                # filter ego vehicle by role name
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':
                    self.ego_vehicle = veh
                else:
                    npc_vehicles.append(veh)

        # check if npc vehicles are right
        id_list = []
        _id_list = []

        for veh in npc_vehicles:
            id_list.append(veh.id)

        for veh in self.npc_vehicles:
            _id_list.append(veh.id)

        if self.debug:
            _difference = set(id_list) ^ set(_id_list)
            if _difference:
                print('These vehicles are not updated correctly: ')
                print(_difference)

    def set_active_tf_direction(self, active_tf_direction: list, clean_vehicles=False):
        """
        This method is responsible to determine which traffic flow is supposed to be activated.

        For multi-task experiments, all traffic flows are activated.

        params: tf_direction: list contains desired traffic flow direction, i.e. [positive_x, positive_y]
        params: clean_vehicles:  if remove vehicles from inactive traffic flows
        """
        # check and update tf_direction
        for tf_direc in active_tf_direction:
            if tf_direc not in self.traffic_flow_info.keys():
                raise ValueError('Wrong traffic flow direction, please check.')
        self.active_tf_direction = active_tf_direction

        # vehicles in inactive traffic flow will be removed
        if clean_vehicles:
            # tf directions to be cleaned
            tf_direction_to_remove = [tf for tf in self.tf_directions if tf not in self.active_tf_direction]

            delete_actors = []  # actors to delete
            # get actors to be removed
            for tf in tf_direction_to_remove:
                _vehicle_sensor_list = self.traffic_flow_info[tf]['vehicle_sensor']
                for item in _vehicle_sensor_list:
                    vehicle = item[0]
                    sensor = item[1]
                    delete_actors.append(vehicle)  # vehicle
                    delete_actors.append(sensor)  # collision sensor actor

                    # remove from class attrubutes
                    for veh in self.npc_vehicles:
                        if veh.id == item[0]:
                            self.npc_vehicles.remove(veh)

                    for tup in self.vehicle_info_list:
                        _veh = tup[0]
                        if _veh.id == vehicle.id:
                            self.vehicle_info_list.remove(tup)

                # reset the tf dict in npc_info
                self.traffic_flow_info[tf]['last_spawn_time'] = 0
                self.traffic_flow_info[tf]['target_spawn_time'] = None
                self.traffic_flow_info[tf]['vehicle_sensor'] = []

            # delete actors from world
            self.delete_actors(delete_actors)
            print('')

        print('Traffic flow is reset, following traffic flow will be activated: ')
        print(self.active_tf_direction)

    def set_collision_detection_rate(self, collision_rate):
        """
        todo merge this method with set_tm_params

        Set the collision detection probability of social vehicles to ego vehicle.
        """
        self.tm_params['collision_probability'] = collision_rate

        print('Collision detection rate to ego vehicle is reset, current value is {}.'.format(collision_rate))

    def set_tm_params(self, tm_params):
        """
        A setter method,
        Set parameters of the traffic flow.

        We use method name from carla PythonAPI as the dict key!
        """
        for key in tm_params:
            if key in self.tm_params.keys():
                # update keys in current instance
                self.tm_params[key] = tm_params[key]

        print('Traffic flow param is reset.')
        print(self.tm_params)

    def check_distance_exceed(self, vehicle, max_dist=75.):
        """
        Check vehicle is too far from junction center.

        params: max_dist: max distance for keeping a vehicle
        """
        junction_center = self.junction.bounding_box.location
        dist = junction_center.distance(vehicle.get_transform().location)
        if dist >= max_dist:
            return True
        else:
            return False

    def check_in_junction(self, actor: carla.Actor, expand=0.0):
        """
        Check if an actor is contained in the target junction

        param actor: actor to be checked if contained in junction
        param expand: expand distance of the junction bounding box
        """
        contain_flag = False

        bbox = self.junction.bounding_box
        actor_loc = actor.get_location()

        relative_vec = actor_loc - bbox.location  # carla.Vector3D

        # todo check junction bbox rotation effect
        if (relative_vec.x <= bbox.extent.x + expand) and \
                (relative_vec.x >= -bbox.extent.x - expand) and \
                (relative_vec.y >= -bbox.extent.y - expand) and \
                (relative_vec.y <= bbox.extent.y + expand):
            contain_flag = True

        return contain_flag

    def check_blocked(self, vehicle_info: dict, max_block_time=5.0):
        """
        Check if a vehicle is blocked,
        usually blocked in junction, or blocked by other traffic lights.

        todo add args to set maximum block time in and out of the junction

        param vehicle: vehicle to be checked
        param max_block_time: the maximum block time for npc vehicle
                              which is outside the junction

        return: bool, True refers that the vehicle is blocked
        """
        vehicle = vehicle_info['vehicle']
        block_time = vehicle_info['block_time']

        # check if vehicle is blocked
        block_flag = False

        # check if vehicle is in the junction
        # True refers to the vehicle is in the junction area
        in_junction = self.check_in_junction(vehicle)

        #
        if in_junction:
            if block_time >= max_block_time:
                self.visualize_actor(vehicle, color=yan, thickness=1.0)
                block_flag = True
        else:
            # # original line
            # if block_time >= self.tls_red_phase_duration:
            if block_time >= self.red_phase_duration + self.yellow_phase_duration:
                self.visualize_actor(vehicle, color=yan, thickness=1.0)
                block_flag = True

        return block_flag

    def check_collision(self, vehicle_info: dict):
        """
        Check if a vehicle collides with other vehicle.
        """
        vehicle = vehicle_info['vehicle']
        sensor_api = vehicle_info['sensor_api']
        collision_flag = sensor_api.collision_flag

        if collision_flag:
            self.visualize_actor(vehicle, color=orange, thickness=1.0)

        return collision_flag

    def update_block_time(self):
        """
        todo add a method to check traffic flow vehicles delete condition

        Update the block time of all active traffic flow vehicles.

        This method is supposed to be called each timestep.
        """
        # find info dict of this vehicle
        for index, item in enumerate(self.vehicle_info_list):
            vehicle = item[0]
            vehicle_info = item[1]
            # get current velocity
            speed = get_speed(vehicle)  # in km/h
            if speed <= 0.05:  # check by a threshold
                # todo this method need to be checked
                vehicle_info['block_time'] += self.timestep_length
            else:
                # reset the block time
                self.vehicle_info_list[index][1]['block_time'] = 0

    def delete_npc(self):
        """
        Check and delete useless npc vehicle.
        """
        # update block time of all NPC vehicles
        self.update_block_time()

        delete_list = []
        # info_tuple is a tuple of vehicle and vehicle_info dict
        for info_tuple in self.vehicle_info_list:
            vehicle = info_tuple[0]
            info_dict = info_tuple[1]
            sensor = info_dict['sensor']

            # conditions for removing this vehicle, delete if any is True
            distance_cond = self.check_distance_exceed(vehicle)
            block_cond = self.check_blocked(info_dict)
            collision_cond = self.check_collision(info_dict)
            conditions = [distance_cond, block_cond, collision_cond]

            if any(conditions):
                delete_list.append(vehicle)
                delete_list.append(sensor)

                # clear storage info
                self.npc_vehicles.remove(vehicle)
                self.vehicle_info_list.remove(info_tuple)
                # remove vehicle sensor tuple from npc_info dict
                for key, item in self.traffic_flow_info.items():
                    veh_sen_list = item['vehicle_sensor']
                    for sensor_tuple in veh_sen_list:  # tup is (vehicle, sensor)
                        veh = sensor_tuple[0]
                        if veh.id == vehicle.id:
                            veh_sen_list.remove(sensor_tuple)

        if delete_list:
            self.delete_actors(delete_list)

    def get_time(self):
        """
        Get current time using timestamp(carla.Timestamp)
        :return: current timestamp in seconds.
        """
        worldsnapshot = self.world.get_snapshot()
        timestamp = worldsnapshot.timestamp
        now_time = timestamp.elapsed_seconds

        return now_time

    def sample_interval_time(self):
        """
        Sample an interval time till next vehicle spawning in seconds.

        :return: time_interval, float
        """
        lower_limit = self.tf_params['time_interval'][0]
        upper_limit = self.tf_params['time_interval'][1]
        time_interval = random.uniform(lower_limit, upper_limit)

        return time_interval

    def get_npc_bp(self, name=None):
        """
        Get single blueprint for social vehicle.
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            color = '0, 255, 255'  # use string to identify a RGB color
            bp.set_attribute('color', color)
        if name:
            bp.set_attribute('role_name', name)  # set actor name

    def spawn_single_collision_sensor(self, vehicle):
        """
        Spawn single collision sensor on a vehicle
        """
        # create collision sensor
        sensor = Sensors(self.world, vehicle)

        # store vehicle info into dict
        info_dict = {
            'id': vehicle.id,
            'vehicle': vehicle,  # carla.Vehicle
            'sensor_api': sensor,  # instance of Sensor
            'sensor': sensor.collision,  # carla.Actor
            'block_time': 0.,  # the time this vehicle has been blocked
        }

        return info_dict

    def spawn_single_vehicle(self, transform, name=None):
        """
        Spawn single NPC vehicle at given transform.

        todo add arg to set vehicle model

        :return: carla.Actor, spawned npc vehicle
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            color = '0, 255, 255'  # use string to identify a RGB color
            bp.set_attribute('color', color)
        if name:
            bp.set_attribute('role_name', name)  # set actor name

        # spawn npc vehicle
        try:
            vehicle = self.world.spawn_actor(bp, transform)  # use spawn method
            # necessary to tick world, seem not necessary to sleep
            # self.world.tick()
            # time.sleep(0.05)
        except:
            raise RuntimeError('Fail to spawn a NPC vehicle, please check.')

        # create collision sensor
        sensor = Sensors(self.world, vehicle)

        # store vehicle info into dict
        info_dict = {
            'id': vehicle.id,
            'vehicle': vehicle,  # carla.Vehicle
            'sensor_api': sensor,  # instance of Sensor
            'sensor': sensor.collision,  # carla.Actor
            'block_time': 0.,  # the time this vehicle has been blocked
        }

        return vehicle, info_dict

    def set_collision_detection(self, reference_actor, other_actor):
        """
        Set the collision detection of a npc vehicle with another actor.
        """

        collision_probability = self.tm_params['collision_probability']
        # ignore conflict with ego vehicle with a probability
        if np.random.random() <= collision_probability:
            collision_detection_flag = False  # refers to enable collision
        else:
            collision_detection_flag = True
        # reference_actor: to be set; other_actor: target vehicle
        self.traffic_manager.collision_detection(reference_actor, other_actor, collision_detection_flag)

        # # print collision detection state
        # collision_statement = '' if collision_detection_flag else 'not'
        # print('Vehicle ', reference_actor.id, ' is ', collision_statement, 'enabled to collide with ego vehicle.')

    def set_traffic_manager(self, vehicle, tm_params: dict):
        """
        Register vehicle to the traffic manager with the given setting.

        todo add api tp set different traffic manager params

        :param vehicle: target vehicle(npc)
        :param tm_params: traffic manager parameters
        """
        # set traffic manager for each vehicle
        vehicle.set_autopilot(True, int(self.tm_port))  # Doc is wrong, the 2nd optional arg is tm_port

        # traffic lights
        per = tm_params['ignore_lights_percentage']
        self.traffic_manager.ignore_lights_percentage(vehicle, per)

        # speed limits
        mean_speed = tm_params['target_speed'][0]
        speed_interval = tm_params['target_speed'][1]
        target_speed = random.uniform(mean_speed - speed_interval, mean_speed + speed_interval)
        per = self.get_percentage_by_target_speed(vehicle, target_speed)
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, per)

        # auto lane change
        auto_lane_change = self.tm_params['auto_lane_change']
        self.traffic_manager.auto_lane_change(vehicle, auto_lane_change)

        # minimum distance to leading vehicle
        distance = self.tm_params['distance_to_leading_vehicle']
        self.traffic_manager.distance_to_leading_vehicle(vehicle, distance)

        # set collision detection for ego
        if self.ego_vehicle:
            self.set_collision_detection(vehicle, self.ego_vehicle)

        # set a initial velocity
        self.set_velocity(vehicle, target_speed=target_speed / 3.6 * 0.75)
        # time.sleep(0.1)
        # print('Vehicle ', vehicle.id, ' is set to traffic manager ', self.tm_port)

    def set_velocity(self, vehicle, target_speed: float):
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
        #     self.world.tick()

    def set_traffic_manager_group(self, vehicle_list, tm_params: dict):
        """
        todo fix this method

        Register a list of vehicles to the traffic manager using given setting.

        :param vehicle: target vehicle(npc)
        :param tm_params: traffic manager parameters
        """
        # # set traffic manager for each vehicle
        # vehicle.set_autopilot(True, int(self.tm_port))  # Doc is wrong, the 2nd optional arg is tm_port

        SetAutopilot = carla.command.SetAutopilot

        # batch = []
        # for vehicle in vehicle_list:
        #     batch.append(SetAutopilot(vehicle, True, int(self.tm_port)))

        response_list = self.client.apply_batch_sync(
            [SetAutopilot(vehicle, True, int(self.tm_port)) for vehicle in vehicle_list],
            False,
        )
        failures = []
        for response in response_list:
            if response.has_error():
                failures.append(response)

        # set specific settings for each vehicle
        for vehicle in vehicle_list:

            # traffic lights
            per = tm_params['ignore_lights_percentage']
            self.traffic_manager.ignore_lights_percentage(vehicle, per)

            # speed limits
            mean_speed = tm_params['target_speed'][0]
            speed_interval = tm_params['target_speed'][1]
            target_speed = random.uniform(mean_speed - speed_interval, mean_speed + speed_interval)
            per = self.get_percentage_by_target_speed(vehicle, target_speed)
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, per)

            # auto lane change
            auto_lane_change = self.tm_params['auto_lane_change']
            self.traffic_manager.auto_lane_change(vehicle, auto_lane_change)

            # minimum distance to leading vehicle
            distance = self.tm_params['distance_to_leading_vehicle']
            self.traffic_manager.distance_to_leading_vehicle(vehicle, distance)

            # set collision detection for ego
            if self.ego_vehicle:
                self.set_collision_detection(vehicle, self.ego_vehicle)

            # set a initial velocity
            self.set_velocity(vehicle, target_speed=target_speed / 3.6 * 0.75)
            # time.sleep(0.1)
            # print('Vehicle ', vehicle.id, ' is set to traffic manager ', self.tm_port)

    def actor_source_2(self):
        """
        In the second one, we will
         -
        """
        for vehicle in self.tm_register_list:
            self.set_traffic_manager(vehicle, self.tm_params)
            # time.sleep(.5)

        self.tm_register_list = []

    def actor_source_1(self):
        """
        Split original spawn_new_vehicles method into 2 methods.

        In the first one, we will
         - check and spawn new vehicles
         - check and delete undesired vehicles
        """
        # # new vehicle to be spawned
        # spawn_batch = []

        for key, item in self.traffic_flow_info.items():

            # # only generate active tf
            # if key not in self.active_tf_direction:
            #     continue

            spawn_transform = item['spawn_transform']

            # get waypoint at the center of lane
            spawn_waypoint = self.map.get_waypoint(
                spawn_transform.location,
                project_to_road=True,  # not in the center of lane(road)
                # lane_type=carla.LaneType.Driving,
            )
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 0.5

            # todo original lines, remove after debug
            # # set spawn height, original height is retrieve from waypoint
            # transform.location.z = 0.5
            # # transform.location.z += 0.5  # sometimes fails

            # todo add param to tune time interval of different traffic flow
            # set initial spawn time
            if not item['target_spawn_time']:
                item['target_spawn_time'] = 0

            # ========== conditions of spawning vehicle ==========
            # condition of distance to start location
            if item['vehicle_sensor']:
                last_vehicle = item['vehicle_sensor'][-1][0]
                # last_spawned_vehicle =self.straight_npc_list[-1]
                distance = spawn_transform.location.distance(last_vehicle.get_transform().location)
                # check if distance gap is large enough
                distance_threshold = self.tf_params['distance_threshold']
                distance_rule = distance >= distance_threshold
                # if distance rule is not satisfied, this direction is skipped
                if not distance_rule:
                    continue
            else:
                distance_rule = True

            # condition of gap time
            now_time = self.get_time()
            if now_time >= item['target_spawn_time']:
                time_rule = True
            else:
                time_rule = False

            # todo add a probability to spawn npc vehicles
            if distance_rule and time_rule:
                # spawn_batch.append(
                #     carla.command.SpawnActor(self.get_npc_bp(), spawn_transform)
                # )

                # original lines to spawn vehicle and its sensor
                try:
                    # # print to debug
                    # print('Try to spawn new NPC vehicle...')

                    vehicle, info_dict = self.spawn_single_vehicle(spawn_transform)
                    sensor = info_dict['sensor']  # sensor actor

                    # # print to debug
                    # print('A new NPC vehicle is spawned successfully. Vehicle id: {}'.format(vehicle.id))

                    # # CAUTION! move this to actor_source_2
                    # # register spawned vehicle to traffic manager
                    # self.set_traffic_manager(vehicle, self.tm_params)

                    self.tm_register_list.append(vehicle)

                    # append to all storage
                    self.npc_vehicles.append(vehicle)  # add to npc vehicles list
                    self.vehicle_info_list.append((vehicle, info_dict))
                    item['vehicle_sensor'].append((vehicle, sensor))

                    # update last spawn time when new vehicle spawned
                    item['last_spawn_time'] = self.get_time()
                    # min time to spawn next vehicle
                    item['target_spawn_time'] = item['last_spawn_time'] + self.sample_interval_time()
                except:
                    print("Fail to spawn a new NPC vehicle and register traffic manager, please check.")
                    # if self.debug:
                    #     raise RuntimeError('Check failure of spawn NPC vehicles...')

    def spawn_new_vehicles(self):
        """
        Spawn all traffic flow in this junction.
        3 flows if crossroad, 2 flows if T-road
        """
        for key, item in self.traffic_flow_info.items():

            # # only generate active tf
            # if key not in self.active_tf_direction:
            #     continue

            spawn_transform = item['spawn_transform']

            # get waypoint at the center of lane
            spawn_waypoint = self.map.get_waypoint(
                spawn_transform.location,
                project_to_road=True,  # not in the center of lane(road)
                lane_type=carla.LaneType.Driving,
            )
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 0.5

            # todo original lines, remove after debug
            # # set spawn height, original height is retrieve from waypoint
            # transform.location.z = 0.5
            # # transform.location.z += 0.5  # sometimes fails

            # todo add param to tune time interval of different traffic flow
            # set initial spawn time
            if not item['target_spawn_time']:
                item['target_spawn_time'] = 0

            # ========== conditions of spawning vehicle ==========
            # condition of distance to start location
            if item['vehicle_sensor']:
                last_vehicle = item['vehicle_sensor'][-1][0]
                # last_spawned_vehicle =self.straight_npc_list[-1]
                distance = spawn_transform.location.distance(last_vehicle.get_transform().location)
                # check if distance gap is large enough
                distance_threshold = self.tf_params['distance_threshold']
                distance_rule = distance >= distance_threshold
                # if distance rule is not satisfied, this direction is skipped
                if not distance_rule:
                    continue
            else:
                distance_rule = True

            # condition of gap time
            now_time = self.get_time()
            if now_time >= item['target_spawn_time']:
                time_rule = True
            else:
                time_rule = False

            # todo add a probability to spawn npc vehicles
            if distance_rule and time_rule:
                try:
                    vehicle, info_dict = self.spawn_single_vehicle(spawn_transform)
                    sensor = info_dict['sensor']  # sensor actor

                    # register spawned vehicle to traffic manager
                    self.set_traffic_manager(vehicle, self.tm_params)

                    # append to all storage
                    self.npc_vehicles.append(vehicle)  # add to npc vehicles list
                    self.vehicle_info_list.append((vehicle, info_dict))
                    item['vehicle_sensor'].append((vehicle, sensor))

                    # update last spawn time when new vehicle spawned
                    item['last_spawn_time'] = self.get_time()
                    # min time to spawn next vehicle
                    item['target_spawn_time'] = item['last_spawn_time'] + self.sample_interval_time()
                except:
                    print("Fail to spawn a new NPC vehicle and register traffic manager, please check.")
                    # if self.debug:
                    #     raise RuntimeError('Check failure of spawn NPC vehicles...')

    def visualize_actor(self, vehicle, thickness=0.5, color=red, duration_time=0.1):
        """
        Visualize an vehicle(or walker) in server by drawing its bounding box.
        """
        bbox = vehicle.bounding_box
        bbox.location = bbox.location + vehicle.get_location()
        transform = vehicle.get_transform()
        # vehicle bbox fixed to ego coordinate system
        rotation = transform.rotation
        # draw the bbox of vehicle
        self.debug_helper.draw_box(
            box=bbox,
            rotation=rotation,
            thickness=thickness,
            color=color,
            life_time=duration_time,
        )

    def get_sink_location(self):
        """
        Get sink point of different traffic flow.

        Only straight flag needs to be considered, because left and right routes are duplicated.
        """

        # get all available sink points
        for key, item in self.traffic_flow_info:
            transform = item['spawn_transform']
            location = transform.location
            start_waypoint = self.map.get_waypoint(location=location,
                                                   project_to_road=True,
                                                   )

            # the waypoint after junction
            exit_waypoint = generate_target_waypoint(
                waypoint=start_waypoint,
                turn=0,
            )

            # the end of the road
            end_waypoint = exit_waypoint.next(1.0)[0]
            while not end_waypoint.is_intersection:
                end_waypoint = end_waypoint.next(1.0)[0]  # end_waypoint refers to the end of the whole route
            # distance gap from the end of the road
            distance_gap = 3.
            end_waypoint = end_waypoint.previous(distance_gap)[0]

            # as the sink location
            end_location = end_waypoint.transform.location

            self.sink_locations.append(end_location)

    # ==================   staticmethod   ==================
    @staticmethod
    def get_percentage_by_target_speed(veh, target_speed):
        """
        Calculate vehicle_percentage_speed_difference according to the given speed.

        :param veh: vehicle
        :param target_speed: target speed of the vehicle in km/h
        :return: per: vehicle_percentage_speed_difference, float value refer to a percentage(per %)
        """
        # target speed in m/s
        target_speed = target_speed / 3.6
        speed_limit = veh.get_speed_limit()  # in m/s
        per = (speed_limit - target_speed) / speed_limit

        return per

    @staticmethod
    def junction_contains(location, bbox, margin: float = 5.0):
        """
        Check if a location is contained in a junction bounding box.

        todo: consider a rotated junction

        :param location: location point to check
        :param bbox: junction bounding box(carla.BoundingBox)
        :param margin: detection is larger than the original bbox with margin distance

        :return: bool
        """
        contain_flag = False

        if margin < 0:
            print('margin value must larger than 0.')
        margin = np.clip(margin, 0, float('inf'))

        # relative location to junction center
        _location = location - bbox.location
        extent = bbox.extent

        if extent.x + margin >= _location.x >= -extent.x - margin and \
                extent.y + margin >= _location.y >= -extent.y - margin:
            contain_flag = True

        return contain_flag

