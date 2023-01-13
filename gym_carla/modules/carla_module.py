"""
This is a parent class of carla module.

Any developed carla module is supposed to inherit this class.
"""

import carla

import math
import numpy as np

from gym_carla.util_development.carla_color import *


class CarlaModule:
    """
    Basic carla module class.

    This class is responsible to manage a function of a simulation.

    A module is initialized by a created carla env.
    """

    def __init__(self, carla_api):
        self.carla_api = carla_api
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        self.npc_vehicles = []
        self.ego_vehicle = None

        self.frame_id = None

    def reload_carla_world(self, carla_api):
        """
        TODO check and fix this method this could cause critical error in carla training

        Reload carla world by reset carla_api.
        """
        self.carla_api = carla_api

        # only update world related attributes
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']

    def __call__(self, *args, **kwargs):
        """
        The module will be called each timestep when carla world ticks.
        """
        pass

    def run_step(self):
        """
        Do a run step to tick all necessary step.
        """
        pass

    def get_junction_by_location(self, center_location):
        """
        Get a junction instance by a location contained in the junction.

        For the junction of which center coordinate is known.

        param center_location: carla.Location of the junction center
        """
        wp = self.map.get_waypoint(location=center_location,
                                   project_to_road=False,  # not in the center of lane(road)
                                   lane_type=carla.LaneType.Driving)
        junction = wp.get_junction()

        return junction

    def update_vehicles(self):
        """
        Update vehicles of current timestep.
        This method is supposed to be called each timestep

        :return:
        """
        self.npc_vehicles = []
        self.ego_vehicle = None

        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # carla.Actorlist, iterable

        if vehicle_list:
            for veh in vehicle_list:  # veh is carla.Vehicle
                attr = veh.attributes  # dict
                role_name = attr['role_name']
                # filter ego vehicle
                if role_name in ['ego', 'hero']:
                    self.ego_vehicle = veh
                else:
                    self.npc_vehicles.append(veh)

    def try_tick_carla(self):
        """
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

                # print('Last frame id is ', self.frame_id)
                # when env is reset, self.frame_id is set to None
                if self.frame_id:
                    print('Last frame id is ', self.frame_id)

                print('*-' * 20)
                tick_counter += 1

        if tick_counter > 0:
            print('carla client is successfully ticked after ', tick_counter, 'times')

    def batch_delete_actors(self, delete_list: list):
        """
        Delete actors in delete_list.

        Using the carla.Client.apply_batch_sync() and carla.command.DestroyActor() method.
        """

        response_list = self.client.apply_batch_sync(
            [carla.command.DestroyActor(x) for x in delete_list],
            False,
        )

        failures = []
        for response in response_list:
            if response.has_error():
                failures.append(response)

        if failures:
            for response in failures:
                actor_id = response.actor_id
                print('Fail to batch delete actor: {}.'.format(actor_id))

    def delete_actors(self, delete_list: list):
        """
        todo fix apply batch sync method and add arg to select usage mode

        Delete actors in delete_list.

        Using the carla.Actor.destroy() method.
        """
        for actor in delete_list:

            # # todo print detail of the actor
            # actor_id = actor.id
            # actor_attributes = actor.attributes

            if_success = actor.destroy()

            # # todo if_success always return False, but it's not a bug
            # if not if_success:
            #     print('Fail to delete actor {}, please check.'.format(actor_id))

    def set_spectator_overhead(self, location, yaw=0, h=50):
        """
        Set spectator from an overview.

        param location: location of the spectator, carla.Location
        param h(float): height of spectator when using the overhead view
        """
        height = h

        location = carla.Location(0, 0, height) + location
        rotation = carla.Rotation(yaw=yaw, pitch=-90)  # rotate to forward direction

        self.spectator.set_transform(carla.Transform(location, rotation))
        self.try_tick_carla()

        print("Spectator is set to overhead view at location: \n{}.".format(location))


class CarlaModule2:
    """
    This is a modified version of the original CarlaModule class.
    This version will serve the lane graph related methods.

    =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    Basic carla module class.

    This class manages a set of functions to setup a CARLA simulation.

    All carla modules are initialized by a created carla env.
    """

    def __init__(self, carla_api):

        self.carla_api = carla_api
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        self.npc_vehicles = []
        self.ego_vehicle = None

        self.frame_id = None

    def reload_carla_world(self, carla_api):
        """
        Reload carla world by reset carla_api.
        """
        self.carla_api = carla_api

        # only update world related attributes
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']

    def __call__(self, *args, **kwargs):
        """
        The module will be called each timestep when carla world ticks.
        """
        pass

    def run_step(self):
        """
        Do a run step to tick all necessary step.
        """
        pass

    def get_junction_by_location(self, center_location):
        """
        Get a junction instance by a location contained in the junction.

        For the junction of which center coordinate is known.

        param center_location: carla.Location of the junction center
        """
        wp = self.map.get_waypoint(location=center_location,
                                   project_to_road=False,  # not in the center of lane(road)
                                   lane_type=carla.LaneType.Driving)
        junction = wp.get_junction()

        return junction

    def update_vehicles(self):
        """
        Update vehicles of current timestep.
        This method is supposed to be called each timestep

        :return:
        """
        self.npc_vehicles = []
        self.ego_vehicle = None

        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # carla.Actorlist, iterable

        if vehicle_list:
            for veh in vehicle_list:  # veh is carla.Vehicle
                attr = veh.attributes  # dict
                role_name = attr['role_name']
                # filter ego vehicle
                if role_name in ['ego', 'hero']:
                    self.ego_vehicle = veh
                else:
                    self.npc_vehicles.append(veh)

    def try_tick_carla(self):
        """
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

                # print('Last frame id is ', self.frame_id)
                # when env is reset, self.frame_id is set to None
                if self.frame_id:
                    print('Last frame id is ', self.frame_id)

                print('*-' * 20)
                tick_counter += 1

        if tick_counter > 0:
            print('carla client is successfully ticked after ', tick_counter, 'times')

    def batch_delete_actors(self, delete_list: list):
        """
        Delete actors in delete_list.

        Using the carla.Client.apply_batch_sync() and carla.command.DestroyActor() method.
        """

        response_list = self.client.apply_batch_sync(
            [carla.command.DestroyActor(x) for x in delete_list],
            False,
        )

        failures = []
        for response in response_list:
            if response.has_error():
                failures.append(response)

        if failures:
            for response in failures:
                actor_id = response.actor_id
                print('Fail to batch delete actor: {}.'.format(actor_id))

    def delete_actors(self, delete_list: list):
        """
        todo fix apply batch sync method and add arg to select usage mode

        Delete actors in delete_list.

        Using the carla.Actor.destroy() method.
        """
        for actor in delete_list:

            # # todo print detail of the actor
            # actor_id = actor.id
            # actor_attributes = actor.attributes

            if_success = actor.destroy()

            # # todo if_success always return False, but it's not a bug
            # if not if_success:
            #     print('Fail to delete actor {}, please check.'.format(actor_id))

    def set_spectator_overhead(self, location, yaw=0, h=50):
        """
        Set spectator from an overview.

        param location: location of the spectator, carla.Location
        param h(float): height of spectator when using the overhead view
        """
        height = h

        location = carla.Location(0, 0, height) + location
        rotation = carla.Rotation(yaw=yaw, pitch=-90)  # rotate to forward direction

        self.spectator.set_transform(carla.Transform(location, rotation))
        self.try_tick_carla()

        print("Spectator is set to overhead view at location: \n{}.".format(location))

    def draw_velocity(self):
        """
        TODO fix this method

        Draw velocity arrow to debug.
        """
        pass

    def draw_waypoint(self, waypoint, color=red, scale=1., z_offset=0., life_time=99999):
        """
        Draw a waypoint in carla world.

        This method is developed based on the one from util_visualization.
        """
        #  waypoint or transform are acceptable
        if isinstance(waypoint, carla.Waypoint):
            transform = waypoint.transform
        elif isinstance(waypoint, carla.Transform):
            transform = waypoint
        else:
            raise ValueError('A waypoint or its transform is required for <draw_waypoint> method.')

        length = 1.0*scale
        yaw = np.deg2rad(transform.rotation.yaw)
        vector = length * np.array([np.cos(yaw), np.sin(yaw)])

        start = transform.location + carla.Location(x=0., y=0., z=z_offset)
        end = start + carla.Location(x=vector[0], y=vector[1], z=0)

        # plot waypoint of the location point
        # debug.draw_point(start, size=0.15, color=color[0], life_time=99999)

        # plot waypoint with its transform
        self.debug_helper.draw_arrow(
            start,
            end,
            thickness=0.15*scale,
            arrow_size=0.15*scale,
            color=color,
            life_time=life_time,
        )

        # print('waypoint is plot, tick the world to visualize it.')


