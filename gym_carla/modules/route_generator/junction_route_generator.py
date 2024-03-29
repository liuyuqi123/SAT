"""
This module is used to generate a route for junction turning scenario.

"""

from gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

# ==================================================
# import carla module
import glob
import os
import sys

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

# original version
from gym_carla.envs.BasicEnv import BasicEnv
from gym_carla.modules.carla_module import CarlaModule
from gym_carla.util_development.util_visualization import draw_waypoint
from gym_carla.util_development.scenario_helper_modified import (generate_target_waypoint,
                                                                 get_waypoint_in_distance
                                                                 )
from gym_carla.util_development.route_manipulation import interpolate_trajectory

# for debug
default_junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

# carla colors
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)


class RouteGenerator(CarlaModule):

    # 2 dim tuple, distance before and after the junction area
    route_distance = (10, 10)

    def __init__(
            self,
            carla_api,
            
    ):

        super(RouteGenerator, self).__init__(carla_api)

        # desired spawn point of ego vehicle
        # this waypoint is generated by generate_route, with a selection of route
        self.spawn_waypoint = None  # carla.Waypoint
        # actual end waypoint of the
        self.end_waypoint = None

        # junction of current scenario
        self.junction = None

    def set_route_distance(self, route_distance: tuple):
        """
        Set route distance manually.
        """
        self.route_distance = route_distance

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

        self.junction = junction

        return junction

    @staticmethod
    def get_start_transform(world, junction, distance, debug=True):
        """
        todo fix this method

        This method is developed based on classmethod get_start_waypoints.

        Assumption:
         - A 4-direction intersection.
         - Enough length of each road connected to the intersection
         -


        :param world: current carla world
        :param junction: carla.Junction
        :param distance: distance to the junction edge
        :param debug: visualization
        :return: dict contains all available start transform(carla.Transform)
        """

        # get carla API
        map = world.get_map()
        debug_helper = world.debug

        bbox = junction.bounding_box
        location = bbox.location
        extent = bbox.extent
        # todo if rotation on junction
        rotation = bbox.rotation  # original rotation of the b box

        # plot the junction
        # transform of the junction center
        transform = carla.Transform(location, rotation)
        # todo use plot coord frame to plot

        if debug:
            draw_waypoint(world, transform)
            # junction bounding box
            debug_helper.draw_box(box=bbox,
                                  rotation=rotation,
                                  thickness=0.5,
                                  color=red,
                                  life_time=-1.0)

        lane_width = map.get_waypoint(location).lane_width

        # start location
        # sequence as [+x -x +y -y] according to local rotation
        x_shift = -1. * lane_width
        x_slack = 0.75  # slack to fit different junction
        x_start = np.array([
            location.x + (extent.x + distance),
            location.x - (extent.x + distance),
            location.x + x_slack * lane_width + x_shift,
            location.x - x_slack * lane_width + x_shift,
        ])

        y_shift = 0. * lane_width
        y_slack = 0.75
        y_start = np.array([
            location.y - y_slack * lane_width + y_shift,
            location.y + y_slack * lane_width + y_shift,
            location.y + (extent.y + distance),
            location.y - (extent.y + distance),
        ])

        # plot the fixed coord center, for locate correct lane
        loc = carla.Location(x=location.x + x_shift,
                             y=location.y
                             )  # default z=0.

        trans = carla.Transform(loc, rotation)

        if debug:
            draw_waypoint(world, trans, color=(white, white))

        # generate waypoints first
        start_waypoints = []
        for i in range(4):
            start_location = carla.Location(x=x_start[i], y=y_start[i], z=0)
            start_waypoint = map.get_waypoint(location=start_location,
                                              project_to_road=True,  # not in the center of lane(road)
                                              lane_type=carla.LaneType.Driving)

            # todo check start point by lane id
            lane_id = start_waypoint.lane_id

            start_waypoints.append(start_waypoint)

            if debug:
                draw_waypoint(world, start_waypoint)
                print('start waypoint is plot. coord: [',
                      start_waypoint.transform.location.x,
                      start_waypoint.transform.location.y,
                      start_waypoint.transform.location.z,
                      ']'
                      )

        # store transform into dict
        start_transform_dict = {
            'positive_x': start_waypoints[0].transform,
            'negative_x': start_waypoints[1].transform,
            'positive_y': start_waypoints[2].transform,
            'negative_y': start_waypoints[3].transform,
        }

        return start_transform_dict

    def get_start_waypoints(self, junction):
        """
        Get available start waypoints of a junction.

        Assuming a 4-direction junction.
        """

        bbox = junction.bounding_box
        location = bbox.location
        extent = bbox.extent
        # todo if rotation on junction
        rotation = bbox.rotation  # original rotation of the b box

        # plot the junction
        # transform of the junction center
        transform = carla.Transform(location, rotation)
        # todo use plot coord frame to plot
        draw_waypoint(self.world, transform)
        # bounding box
        self.debug_helper.draw_box(box=bbox,
                                   rotation=rotation,
                                   thickness=0.5,
                                   color=red,
                                   life_time=-1.0)

        lane_width = self.map.get_waypoint(location).lane_width

        # start location
        # sequence as [+x -x +y -y] according to local rotation
        x_shift = -1. * lane_width
        x_slack = 0.75  # slack to fit different junction
        x_start = np.array([
            location.x + (extent.x + self.route_distance[0]),
            location.x - (extent.x + self.route_distance[0]),
            location.x + x_slack * lane_width + x_shift,
            location.x - x_slack * lane_width + x_shift,
        ])

        y_shift = 0. * lane_width
        y_slack = 0.75
        y_start = np.array([
            location.y - y_slack * lane_width + y_shift,
            location.y + y_slack * lane_width + y_shift,
            location.y + (extent.y + self.route_distance[0]),
            location.y - (extent.y + self.route_distance[0]),
        ])

        # plot the fixed coord center, for locate correct lane
        loc = carla.Location(x=location.x + x_shift,
                             y=location.y
                             )

        trans = carla.Transform(loc, rotation)
        draw_waypoint(self.world, trans, color=(white, white))

        start_waypoints = []
        for i in range(4):
            start_location = carla.Location(x=x_start[i], y=y_start[i], z=0)
            start_waypoint = self.map.get_waypoint(location=start_location,
                                                   project_to_road=True,  # not in the center of lane(road)
                                                   lane_type=carla.LaneType.Driving)

            lane_id = start_waypoint.lane_id

            start_waypoints.append(start_waypoint)
            draw_waypoint(self.world, start_waypoint)
            print('start waypoint is plot. coord: [',
                  start_waypoint.transform.location.x,
                  start_waypoint.transform.location.y,
                  start_waypoint.transform.location.z,
                  ']'
                  )

        return start_waypoints

    def generate_route(self, start_waypoint, turn_flag=-1):
        """
        Generate a route by given start waypoint.

        param turn_flag: left -1, straight 0, right 1
        """
        # we set a small gap between spawn point and route beginning waypoint
        self.spawn_waypoint = start_waypoint
        # plot route start waypoint
        draw_waypoint(self.world, self.spawn_waypoint, color=(magenta, magenta))

        dist_gap = 3.0  # gap distance
        wp_choice = start_waypoint.next(dist_gap)  # return -> list: waypoint choice
        start_waypoint = wp_choice[0]  # assuming that next waypoint is not in junction

        # 1. get key waypoint, exit point of the junction
        # get exit waypoint of the junction, with a turn flag
        exit_waypoint = generate_target_waypoint(waypoint=start_waypoint, turn=turn_flag)
        draw_waypoint(self.world, exit_waypoint, color=(blue, yellow))

        # end_waypoint is the last point of the continuous route
        end_waypoint, _ = get_waypoint_in_distance(exit_waypoint, self.route_distance[1])
        draw_waypoint(self.world, end_waypoint, color=(magenta, yan))

        # additional method to return the complete list of the route in junction
        # return format: (list([waypoint, RoadOption]...), destination_waypoint)
        # wp_list, target_waypoint = generate_target_waypoint_list(waypoint=start_wapoints[2], turn=-1)

        # 2. generate a route by the keypoint
        # this method requires list of carla.Location format
        waypoint_list = [start_waypoint.transform.location,
                         exit_waypoint.transform.location,
                         end_waypoint.transform.location,
                         ]
        # gps_route is not used in current experiment
        gps_route, route = interpolate_trajectory(world=self.world,
                                                  waypoints_trajectory=waypoint_list,
                                                  hop_resolution=1.0)

        # get an extra end point to avoid buffer being empty
        self.end_waypoint = end_waypoint.next(dist_gap)[-1]  # carla.Waypoint
        route.append((self.end_waypoint.transform, route[-1][1]))  # route[-1][1] RoadOption

        # plot generated route
        for i, item in enumerate(route):
            trans = item[0]
            draw_waypoint(self.world, trans, color=(green, green))

        # plot actual end point of the route
        for wp in [self.spawn_waypoint, self.end_waypoint]:
            trans = wp.transform
            draw_waypoint(self.world, trans, color=(red, red))

        return route

    def get_route(self, junction_center, route_option='left'):
        """
        This is the API method for a gym env to get single route
        via route option.

        Route option is responsible for determining start point
        and turning direction.

        There are 2 route of straight task.

        :param junction_center: location, todo need to fix api
        :param route_option: []
        :return:
        """
        # junction
        junction = self.get_junction_by_location(junction_center)

        # get all available spawn point around the junction
        # start_waypoints is the original spawn point, 4 in total
        start_waypoints = self.get_start_waypoints(junction)

        # todo improve this part
        # additional start waypoint on +y and -y direction
        # +y direction
        positive_y_1_loc = start_waypoints[2].transform.location
        positive_y_1_loc.x += start_waypoints[2].lane_width
        positive_y_1_wp = self.map.get_waypoint(positive_y_1_loc, project_to_road=True)
        # -y direction
        # this is for the traffic flow on right turning lane
        negative_y_1_loc = start_waypoints[3].transform.location
        negative_y_1_loc.x -= start_waypoints[3].lane_width
        negative_y_1_wp = self.map.get_waypoint(negative_y_1_loc, project_to_road=True)

        # get route, start and end point through route option
        if route_option == 'right':
            start_waypoint = positive_y_1_wp
            turn_flag = 1
        elif route_option == 'straight_0' or route_option == 'straight':
            start_waypoint = start_waypoints[2]
            turn_flag = 0
        elif route_option == 'straight_1':
            start_waypoint = positive_y_1_wp
            turn_flag = 0
        elif route_option == 'left':
            start_waypoint = start_waypoints[2]
            turn_flag = -1
        else:
            raise ValueError('Route option is wrong for the route manager.')

        # todo improve api, if wish to start from other direction of junction
        # selection route by selecting transform in list
        # route is a list of tuple (carla.Transform, RoadOption)
        route = self.generate_route(start_waypoint=start_waypoint,
                                    turn_flag=turn_flag,
                                    )

        return route, self.spawn_waypoint, self.end_waypoint


def test():

    env = BasicEnv()

    env.set_spectator_overhead(default_junction_center, yaw=-90, h=70)

    rou_gen = RouteGenerator(env.get_env_api())

    rou_gen.set_route_distance((20, 10))

    junction = rou_gen.get_junction_by_location(default_junction_center)
    start_waypoints = rou_gen.get_start_waypoints(junction)

    negative_y_transform = start_waypoints[3].transform

    # route, spawn_waypoint = rou_gen.get_route(default_junction_center)

    wp_2 = env.map.get_waypoint(carla.Location(x=-9.721644, y=98.222672, z=0.000000))

    draw_waypoint(env.world, wp_2, color=(magenta, magenta))

    env.world.tick()

    print('d')


if __name__ == '__main__':
    test()
