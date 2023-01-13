"""
todo roll out policy requires modification

Roll out policy for env run loop.
"""

import carla

import numpy as np


class BasePolicy:
    """
    todo fix this as the most general roll out policy

    Base class for roll out policy.
    """
    def __init__(self, action_space):

        # action_space refers to a gym.Box instance
        self.action_space = action_space

        # # todo retrieve dimension bound
        # self.space_dim =
        # self.upper_bound =
        # self.lower_bound =

        # ndarray
        self.action = None

    def predict(self, state):
        """
        Default input is state array.

        :param state:
        :return: action
        """

        action = None

        return action


class RandomPolicy(BasePolicy):

    def __init__(self):


    def predict(self, state):
        """
        todo add action space dim check

        :param state:
        :return:
        """

        action =

        return action



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


# todo finish this class with RL model loading
class RLPolicy(BasePolicy):

    def __init__(self):

        pass


