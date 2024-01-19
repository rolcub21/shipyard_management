import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium import ActionWrapper

class CustomActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(9)


        self.new_action_mapping = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 1]), # Example mapping, modify as needed
            5: np.array([1, 0]),
            6: np.array([-1, 0]),
            7: np.array([0, -1]),
        }

    def action(self, action):
        
        action = self.new_action_mapping[action]
        return action


