import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info= self.env.step(action)  # Get the original results from the environment
        agent_location = obs['agent']
        # Custom reward logic
        distance = info['distance']
        if terminated:
           reward = 1 

        
       #elif np.array_equal(agent_location, np.array([self.env.size_rows, 0])):
            #reward = 0.1
            #print("it's working")
        #elif np.array_equal(agent_location, np.array([self.env.size_rows, 3])):
            #reward = 0.3

        else:
            reward = -(distance*0.5)

        return obs, reward, terminated, truncated, info


# Now you can use wrapped_env as you would use a regular environment
