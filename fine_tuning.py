#Run it out of any environment for rendering function to work through the virtual machine
import gymnasium as gym
import numpy as np
import sys
import random
import datetime
import gym_examples
from gym import ActionWrapper
from gym_examples.wrappers.customactionwrapper import CustomActionSpaceWrapper
from gym_examples.wrappers.reward_wraper import RewardWrapper

#from pynput import keyboard
from stable_baselines3 import A2C, DQN, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

env = make_vec_env('gym_examples/GridWorld-v0', n_envs=4)

model = PPO.load("best_model_20240119-133849.zip", env=env)
model.learn(total_timesteps=600000,  progress_bar=True)

model.save("best_model_20240119-133849.zip")
del model
