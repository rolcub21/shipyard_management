#Run it out of any environment for rendering function to work through the virtual machine
import gymnasium as gym
import numpy as np
import sys
import time
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



vec_env = make_vec_env('gym_examples/GridWorld-v0') 
model = PPO.load("best_model_20240119-145021.zip", env=vec_env)

num_episodes = 30  # Number of episodes to evaluate


for episode in range(num_episodes):
    obs = vec_env.reset()


    while True:  # Check if any environment is done
        actions, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(actions)
        vec_env.render("human")
        time.sleep(0.15)

vec_env.close()