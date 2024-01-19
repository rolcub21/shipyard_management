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
#eval_env = make_vec_env('gym_examples/GridWorld-v0', n_envs=4)




#env = RewardWrapper(env)
#obs = env.reset()
#num_steps = 300

#def on_press(key):
    #try:
        #if key.char in ['0', '1', '2', '3', '4', '5']:
            #action = int(key.char)
            #print(action)
            #observation, reward, done, info, trunc = env.step(action)
            #if done:
                #observation = env.reset()

    #except AttributeError:
        #print("Wrong key")

#def on_release(key):
    #pass

#Collect events until released
#with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    #listener.join()




#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
render_mode = "None"
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=dir)
#model = PPO.load("best_model_20240118-160043.zip", env=env)
model.learn(total_timesteps=800000,  progress_bar=True)

model.save("best_model_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
del model

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -





