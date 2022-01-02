'''
Auhor: Rodrigo Moreira rodrigo at ufv dot br
Based on: https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952
'''


import gym
from stable_baselines.common.env_checker import check_env
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import matplotlib.pyplot as plt
import random
from gym.spaces import Discrete
from gym.spaces import Box

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/rodrigo/adaptative-monitoring/')
from pooling import main


class BasicEnv(gym.Env):

    def __init__(self):
        #print("Creating environment Adaptative Sampling")
        self.action_space = Discrete(2000)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 3 + random.randint(-3,3)
        #print("\ninit: "+str(self.state)+"\n")
        self.pooling_times = 10
    def step(self, action):
        print("\nStep Action Required: "+str(action))
        self.state = main(action, 2, 'enp0s3')
        print("\nNew State after pooling: "+str(self.state))
        self.pooling_times -= 1

        print("pooling times: "+str(self.pooling_times))

        if self.state >= 90:#If IoT sampling is bigger than 90% that is correct
            reward = 1
        else:
            reward = -1

        if self.pooling_times <= 0:
            done = True
            self.pooling_times = 10
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = 38 + random.randint(-3,3)
        self.shower_lenght = 60
        return self.state

    def render(slef, mode='human'):

        if mode == 'human':
            plt.imshow(np.asarray(im))
            plt.axis('off')
        elif mode == 'rgb_array':
            return np.asarray(im)
