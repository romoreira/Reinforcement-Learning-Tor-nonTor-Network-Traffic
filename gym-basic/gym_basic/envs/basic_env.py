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


class BasicEnv(gym.Env):

    def __init__(self):
        self.pooling_max = 100
        self.pooling_min = 1
        self.classification_acc_max = 100
        self.classification_acc_min = 0
        
        low = np.array([self.pooling_min, self.classification_acc_min], dtype=np.float32,)
        high = np.array([self.pooling_max, self.classification_acc_max], dtype=np.float32,)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        
        self.state = None

    def step(self, action):
        
        if action == 1:
            self.state = self.state + 1
            reward = 1
        elif action == 0:
            self.state = self.state + 0
            reward = 0.5
        
        done = True

        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = 0
        return self.state

    def render(slef, mode='human'):

        if mode == 'human':
            plt.imshow(np.asarray(im))
            plt.axis('off')
        elif mode == 'rgb_array':
            return np.asarray(im)
