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
        high = np.array([1])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = gym.spaces.Discrete(100)
        self.state = 0

    def step(self, action):
        done = False
        if action == 1:
            self.state = self.state + 1
            reward = 1
            if self.state == 5:
                done = True
        elif action == 0:
            self.state = self.state + 0
            reward = 0.5
        else:
            self.state = self.state - 1
            reward = -1


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
