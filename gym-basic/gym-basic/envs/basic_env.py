import gym
from stable_baselines.common.env_checker import check_env

env = gym.make("gym_basic:basic-v0")

class BasicEnv(gym.Env):

    def __init__(self):
            self.action_space = gym.spaces.Discrete(5)
            self.observation_space = gym.spaces.Discrete(2)
            
    def step(self, action):
            state = 1
    
            if action == 2:
                reward = 1
            else:
                reward = -1
            
            done = True
            info = {}
            return state, reward, done, info

    def reset(self):
            state = 0
            return state

check_env(env)
