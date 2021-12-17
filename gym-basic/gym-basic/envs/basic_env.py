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

    def render(slef, mode='human'):
            # Render the environment to the screen
            im = <obtain image from env>        
        
            if mode == 'human':    
                plt.imshow(np.asarray(im))
                plt.axis('off')
            elif mode == 'rgb_array':
                return np.asarray(im)

#env = gym.make("gym_basic:basic-v0")
env = BasicEnv()
check_env(env)

action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
print(q_table)


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()

for i in range(10):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
