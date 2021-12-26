import gym
import stable_baselines
import torch
from agent import DQN_Agent
from tqdm import tqdm
import random
from torch import randint

seed = 1432

env = gym.make('basic-v0')
#env = gym.make('gym_basic:basic-v0')

episode_count = 10
rew_arr = []

print("Action space: "+str(env.action_space.shape[0]))

for i in range(episode_count):
    obs, done, rew = env.reset(), False, 0
    while (done != True):
        A = randint(-1, env.action_space.shape[0]+1, (1,))
        print("Random number: "+str(A))
        obs, reward, done, info = env.step(A.item())
        rew += reward
        print("Reward: "+str(rew))
        print("Observation: "+str(obs))
    rew_arr.append(rew)

print("average reward per episode :", sum(rew_arr) / len(rew_arr))
