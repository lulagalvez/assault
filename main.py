import gymnasium as gym
from stable_baselines3 import a2c
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

env = gym.make("ALE/Assault-v5", render_mode="human")
observation, info = env.reset()

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0
    
    while not terminated:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        print('Episode: {} Score: {}'. format(episode, score))
#env.close() 