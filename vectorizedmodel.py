import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

learning_rate = 0.0008


env = make_atari_env("ALE/Assault-v5", n_envs=4, seed=0)

env = VecFrameStack(env, n_stack=4)

log_path = os.path.join('Training', 'Logs', 'new_logs')

model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path, learning_rate=learning_rate)

model.learn(total_timesteps=500000) 

A2C_Path = os.path.join('Training', 'Saved Models', 'a2c', 'A2C_LR_08_500k')

model.save(A2C_Path)