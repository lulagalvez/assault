import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os


env = make_atari_env("ALE/Assault-v5", n_envs=4, seed=0)

env = VecFrameStack(env, n_stack=4)

log_path = os.path.join('Training', 'Logs')

model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=3000000) 

A2C_Path = os.path.join('Training', 'Saved Models', 'A2C_Model_Assault_3M')

model.save(A2C_Path)