import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

log_path = os.path.join('Training', 'Logs')

env = gym.make("ALE/Assault-v5", render_mode="human")

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=1000) 

PPO_Path = os.path.join('Training', 'Saved Models', 'test_run')

model.save(PPO_Path)