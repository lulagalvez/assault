import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

env = make_atari_env("ALE/Assault-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

A2C_Path = os.path.join('Training', 'Saved Models', 'A2C_Model_Assault_3M')

model = A2C.load(A2C_Path, env=env)

res = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print(f"Result: {res}")








