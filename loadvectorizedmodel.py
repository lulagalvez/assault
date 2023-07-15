import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

_PPO = False
N_EPISODES = 10
N_FRAMES = 4

model_name = input("Ingrese nombre del modelo: ")

env = make_atari_env("ALE/Assault-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=N_FRAMES)

if _PPO:
    model_path = os.path.join('Training', 'Saved Models', 'ppo', model_name)
    model = PPO.load(model_path, env=env)
else:
    model_path = os.path.join('Training', 'Saved Models', 'a2c', model_name)
    model = A2C.load(model_path, env=env)

res = evaluate_policy(model, env, n_eval_episodes=N_EPISODES, render=True)

print(f"Result: {res}")








