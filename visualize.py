import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import time

_PPO = False
N_FRAMES = 4

MODEL_NAME = input("Enter name of model: ")

env = make_atari_env("ALE/Assault-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=N_FRAMES)

if _PPO:
    model_path = os.path.join('Training', 'Saved Models','ppo', MODEL_NAME)
    model = PPO.load(model_path, env)
else:
    model_path = os.path.join('Training', 'Saved Models','a2c', MODEL_NAME)
    model = A2C.load(model_path, env)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(.08)

env.close()

