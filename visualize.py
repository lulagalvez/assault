import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import time

env = make_atari_env("ALE/Assault-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

a2c_path = os.path.join('Training', 'Saved Models', 'PPO_400k')

model = PPO.load(a2c_path, env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(.08)

env.close()

