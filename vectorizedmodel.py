import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

learning_rate = 0.00025
gamma = 0.99
ent_coef = 0.1


env = make_atari_env("ALE/Assault-v5", n_envs=4, seed=0)

env = VecFrameStack(env, n_stack=4)

log_path = os.path.join('Training', 'Logs', 'new_logs')

model = PPO('CnnPolicy', env,
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            n_epochs=4,
            batch_size=32,
            vf_coef=1,
            gae_lambda=0.95,
            clip_range=0.5
            )

model.learn(total_timesteps=1000000) 

model_path = os.path.join('Training', 'Saved Models', 'ppo', 'PPO_LR00025_1M')

model.save(model_path)