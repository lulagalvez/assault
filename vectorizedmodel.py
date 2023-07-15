import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

# PPO = False si se desea utilizar A2C
_PPO = False
TIMESTEPS = 400_000
MODEL_NAME = "A2C_400k_18"
LEARNING_RATE = 0.0007
GAMMA = 0.995
ENT_COEF = 0.0
VF_COEF = 0.5
N_FRAMES = 4

env = make_atari_env("ALE/Assault-v5", n_envs=N_FRAMES, seed=0)
env = VecFrameStack(env, n_stack=N_FRAMES)

log_path = os.path.join('Training', 'Logs', 'new_logs')

if _PPO:
    model = PPO('CnnPolicy', env,
                verbose=1,
                tensorboard_log=log_path,
                learning_rate=LEARNING_RATE,
                )
else:
    model = A2C('CnnPolicy', env,
                verbose=1,
                tensorboard_log=log_path,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF
                )

model.learn(total_timesteps=TIMESTEPS)

if _PPO:
    model_path = os.path.join('Training', 'Saved Models', 'ppo', MODEL_NAME)
else:
    model_path = os.path.join('Training', 'Saved Models', 'a2c', MODEL_NAME)

model.save(model_path)