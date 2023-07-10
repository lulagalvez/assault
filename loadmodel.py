import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

n_showcase_episodes = 3

for episode in range(n_showcase_episodes):
    print(f"starting episode {episode}...")

    env = gym.make("ALE/Assault-v5", render_mode="human")

    env = VecFrameStack(env, n_stack=4)

    print(env.observation_space)
    
    PPO_Path = os.path.join('Training', 'Saved Models','old', 'test_run')
    
    state, info = env.reset(seed=0)

    model = PPO.load(PPO_Path, env=env)

    # get an initial state

    # play one episode
    done = False
    score = 0
    while not done:
        action, _states = model.predict(state) # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        print('Episode: {} Score: {}'. format(episode, score))
        
        done = terminated or truncated

env.close()