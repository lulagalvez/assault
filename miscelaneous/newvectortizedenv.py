from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C


vec_env = make_atari_env( "ALE/Assault-v5", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)


model = A2C("CnnPolicy", vec_env, verbose=1, learning_rate=0.0006)
model.learn(total_timesteps=5000)

vec_env = make_atari_env("ALE/Assault-v5", n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)



obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")