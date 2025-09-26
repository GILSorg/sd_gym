from stable_baselines3 import PPO2
from stable_baselines3.common.vec_env import DummyVecEnv
from sd_gym import core, env as env_lib
import numpy as np

class CustomSDEnv(env_lib.SDEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        custom_reward = -np.sum(obs**2) + np.sum(action**2)
        return obs, custom_reward, done, info

def initialize_environment(sd_model_filename):
    pysd_params = core.Params(sd_model_filename, env_dt=1.0, sd_dt=.1, simulator='PySD')
    pysd_sd_env = CustomSDEnv(pysd_params)
    return DummyVecEnv([lambda: pysd_sd_env])

def train_agent(env, total_timesteps=10000):
    model = PPO2('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)