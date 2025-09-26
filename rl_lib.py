from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sd_gym import core, env as env_lib
import numpy as np
from stable_baselines3.common.logger import configure
import gymnasium as gym

class FlattenDictWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        flat_space = gym.spaces.flatten_space(env.action_space)
        
        # Set finite bounds for the action space
        # Using -1 to 1 as standard bounds, which will be scaled appropriately
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=flat_space.shape,
            dtype=np.float32
        )
        
        # Store original bounds for scaling
        self.original_low = flat_space.low
        self.original_high = flat_space.high
        
    def step(self, action):
        # Scale the action from [-1, 1] to original bounds
        scaled_action = self._scale_action(action)
        unflattened_action = gym.spaces.unflatten(self.env.action_space, scaled_action)
        return super().step(unflattened_action)
    # Function to scale the action from [-1, 1] to original bounds
    def _scale_action(self, action):
        scaled_action = action.copy()
        for i in range(len(action)):
            scaled_action[i] = (
                (action[i] + 1.0) * (self.original_high[i] - self.original_low[i]) / 2.0
                + self.original_low[i]
            )
        return scaled_action

class CustomSDEnv(env_lib.SDEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        custom_reward = -np.sum(obs**2) + np.sum(action**2)
        return obs, custom_reward, done, info

# Configure PySD environment and wrap it in DummyVecEnv using Stable Baselines3
def initialize_environment(sd_model_filename):
    pysd_params = core.Params(sd_model_filename, env_dt=1.0, sd_dt=.1, simulator='PySD')
    pysd_sd_env = FlattenDictWrapper(CustomSDEnv(pysd_params))
    return DummyVecEnv([lambda: pysd_sd_env])

# Train an RL agent using PPO2 and log the output to TensorBoard
def train_agent(env, total_timesteps=10000):
    model = PPO('MultiInputPolicy', env, verbose=1)
    logger = configure("logs/", ["stdout", "tensorboard"])
    model.set_logger(logger)
    model.learn(total_timesteps=total_timesteps)

