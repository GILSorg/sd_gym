from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from sd_gym import core, env as env_lib
import numpy as np
import gymnasium as gym
import logging
from datetime import datetime
import os
import torch
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

def setup_debug_logging():
    """Set up logging configuration for debugging action scaling"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_log_file = os.path.join(log_dir, f'action_debug_{timestamp}.log')

    # Configure logger
    logger = logging.getLogger('action_debug')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(debug_log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class FlattenDictWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.logger = setup_debug_logging()
        self.original_action_space = env.action_space
        self.categorical_vars = env.params.categorical_sd_vars

        # Create a flattened Box space instead of Dict space
        self.total_dims = 0
        self.var_indices = {}
        current_idx = 0

        # Map variables to indices in the flattened space
        for key, space in self.original_action_space.spaces.items():
            if key in self.categorical_vars:
                size = len(self.categorical_vars[key])
                self.var_indices[key] = (current_idx, current_idx + size)
                current_idx += size
            else:
                self.var_indices[key] = (current_idx, current_idx + 1)
                current_idx += 1
        
        self.total_dims = current_idx
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(self.total_dims,),
            dtype=np.float32
        )

    def _scale_action(self, action):
        """Convert flattened action back to Dict format"""
        scaled_actions = {}
        
        try:
            for key, (start_idx, end_idx) in self.var_indices.items():
                space = self.original_action_space.spaces[key]
                
                if key in self.categorical_vars:
                    # For categorical variables, get probabilities
                    probs = action[start_idx:end_idx]
                    idx = int(np.argmax(probs))  # Get most likely category
                    idx = np.clip(idx, 0, len(self.categorical_vars[key]) - 1)
                    
                    # Always store the index for categorical variables
                    scaled_actions[key] = np.array(idx, dtype=np.int32)
                    
                    self.logger.debug(f"Categorical {key}: value={self.categorical_vars[key][idx]}, "
                                    f"index={idx}, options={self.categorical_vars[key]}")
                else:
                    # For continuous variables, scale from [-1,1] to variable range
                    norm_action = np.clip(action[start_idx], -1.0, 1.0)
                    low, high = float(space.low[0]), float(space.high[0])
                    scaled_value = low + (high - low) * (norm_action + 1) / 2
                    scaled_value = np.clip(scaled_value, low, high)
                    scaled_actions[key] = np.array([float(scaled_value)], dtype=space.dtype)

            # Verify all values are within their spaces
            for key, value in scaled_actions.items():
                space = self.original_action_space.spaces[key]
                if not space.contains(value):
                    if key in self.categorical_vars:
                        # For categorical, ensure index is valid
                        valid_idx = np.clip(value.item(), 0, len(self.categorical_vars[key]) - 1)
                        scaled_actions[key] = np.array(valid_idx, dtype=np.int32)
                    else:
                        # For continuous, clip to space bounds
                        scaled_actions[key] = np.clip(
                            value, 
                            space.low, 
                            space.high
                        ).astype(space.dtype)

            self.logger.info("Step Actions: %s", scaled_actions)
            return scaled_actions
        
        except Exception as e:
            self.logger.error("Error in _scale_action: %s", str(e))
            raise

    def step(self, action):
        scaled_action = self._scale_action(action)
        return self.env.step(scaled_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class CustomSDEnv(env_lib.SDEnv):
    def __init__(self, params):
        super().__init__(params)
        self.var_limits = params.sd_var_limits_override
        self.logger = logging.getLogger(__name__)
        
        # Initialize state first
        self.reset()
        
        # Create observation space with wide bounds
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64
            ) for k in self.observables
        })

    def _process_observation(self, obs):
        """Process observation to match space requirements"""
        processed_obs = {}
        
        try:
            for key in self.observables:
                # Get the original observation value and ensure it's a scalar
                value = obs.get(key, 0.0)
                if isinstance(value, (np.ndarray, list)):
                    value = float(value[0]) if len(value) > 0 else 0.0
                elif isinstance(value, (int, float)):
                    value = float(value)
                else:
                    value = 0.0
                
                # Handle invalid values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                
                # Create array with correct shape and dtype
                processed_obs[key] = np.array([value], dtype=np.float64)
            
            return processed_obs
            
        except Exception as e:
            self.logger.error(f"Error processing observation: {str(e)}")
            return {key: np.array([0.0], dtype=np.float64) for key in self.observables}

    def step(self, action):
        """Override step to ensure proper observation and reward handling"""
        try:
            result = super().step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                obs, reward, done, truncated, info = result
            
            processed_obs = self._process_observation(obs)
            
            try:
                if self.reward_fn is not None:
                    scalar_obs = {k: float(v[0]) for k, v in processed_obs.items()}
                    reward = float(self.reward_fn(scalar_obs))
                    reward = np.clip(reward, -1.0, 1.0)
                    self.logger.info("Step Reward: %.4f", reward)
                    
                    # Add actions to info dictionary
                    info['last_actions'] = action
            except Exception as e:
                self.logger.error("Reward calculation error: %s", str(e))
                reward = -0.1
            
            return processed_obs, reward, done, truncated, info
            
        except Exception as e:
            self.logger.error("Error in step: %s", str(e))
            return (
                {key: np.array([0.0], dtype=np.float64) for key in self.observables},
                -0.1,
                True,
                False,
                {}
            )

# Configure PySD environment and wrap it in DummyVecEnv using Stable Baselines3
def initialize_environment(sd_model_filename, var_limits=None, categorical_vars=None, initial_conditions=None, params=None, reward_function=None):
    """Initialize parallel environments with vectorized processing"""
    def make_env():
        def _init():
            # Get actionable variables
            actionable_vars = []
            if var_limits:
                actionable_vars.extend(list(var_limits.keys()))
            if categorical_vars:
                actionable_vars.extend(list(categorical_vars.keys()))
            
            # Combine initial conditions
            all_initial_conditions = {}
            if initial_conditions:
                all_initial_conditions.update(initial_conditions)
            if params:
                all_initial_conditions.update(params)
            
            pysd_params = core.Params(
                sd_model_filename, 
                env_dt=1.0, 
                sd_dt=.1, 
                simulator='PySD',
                sd_var_limits_override=var_limits if var_limits is not None else {},
                categorical_sd_vars=categorical_vars if categorical_vars is not None else {},
                actionables=actionable_vars,
                initial_conditions_override=all_initial_conditions,
                reward_function=reward_function
            )
            env = FlattenDictWrapper(CustomSDEnv(pysd_params))
            return env
        return _init

    # Create single environment for evaluation
    env = DummyVecEnv([make_env()])
    
    # Add normalization wrapper with disabled reward normalization
    normalized_env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,  # Disable reward normalization
        clip_obs=10.0,
        clip_reward=1.0,    # Clip rewards to [-1,1]
        gamma=0.99
    )
    
    # Store reward threshold in the normalized environment
    if reward_function is not None and hasattr(reward_function, 'reward_threshold'):
        normalized_env.reward_threshold = reward_function.reward_threshold
    
    return normalized_env

def train_agent(env, logger=None, total_timesteps=10000):
    """Train an RL agent using optimized PPO configuration"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Custom logging callback that inherits from BaseCallback
        class EfficientLoggingCallback(BaseCallback):
            def __init__(self, logger, verbose=0):
                super().__init__(verbose)
                self.logger = logger
                self.step_count = 0
                self.episode_count = 0
                self.log_interval = 100

            def _on_step(self):
                """Called after each step"""
                try:
                    self.step_count += 1
                    
                    if self.logger and self.step_count % self.log_interval == 0:
                        # Get episode rewards if available
                        if len(self.model.ep_info_buffer) > 0:
                            mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                            self.logger.info(f"Training Step {self.step_count}: mean_reward={mean_reward:.4f}")
                        else:
                            self.logger.info(f"Training Step {self.step_count}")
                
                    # Log episode completion
                    if self.locals.get("done", False):
                        self.episode_count += 1
                        reward = self.locals.get("rewards", [0])[0]
                        self.logger.info(f"Episode {self.episode_count} completed - Final reward: {reward:.4f}")
                        
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error in callback: {str(e)}")
                    return True

            def _on_training_start(self):
                """Called at start of training"""
                self.logger.info("=== Starting Training ===")
                self.step_count = 0
                self.episode_count = 0

            def _on_training_end(self):
                """Called at end of training"""
                self.logger.info(f"=== Training Complete ===")
                self.logger.info(f"Total steps: {self.step_count}")
                self.logger.info(f"Total episodes: {self.episode_count}")

        class BestConfigCallback(BaseCallback):
            def __init__(self, logger, verbose=0):
                super().__init__(verbose)
                self.logger = logger
                self.best_reward = float('-inf')
                self.best_config = None
                self.best_config_step = 0
                self.last_actions = None
                self.current_reward = None
            
            def _on_step(self):
                try:
                    # Get current reward from rollout buffer
                    if len(self.model.ep_info_buffer) > 0:
                        self.current_reward = self.model.ep_info_buffer[-1]['r']
                        
                        # If we have actions and this is the best reward, save config
                        if self.last_actions is not None and self.current_reward > self.best_reward:
                            self.best_reward = self.current_reward
                            self.best_config = self.last_actions.copy()
                            self.best_config_step = self.num_timesteps
                            self.logger.info(f"\n=== New Best Configuration at Step {self.num_timesteps} ===")
                            self.logger.info(f"Reward: {self.best_reward:.4f}")
                            self.logger.info("Actions:")
                            for key, value in self.best_config.items():
                                self.logger.info(f"  {key}: {value}")
                    
                    # Store current actions for next step
                    if len(self.locals['infos']) > 0 and 'last_actions' in self.locals['infos'][0]:
                        self.last_actions = self.locals['infos'][0]['last_actions']
                        
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error tracking best config: {str(e)}")
                    return True
            
            def _on_training_end(self):
                """Log the best configuration at end of training"""
                if self.best_config is not None:
                    self.logger.info(f"\n=== Best Configuration Overall ===")
                    self.logger.info(f"Found at training step: {self.best_config_step}")
                    self.logger.info(f"Best Reward: {self.best_reward:.4f}")
                    self.logger.info("Best Actions Configuration:")
                    for key, value in self.best_config.items():
                        if key in self.training_env.get_attr('categorical_vars')[0]:
                            # For categorical vars, show both index and actual value
                            cat_values = self.training_env.get_attr('categorical_vars')[0][key]
                            actual_value = cat_values[int(value)]
                            self.logger.info(f"  {key}: index={value}, value={actual_value}")
                        else:
                            self.logger.info(f"  {key}: {value}")

        model = PPO(
            'MultiInputPolicy', 
            env,
            learning_rate=1e-3,          
            n_steps=512,                 
            batch_size=64,               
            n_epochs=5,                  
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            max_grad_norm=1.0,           
            ent_coef=0.005,             
            vf_coef=0.5,
            target_kl=0.02,
            use_sde=True,
            verbose=1,                   # Keep this for console output
            device=device,
            policy_kwargs={
                "net_arch": [64, 64],    
                "optimizer_class": torch.optim.AdamW,  
                "activation_fn": torch.nn.ReLU,
                "full_std": True,
                "log_std_init": -2,
            }
        )
        
        # Get reward threshold from environment
        reward_threshold_value = getattr(env, 'reward_threshold', 0.82)
        
        # Create callbacks with dynamic threshold
        reward_threshold = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold_value,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            env,
            callback_on_new_best=reward_threshold,
            eval_freq=1000,
            n_eval_episodes=10,
            verbose=1,
            best_model_save_path="./best_model/",
            log_path="./eval_logs/",
            deterministic=True
        )
        
        # Create logging callback instance
        logging_callback = EfficientLoggingCallback(logger) if logger else None
        best_config_callback = BestConfigCallback(logger) if logger else None
        
        # Combine callbacks
        callbacks = [
            eval_callback,
            logging_callback,
            best_config_callback
        ]
        
        # Create necessary directories
        os.makedirs("./best_model/", exist_ok=True)
        os.makedirs("./eval_logs/", exist_ok=True)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        return model
        
    except Exception as e:
        if logger:
            logger.error(f"Training failed: {str(e)}")
        raise

class EvalLoggingCallback(BaseCallback):
    """Custom callback for evaluation logging"""
    def __init__(self, logger, eval_freq=100, n_eval_episodes=5):
        super().__init__()
        self.logger = logger
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.training_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            self.logger.info(f"\n=== Evaluation at step {self.n_calls} ===")
            self.logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Track best performance
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.logger.info(f"New best mean reward: {mean_reward:.2f}")
            
        return True

def evaluate_agent(model, env, logger, n_eval_episodes=5):
    """Evaluate a trained agent and log detailed metrics."""
    try:
        logger.info("Evaluating trained policy...")
        
        # Track best configuration
        best_reward = float('-inf')
        best_config = None
        best_adoption = 0.0
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            reset_result = env.reset()
            obs = reset_result if isinstance(reset_result, dict) else reset_result[0]
            done = False
            episode_reward = 0
            episode_length = 0
            current_actions = None
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                
                # Handle both old and new gym API formats
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                
                # Handle info from vectorized environment (it's a list)
                if isinstance(info, list):
                    info = info[0] if info else {}
                
                # Store current actions
                current_actions = info.get('last_actions', None)
                    
                episode_reward += float(reward)
                episode_length += 1
                
                # Calculate EV adoption rate - handle vectorized observations
                if isinstance(obs, dict):
                    ec_in_use = float(obs.get('ec_in_use', [0])[0])
                    pc_in_use = float(obs.get('pc_in_use', [0])[0])
                elif isinstance(obs, (list, tuple)):
                    obs_dict = obs[0] if obs else {}
                    ec_in_use = float(obs_dict.get('ec_in_use', [0])[0])
                    pc_in_use = float(obs_dict.get('pc_in_use', [0])[0])
                else:
                    ec_in_use, pc_in_use = 0.0, 0.0
                    
                total_cars = ec_in_use + pc_in_use
                ev_adoption = ec_in_use / total_cars if total_cars > 0 else 0
                
                # Convert reward to float before comparison
                reward_float = float(reward)
                
                # Update best configuration if this is the best reward
                if reward_float > best_reward and current_actions is not None:
                    best_reward = reward_float  # Store as float
                    best_config = current_actions
                    best_adoption = ev_adoption
                
                logger.info(f"Episode {episode + 1}, Step {episode_length}:")
                logger.info(f"  Action taken: {action.tolist()}")
                logger.info(f"  Reward: {reward_float:.4f}")
                logger.info(f"  EV Adoption: {ev_adoption:.4f}")
            
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(episode_length)
            
        # Log best configuration found with explicit float conversion
        logger.info("\n=== Best Configuration During Evaluation ===")
        logger.info(f"Best Reward: {float(best_reward):.4f}")
        logger.info(f"Best EV Adoption Rate: {float(best_adoption):.4f}")
        logger.info("Best Actions Configuration:")
        if best_config:
            categorical_vars = env.get_attr('categorical_vars')[0]
            for key, value in best_config.items():
                if isinstance(value, np.ndarray):
                    value = value.item()  # Convert numpy array to scalar
                if key in categorical_vars:
                    actual_value = categorical_vars[key][int(value)]
                    logger.info(f"  {key}: index={value}, value={actual_value}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Log summary statistics
        logger.info("\nEvaluation Summary:")
        logger.info(f"Average episode reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        logger.info(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'best_reward': best_reward,
            'best_config': best_config,
            'best_adoption': best_adoption,
            'n_eval_episodes': n_eval_episodes,
            'average_episode_length': np.mean(episode_lengths),
            'total_timesteps': model.num_timesteps,
            'training_env': str(env),
            'model': str(model)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

