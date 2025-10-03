import os
from urllib import request
from rl_lib import initialize_environment, train_agent, evaluate_agent
import pysd
import logging
from datetime import datetime
from sd_gym.core import RewardFn
import numpy as np
from get_model_info import analyze_model
from stable_baselines3.common.evaluation import evaluate_policy


# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a log file with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'training_{timestamp}.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will still print to console
    ]
)

logger = logging.getLogger(__name__)

# Custom reward function for EV adoption model
class EVAdoptionReward(RewardFn):
    def __init__(self):
        self.prev_ev_adoption = 0.0
        self.steps = 0
        self.logger = logging.getLogger(__name__)
        self.reward_threshold = 0.85
        
    def __call__(self, observation):
        try:
            # Calculate EV adoption rate
            ec_in_use = float(observation.get('ec_in_use', 0.0))
            pc_in_use = float(observation.get('pc_in_use', 0.0))
            total_cars = ec_in_use + pc_in_use
            
            # Calculate current EV adoption rate
            ev_adoption = ec_in_use / total_cars if total_cars > 0 else 0.0
            ev_adoption = max(0.0, min(1.0, ev_adoption))
            
            # Make reward directly proportional to EV adoption with penalties
            reward = ev_adoption  # Base reward is just the adoption rate
            
            # Apply strong penalties for low adoption
            if ev_adoption < 0.01:  # Less than 1% adoption
                reward = -0.5  # Strong negative reward
            elif ev_adoption < 0.05:  # Less than 5% adoption
                reward = -0.25  # Moderate negative reward
            
            # Small improvement bonus (only if adoption is non-zero)
            improvement = ev_adoption - self.prev_ev_adoption if self.steps > 0 else 0.0
            if ev_adoption > 0.01 and improvement > 0:
                reward += 0.1 * improvement
            
            self.logger.debug(
                f"EV Adoption: {ev_adoption:.4f}, "
                f"Raw Reward: {reward:.4f}"
            )
            
            # Update previous values
            self.prev_ev_adoption = ev_adoption
            self.steps += 1
            
            return float(np.clip(reward, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Reward calculation error: {str(e)}")
            return -1.0  # Stronger penalty for errors
    
    def reset(self, observation):
        """Reset reward state"""
        self.prev_ev_adoption = 0.0
        self.steps = 0
        return self

# Rest of your imports and constants
SD_MODEL_URL = 'https://exchange.iseesystems.com/model/mindsproject/electric-vehicles-in-norway'
SD_MODEL_FILE_PATH = '/tmp/electric_vehicles_norway.stmx'
GENERATED_MODEL_PATH = '/tmp/generated_sd_models'

if __name__ == '__main__':
    try:
        # Download and modify the model file
        logger.info("Downloading model file...")
        sd_model_filename, _ = request.urlretrieve(SD_MODEL_URL, SD_MODEL_FILE_PATH)
        if not os.path.exists(GENERATED_MODEL_PATH):
            os.mkdir(GENERATED_MODEL_PATH)

        # PySD doesn't support random, so replace with a fixed value
        logger.info("Modifying model file...")
        with open(sd_model_filename, 'r') as f:
            data = f.read()
            data = data.replace('RANDOM(0.98, 1.02)', '0.99')

        with open(sd_model_filename, 'w') as f:
            f.write(data)

        # Analyze model structure
        logger.info("Analyzing model structure...")
        all_vars, constants, stocks = analyze_model(sd_model_filename)

        # Separate initial conditions into stocks and parameters
        stocks = {
            'ec_in_use': 1840.0,
            'pc_in_use': 2194267.0,
            'oil_reserves': 7.40333e9,
            'unproven_reserves_of_oil': 1.24542e10
        }

        # Move other variables to params
        params = {
            'costs_of_1_km_ec': 0.5625,
            'charging_stations_per_km_of_road': 0.00105876,
            'charging_stations': 100.0,  # Moved from stocks
            'overpaym_effect': 311.685,
            'fuel_price': 10.6,
            'change_in_price_of_fuel': 0.92173913,
            'price_of_pc': 532500.0,
            'demand_on_charging_stations': 60.72,
            'ec_sales': 3151.7724468,
            'oil_industry_confidence_in_tomorrow': 1.0,
            'oil_production': 4.18866556e8
        }

        # Define variable limits for continuous variables
        var_limits = {
            'km_per_one_battery': (10, 1000),          # range of electric cars
            'kwh_per_battery': (10, 100),              # charge for battery
            'electricity_price': (1, 10),              # cost of electricity
            'price_pc_without_taxes': (10000, 70000),  # price of petrol cars
            'price_ec_without_taxes': (20000, 100000), # price of electric cars
            'gov_policy_on_taxes': (0, 1),             # government tax incentives
        }

        # Define categorical variables with their valid values
        categorical_vars = {
            'average_car_lifetime': [1, 3, 5, 7, 10, 15],  # discrete years
            'vat': [0.15, 0.3, 0.44, 0.5],  # discrete tax rates
            # Reduce the range and step size for oil_industry_capacity
            'oil_industry_capacity': list(range(1, 31))  # 0 to 30 in steps of 1
        }

        # Initialize environment with both continuous and categorical variables
        logger.info("Initializing environment...")
        reward_function = EVAdoptionReward()
        env = initialize_environment(
            sd_model_filename,
            var_limits=var_limits,
            categorical_vars=categorical_vars,
            initial_conditions=stocks,
            reward_function=reward_function
        )
        
        logger.info(f"Training with reward threshold: {reward_function.reward_threshold}")
        logger.info("Starting training...")
        model = train_agent(env, logger=logger, total_timesteps=10000)
        
        logger.info("Training completed successfully")
        
        # Use the new evaluation function
        eval_results = evaluate_agent(model, env, logger)
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)