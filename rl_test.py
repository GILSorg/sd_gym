import os
from urllib import request
from rl_lib import initialize_environment, train_agent

SD_MODEL_URL = 'https://exchange.iseesystems.com/model/mindsproject/electric-vehicles-in-norway'
SD_MODEL_FILE_PATH = '/tmp/electric_vehicles_norway.stmx'
GENERATED_MODEL_PATH = '/tmp/generated_sd_models'

sd_model_filename, _ = request.urlretrieve(SD_MODEL_URL, SD_MODEL_FILE_PATH)
if not os.path.exists(GENERATED_MODEL_PATH):
    os.mkdir(GENERATED_MODEL_PATH)

# PySD doesn't support random, so replace with a fixed value
with open(sd_model_filename, 'r') as f:
    data = f.read()
    data = data.replace('RANDOM(0.98, 1.02)', '0.99')

with open(sd_model_filename, 'w') as f:
    f.write(data)

env = initialize_environment(sd_model_filename)
train_agent(env)