#!pip install dm-env
#!pip install dopamine-rl==3.1.10

import numpy as np
import os

import dopamine
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf

import matplotlib
import time

start = time.time()

path = '/home/johan/ICRC/Dopamine_DQN_IRCR_grid_xl/'
#LOG_PATH = os.path.join(path, 'dqn')
#LOG_PATH = os.path.join(path, 'dqn_test')

import sys
sys.path.append(path)

from dqn_agent_new import*
import run_experiment_new

num_runs = 8 
for i in range(1, num_runs + 1):

  def create_random_dqn_agent(sess, environment, summary_writer=None):
    """The Runner class will expect a function of this type to create an agent."""
    return JaxDQNAgentNew(num_actions=environment.action_space.n)

  gin.parse_config_file(path+'/grid_xl_1.gin')
  #gin.parse_config_file(path+'/dqn_cartpole.gin')
  LOG_PATH = os.path.join(path, f'dqn_test{i}')
  random_dqn_runner = run_experiment_new.TrainRunner_new(LOG_PATH, create_random_dqn_agent)

  print('Will train agent, please be patient, may be a while...')
  random_dqn_runner.run_experiment()
  print('Done training!')

end = time.time()
seconds = end - start

minutes, seconds = divmod(seconds, 60)
hours, minutes = divmod(minutes, 60)
print('seconds:',seconds, 'hour:',hours, 'minute:', minutes)
