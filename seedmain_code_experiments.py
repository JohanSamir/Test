import numpy as np
import os
import dopamine
#from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import sys

import matplotlib
from dqn_agent_new import *
#from rainbow_agent_new import *
#from quantile_agent_new import *
#from implicit_quantile_agent_new import *
import networks_new
import external_configurations
import run_experiment_new

agents = {
    'dqn': JaxDQNAgentNew,
    #rainbow': JaxRainbowAgentNew,
    #'quantile': JaxQuantileAgentNew,
    #'implicit': JaxImplicitQuantileAgentNew,
}

'''
inits = {
    'conf_1': {'neurons':128,'gamma':0.9},
    'conf_2': {'neurons':128,'gamma':0.99},
    'conf_3': {'neurons':256,'gamma':0.9},
    'conf_4': {'neurons':256,'gamma':0.99},
    'conf_5': {'neurons':512,'gamma':0.9},
    'conf_6': {'neurons':512,'gamma':0.99}
    }
'''

'''
inits = {
    'conf_1': {'neurons':128,'gamma':0.9},
    'conf_2': {'neurons':256,'gamma':0.9},
    'conf_3': {'neurons':512,'gamma':0.9},
    }
'''

num_runs = 8
#environments = ['cartpole', 'acrobot','lunarlander','mountaincar']
environments = ['grid_xl_1']
#seeds = [True, False]
seeds = [True]
path= '/home/johan/ICRC/Hindsight_Q_Learning_gridxl_copy1/'

#for seed in seeds:
for agent in agents:
  for env in environments:
  #for init in inits:
      for i in range (1, num_runs + 1):  
        
        def create_agent(sess, environment, summary_writer=None):
          return agents[agent](num_actions=environment.action_space.n)

        #agent_name = agents[agent].__name__
        LOG_PATH = os.path.join(f'{path}{i}_{agent}_{env}', f'dqn_test{i}')
        sys.path.append(path)    
        gin_file = f'{path}{env}.gin'

        #neurons = inits[init]['neurons']
        #gamma = inits[init]['gamma']
        #print('neurons:',neurons,'gamma:',gamma)

        '''
        gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                        f"{agent_name}.neurons = {neurons}",
                        f"{agent_name}.gamma = {gamma}"]
        gin.clear_config()
        gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
        '''
        gin.parse_config_file(gin_file)
        agent_runner = run_experiment_new.TrainRunner_new(LOG_PATH, create_agent)

        print(f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...')
        agent_runner.run_experiment()
        print('Done training!')
print('Finished!')
