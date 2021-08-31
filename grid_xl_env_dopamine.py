"""Custom key_to_door environment made compatible for Dopamine."""

from dopamine.discrete_domains import atari_lib
from flax import nn
import gin 
import jax 
import jax.numpy as jnp 
import grid_xl #Custom key_to_door_5x5

#gin.constant('grid_xl_env_dopamine.KEYTODOOR_SHAPE', (2500,1))
gin.constant('grid_xl_env_dopamine.KEYTODOOR_SHAPE', (1,1)) 
gin.constant('grid_xl_env_dopamine.DTYPE', jnp.float64)


class grid_xl_Env(object):
  def __init__(self):
    self.env = grid_xl.GridXL()
    #self.env = key_to_door_modi.KeyToDoorEnv()
    self.env.n = self.env.num_actions
    self.game_over = False

  @property
  def observation_space(self):
    return self.env.num_states

  @property
  def action_space(self):
    return self.env # Only used for the `n` parameter.

  @property
  def reward_range(self):
    pass  # Unused

  @property
  def metadata(self):
    pass  # Unused

  def reset(self):
    self.game_over = False
    return self.env.reset(reset_counts=False)

  def step(self, action):
    state, reward, done, info  = self.env.step(action)
    self.game_over = done
    #print('reward env:', reward)
    return state, reward, done, info

@gin.configurable
def create_custom_key_env():
  return grid_xl_Env()
