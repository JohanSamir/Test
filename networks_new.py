"""Various networks for Jax Dopamine agents."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from jax import random
import math

from jax.tree_util import tree_flatten, tree_map

#---------------------------------------------------------------------------------------------------------------------


env_inf = {"CartPole":{"MIN_VALS": onp.array([-2.4, -5., -math.pi/12., -math.pi*2.]),"MAX_VALS": onp.array([2.4, 5., math.pi/12., math.pi*2.])},
            "Acrobot":{"MIN_VALS": onp.array([-1., -1., -1., -1., -5., -5.]),"MAX_VALS": onp.array([1., 1., 1., 1., 5., 5.])},
            "MountainCar":{"MIN_VALS":onp.array([-1.2, -0.07]),"MAX_VALS": onp.array([0.6, 0.07])}
            }
#---------------------------------------------< DQNNetwork >----------------------------------------------------------

@gin.configurable
class DQNNetwork(nn.Module):

  num_actions:int
  net_conf: str
  env: str
  normalize_obs:bool
  noisy: bool
  dueling: bool
  initzer:str
  hidden_layer: int
  neurons: int

  @nn.compact
  def __call__(self, x , rng):

    if self.net_conf == 'minatar':
      print('x:', x.shape)
      x = x.squeeze(3)
      print('x1:', x.shape)
      #x = x[None, ...]
      #print('x2:', x.shape)
      x = x.astype(jnp.float32)
      print('x3:', x.shape)
      x = nn.Conv(features=16, kernel_size=(3, 3),  strides=(1, 1), kernel_init=self.initzer)(x)
      print('x4:', x.shape)
      x = jax.nn.relu(x)
      print('x5:', x.shape)
      x = x.reshape((-1))
      print('x6:', x.shape)
      #x = x.reshape((x.shape[0], -1))
      #print('x7:', x.shape)
      
    elif self.net_conf == 'classic':
      #classic environments
      x = x.astype(jnp.float32)
      x = x.reshape((-1))

    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=self.initzer)(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    adv = net(x, features=self.num_actions, rng=rng)
    val = net(x, features=1, rng=rng)

    dueling_q = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    non_dueling_q = net(x, features=self.num_actions, rng=rng)
    #print('non_dueling_q:', non_dueling_q.shape)
    #non_dueling_q =  jax.nn.softmax(non_dueling_q, axis=0)
    #print('non_dueling_q_modi:', non_dueling_q.shape)
    q_values = jnp.where(self.dueling, dueling_q, non_dueling_q)

    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class DQNNetwork_H(nn.Module):

  num_actions:int
  net_conf: str
  env: str
  normalize_obs:bool
  noisy: bool
  dueling: bool
  initzer:str
  hidden_layer: int
  neurons: int

  @nn.compact
  def __call__(self, x, rng):

    #print('x--:', x.shape)
    if self.net_conf == 'minatar':
      print('x:', x.shape)
      x = x.squeeze(3)
      print('x1:', x.shape)
      #x = x[None, ...]
      #print('x2:', x.shape)
      x = x.astype(jnp.float32)
      print('x3:', x.shape)
      x = nn.Conv(features=16, kernel_size=(3, 3),  strides=(1, 1), kernel_init=self.initzer)(x)
      print('x4:', x.shape)
      x = jax.nn.relu(x)
      print('x5:', x.shape)
      x = x.reshape((-1))
      print('x6:', x.shape)
      #x = x.reshape((x.shape[0], -1))
      #print('x7:', x.shape)
      
    elif self.net_conf == 'classic':
      #classic environments
      #print('classic:', x)
      x = x.astype(jnp.float32)
      x = x.reshape((-1))

    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=self.initzer)(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    adv = net(x, features=self.num_actions, rng=rng)
    val = net(x, features=1, rng=rng)

    dueling_q = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    non_dueling_q = net(x, features=self.num_actions, rng=rng)

    q_values = jnp.where(self.dueling, dueling_q, non_dueling_q)
    #print('q-values:', q_values)
    q_values = jax.nn.softmax(q_values)
    #print('softmax-q_values:', q_values)

    return atari_lib.DQNNetworkType(q_values)
