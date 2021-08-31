"""DQN replay memory with SIL support.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

#import google3

from dopamine.replay_memory import circular_replay_buffer as crb
from dopamine.replay_memory import prioritized_replay_buffer as prb
from dopamine.replay_memory import sum_tree
import numpy as np
#import tensorflow.compat.v2 as tf

import gin
import json  
#from tensorflow.contrib import staging as contrib_staging

PLACEHOLDER_RETURN_VALUE = np.finfo(np.float32).min

@gin.configurable
class SILOutOfGraphReplayBuffer(crb.OutOfGraphReplayBuffer):
  """A simple out-of-graph Replay Buffer with support for SIL.

  Stores transitions, state, action, reward, next_state, terminal (and any
  extra contents specified) in a circular buffer and provides a uniform
  transition sampling function.

  When the states consist of stacks of observations storing the states is
  inefficient. This class writes observations and constructs the stacked states
  at sample time.

  Attributes:
    add_count: int, counter of how many transitions have been added (including
      the blank ones at the beginning of an episode).
    invalid_range: np.array, an array with the indices of cursor-related invalid
      transitions
  """

  def __init__(self, observation_shape, stack_size,
               replay_capacity, batch_size, **kwargs):
    self.episode_num_last_completed = -1
    self.curr_episode_start = stack_size - 1
    self.curr_episode_end = stack_size - 2
    self.min_episode_return = 10000#np.inf
    self.max_episode_return= -10000#-np.inf
    self.rmin_inf = dict()
    self.rmax_inf = dict()
    self.episode_return_inf = dict()


    super(SILOutOfGraphReplayBuffer, self).__init__(
        observation_shape, stack_size, replay_capacity, batch_size, **kwargs)

  def get_storage_signature(self):
    """Returns a default list of elements to be stored in this replay memory.

    Note - Derived classes may return a different signature.

    Returns:
      list of ReplayElements defining the type of the contents stored.
    """
    storage_elements = [
        crb.ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
        crb.ReplayElement('action', self._action_shape, self._action_dtype),
        crb.ReplayElement('reward', self._reward_shape, self._reward_dtype),
        crb.ReplayElement('terminal', (), self._terminal_dtype),
        crb.ReplayElement('return', (), np.float32),
        crb.ReplayElement('episode_num', (), np.int32),
    ]

    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

  def add(self, observation, action, reward, terminal, *args):
    """Adds a transition to the replay memory.

    This function checks the types and handles the padding at the beginning of
    an episode. Then it calls the _add function.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
                was terminal (1) or not (0).
      *args: extra contents with shapes and dtypes according to
        extra_storage_types.
    """
    self._check_add_types(observation, action, reward, terminal, *args)
    if self.is_empty() or self._store['terminal'][self.cursor() - 1] == 1:
      for _ in range(self._stack_size - 1):
        # Child classes can rely on the padding transitions being filled with
        # zeros. This is useful when there is a priority argument.
        self._add_zero_transition()
    #print('reward add:', reward)
    self._add(observation, action, reward, terminal, *args)
    self.curr_episode_end = (
        self.curr_episode_end + 1) % self._replay_capacity

  def is_valid_transition(self, index, sample_from_complete_episodes=False):
    """Checks if the index contains a valid transition.

    Checks for collisions with the end of episodes and the current position
    of the cursor.

    Args:
      index: int, the index to the state in the transition.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

    Returns:
      Is the index valid: Boolean.

    """
    # Check the index is in the valid range
    if index < 0 or index >= self._replay_capacity:
      return False
    if not self.is_full():
      # The indices and next_indices must be smaller than the cursor.
      if index >= self.cursor() - self._update_horizon:
        return False
      # The first few indices contain the padding states of the first episode.
      if index < self._stack_size - 1:
        return False

    # Skip transitions that straddle the cursor.
    if index in set(self.invalid_range):
      return False

    # If there are terminal flags in any other frame other than the last one
    # the stack is not valid, so don't sample it.
    if self.get_terminal_stack(index)[:-1].any():
      return False

    # If the episode the transition is from was not inserted entirely,
    # optionally don't sample it.
    if sample_from_complete_episodes:
      episode_num = self._store['episode_num'][index]
      if episode_num > self.episode_num_last_completed:
        return False

    return True

  def sample_index_batch(self, batch_size, sample_from_complete_episodes=False):
    """Returns a batch of valid indices sampled uniformly.

    If sample_complete_episodes=True, only returns indices of transitions where
    the full episode was inserted in the replay buffer.

    Args:
      batch_size: int, number of indices returned.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      RuntimeError: If the batch was not constructed after maximum number of
        tries.
    """
    if self.is_full():
      # add_count >= self._replay_capacity > self._stack_size
      min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
      max_id = self.cursor() - self._update_horizon
    else:
      # add_count < self._replay_capacity
      min_id = self._stack_size - 1
      max_id = self.cursor() - self._update_horizon
      if max_id <= min_id:
        raise RuntimeError('Cannot sample a batch with fewer than stack size '
                           '({}) + update_horizon ({}) transitions.'.
                           format(self._stack_size, self._update_horizon))

    indices = []
    attempt_count = 0
    while (len(indices) < batch_size and
           attempt_count < self._max_sample_attempts):
      index = np.random.randint(min_id, max_id) % self._replay_capacity
      if self.is_valid_transition(index, sample_from_complete_episodes):
        indices.append(index)
      else:
        attempt_count += 1
    if len(indices) != batch_size:
      raise RuntimeError(
          'Max sample attempts: Tried {} times but only sampled {}'
          ' valid indices. Batch size is {}'.
          format(self._max_sample_attempts, len(indices), batch_size))

    return indices

  def sample_transition_batch(self, batch_size=None, indices=None,
                              sample_from_complete_episodes=True):
    """Returns a batch of transitions (including any extra contents).

    If get_transition_elements has been overridden and defines elements not
    stored in self._store, an empty array will be returned and it will be
    left to the child class to fill it. For example, for the child class
    OutOfGraphPrioritizedReplayBuffer, the contents of the
    sampling_probabilities are stored separately in a sum tree.

    When the transition is terminal next_state_batch has undefined contents.

    NOTE: This transition contains the indices of the sampled elements. These
    are only valid during the call to sample_transition_batch, i.e. they may
    be used by subclasses of this replay buffer but may point to different data
    as soon as sampling is done.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.
      sample_from_complete_episodes: bool, whether to sample only transitions
        from completed episodes.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().

    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(
          batch_size, sample_from_complete_episodes)
    assert len(indices) == batch_size

    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = self._create_batch_arrays(batch_size)
    for batch_element, state_index in enumerate(indices):
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()
      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        # np.argmax of a bool array returns the index of the first True.
        trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
                                      0) + 1
      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (
          self._cumulative_discount_vector[:trajectory_length])
      trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                          next_state_index)

      # Fill the contents of each array in the sampled batch.
      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'state':
          element_array[batch_element] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_element] = np.sum(
              trajectory_discount_vector * trajectory_rewards, axis=0)
        elif element.name == 'next_state':
          element_array[batch_element] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name in ('next_action', 'next_reward'):
          element_array[batch_element] = (
              self._store[element.name.lstrip('next_')][(next_state_index) %
                                                        self._replay_capacity])
        elif element.name == 'terminal':
          element_array[batch_element] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_element] = state_index
        elif element.name in self._store.keys():
          element_array[batch_element] = (
              self._store[element.name][state_index])
        # We assume the other elements are filled in by the subclass.

    return batch_arrays

  def get_transition_elements(self, batch_size=None):
    """Returns a 'type signature' for sample_transition_batch.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
    Returns:
      signature: A namedtuple describing the method's return type signature.
    """
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        crb.ReplayElement('state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
        crb.ReplayElement('action', (batch_size,) + self._action_shape,
                          self._action_dtype),
        crb.ReplayElement('reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
        crb.ReplayElement('next_state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
        crb.ReplayElement('next_action', (batch_size,) + self._action_shape,
                          self._action_dtype),
        crb.ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                          self._reward_dtype),
        crb.ReplayElement('terminal', (batch_size,), self._terminal_dtype),
        crb.ReplayElement('indices', (batch_size,), np.int32),
        crb.ReplayElement('return', (batch_size,), np.float32),
        crb.ReplayElement('episode_num', (batch_size,), np.int32),
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          crb.ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                            element.type))
    return transition_elements

  def _calculate_discounted_returns(self, rewards):
    returns_reversed = itertools.accumulate(rewards[::-1],
                                            lambda x, y: x * self._gamma + y)
    return np.array(list(returns_reversed))[::-1] \

  def _calculate_returns(self, rewards):
    return_ = np.sum(rewards)
    return return_ * np.ones_like(rewards) 

  def _get_circular_slice(self, array, start, end):
    assert array.ndim == 1
    if end >= start:
      return array[start: end + 1]
    else:
      return np.concatenate([array[start:], array[:end + 1]])

  def _set_circular_slice(self, array, start, end, values):
    assert array.ndim == 1
    if end >= start:
      assert len(values) == end - start + 1, 'len(values):{} vs end-start+1:{} start:() end:()'.format(len(values), end - start + 1, start, end)
      array[start: end + 1] = values
    else:
      length_left = len(array) - start
      assert len(values) == end + length_left + 1
      array[start:] = values[:length_left]
      array[:end + 1] = values[length_left:]

  def calculate_and_store_return(self, episode_num, idx_iter):
    """Calculates and updates the return of a given episode based on stored rewards.

    Args:
      episode_num: int, identifier of the episode.
    Raises:
      RuntimeError: if the episode queried does not exist or is unfinished.
    """
    #print('ÑÑÑÑÑÑÑÑÑÑÑÑÑÑ')
    if episode_num != self.episode_num_last_completed + 1:
      raise RuntimeError(
          'The next completed episode should have number {}. '
          'Found `episode_num`={}.'.format(
              self.episode_num_last_completed + 1, episode_num))
    if self._store['terminal'][self.curr_episode_end] != 1:
      raise RuntimeError(
          'Trying to calculate the return of an unfinished episode.')
    rewards = self._get_circular_slice(
        self._store['reward'],
        self.curr_episode_start,
        self.curr_episode_end)

    #print('self._store[reward]', self._store['reward'], 'self.curr_episode_start:',self.curr_episode_start,'self.curr_episode_end):', self.curr_episode_end)
    #print('rewards:', rewards, rewards.shape)
    returns = self._calculate_returns(rewards)
    #print('returns:', returns, returns.shape)
    episode_return = np.sum(rewards)
    #print('episode_return:', episode_return)

    #print('self._replay.max_episode_return_init:',self.max_episode_return)
    #print('self._replay.min_episode_return_init:',self.min_episode_return)

    if idx_iter not in self.rmin_inf.keys():
        self.rmin_inf[idx_iter] = 0 #10000

    if idx_iter not in self.rmax_inf.keys():
        self.rmax_inf[idx_iter] = 0 #-10000

    if idx_iter not in self.episode_return_inf.keys():
        self.episode_return_inf[idx_iter] = 0

    if episode_return < self.min_episode_return: #NEW
       self.min_episode_return = episode_return  #NEW
    if episode_return > self.max_episode_return: #NEW
       self.max_episode_return = episode_return  #NEW 

    self.rmin_inf[idx_iter] = np.append(self.rmin_inf[idx_iter], np.array(self.min_episode_return))
    self.rmax_inf[idx_iter] = np.append(self.rmax_inf[idx_iter], np.array(self.max_episode_return))
    self.episode_return_inf[idx_iter] = np.append(self.episode_return_inf[idx_iter], np.array(episode_return))
    
    #if idx_iter>29:
    #print('idx_iter:',idx_iter)
    #Num itera:4 -> [0, 1, 2, 3]
    #if idx_iter>28:
    #if idx_iter>18:
    #if idx_iter>32:
    #if idx_iter>11:
    #  print('SAVINF FILESSS')

    #  np.save('/content/drive/My Drive/SaveFiles/Data/Dopamine_github/flax_linen/plot_rmin_and_rmax/rmin_inf.npy', self.rmin_inf)
    #  np.save('/content/drive/My Drive/SaveFiles/Data/Dopamine_github/flax_linen/plot_rmin_and_rmax/rmax_inf.npy', self.rmax_inf)
    #  np.save('/content/drive/My Drive/SaveFiles/Data/Dopamine_github/flax_linen/plot_rmin_and_rmax/episode_return_inf.npy', self.episode_return_inf)
    #print('self._replay.max_episode_return_output:', self.max_episode_return)
    #print('self._replay.min_episode_return_output:', self.min_episode_return)

    self._set_circular_slice(
        self._store['return'],
        self.curr_episode_start,
        self.curr_episode_end,
        returns)

    self.episode_num_last_completed = episode_num
    self.curr_episode_start = (
        self.curr_episode_end + self._stack_size) % self._replay_capacity
    self.curr_episode_end = (
        self.curr_episode_end + self._stack_size - 1) % self._replay_capacity