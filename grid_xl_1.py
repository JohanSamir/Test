"""Key-Door-Treasure environment."""
import collections
import random

import numpy as np


class GridXL:
  """GridXL environment."""

  def __init__(self, size=50, gamma=0.99, timeout=500, std=0.):
  #def __init__(self, size=50, gamma=0.99, timeout=400, std=0.):  
  #def __init__(self, size=3, gamma=0.99, timeout=20, std=0.):
    # discount
    self.gamma = gamma
    self.timeout = timeout
    # proportions
    self.size = size
    # treasure
    self.t_map = np.zeros((self.size, self.size))
    self.t_map[0, -1] = 1
    #print('self.t_map:\n',self.t_map)
    self.std = std
    # agent
    self.x = self.size - 1
    self.y = 0
    #print('position:',self.x, self.y)
    self.timer = 0
    # actions_map
    self.actions_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    # states_map
    self.compute_coord_to_state()
    # global state-action counts
    self.state_action_counts = collections.defaultdict(int)
    # episodic state-action counts
    self.episodic_state_action_counts = collections.defaultdict(int)
    # return-conditioned state-action counts
    self.episodic_states = []
    self.episodic_actions = []
    self.episodic_reward = 0.
    self.state_action_returns = collections.defaultdict(list)

  @property
  def num_states(self):
    #return np.max(self.coord_to_state) + 1
    #num_states_= np.max(self.coord_to_state) + 1
    #print('num_states_inf:', num_states_, type(num_states_), num_states_.shape)
    num_statest = np.ones(self.size*self.size)
    #print('num_statest:', num_statest, type(num_statest), num_statest.shape)
    return num_statest.shape
    
  @property
  def num_actions(self):
    return len(self.actions_map)
    #num_actions_ = len(self.actions_map)
    #return (len(self.actions_map), )
    #return num_actions_.reshape(num_actions_,)

  def compute_coord_to_state(self):
    """Computes index to state mapping."""
    # give x, y and k, return ordered state
    self.coord_to_state = - np.ones(
        (self.size, self.size), dtype=np.int64)
    print('self.coord_to_state:',self.coord_to_state)
    self.state_to_coord = []
    state = 0
    for x in range(self.size):
      for y in range(self.size):
        self.state_to_coord.append((x, y,))
        self.coord_to_state[x, y] = state  # np style x-y.
        state += 1
    print('self.coord_to_state:',self.coord_to_state)

  def state(self):
    #return self.coord_to_state[self.x, self.y]
    state_ = self.coord_to_state[self.x, self.y]
    return state_

  def reset(self, reset_counts=False):
    """Reset environment."""
    self.x = self.size - 1
    self.y = 0
    self.timer = 0
    self.episodic_states = []
    self.episodic_actions = []
    self.episodic_reward = 0.
    if reset_counts:
      self.state_action_counts = collections.defaultdict(int)
    self.episodic_state_action_counts = collections.defaultdict(int)
    return self.state()

  def step(self, action):
    """Compute one step of interaction in the environment."""
    state = self.state()
    action = np.int(action)
    #print('action:', action)
    dx, dy = self.actions_map[action]
    #print('----------------------------')
    #print('state:', state)
    #print('move:', dx, dy)
    self.x = max(0, min(self.size - 1, (self.x + dx)))
    self.y = max(0, min(self.size - 1, (self.y + dy)))
    #print('post info:', self.x, self.y)

    reward = 0.
    done = False
    info = {}
    if self.t_map[self.x, self.y]:
      reward = random.gauss(0., self.std)
      #reward = 100
      done = True
    self.timer += 1
    if self.timer == self.timeout:
      if not done:
        distance = float(abs(self.x) + abs(self.y - self.size + 1))
        reward = random.gauss(-distance, self.std)
      done = True

    # Update episodic quantities.
    self.episodic_states.append(self.state())
    self.episodic_actions.append(action)
    self.episodic_reward += reward

    # Update state-action counts.
    self.state_action_counts[state, action] += 1
    self.episodic_state_action_counts[state, action] += 1
    if done:
      state_action_set = set()
      for state, action in zip(self.episodic_states, self.episodic_actions):
        if (state, action) not in state_action_set:
          state_action_set.add((state, action))
          self.state_action_returns[state, action].append(self.episodic_reward)

    return self.state(), reward, done, info

  def expected_reward(self, unused_state, reward, new_state):
    x, y = self.state_to_coord[new_state]
    if self.t_map[x, y]:
      return 1.
    else:
      return reward

  def value_iteration(self):
    """Value iteration."""
    epsilon = 1e-15
    # optimal value
    q = np.zeros((self.num_states, self.num_actions))
    for i in range(1000):
      new_q = np.zeros((self.num_states, self.num_actions))
      for state in range(self.num_states):
        for action in range(self.num_actions):
          self.x, self.y = self.state_to_coord[state]
          new_state, reward, done, _ = self.step(action)
          exp_reward = self.expected_reward(state, reward, new_state)
          new_q[state, action] = (
              exp_reward + self.gamma * np.max(q[new_state, :]) * (1 - done))
      if np.sum((new_q - q) ** 2) < epsilon:
        print('Converged to q* in ', i, 'iterations')
        break
      else:
        q = new_q
    return new_q