"""Compact implementation of a DQN agent

Specifically, we implement the following components:

  * prioritized replay
  * huber_loss
  * mse_loss
  * double_dqn
  * noisy
  * dueling
  * Munchausen

Details in: 
"Human-level control through deep reinforcement learning" by Mnih et al. (2015).
"Noisy Networks for Exploration" by Fortunato et al. (2017).
"Deep Reinforcement Learning with Double Q-learning" by Hasselt et al. (2015).
"Dueling Network Architectures for Deep Reinforcement Learning" by Wang et al. (2015).
"Munchausen Reinforcement Learning" by Vieillard et al. (2020).
"""
import time
import copy
import functools
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
import jax.scipy.special as scp
import circular_replay_buffer_new


def mse_loss(targets, predictions):
  return jnp.mean(jnp.power((targets - (predictions)),2))

def cross_entropy_loss(target, pred, num_actions=4, eps=1e-08):
  #print('target:',target,  target.shape)
  #print('pred:', pred,pred.shape)
  one_hot = jax.nn.one_hot(target, num_actions)
  #print('one_hot:',one_hot, one_hot.shape)
  
  log_pre = jnp.log(pred) + eps
  multi = one_hot * log_pre
  sum_op = jnp.sum(one_hot * log_pre)
  #-jnp.mean(sum_op)
  return -sum_op

#@functools.partial(jax.jit, static_argnums=(0))
def train_H(network_def, optimizer, states, actions, z, rng):

  #print('states:',states)
  #print('actions:', actions)
  #print('z:',z)

  online_params_H = optimizer.target
  states = states.reshape(128,1)

  z = z.reshape(z.shape[0],1)
  concatenate = jnp.hstack((states, z))
  states = jnp.array([[concatenate]])
  states = states.reshape(concatenate.shape[0],concatenate.shape[1],1,1)
  #print('states:', states)

  def loss_fn(params, rng_input, target_actions):
    def q_online(state):
      return network_def.apply(params, state, rng=rng_input)

    q_vals = jax.vmap(q_online, in_axes=(0))(states).q_values
    q_vals = jnp.squeeze(q_vals)

    loss = cross_entropy_loss(target_actions, q_vals)    
    mean_loss = jnp.mean(loss)
    return mean_loss, loss

  rng, rng2 = jax.random.split(rng, 2)

  # Concatenate TODO    
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (mean_loss, loss), grad = grad_fn(online_params_H, rng2, actions)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, mean_loss


@functools.partial(jax.jit, static_argnums=(0, 9,10,11,12,13, 14))
def train(network_def, target_params, optimizer, states, actions, next_states, rewards,
          terminals, loss_weights, cumulative_gamma, target_opt, mse_inf,tau,alpha,clip_value_min, rng):


  """Run the training step."""
  online_params = optimizer.target
  def loss_fn(params, rng_input, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state, rng=rng_input)

    #print('statesQnetwork.shape:', states.shape)
    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    
    if mse_inf:
      loss = jax.vmap(mse_loss)(target, replay_chosen_q)
    else:
      loss = jax.vmap(dqn_agent.huber_loss)(target, replay_chosen_q)

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss

  rng, rng2, rng3, rng4  = jax.random.split(rng, 4)

  def q_target(state):
    return network_def.apply(target_params, state, rng=rng2)

  def q_target_online(state):
    return network_def.apply(online_params, state, rng=rng4)

  if target_opt == 0:
    target = target_Q(q_target, states, next_states, rewards, terminals, cumulative_gamma) 
  elif target_opt == 1:
    #Double DQN
    target = target_DDQN(q_target_online, q_target, next_states, rewards,  terminals, cumulative_gamma)

  elif target_opt == 2:
    #Munchausen
    target = target_m_dqn(q_target_online, q_target, states,next_states,actions,rewards,terminals,
                cumulative_gamma,tau,alpha,clip_value_min)
  else:
    print('error')

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (mean_loss, loss), grad = grad_fn(online_params, rng3, target, loss_weights)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, mean_loss

def target_Q(target_network, states, next_states, rewards, terminals, cumulative_gamma):
  """Compute the target Q-value."""

  q_vals = jax.vmap(target_network, in_axes=(0))(next_states).q_values
  q_vals = jnp.squeeze(q_vals)
  replay_next_qt_max = jnp.max(q_vals, 1)

  #ValueJumps
  q_vals_st = jax.vmap(target_network, in_axes=(0))(states).q_values
  q_vals_st = jnp.squeeze(q_vals_st)
  replay_qt_max = jnp.max(q_vals_st, 1)

  rewards = rewards + (replay_next_qt_max-((1/cumulative_gamma)*replay_qt_max))

  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                               (1. - terminals))

#@functools.partial(jax.jit, static_argnums=(0, 1, 6, 7, 8, 9, 10, 12, 13))
def select_action(network_def, network_def_H, params, params_H, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, epsilon):

  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))
  
  #print('------------------------------------------------------------------------')
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  Qnetwork = network_def.apply(params, state, rng=rng3).q_values
  selected_action = jnp.argmax(Qnetwork)

  z = jnp.array([0]) #TODO
  concatenate = jnp.hstack((state, z))
  distri_action_H = network_def_H.apply(params_H, concatenate, rng=rng3).q_values
  selected_action_H = jnp.argmax(distri_action_H)

  #array_eval = jnp.array([48, 98, 99, 52, 2447, 2448])
  #array_eval = jnp.array([48, 98, 99])
  '''

  array_eval = jnp.array([48, 99])
  n = state in array_eval

  if n is True:
    print('------------------------------------------------------------------------')
    print('state:', state)
    print('distri_Qnetwork:', Qnetwork)
    selected_action = jnp.argmax(Qnetwork)
    print("action-Q:", selected_action)
    print("distri_action_H", distri_action_H)
    selected_action_H = jnp.argmax(distri_action_H)
    print("action-H:", selected_action_H)
    print('------------------------------------------------------------------------')
  '''

  p = jax.random.uniform(rng1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        selected_action)

@gin.configurable
class JaxDQNAgentNew(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,

               tau,
               alpha=1,
               clip_value_min=-10,

               net_conf = None,
               env = "CartPole", 
               normalize_obs = True,
               hidden_layer=2, 
               neurons=512,
               replay_scheme='prioritized',
               noisy = False,
               dueling = False,
               initzer = 'xavier_uniform',
               target_opt=0,
               mse_inf=False,
               network=networks.NatureDQNNetwork,
               network_H=networks.NatureDQNNetwork,
               optimizer='adam',
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               seed=None):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    # We need this because some tools convert round floats into ints.
    seed = int(time.time() * 1e6) if seed is None else seed
    self._net_conf = net_conf
    self._env = env 
    self._normalize_obs = normalize_obs
    self._hidden_layer = hidden_layer
    self._neurons=neurons 
    self._noisy = noisy
    self._dueling = dueling
    self._initzer = initzer
    self._target_opt = target_opt
    self._mse_inf = mse_inf
    self._tau = tau
    self._alpha = alpha
    self._clip_value_min = clip_value_min
    self._rng = jax.random.PRNGKey(seed)
    self.return_rg =-30
    #self.eps=1

    super(JaxDQNAgentNew, self).__init__(
        num_actions= num_actions,
        network= functools.partial(network, 
                                num_actions=num_actions,
                                net_conf=self._net_conf,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling,
                                initzer=self._initzer),
        optimizer=optimizer,
        epsilon_fn=dqn_agent.identity_epsilon if self._noisy == True else epsilon_fn)

    self.network_def_H = network_H(num_actions=num_actions,
                                net_conf=self._net_conf,
                                env=self._env,
                                normalize_obs=self._normalize_obs,
                                hidden_layer=self._hidden_layer, 
                                neurons=self._neurons,
                                noisy=self._noisy,
                                dueling=self._dueling,
                                initzer=self._initzer)

    self._replay_scheme = replay_scheme
    self._curr_episode = 0
    self.actions = []
    self.idx_iter= 0
    self._build_networks_and_optimizer_H()

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    online_network_params = self.network_def.init(
        rng, x=self.state,  rng=self._rng)
    optimizer_def = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer = optimizer_def.create(online_network_params)
    self.target_network_params = copy.deepcopy(online_network_params)

  def _build_networks_and_optimizer_H(self):
    self._rng, rng = jax.random.split(self._rng)
    self.state = self.state.reshape(1)
    concatenate = jnp.hstack((self.state, self.return_rg))
    concatenate = jnp.array([[concatenate]])
    #print('concatenate.shape:', concatenate, concatenate.shape)

    online_network_params_H = self.network_def_H.init(rng, x=concatenate, rng=self._rng)
    optimizer_def_H = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_H = optimizer_def_H.create(online_network_params_H)


  @property
  def online_params_H(self):
    return self.optimizer_H.target


  def _build_replay_buffer(self):
    """Creates the prioritized replay buffer used by the agent."""
    #return circular_replay_buffer_new.SILWrappedReplayBuffer(
    return circular_replay_buffer_new.SILOutOfGraphReplayBuffer(      
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)
    
  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          # Weight the loss by the inverse priorities.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])
 
        f = (self._replay.max_episode_return-self._replay.min_episode_return)
        if f > 1e-3:
          return_rg=(self.replay_elements['return']-self._replay.min_episode_return)/f
        else:
          return_rg=0

        self.return_rg = jnp.ones(self.replay_elements['return'].shape[0])*return_rg

        self.optimizer, loss, mean_loss = train(self.network_def,
                                     self.target_network_params,
                                     self.optimizer,
                                     self.replay_elements['state'],
                                     self.replay_elements['action'],
                                     self.replay_elements['next_state'],
                                     #self.replay_elements['reward'],
                                     self.return_rg,
                                     self.replay_elements['terminal'],
                                     loss_weights,
                                     self.cumulative_gamma,
                                     self._target_opt,
                                     self._mse_inf,
                                     self._tau,
                                     self._alpha,
                                     self._clip_value_min,
                                     self._rng)

        #self._train_step_H()
        if self._replay_scheme == 'prioritized':
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))
        
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='HuberLoss', simple_value=mean_loss)])
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1


  def _train_step_H(self):
    if self._replay.add_count > self.min_replay_history:
        self._sample_from_replay_buffer()
        #print('/////////////////////////////////////////////////////////////////')
        #print('self.replay_elements[state]:', self.replay_elements['state'])
        #print('self.replay_elements[action]:', self.replay_elements['action'])
        #self.return_rg = jnp.ones(self.replay_elements['return'].shape[0])*return_rg
        self.optimizer_H, loss_H, mean_loss_H = train_H(self.network_def_H,
                                     self.optimizer_H,
                                     self.replay_elements['state'],
                                     self.replay_elements['action'],
                                     self.replay_elements['return'],
                                     self._rng)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
              summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='cross_entropy_loss', simple_value=loss_H)])
            self.summary_writer.add_summary(summary, self.training_steps)


  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        episode_return, 
                        episode_num,
                        priority=None):

    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.sum_tree.max_recorded_priority

    if not self.eval_mode:
     self._replay.add(last_observation, action, reward, is_terminal,
                       episode_return, episode_num)

  def begin_episode(self, observation, iteration):
    """Returns the agent's first action for this episode.
    Args:
      observation: numpy array, the environment's initial observation.
    Returns:
      int, the selected action.
    """

    self.idx_iter = iteration
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()
      self._train_step_H()

    epsilon = self.pers_decaying_epsilon()
    self._rng, self.action = select_action(self.network_def,
    	                                   self.network_def_H,
                                           self.online_params,
                                           self.online_params_H,
                                           self.state,
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn,
                                           epsilon)
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.
    We store the observation of the last time step since we want to store it
    with the reward.
    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.
    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(
          self._last_observation, self.action, reward, False,
          circular_replay_buffer_new.PLACEHOLDER_RETURN_VALUE, self._curr_episode)
      self._train_step()
      self._train_step_H()

    epsilon = self.pers_decaying_epsilon()

    self._rng, self.action = select_action(self.network_def,
    	                                   self.network_def_H,
                                           self.online_params,
                                           self.online_params_H,
                                           self.state,
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn,
                                           epsilon)
    self.action = onp.asarray(self.action)
    self.actions.append(self.action)
    return self.action

  def end_episode(self, reward, terminal=True):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(
          self._observation, self.action, reward, terminal,
          circular_replay_buffer_new.PLACEHOLDER_RETURN_VALUE, self._curr_episode)
      #print('self.idx_iter',self.idx_iter)
      self._update_episodic_return(self._curr_episode, self.idx_iter)
      self._curr_episode += 1

  def _update_episodic_return(self, num_episode, idx_iter):
    """Calculates and sets the episodic return for the last episode completed.

    Executes a tf session and executes replay buffer ops in order to store the
    episodic returns.

    Args:
      num_episode: int, identifier of the episode to be updated.
    """
    self._replay.calculate_and_store_return(num_episode, idx_iter)


  def init_eps(self):
    # TODO: I am not using this metod to explore and exploit
    self.eps=1

  def pers_decaying_epsilon(self, decay_period=0.99987, EPS_MIN=0.01):
    self.eps = max(self.eps * decay_period, EPS_MIN)
    return self.eps


  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.
    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.
    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.
    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {
        'state': self.state,
        'training_steps': self.training_steps,
        'online_params': self.online_params,
        'online_params_H': self.online_params_H
    }
    return bundle_dictionary


  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.
    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.
    Args:
      checkpoint_dir: str, path to the checkpoint saved.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.
    Returns:
      bool, True if unbundling was successful.
    """
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      self.state = bundle_dictionary['state']
      self.training_steps = bundle_dictionary['training_steps']
      if isinstance(bundle_dictionary['online_params'], core.FrozenDict):
        online_network_params = bundle_dictionary['online_params']
        online_network_params_H = bundle_dictionary['online_params_H']
      else:  # Load pre-linen checkpoint.
        online_network_params = core.FrozenDict({
            'params': checkpoints.convert_pre_linen(
                bundle_dictionary['online_params']).unfreeze()
        })
        online_network_params_H = core.FrozenDict({
            'params': checkpoints.convert_pre_linen(
                bundle_dictionary['online_params_H']).unfreeze()
        })
      # We recreate the optimizer with the new online weights.
      optimizer_def = create_optimizer(self._optimizer_name)
      self.optimizer = optimizer_def.create(online_network_params)

      optimizer_def_H = create_optimizer(self._optimizer_name)
      self.optimizer_H = optimizer_def_H.create(online_network_params_H)

    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    return True
