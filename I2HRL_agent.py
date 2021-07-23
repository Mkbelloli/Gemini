import tensorflow as tf

import gin.tf
from agents import ddpg_agent
# pylint: disable=unused-import
import cond_fn
from utils import utils as uvf_utils
from context import gin_imports
# pylint: enable=unused-import
slim = tf.contrib.slim
import numpy as np
from tf_agents import specs
import agent


@gin.configurable
class I2HRL_UvfAgent(agent.UvfAgent):
    def __init__(self, *args, **kwargs):
        agent.UvfAgent.__init__(self, *args, **kwargs)


@gin.configurable
class I2HRL_MetaAgent(agent.MetaAgent):
    def __init__(self, *args, **kwargs):
        kwargs['additional_state_specs'] = specs.TensorSpec(
            dtype = kwargs['additional_state_dtype'],
            shape = kwargs['additional_state_shape'])
        del kwargs['additional_state_dtype'] # not needed anymore
        del kwargs['additional_state_shape'] # not needed anymore
        agent.MetaAgent.__init__(self, *args, **kwargs)


    def set_subagent(self, agent=None):
        pass

    def action(self, state, context=None, **kwargs):
        """Returns the next action for the state.

        Args:
          state: A [num_state_dims] tensor representing a state.
          context: A list of [num_context_dims] tensor representing a context.
          additional_state: A list of [num_context_dims] tensor representing additional info like embedding.
        Returns:
          A [num_action_dims] tensor representing the action.
        """

        additional_state = None
        if "additional_state" in kwargs:
            additional_state = kwargs["additional_state"]

        merged_state = self.merged_state(state, context, additional_state)
        return self.BASE_AGENT_CLASS.action(self, merged_state)

    def actions(self, state, context=None, **kwargs):
        """Returns the next action for the state.

        Args:
          state: A [-1, num_state_dims] tensor representing a state.
          context: A list of [-1, num_context_dims] tensor representing a context.
          additional_state: A list of [num_context_dims] tensor representing additional info like embedding.
        Returns:
          A [-1, num_action_dims] tensor representing the action.
        """

        additional_state = None
        if "additional_state" in kwargs:
            additional_state = kwargs["additional_state"]

        merged_states = self.merged_states(state, context, additional_state)
        return self.BASE_AGENT_CLASS.actor_net(self, merged_states)

    def begin_episode_ops(self, mode, action_fn=None, state=None):
        ops = agent.UvfAgentCore.begin_episode_ops(mode, action_fn, state)


@gin.configurable()
def policy_representation_net(
    input,
    #actions,
    #low_actions,
    num_output_dims=30,
    policy_hidden_layers=(100,),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True):
  """Creates a simple feed forward net for embedding states.
  """
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):

    #input = tf.concat(states, actions, low_actions)
    input_shape = tf.shape(input)
    input_dtype = input.dtype
    inputs = tf.to_float(input)

    orig_inputs = inputs
    embed = inputs
    if policy_hidden_layers:
      embed = slim.stack(embed, slim.fully_connected, policy_hidden_layers, scope='emb_pol_inputs')

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      embed = slim.fully_connected(embed, num_output_dims,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='emb_pol_value')

    output = embed
    output = tf.cast(output, input_dtype)
    return output

@gin.configurable()
def imitation_net(
    input,
    #state,
    #meta_action,
    #low_action,
    #policy_emb,
    num_output_dims=80,
    imitation_hidden_layers=(50,),
    normalizer_fn=None,
    activation_fn=tf.nn.sigmoid,
    zero_time=True):
  """Creates a simple feed forward net for embedding states.
  """
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):

    input_shape = tf.shape(input)
    input_dtype = input.dtype
    inputs = tf.to_float(input)

    orig_inputs = inputs
    embed = inputs
    if imitation_hidden_layers:
      embed = slim.stack(embed, slim.fully_connected, imitation_hidden_layers,
                         scope='imi_fun_inputs')

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      embed = slim.fully_connected(embed, num_output_dims,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='imi_fun_value')

    output = embed
    output = tf.cast(output, input_dtype)
    return output

@gin.configurable()
class PolicyRepresentationModule(object):
  EMBEDDING_NET_SCOPE = 'embedding_net'
  IMITATION_NET_SCOPE = 'imitation_net'

  def __init__(self,
               policy_embedding_size=30,
               transition_history=5):

    self._scope = tf.get_variable_scope().name
    self._transition_history = transition_history
    self.dtype = tf.float32
    self._policy_embedding_size = policy_embedding_size

    self._policy_representation_net = tf.make_template(
        self.EMBEDDING_NET_SCOPE, policy_representation_net,
        create_scope_now_=True)
    self._imitation_function_net = tf.make_template(
        self.IMITATION_NET_SCOPE, imitation_net,
        create_scope_now_=True)
    self.__transition_history = transition_history
    self.steps_history =[tf.cast(tf.stack([0]*53), tf.float32)] * self.__transition_history
    self.current_step = 0
    self.__num_steps = 0
    ## initialize buffer
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        self.steps_history = tf.get_variable( name='TransitionBuffer', shape=(10,53), dtype=tf.float32, trainable=False)
        self.current_step = tf.get_variable(name='CurrentPosition',  dtype=tf.int32, \
                                              trainable=False, initializer=tf.zeros(1, dtype=tf.int32))
        self.__current_step = 0


  def load_step(self, state, low_action, meta_action):
    input_c = tf.concat([state, low_action, meta_action], 0)  # TODO: da mettere in vettore 1D
    input = tf.reshape(input_c, [1, state.shape[0] + low_action.shape[0] + meta_action.shape[0]])
    update_buffer_ops = []
    update_buffer_upd = tf.scatter_update(self.steps_history, [self.__current_step], input)

    with tf.control_dependencies([update_buffer_upd]):
        update_pointer_upd = tf.math.mod(tf.math.add(self.current_step,1), tf.constant(self.__transition_history, tf.int32))
    update_buffer_ops.append(update_buffer_upd)
    update_buffer_ops.append(update_pointer_upd)
    self.__current_step = (self.__current_step+1)%self.__transition_history
    return tf.group(*update_buffer_ops)

  def __get_step_history(self):
      #buff = self.steps_history[self.current_step:] + self.steps_history[:self.current_step]
      input_c = tf.concat([self.steps_history[self.__current_step:], self.steps_history[:self.__current_step]], 0)  # TODO: da mettere in vettore 1D
      return tf.reshape(input_c, [1, 53*self.__transition_history])

  def get_embedding_policy(self):
        history = self.__get_step_history()
        with tf.control_dependencies([history]):
            return tf.cast(self._policy_representation_net(history), tf.float32)

  def __call__(self, states, actions, low_actions):
      return self.__call_embedding(states, actions, low_actions)

  def __call_embedding(self, states, meta_actions, low_actions):
    input_c = tf.concat([states, meta_actions, low_actions],1 )
    input = tf.reshape(input_c, [1,states.shape[0]*states.shape[1]
                                    + meta_actions.shape[0] * meta_actions.shape[1]
                                    + low_actions.shape[0] * low_actions.shape[1]])
    embedded = tf.cast(self._policy_representation_net(input), tf.float32)
    return embedded

  def __call_imitation(self, state, meta_action, embed):
      input_c = tf.concat([state, meta_action, embed],0)  # TODO: da mettere in vettore 1D
      input = tf.reshape(input_c, [1, state.shape[0] + meta_action.shape[0] + embed.shape[0]])
      f_phi = self._imitation_function_net(input)
      return f_phi

  def loss(self, states, meta_actions, low_actions):
    ds = tf.concat([states, meta_actions, low_actions], 1)

    # len(e1) + len(e2)=1
    batch_size = self._transition_history + 1
    num_batches = ds.shape[0].value // batch_size
    batches_ds = tf.split(ds[:num_batches * batch_size, :], num_batches, 0)

    r = []
    for batch_ds in batches_ds:
    # TODO: da fare per ogni batch
        batch_states, batch_meta_actions, batch_low_actions = tf.split(batch_ds[:-1],
                                                                       [states.shape[1].value,
                                                                        meta_actions.shape[1].value,
                                                                        low_actions.shape[1].value], 1)
        ms = self.__call_embedding(batch_states, batch_meta_actions, batch_low_actions)
        state, meta_action, low_action = tf.split(batch_ds[-1], [states.shape[1].value,
                                                                    meta_actions.shape[1].value,
                                                                    low_actions.shape[1].value], 0)
        pred_low_action = self.__call_imitation(state, meta_action, tf.squeeze(ms))
        value = tf.math.divide(
            tf.norm(pred_low_action-low_action, ord='euclidean'),
            tf.math.minimum(0.1, tf.norm(low_action, ord='euclidean')))
        r.append( tf.clip_by_value(value, 1e-8, 1.0) )


    return tf.reduce_mean(tf.stack(r))

  def get_trainable_vars(self):
    return (
        slim.get_trainable_variables(
            uvf_utils.join_scope(self._scope, self.EMBEDDING_NET_SCOPE)) +
        slim.get_trainable_variables(
            uvf_utils.join_scope(self._scope, self.IMITATION_NET_SCOPE)))