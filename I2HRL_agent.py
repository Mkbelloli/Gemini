import tensorflow as tf
import gin.tf
from agents import ddpg_agent
# pylint: disable=unused-import
import cond_fn
from utils import utils as uvf_utils
from context import gin_imports
# pylint: enable=unused-import
slim = tf.contrib.slim
import confvalues
import numpy as np

import agent


@gin.configurable
class I2HRL_UvfAgent(agent.UvfAgent):
    def __init__(self, *args, **kwargs):
        agent.UvfAgent.__init__(self, *args, **kwargs)
        # TODO da spostare in gin
        self._replay_buffer = None
        self._policy_embedder = None

    def set_replaybuffer(self, replay):
        self._replay_buffer = replay

    def load_policy_embedder(self, policy_embedder):
        self._policy_embedder = policy_embedder

    def get_embedding_policy(self):

        if self._policy_embedder is None:
            return None

        def calculated_embedding():
            crnt_idx = self._replay_buffer.get_position()
            transitions = self._replay_buffer.gather_nstep(self._policy_embedder.transition_history, [crnt_idx])
            states, low_actions, _, _, _, meta_actions = transitions[0][:6]
            return self._policy_embedder(states, meta_actions, low_actions)

        def default_embedding():
            return tf.zeros_like(tf.zeros(self._policy_embedder.policy_embedding_size, tf.int32)),

        emb_op = tf.cond(tf.less(self._replay_buffer.get_num_adds(), self._policy_embedder.transition_history),
                default_embedding,
                calculated_embedding)
        return emb_op



@gin.configurable
class I2HRL_MetaAgent(agent.MetaAgent):
    def __init__(self, *args, **kwargs):

        agent.MetaAgent.__init__(self, *args, **kwargs)
        # TODO da spostare in gin
        self._sub_agent = None

    def set_subagent(self, agent=None):
        pass
        #self._sub_agent = agent

    def action(self, state, context=None, additional_state=None):
        merged_state = self.merged_state(state, additional_state)

        if self._sub_agent is not None:
            additional_state = agent.get_embedding_policy()
            # using merged_state, in case additional_state is None
            # (even it should not happen), context_var are added
            # and it is wrong.
            merged_state = tf.concat([state, ] + additional_state, axis=-1)

        return self.BASE_AGENT_CLASS.action(self, merged_state)

    def actions(self, state, context=None, additional_state=None):
        merged_state = self.merged_state(state, additional_state)

        if self._sub_agent is not None:
            additional_state = agent.get_embedding_policy()
            # using merged_state, in case additional_state is None
            # (even it should not happen), context_var are added
            # and it is wrong.
            merged_state = tf.concat([state, ] + additional_state, axis=-1)
        return self.BASE_AGENT_CLASS.actor_net(self, merged_states)

    def begin_episode_ops(self, mode, action_fn=None, state=None):
        ops = agent.UvfAgentCore.begin_episode_ops(mode, action_fn, state)


@gin.configurable()
def policy_representation_net(
    states,
    actions,
    low_actions,
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

    input = tf.concat(states, actions, low_actions)
    input_shape = tf.shape(input)
    input_dtype = input.dtype
    inputs = tf.to_float(input)

    orig_inputs = inputs
    embed = inputs
    if policy_hidden_layers:
      embed = slim.stack(embed, slim.fully_connected, policy_hidden_layers,
                         scope='emb_pol_inputs')

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
    state,
    meta_action,
    low_action,
    policy_emb,
    num_output_dims=30,
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

    input = tf.concat(state, meta_action, low_action,policy_emb)
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
    self._policy_embedding_size = policy_embedding_size

    self._policy_representation_net = tf.make_template(
        self.EMBEDDING_NET_SCOPE, policy_representation_net,
        create_scope_now_=True)
    self._imitation_function_net = tf.make_template(
        self.EMBEDDING_NET_SCOPE, policy_representation_net,
        create_scope_now_=True)
    self.policy_embedding_size = 100
    self.transition_history = 10

  def __call__(self, states, actions, low_actions):
      self.__call_embedding(states, actions, low_actions)

  def __call_embedding(self, states, meta_actions, low_actions):
    input_c = tf.concat([states, meta_actions, low_actions],1 )
    input = tf.reshape(input_c, [1,states.shape[0]*states.shape[1]
                                    + meta_actions.shape[0] * meta_actions.shape[1]
                                    + low_actions.shape[0] * low_actions.shape[1]])
    embedded = self._policy_representation_net(input)
    return embedded

  def __call_imitation(self, state, meta_action, low_action, embed):
      input = tf.concat([state, meta_action, low_action, embed],1)  # TODO: da mettere in vettore 1D

      f_phi = self._imitation_function_net(input)
      return f_phi

  def loss(self, states, meta_actions, low_actions):
    ds = tf.concat([states, meta_actions, low_actions], 1)

    # len(e1) + len(e2)=1
    batch_size = self._transition_history + 1
    num_batches = ds.shape[0].value // batch_size
    batches_ds = tf.split(ds[:num_batches * batch_size, :], num_batches, 0)

    r = 0
    for batch_ds in batches_ds:
    # TODO: da fare per ogni batch
        batch_states, batch_meta_actions, batch_low_actions = tf.split(batch_ds[:-1],
                                                                       [states.shape[1].value,
                                                                        meta_actions.shape[1].value,
                                                                        low_actions.shape[1].value], 1)
        ms = self.__call_embedding(batch_states, batch_meta_actions, batch_low_actions)
        state, meta_action, low_action = tf.split(batch_ds[-1], [states.shape[1].value,
                                                                    meta_actions.shape[1].value,
                                                                    low_actions.shape[1].value], 1)
        r += np.log( self.__call_imitation(state, meta_action, low_action, ms))

    return r/num_batches

  def get_trainable_vars(self):
    return (
        slim.get_trainable_variables(
            uvf_utils.join_scope(self._scope, self.EMBEDDING_NET_SCOPE)) +
        slim.get_trainable_variables(
            uvf_utils.join_scope(self._scope, self.IMITATION_NET_SCOPE)))