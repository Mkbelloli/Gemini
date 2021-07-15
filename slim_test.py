


with slim.arg_scope(
        [slim.fully_connected],
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        weights_initializer=slim.variance_scaling_initializer(
            factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):
    input_shape = tf.shape(states)
    states_dtype = states.dtype
    states = tf.to_float(states)

    orig_states = states

    embed = input_steps
    embed = slim.stack(embed, slim.fully_connected, self.policy_layers,
                       scope='hidden')

with slim.arg_scope([slim.fully_connected],
                    weights_regularizer=None,
                    weights_initializer=tf.random_uniform_initializer(
                        minval=-0.003, maxval=0.003)):
    embed = slim.fully_connected(embed, num_output_dims,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='value')