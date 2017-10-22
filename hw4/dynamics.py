import tensorflow as tf
import numpy as np

EPS = np.finfo(np.float32).eps


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    # Predefined function to build a feedforward neural network
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self._sess = sess
        self._normalization = normalization
        self._batch_size = batch_size
        self._iterations = iterations
        self._learning_rate = learning_rate

        # assuming continuous obs and actions
        obs_dim = env.observation_space.shape[0]
        # obs_dim = 1 if len(obs_dim) == 0 else obs_dim[0]
        ac_dim = env.action_space.shape[0]
        # ac_dim = 1 if len(ac_dim) == 0 else ac_dim[0]
        self._input_obs = tf.placeholder(
            tf.float32, shape=[None, obs_dim])
        self._input_acs = tf.placeholder(
            tf.float32, shape=[None, ac_dim])
        self._input = tf.concat(
            [self._input_obs, self._input_acs], axis=1)
        self._delta = tf.placeholder(
            tf.float32, shape=[None, obs_dim])
        self.network = build_mlp(
            self._input, output_size=obs_dim,
            scope='dynamics', n_layers=n_layers, size=size,
            activation=activation, output_activation=output_activation)

        self.loss = tf.reduce_sum(tf.reduce_mean(
            tf.square(self.network - self._delta), axis=0))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states,
        (unnormalized)actions, (unnormalized)next_states and fit the
        dynamics model going from normalized states, normalized actions
        to normalized state differences (s_t+1 - s_t)
        """
        # self._normalization takes the following form
        # mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action
        obs = np.concatenate(
            list(map(lambda x: x['observations'], data)))
        acs = np.concatenate(
            list(map(lambda x: x['actions'], data)))
        next_obs = np.concatenate(
            list(map(lambda x: x['next_observations'], data)))

        norm_states = (obs - self._normalization[0]) /\
            (self._normalization[1] + EPS)
        deltas = obs - next_obs
        norm_deltas = (deltas - self._normalization[2]) /\
            (self._normalization[3] + EPS)
        norm_actions = (acs - self._normalization[4]) /\
            (self._normalization[5] + EPS)

        for i in range(self._iterations):
            dataset = tf.data.Dataset.from_tensor_slices(
                (norm_states, norm_actions, norm_deltas))
            batch_dataset = dataset.shuffle(
                    norm_states.shape[0]
                ).batch(self._batch_size)
            iterator = batch_dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            loss = 0
            idx = 0
            while True:
                try:
                    o, a, d = self._sess.run(next_element)
                except tf.errors.OutOfRangeError:
                    break
                feed_dict = {
                    self._input_obs: o,
                    self._input_acs: a,
                    self._delta: d
                }
                loss += self._sess.run(self.loss, feed_dict)
                idx += 1
                self._sess.run(self.optimizer, feed_dict)
            print('    Dyn fit -- avg loss {} :: epoch {}'.format(
                loss / idx, i))

    def predict(self, states, actions):
        """
        Write a function to take in a batch of (unnormalized) states
        and (unnormalized) actions and return the (unnormalized) next
        states as predicted by using the model
        """
        norm_states = (states - self._normalization[0]) /\
            (self._normalization[1] + EPS)
        norm_actions = (actions - self._normalization[4]) /\
            (self._normalization[5] + EPS)

        feed_dict = {
            self._input_obs: norm_states,
            self._input_acs: norm_actions,
        }
        norm_delta = self._sess.run(self.network, feed_dict)
        next_states = (norm_delta * self._normalization[3]) +\
            self._normalization[2] + states
        return next_states
