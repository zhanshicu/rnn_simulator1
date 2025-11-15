"""
Encoder RNN for continuous behavioral features.
Uses bidirectional LSTM to encode sequences into latent representations.
"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

from model.consts import Const
from model.model_beh_continuous_7d import ModelBehContinuous7D
from util import DLogger


class ENCRNNContinuous(ModelBehContinuous7D):
    """
    Encoder that processes continuous feature sequences.

    Args:
        n_cells: Number of LSTM units
        feature_dim: Dimensionality of continuous features
        s_size: State space size (0 if no external states)
        latent_size: Dimensionality of latent representation
        n_samples: Number of samples (typically 1 for encoder)
        n_T: Maximum sequence length
        static_loop: Whether to use static RNN (True) or dynamic RNN (False)
    """

    def __init__(self, n_cells, feature_dim, s_size, latent_size, n_samples, n_T, static_loop):
        super().__init__(feature_dim, reward_dim=feature_dim, s_size=s_size)
        DLogger.logger().debug("Encoder created with n_cells: " + str(n_cells))

        self.static_loop = static_loop
        self.n_T = n_T
        self.n_samples = n_samples
        self.n_cells = n_cells
        self.latent_size = latent_size

        self.seq_lengths = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])

        self.rnn_out, _ = self.rnn_cells(n_cells, self.rnn_in)

    def rnn_cells(self, n_cells, rnn_in):
        """
        Bidirectional LSTM encoder.

        Args:
            n_cells: Number of LSTM units
            rnn_in: Input tensor [n_batches, n_timesteps+1, input_dim]

        Returns:
            output: Concatenated forward and backward outputs [n_batches, n_timesteps+1, 2*n_cells]
            state: Final states
        """
        with tf.compat.v1.variable_scope('enc'):
            fw_gru = LSTMCell(n_cells)
            bw_gru = LSTMCell(n_cells)

            if not self.static_loop:
                output, state = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    fw_gru, bw_gru, rnn_in,
                    time_major=False,
                    sequence_length=self.seq_lengths + 1,
                    dtype=Const.FLOAT
                )

                return tf.concat(output, axis=2), tf.stack(
                    (tf.concat(state[0], axis=1), tf.concat(state[1], axis=1)), axis=0
                )
            else:
                output_static, state_fw, state_bw = tf.compat.v1.nn.static_bidirectional_rnn(
                    fw_gru, bw_gru,
                    tf.unstack(rnn_in, num=self.n_T + 1, axis=1),
                    sequence_length=self.seq_lengths + 1,
                    dtype=Const.FLOAT
                )

                st1, st2 = tf.concat(tf.stack(output_static, axis=1), axis=2), \
                           tf.stack((tf.concat(state_fw, axis=1), tf.concat(state_bw, axis=1)), axis=0)

                return st1, st2

    def enc_beh_feed(self, features, rewards, states, seq_lengths):
        """
        Create feed dict for encoder.

        Args:
            features: [n_batches, n_timesteps, feature_dim]
            rewards: [n_batches, n_timesteps]
            states: [n_batches, n_timesteps, s_size] or None
            seq_lengths: [n_batches] - actual sequence lengths

        Returns:
            feed_dict
        """
        feed_dict = super().beh_feed(features, rewards, states)
        feed_dict[self.seq_lengths] = seq_lengths
        return feed_dict
