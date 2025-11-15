"""
Decoder RNN for continuous behavioral features.
Uses hypernetwork: latent code z -> RNN weights -> continuous predictions.
"""
import os
# Force TensorFlow to use Keras 2 (required for compatibility with TF 1.x code)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import numpy as np

from model.consts import Const
from model.model_beh_continuous import ModelBehContinuous
from model.rnn_cell import GRUCell2
from util import DLogger


class DECRNNContinuous(ModelBehContinuous):
    """
    Decoder that generates continuous feature predictions.

    The key innovation: RNN weights are generated from latent code z via hypernetwork.
    This allows individual-specific behavioral dynamics to be encoded in z.

    Args:
        n_cells: Number of GRU units
        feature_dim: Dimensionality of continuous features
        s_size: State space size (0 if no external states)
        n_samples: Number of latent samples to use
        z: Latent representation [n_samples, n_batches, latent_size]
        n_T: Maximum sequence length
        static_loop: Whether to use static RNN
    """

    def __init__(self, n_cells, feature_dim, s_size, n_samples, z, n_T, static_loop):
        super().__init__(feature_dim, s_size)

        DLogger.logger().debug("Continuous decoder created with n_cells: " + str(n_cells))

        self.z = z
        self.seq_lengths = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.n_cells = n_cells
        self.n_samples = n_samples
        self.feature_dim = feature_dim

        # Placeholder for observed features (for computing loss)
        self.observed_features = tf.compat.v1.placeholder(
            shape=[None, None, feature_dim], dtype=Const.FLOAT
        )

        # Random permutation for baseline comparison
        self.rand_gather = self.rand_without_replacement(self.n_batches)

        # Create GRU cell with hypernetwork-generated weights
        self.cell = GRUCell2(n_cells, n_samples, n_T)

        # Generate RNN weights from latent code z
        W1, W2, W_out, b1, b2, b_out = self.z_to_RNNweights(feature_dim, n_cells, s_size, self.z)

        self.cell.set_weights(W1, b1, W2, b2)
        self.W_out = W_out
        self.b_out = b_out

        # Random baseline cell (for comparison)
        self.scell = GRUCell2(n_cells, n_samples, n_T)
        self.scell.set_weights(
            tf.gather(W1, self.rand_gather, axis=1),
            tf.gather(b1, self.rand_gather, axis=1),
            tf.gather(W2, self.rand_gather, axis=1),
            tf.gather(b2, self.rand_gather, axis=1)
        )
        self.W_out_random = tf.gather(W_out, self.rand_gather, axis=1)
        self.b_out_random = tf.gather(b_out, self.rand_gather, axis=1)

        step_size = tf.ones([tf.shape(self.rnn_in)[0]], dtype=tf.int32) * tf.shape(self.rnn_in)[1]

        # Run RNN forward
        if static_loop:
            self.state_track, self.last_state = self.cell.static_rnn(self.rnn_in, step_size)
        else:
            self.state_track, self.last_state = self.cell.dynamic_rnn(self.rnn_in, step_size)
        # state_track: [n_samples, n_batches, n_timesteps+1, n_cells]

        sstate_track, slast_state = self.scell.dynamic_rnn(self.rnn_in, step_size)

        # Generate continuous feature predictions
        # Predictions: linear projection from hidden state to feature space
        self.feature_predictions = self._compute_predictions(self.state_track, self.W_out, self.b_out)
        # Shape: [n_samples, n_batches, n_timesteps+1, feature_dim]

        self.feature_predictions_random = self._compute_predictions(
            sstate_track, self.W_out_random, self.b_out_random
        )

        # Compute reconstruction loss (MSE)
        # Mask for valid timesteps
        mask = tf.cast(
            tf.sequence_mask(self.seq_lengths, maxlen=n_T)[np.newaxis, :, :, np.newaxis],
            Const.FLOAT
        )

        # Predicted vs observed features (excluding last timestep which has no target)
        pred_features = self.feature_predictions[:, :, :-1, :] * mask
        obs_features = self.observed_features[np.newaxis, :, :, :] * mask

        # MSE loss per sample, per batch
        reconstruction_errors = tf.reduce_sum(
            tf.square(pred_features - obs_features), axis=[2, 3]
        )
        # Normalize by sequence length
        reconstruction_errors = reconstruction_errors / tf.cast(
            self.seq_lengths[np.newaxis, :], Const.FLOAT
        )

        self.beh_loss = tf.reduce_mean(reconstruction_errors, axis=0)
        self.loss = tf.reduce_mean(self.beh_loss)

        # Random baseline loss
        pred_features_rand = self.feature_predictions_random[:, :, :-1, :] * mask
        reconstruction_errors_rand = tf.reduce_sum(
            tf.square(pred_features_rand - obs_features), axis=[2, 3]
        )
        reconstruction_errors_rand = reconstruction_errors_rand / tf.cast(
            self.seq_lengths[np.newaxis, :], Const.FLOAT
        )
        self.sloss = tf.reduce_mean(reconstruction_errors_rand)

        # Gradient statistics for analysis
        self.z_grad = tf.reduce_mean(
            tf.abs(tf.gradients(self.feature_predictions[:, :, :, 0], self.z)[0]),
            axis=[0, 1]
        )

    def _compute_predictions(self, state_track, W_out, b_out):
        """
        Compute feature predictions from hidden states.

        Args:
            state_track: [n_samples, n_batches, n_timesteps+1, n_cells]
            W_out: [n_samples, n_batches, n_cells, feature_dim]
            b_out: [n_samples, n_batches, feature_dim]

        Returns:
            predictions: [n_samples, n_batches, n_timesteps+1, feature_dim]
        """
        # Expand dimensions for broadcasting
        state_expanded = state_track[:, :, :, :, np.newaxis]  # [S, B, T, C, 1]
        W_expanded = W_out[:, :, np.newaxis, :, :]  # [S, B, 1, C, F]

        # Matrix multiplication: [S, B, T, C, 1] * [S, B, 1, C, F] -> [S, B, T, F]
        predictions = tf.reduce_sum(state_expanded * W_expanded, axis=3)

        # Add bias
        predictions = predictions + b_out[:, :, np.newaxis, :]

        return predictions

    def beh_feed(self, features, rewards, states=None):
        """
        Create feed dict for decoder.

        Args:
            features: [n_batches, n_timesteps, feature_dim]
            rewards: [n_batches, n_timesteps]
            states: [n_batches, n_timesteps, s_size] or None

        Returns:
            feed_dict
        """
        feed_dict = super().beh_feed(features, rewards, states)
        feed_dict[self.cell.state_in] = np.zeros(
            (self.n_samples, features.shape[0], self.n_cells), dtype=np.float64
        )
        feed_dict[self.scell.state_in] = np.zeros(
            (self.n_samples, features.shape[0], self.n_cells), dtype=np.float64
        )
        return feed_dict

    def rand_without_replacement(self, n_rands):
        """Random permutation for baseline comparison"""
        probs = tf.ones(shape=(n_rands,), dtype=Const.FLOAT) / tf.cast(n_rands, dtype=Const.FLOAT)
        ztmp = -tf.log(-tf.log(tf.compat.v1.random_uniform(tf.shape(probs), 0, 1, dtype=Const.FLOAT)))
        _, indices = tf.nn.top_k(probs + ztmp, n_rands)
        return indices

    def z_to_RNNweights(self, feature_dim, n_cells, s_size, z):
        """
        Hypernetwork: Generate RNN weights from latent code z.

        This is the key innovation - individual behavioral patterns are encoded as
        different RNN weight configurations.

        Args:
            feature_dim: Output feature dimensionality
            n_cells: Number of GRU units
            s_size: State space size
            z: Latent codes [n_samples, n_batches, latent_size]

        Returns:
            W1, W2, W_out, b1, b2, b_out: RNN weight matrices
        """
        W1_dim, b1_dim, W2_dim, b2_dim = GRUCell2.get_weight_dims(
            self.feature_dim + s_size + 1, n_cells
        )
        W_out_dim = [n_cells, feature_dim]
        b_out_dim = [feature_dim]

        total_weight_dim = (W1_dim[0] * W1_dim[1] + b1_dim[0] +
                           W2_dim[0] * W2_dim[1] + b2_dim[0] +
                           W_out_dim[0] * W_out_dim[1] + b_out_dim[0])

        # Hypernetwork: 3-layer MLP from z to weights
        with tf.compat.v1.variable_scope('dec'):
            dense1 = tf.compat.v1.layers.dense(inputs=z, units=100, activation=tf.nn.tanh)
            dense2 = tf.compat.v1.layers.dense(inputs=dense1, units=100, activation=tf.nn.tanh)
            dense3 = tf.compat.v1.layers.dense(inputs=dense2, units=100, activation=tf.nn.tanh)
            output = tf.compat.v1.layers.dense(inputs=dense3, units=total_weight_dim, activation=None)

        # Unpack weights
        with tf.compat.v1.variable_scope('dec_init'):
            last_ix = 0

            W1 = output[:, :, last_ix:(last_ix + W1_dim[0] * W1_dim[1])]
            last_ix += W1_dim[0] * W1_dim[1]

            b1 = output[:, :, last_ix:(last_ix + b1_dim[0])]
            last_ix += b1_dim[0]

            W2 = output[:, :, last_ix:(last_ix + W2_dim[0] * W2_dim[1])]
            last_ix += W2_dim[0] * W2_dim[1]

            b2 = output[:, :, last_ix:(last_ix + b2_dim[0])]
            last_ix += b2_dim[0]

            W_out = output[:, :, last_ix:(last_ix + W_out_dim[0] * W_out_dim[1])]
            last_ix += W_out_dim[0] * W_out_dim[1]

            b_out = output[:, :, last_ix:(last_ix + b_out_dim[0])]

            # Reshape to proper dimensions
            W1 = tf.reshape(W1, [tf.shape(W1)[0], tf.shape(W1)[1]] + W1_dim)
            b1 = tf.reshape(b1, [tf.shape(b1)[0], tf.shape(b1)[1]] + b1_dim)
            W2 = tf.reshape(W2, [tf.shape(W2)[0], tf.shape(W2)[1]] + W2_dim)
            b2 = tf.reshape(b2, [tf.shape(b2)[0], tf.shape(b2)[1]] + b2_dim)
            W_out = tf.reshape(W_out, [tf.shape(W_out)[0], tf.shape(W_out)[1]] + W_out_dim)
            b_out = tf.reshape(b_out, [tf.shape(b_out)[0], tf.shape(b_out)[1]] + b_out_dim)

        return W1, W2, W_out, b1, b2, b_out

    def dec_beh_feed(self, features, rewards, states, seq_lengths):
        """
        Create complete feed dict for decoder including sequence lengths.

        Args:
            features: [n_batches, n_timesteps, feature_dim]
            rewards: [n_batches, n_timesteps]
            states: [n_batches, n_timesteps, s_size] or None
            seq_lengths: [n_batches]

        Returns:
            feed_dict
        """
        feed_dict = self.beh_feed(features, rewards, states)
        feed_dict[self.seq_lengths] = seq_lengths
        feed_dict[self.observed_features] = features
        return feed_dict
