"""
MMD Autoencoder for continuous behavioral features.
Learns disentangled latent representations using Maximum Mean Discrepancy.
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from model.consts import Const
from model.enc_continuous import ENCRNNContinuous
from util.helper import tf_cov
from util.losses import mmd_loss


class MMDAEContinuous(ENCRNNContinuous):
    """
    MMD-based autoencoder for learning disentangled behavioral representations.

    Uses Maximum Mean Discrepancy to match latent distribution to Gaussian prior,
    combined with KL divergence for regularization.

    Args:
        n_cells: Number of LSTM units in encoder
        feature_dim: Dimensionality of continuous features
        s_size: State space size (0 if no external states)
        latent_size: Dimensionality of latent space
        n_T: Maximum sequence length
        static_loops: Whether to use static RNN
        mmd_loss_coef: Coefficient for MMD loss term
    """

    def __init__(self, n_cells, feature_dim, s_size, latent_size, n_T, static_loops, mmd_loss_coef):
        super().__init__(n_cells, feature_dim, s_size, latent_size, 1, n_T, static_loops)

        with tf.compat.v1.variable_scope('enc'):
            # Force CPU execution to bypass XLA JIT compilation
            # TensorFlow 2.20.0 aggressively tries to JIT compile dense layers and activations
            with tf.device('/CPU:0'):
                # Extract final hidden state for each sequence
                batch_range = tf.range(tf.shape(self.rnn_in)[0])
                indices = tf.stack([batch_range, self.seq_lengths], axis=1)
                out = tf.gather_nd(self.rnn_out[:, :, :], indices)

                # Encoder network: hidden state -> latent code
                dense1 = tf.compat.v1.layers.dense(inputs=out, units=n_cells, activation=tf.nn.relu)
                dense2 = tf.compat.v1.layers.dense(inputs=dense1, units=n_cells, activation=tf.nn.relu)
                dense3 = tf.compat.v1.layers.dense(inputs=dense2, units=10, activation=tf.nn.softplus)

                # Latent representation: [1, n_batches, latent_size]
                self.z = tf.compat.v1.layers.dense(inputs=dense3, units=latent_size, activation=None)[np.newaxis]

                # Compute covariance matrix for latent codes
                self.z_cov = tf_cov(self.z[0]) + tf.eye(num_rows=tf.shape(self.z)[2], dtype=Const.FLOAT) * 1e-6

                # Sample from prior (standard Gaussian)
                true_samples = tf.compat.v1.random_normal(tf.stack([3000, self.z.shape[2]]), dtype=Const.FLOAT)

                # MMD loss: match latent distribution to Gaussian
                # KL loss: regularize covariance to be identity
                self.loss = mmd_loss_coef * mmd_loss(true_samples, self.z[0], 1) + \
                            tfp.distributions.kl_divergence(
                                tfp.distributions.MultivariateNormalFullCovariance(
                                    loc=tf.reduce_mean(self.z[0], axis=0),
                                    covariance_matrix=self.z_cov
                                ),
                                tfp.distributions.MultivariateNormalDiag(
                                    loc=tf.zeros((self.z.shape[2]), dtype=Const.FLOAT)
                                )
                            )

            self.dc_loss = tf.constant(0)
            self.z_pred = self.z[0]
