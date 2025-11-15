"""
Base model for continuous behavioral features.
Instead of discrete actions, this model handles continuous feature vectors.
"""
from model.consts import Const
from util import DLogger
import tensorflow as tf
import numpy as np


class ModelBehContinuous:
    """
    Base model for continuous behavioral modeling.

    Args:
        feature_dim: Dimensionality of continuous feature vectors
                    (e.g., 7 for [knowledge, style, clarity, energy, filler, length, pace])
        s_size: State space size (set to 0 if no external states)
    """

    def __init__(self, feature_dim, s_size=0):
        DLogger.logger().debug("feature dimension: " + str(feature_dim))
        DLogger.logger().debug("number of states: " + str(s_size))

        self.s_size = s_size
        self.feature_dim = feature_dim

        # Placeholders for continuous features and rewards
        # Rewards can be scalar (dim=1) or multi-dimensional (dim=feature_dim)
        self.prev_rewards = self.get_pre_reward()
        # DIM: nBatches x (nTimesteps + 1) x reward_dim

        self.prev_features = self.get_pre_features()
        # DIM: nBatches x (nTimesteps + 1) x feature_dim

        self.n_batches = tf.shape(self.prev_rewards)[0]

        if s_size != 0:
            self.prev_states = self.get_pre_state()
            rnn_in = tf.concat(values=[self.prev_rewards, self.prev_features, self.prev_states], axis=2)
            # DIM: nBatches x (nTimesteps + 1) x (reward_dim + feature_dim + s_size)
        else:
            rnn_in = tf.concat(values=[self.prev_rewards, self.prev_features], axis=2)
            # DIM: nBatches x (nTimesteps + 1) x (reward_dim + feature_dim)

        self.rnn_in = rnn_in

    def beh_feed(self, features, rewards, states=None):
        """
        Create feed dict for TensorFlow by adding dummy timesteps.

        Args:
            features: [n_batches, n_timesteps, feature_dim] - continuous features
            rewards: [n_batches, n_timesteps] or [n_batches, n_timesteps, reward_dim]
                    - scalar or multi-dimensional rewards
            states: [n_batches, n_timesteps, s_size] - optional external states

        Returns:
            feed_dict for TensorFlow placeholders
        """
        # Add dummy feature at beginning
        prev_features = np.concatenate(
            (np.zeros((features.shape[0], 1, features.shape[2])), features),
            axis=1
        )

        # Handle both scalar (2D) and multi-dimensional (3D) rewards
        if len(rewards.shape) == 2:
            # Scalar rewards: [n_batches, n_timesteps]
            prev_rewards = np.hstack((np.zeros((rewards.shape[0], 1)), rewards))
            prev_rewards = prev_rewards[:, :, np.newaxis]  # Add dimension for concat
        else:
            # Multi-dimensional rewards: [n_batches, n_timesteps, reward_dim]
            prev_rewards = np.concatenate(
                (np.zeros((rewards.shape[0], 1, rewards.shape[2])), rewards),
                axis=1
            )

        feed_dict = {
            self.prev_rewards: prev_rewards,
            self.prev_features: prev_features
        }

        if states is not None:
            prev_states = np.hstack((states, np.zeros(states[:, 0:1].shape)))
            feed_dict[self.prev_states] = prev_states

        return feed_dict

    def get_pre_reward(self):
        """Placeholder for previous rewards (scalar or multi-dimensional)"""
        return tf.compat.v1.placeholder(shape=[None, None, None], dtype=Const.FLOAT)

    def get_pre_features(self):
        """Placeholder for previous continuous features"""
        return tf.compat.v1.placeholder(shape=[None, None, self.feature_dim], dtype=Const.FLOAT)

    def get_pre_state(self):
        """Placeholder for previous states (if using external states)"""
        return tf.compat.v1.placeholder(shape=[None, None, self.s_size], dtype=Const.FLOAT)
