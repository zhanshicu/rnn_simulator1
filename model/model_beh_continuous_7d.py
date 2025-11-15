"""
Base model for continuous behavioral features with 7D rewards.

Key difference: Rewards are 7-dimensional vectors (one score per feature)
instead of scalar values.
"""
from model.consts import Const
from util import DLogger
import tensorflow as tf
import numpy as np


class ModelBehContinuous7D:
    """
    Base model for continuous behavioral modeling with multi-dimensional rewards.

    Args:
        feature_dim: Dimensionality of continuous feature vectors (actions)
        reward_dim: Dimensionality of reward vectors (typically same as feature_dim)
        s_size: State space size (set to 0 if no external states)
    """

    def __init__(self, feature_dim, reward_dim=None, s_size=0):
        if reward_dim is None:
            reward_dim = feature_dim  # Default: one reward per feature

        DLogger.logger().debug("feature dimension: " + str(feature_dim))
        DLogger.logger().debug("reward dimension: " + str(reward_dim))
        DLogger.logger().debug("number of states: " + str(s_size))

        self.s_size = s_size
        self.feature_dim = feature_dim
        self.reward_dim = reward_dim

        # Placeholders for continuous features and rewards
        self.prev_rewards = self.get_pre_reward()
        # DIM: nBatches x (nTimesteps + 1) x reward_dim  ← NOW MULTI-DIMENSIONAL!

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
            features: [n_batches, n_timesteps, feature_dim] - continuous features (actions)
            rewards: [n_batches, n_timesteps, reward_dim] - multi-dimensional rewards
            states: [n_batches, n_timesteps, s_size] - optional external states

        Returns:
            feed_dict for TensorFlow placeholders
        """
        # Add dummy reward and feature at beginning
        # rewards shape: [n_batches, n_timesteps, reward_dim]
        prev_rewards = np.concatenate(
            (np.zeros((rewards.shape[0], 1, rewards.shape[2])), rewards),
            axis=1
        )
        # prev_features shape: [n_batches, n_timesteps+1, feature_dim]
        prev_features = np.concatenate(
            (np.zeros((features.shape[0], 1, features.shape[2])), features),
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
        """Placeholder for previous rewards (multi-dimensional)"""
        return tf.compat.v1.placeholder(shape=[None, None, self.reward_dim], dtype=Const.FLOAT)

    def get_pre_features(self):
        """Placeholder for previous continuous features (actions)"""
        return tf.compat.v1.placeholder(shape=[None, None, self.feature_dim], dtype=Const.FLOAT)

    def get_pre_state(self):
        """Placeholder for previous states (if using external states)"""
        return tf.compat.v1.placeholder(shape=[None, None, self.s_size], dtype=Const.FLOAT)
