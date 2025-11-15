"""
Training script for continuous behavioral hypernetwork model.
Trains on simulated salesperson learning data.
"""
import sys
import os

# Force TensorFlow to use Keras 2 (required for compatibility with TF 1.x code)
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Disable XLA JIT compilation (incompatible with TF 1.x-style code)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
# Disable JIT compilation (XLA) - incompatible with TF 1.x code
tf.config.optimizer.set_jit(False)
# Enable TF1.x compatibility mode (required for Session-based code)
tf.compat.v1.disable_eager_execution()

import numpy as np
from expr.simulate_salesperson_data_7d_rewards import SalespersonSimulator7D
from model.rnn2rnn_continuous import HYPMMDContinuous
from model.consts import Const
from util import DLogger


def split_train_test(batched_data, train_ratio=0.8):
    """
    Split data into train and test sets.

    Args:
        batched_data: Dictionary with actions, rewards, seq_lengths, etc.
        train_ratio: Fraction of data for training

    Returns:
        train_data, test_data: Dictionaries with split data
    """
    n_samples = batched_data['actions'].shape[0]
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_data = {
        'actions': batched_data['actions'][train_indices],
        'rewards': batched_data['rewards'][train_indices],
        'seq_lengths': batched_data['seq_lengths'][train_indices],
        'ids': [batched_data['ids'][i] for i in train_indices],
        'archetypes': [batched_data['archetypes'][i] for i in train_indices],
        'latents': batched_data['latents'][train_indices]
    }

    test_data = {
        'actions': batched_data['actions'][test_indices],
        'rewards': batched_data['rewards'][test_indices],
        'seq_lengths': batched_data['seq_lengths'][test_indices],
        'ids': [batched_data['ids'][i] for i in test_indices],
        'archetypes': [batched_data['archetypes'][i] for i in test_indices],
        'latents': batched_data['latents'][test_indices]
    }

    return train_data, test_data


def create_batches(data, batch_size):
    """
    Create mini-batches from data.

    Args:
        data: Dictionary with actions, rewards, seq_lengths
        batch_size: Batch size

    Returns:
        List of batch dictionaries
    """
    n_samples = data['actions'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        batch = {
            'actions': data['actions'][start_idx:end_idx],
            'rewards': data['rewards'][start_idx:end_idx],
            'seq_lengths': data['seq_lengths'][start_idx:end_idx]
        }
        batches.append(batch)

    return batches


def train_model(
    train_data,
    test_data,
    n_epochs=100,
    batch_size=10,
    learning_rate=0.001,
    beta=1.0,
    latent_size=3,
    enc_cells=20,
    dec_cells=20
):
    """
    Train the hypernetwork model.

    Args:
        train_data: Training data dictionary
        test_data: Test data dictionary
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate for Adam optimizer
        beta: Weight for encoder loss
        latent_size: Dimensionality of latent space
        enc_cells: Number of encoder LSTM units
        dec_cells: Number of decoder GRU units

    Returns:
        Trained model session and model object
    """
    # Reset TensorFlow graph
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(42)

    # Get data dimensions
    feature_dim = train_data['actions'].shape[2]
    max_seq_length = train_data['actions'].shape[1]

    DLogger.logger().info(f"Feature dimension: {feature_dim}")
    DLogger.logger().info(f"Max sequence length: {max_seq_length}")
    DLogger.logger().info(f"Training samples: {train_data['actions'].shape[0]}")
    DLogger.logger().info(f"Test samples: {test_data['actions'].shape[0]}")

    # Create model
    DLogger.logger().info("Building model...")
    model = HYPMMDContinuous(
        enc_cells=enc_cells,
        dec_cells=dec_cells,
        feature_dim=feature_dim,
        s_size=0,  # No external states
        latent_size=latent_size,
        n_T=max_seq_length,
        static_loops=False,  # Use dynamic RNN for flexibility
        mmd_coef=2.0
    )

    # Define loss and optimizer
    trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    beta_ph = tf.compat.v1.placeholder(Const.FLOAT, shape=())
    lr_ph = tf.compat.v1.placeholder(Const.FLOAT, shape=())

    # Combined loss: decoder reconstruction + beta * encoder MMD
    total_loss = model.dec.loss + beta_ph * model.enc.loss

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph)
    train_op = optimizer.minimize(total_loss, var_list=trainables)

    # Initialize session with XLA JIT and all optimizations disabled
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.OFF
    config.graph_options.optimizer_options.opt_level = tf.compat.v1.OptimizerOptions.L0
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    config.graph_options.rewrite_options.constant_folding = tf.compat.v1.rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.arithmetic_optimization = tf.compat.v1.rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.auto_mixed_precision = tf.compat.v1.rewriter_config_pb2.RewriterConfig.OFF
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    DLogger.logger().info("Starting training...")

    # Training loop
    for epoch in range(n_epochs):
        # Create batches
        batches = create_batches(train_data, batch_size)

        epoch_enc_loss = 0
        epoch_dec_loss = 0
        epoch_total_loss = 0

        for batch_idx, batch in enumerate(batches):
            # Create feed dict
            enc_dict = model.enc.enc_beh_feed(
                batch['actions'],
                batch['rewards'],
                None,  # No states
                batch['seq_lengths']
            )

            dec_dict = model.dec.dec_beh_feed(
                batch['actions'],
                batch['rewards'],
                None,  # No states
                batch['seq_lengths']
            )

            feed_dict = {
                **enc_dict,
                **dec_dict,
                beta_ph: beta,
                lr_ph: learning_rate
            }

            # Train step
            _, enc_loss, dec_loss, tot_loss, z_cov = sess.run(
                [train_op, model.enc.loss, model.dec.loss, total_loss, model.enc.z_cov],
                feed_dict=feed_dict
            )

            epoch_enc_loss += enc_loss
            epoch_dec_loss += dec_loss
            epoch_total_loss += tot_loss

        # Average losses
        n_batches = len(batches)
        epoch_enc_loss /= n_batches
        epoch_dec_loss /= n_batches
        epoch_total_loss /= n_batches

        # Evaluate on test set every 10 epochs
        if epoch % 10 == 0:
            # Test evaluation
            test_batches = create_batches(test_data, batch_size=test_data['actions'].shape[0])
            test_batch = test_batches[0]

            enc_dict_test = model.enc.enc_beh_feed(
                test_batch['actions'],
                test_batch['rewards'],
                None,
                test_batch['seq_lengths']
            )

            dec_dict_test = model.dec.dec_beh_feed(
                test_batch['actions'],
                test_batch['rewards'],
                None,
                test_batch['seq_lengths']
            )

            feed_dict_test = {
                **enc_dict_test,
                **dec_dict_test,
                beta_ph: beta,
                lr_ph: learning_rate
            }

            test_dec_loss, test_enc_loss = sess.run(
                [model.dec.loss, model.enc.loss],
                feed_dict=feed_dict_test
            )

            DLogger.logger().info(
                f"Epoch {epoch:4d} | "
                f"Train - Total: {epoch_total_loss:7.2f} "
                f"Dec: {epoch_dec_loss:7.2f} "
                f"Enc: {epoch_enc_loss:7.2f} | "
                f"Test - Dec: {test_dec_loss:7.2f} "
                f"Enc: {test_enc_loss:7.2f}"
            )
        else:
            DLogger.logger().info(
                f"Epoch {epoch:4d} | "
                f"Total: {epoch_total_loss:7.2f} "
                f"Dec: {epoch_dec_loss:7.2f} "
                f"Enc: {epoch_enc_loss:7.2f}"
            )

    DLogger.logger().info("Training complete!")

    return sess, model


def analyze_latent_representations(sess, model, data, n_samples=10):
    """
    Analyze learned latent representations.

    Args:
        sess: TensorFlow session
        model: Trained model
        data: Data dictionary
        n_samples: Number of samples to analyze

    Returns:
        latent_codes: Learned latent codes
    """
    DLogger.logger().info("\n=== Analyzing Latent Representations ===")

    # Extract latent codes for all data
    enc_dict = model.enc.enc_beh_feed(
        data['actions'],
        data['rewards'],
        None,
        data['seq_lengths']
    )

    latent_codes = sess.run(model.enc.z_pred, feed_dict=enc_dict)

    DLogger.logger().info(f"Latent codes shape: {latent_codes.shape}")
    DLogger.logger().info(f"Latent codes mean: {np.mean(latent_codes, axis=0)}")
    DLogger.logger().info(f"Latent codes std: {np.std(latent_codes, axis=0)}")

    # Show examples
    for i in range(min(n_samples, len(data['ids']))):
        DLogger.logger().info(
            f"\n{data['ids'][i]} ({data['archetypes'][i]}): "
            f"True latent: {data['latents'][i]}, "
            f"Learned latent: {latent_codes[i]}"
        )

    # Compute correlation between true and learned latents
    correlation = np.corrcoef(data['latents'].T, latent_codes.T)
    DLogger.logger().info(f"\nCorrelation matrix (true vs learned):")
    DLogger.logger().info(f"{correlation[:3, 3:]}")

    return latent_codes


def test_predictions(sess, model, data, person_idx=0):
    """
    Test model predictions on specific salesperson.

    Args:
        sess: TensorFlow session
        model: Trained model
        data: Data dictionary
        person_idx: Index of person to test
    """
    DLogger.logger().info(f"\n=== Testing Predictions ===")
    DLogger.logger().info(f"Salesperson: {data['ids'][person_idx]} ({data['archetypes'][person_idx]})")

    # Get single person data
    actions = data['actions'][person_idx:person_idx+1]
    rewards = data['rewards'][person_idx:person_idx+1]
    seq_lengths = data['seq_lengths'][person_idx:person_idx+1]

    # Create feed dict
    enc_dict = model.enc.enc_beh_feed(actions, rewards, None, seq_lengths)
    dec_dict = model.dec.dec_beh_feed(actions, rewards, None, seq_lengths)
    feed_dict = {**enc_dict, **dec_dict}

    # Get predictions
    predictions, latent = sess.run(
        [model.dec.feature_predictions, model.enc.z_pred],
        feed_dict=feed_dict
    )

    # predictions shape: [n_samples=1, n_batches=1, n_timesteps+1, feature_dim]
    predictions = predictions[0, 0, :-1, :]  # Remove last timestep, take first sample and batch
    true_actions = actions[0, :seq_lengths[0], :]

    DLogger.logger().info(f"Learned latent: {latent[0]}")
    DLogger.logger().info(f"True latent: {data['latents'][person_idx]}")

    # Compute MSE
    mse = np.mean((predictions[:seq_lengths[0]] - true_actions) ** 2)
    DLogger.logger().info(f"Reconstruction MSE: {mse:.2f}")

    DLogger.logger().info(f"\nFirst 3 timesteps comparison:")
    feature_names = ['knowledge', 'style', 'clarity', 'energy', 'filler', 'sent_len', 'pace']
    for t in range(min(3, seq_lengths[0])):
        DLogger.logger().info(f"\nTimestep {t}:")
        for feat_idx, feat_name in enumerate(feature_names):
            DLogger.logger().info(
                f"  {feat_name:10s}: True={true_actions[t, feat_idx]:5.1f}, "
                f"Pred={predictions[t, feat_idx]:5.1f}"
            )


def main():
    """Main training pipeline."""
    DLogger.logger().info("=== Salesperson Learning Model Training ===\n")

    # 1. Generate simulated data
    DLogger.logger().info("Step 1: Generating simulated data...")
    sim = SalespersonSimulator7D(seed=42)
    dataset = sim.simulate_dataset(
        n_salespeople=60,
        sessions_per_person=20
    )
    batched_data = sim.dataset_to_arrays(dataset)

    # 2. Split train/test
    DLogger.logger().info("\nStep 2: Splitting train/test...")
    train_data, test_data = split_train_test(batched_data, train_ratio=0.8)

    # 3. Train model
    DLogger.logger().info("\nStep 3: Training model...")
    sess, model = train_model(
        train_data,
        test_data,
        n_epochs=50,
        batch_size=10,
        learning_rate=0.001,
        beta=1.0,
        latent_size=3,
        enc_cells=20,
        dec_cells=20
    )

    # 4. Analyze results
    DLogger.logger().info("\nStep 4: Analyzing results...")
    latent_codes = analyze_latent_representations(sess, model, test_data, n_samples=5)

    # 5. Test predictions
    test_predictions(sess, model, test_data, person_idx=0)

    DLogger.logger().info("\n=== Training Pipeline Complete ===")

    return sess, model, train_data, test_data


if __name__ == '__main__':
    sess, model, train_data, test_data = main()
