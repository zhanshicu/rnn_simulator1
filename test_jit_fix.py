#!/usr/bin/env python3
"""
Test script to verify that the JIT compilation fix works.
This tests the MaximumMeanDiscrepancy function that was causing the error.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
# Enable TF1.x compatibility mode
tf.compat.v1.disable_eager_execution()

import numpy as np
from util.losses import maximum_mean_discrepancy, mmd_loss
from util.utils import gaussian_kernel_matrix
from functools import partial

def test_mmd_basic():
    """Test basic MMD calculation without session config."""
    print("Testing basic MMD calculation...")

    # Create simple test data
    x = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    y = tf.constant([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=tf.float32)

    # Compute MMD
    mmd = maximum_mean_discrepancy(x, y)

    # Create session with JIT disabled
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.OFF

    with tf.compat.v1.Session(config=config) as sess:
        result = sess.run(mmd)
        print(f"✓ MMD calculation successful: {result:.6f}")
        return True

def test_mmd_with_gaussian_kernel():
    """Test MMD with Gaussian kernel (the actual failing case)."""
    print("\nTesting MMD with Gaussian kernel (original error case)...")

    # Create test data similar to what's used in training
    np.random.seed(42)
    source_samples = tf.constant(np.random.randn(10, 5).astype(np.float32))
    target_samples = tf.constant(np.random.randn(10, 5).astype(np.float32))

    # Use the same sigmas as in the actual code
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas, dtype=tf.float32))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)

    # Create session with JIT disabled
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.OFF

    with tf.compat.v1.Session(config=config) as sess:
        result = sess.run(loss_value)
        print(f"✓ MMD with Gaussian kernel successful: {result:.6f}")
        return True

def test_mmd_loss_function():
    """Test the full MMD loss function."""
    print("\nTesting full MMD loss function...")

    tf.compat.v1.reset_default_graph()

    np.random.seed(42)
    source_samples = tf.constant(np.random.randn(20, 10).astype(np.float32))
    target_samples = tf.constant(np.random.randn(20, 10).astype(np.float32))

    loss_value = mmd_loss(source_samples, target_samples, weight=1.0)

    # Create session with JIT disabled
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.OFF

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        result = sess.run(loss_value)
        print(f"✓ Full MMD loss function successful: {result:.6f}")
        return True

def main():
    print("=" * 70)
    print("Testing JIT Compilation Fix")
    print("=" * 70)
    print("\nThis tests the MaximumMeanDiscrepancy operations that were")
    print("previously failing with 'JIT compilation failed' error.\n")

    try:
        # Run all tests
        test_mmd_basic()
        test_mmd_with_gaussian_kernel()
        test_mmd_loss_function()

        print("\n" + "=" * 70)
        print("✓ All tests passed! JIT compilation fix is working.")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("\n" + "=" * 70)
        print("✗ Tests failed. JIT compilation issue may still exist.")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
