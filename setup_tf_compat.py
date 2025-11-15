"""
TensorFlow Compatibility Setup for Keras 3 / TF 2.x
====================================================

This module MUST be imported FIRST, before any TensorFlow imports,
to ensure proper compatibility with legacy TF-Keras instead of Keras 3.

Usage in Jupyter Notebooks:
---------------------------
Place this as the FIRST import in your notebook:

    import setup_tf_compat  # Must be first!
    import tensorflow as tf
    # ... rest of your imports

This sets up the environment variables needed for:
1. Using legacy Keras (TF-Keras) instead of Keras 3
2. Disabling XLA JIT compilation (prevents JIT errors)
3. Ensuring compatibility with tf.compat.v1 APIs

If you get errors like:
- AttributeError: `dense` is not available with Keras 3
- JIT compilation failed errors

Then you need to RESTART YOUR KERNEL and import this module first.
"""

import os
import sys

# Check if TensorFlow is already imported
if 'tensorflow' in sys.modules:
    import warnings
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "WARNING: TensorFlow was already imported before setup_tf_compat!\n"
        "=" * 70 + "\n"
        "The compatibility settings may not take effect.\n"
        "\n"
        "To fix this:\n"
        "1. RESTART your Python kernel/runtime\n"
        "2. Import setup_tf_compat FIRST, before any other imports\n"
        "3. Then import tensorflow and other modules\n"
        "\n"
        "Example (in Jupyter notebook):\n"
        "  import setup_tf_compat  # MUST be first!\n"
        "  import tensorflow as tf\n"
        "  # ... rest of your code\n"
        "=" * 70,
        RuntimeWarning,
        stacklevel=2
    )

# Set environment variables for TensorFlow compatibility
print("Setting up TensorFlow compatibility...")

# Use legacy Keras (TF-Keras) instead of Keras 3
# This prevents "AttributeError: `dense` is not available with Keras 3"
os.environ['TF_USE_LEGACY_KERAS'] = '1'
print("✓ Set TF_USE_LEGACY_KERAS=1 (using TF-Keras instead of Keras 3)")

# Disable XLA JIT compilation
# This prevents "JIT compilation failed" errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = 'false'
print("✓ Disabled XLA JIT compilation")

print("\nTensorFlow compatibility setup complete!")
print("You can now safely import tensorflow and other modules.")
print()
