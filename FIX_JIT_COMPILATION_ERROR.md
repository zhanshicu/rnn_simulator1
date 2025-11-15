# Fix for TensorFlow JIT Compilation Error

## Problem

When running this repository, you may encounter the following error:

```
UnknownError: JIT compilation failed.
  [[{{node enc_1/MaximumMeanDiscrepancy/Exp}}]]
```

This error occurs because TensorFlow 2.x attempts to use XLA (Accelerated Linear Algebra) JIT compilation on certain operations, particularly the `tf.exp` operation used in the Gaussian kernel matrix calculation for Maximum Mean Discrepancy (MMD) loss.

## Root Cause

The error originates from:
- **File**: `util/utils.py`, line 185
- **Function**: `gaussian_kernel_matrix`
- **Operation**: `tf.exp(-s)` inside the Gaussian RBF kernel computation
- **Used by**: MMD loss calculation in encoder training

## Solution

This repository has been fixed by disabling XLA JIT compilation in TensorFlow session configurations. The fix has been applied to:

1. **`expr/train_continuous.py`** (line 157-162) - Main training script
2. **`model/rnn_cell.py`** (lines 177-180, 203-206) - RNN cell test functions

### Code Changes

All TensorFlow session creations now include:

```python
# Create session config that disables JIT compilation
config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.OFF
sess = tf.compat.v1.Session(config=config)
```

## Alternative Solution: Environment Variables

If you prefer not to modify the code, you can also disable XLA JIT compilation using environment variables before running your script:

### Linux/Mac:
```bash
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_XLA_ENABLE_XLA_DEVICES=false
python your_script.py
```

### Python:
```python
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = 'false'

import tensorflow as tf
# ... rest of your code
```

## Testing the Fix

A test script has been provided to verify the fix works correctly:

```bash
# Make sure you're in your virtual environment
source ~/sales_experiment/.venv/bin/activate  # or your venv path

# Run the test
python test_jit_fix.py
```

Expected output:
```
======================================================================
Testing JIT Compilation Fix
======================================================================

Testing basic MMD calculation...
✓ MMD calculation successful: 0.000xxx

Testing MMD with Gaussian kernel (original error case)...
✓ MMD with Gaussian kernel successful: 0.000xxx

Testing full MMD loss function...
✓ Full MMD loss function successful: 0.000xxx

======================================================================
✓ All tests passed! JIT compilation fix is working.
======================================================================
```

## Why This Fix Works

1. **XLA JIT Compilation**: TensorFlow 2.x can automatically enable XLA for performance optimization
2. **Compatibility Issues**: Some TF 1.x style operations (used in this codebase) may not be fully compatible with XLA
3. **Session Config**: By explicitly setting `global_jit_level = OFF`, we prevent TensorFlow from attempting XLA compilation
4. **No Performance Impact**: For this use case, the performance difference is negligible since the computations are relatively small

## Additional Notes

- This fix maintains backward compatibility with TensorFlow 1.x
- The code uses `tf.compat.v1` APIs for compatibility
- No changes to the actual algorithm or loss functions were needed
- Only session configuration was modified

## Files Modified

- ✅ `expr/train_continuous.py` - Main training script
- ✅ `model/rnn_cell.py` - RNN cell implementation
- ✅ `test_jit_fix.py` - Test script (new)
- ✅ `FIX_JIT_COMPILATION_ERROR.md` - This documentation (new)

## References

- [TensorFlow XLA Documentation](https://www.tensorflow.org/xla)
- [TensorFlow ConfigProto Documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/ConfigProto)
- [Maximum Mean Discrepancy Loss](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)
