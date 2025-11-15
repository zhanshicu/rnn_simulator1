# Jupyter Notebook Setup Guide

## Quick Start for Jupyter Notebooks

If you're running this code in a Jupyter notebook (like `main.ipynb`), follow these steps:

### Step 1: Restart Kernel (Important!)

If you've already run any cells that import TensorFlow, **you MUST restart your kernel first**.

In Jupyter:
- Click: **Kernel → Restart Kernel**
- Or use the restart button in the toolbar

### Step 2: First Cell - Import Compatibility Setup

Make this your **FIRST cell** in the notebook:

```python
# MUST BE FIRST CELL! Import before TensorFlow
import setup_tf_compat
```

You should see output like:
```
Setting up TensorFlow compatibility...
✓ Set TF_USE_LEGACY_KERAS=1 (using TF-Keras instead of Keras 3)
✓ Disabled XLA JIT compilation
```

### Step 3: Continue with Normal Imports

Now you can import TensorFlow and other modules:

```python
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

### Step 4: Import TensorFlow

```python
import tensorflow as tf
import tensorflow_probability as tfp

print("TF:", tf.__version__)
print("TFP:", tfp.__version__)
```

### Step 5: Run the Rest of Your Notebook

Continue with your data generation, training, etc.

## Common Errors and Solutions

### Error: `AttributeError: 'dense' is not available with Keras 3`

**Cause**: TensorFlow 2.16+ uses Keras 3 by default, which removed many legacy APIs.

**Solution**:
1. **Restart your kernel** (Kernel → Restart)
2. Import `setup_tf_compat` as the FIRST import
3. Then import TensorFlow

### Error: `JIT compilation failed`

**Cause**: XLA JIT compilation is enabled and failing on certain operations.

**Solution**:
- Already handled by `setup_tf_compat`
- If still occurs, check that you imported `setup_tf_compat` BEFORE TensorFlow

### Error: `KeyError: 'features'`

**Cause**: The data uses 'actions' key, not 'features'.

**Solution**:
- Already fixed in latest version
- Make sure you pulled the latest code

### Error: `AttributeError: module 'tensorflow' has no attribute 'is_finite'`

**Cause**: `tf.is_finite` was removed in TF 2.x

**Solution**:
- Already fixed in latest version (uses `tf.math.is_finite`)
- Make sure you pulled the latest code

## Complete Example Notebook Structure

Here's the recommended structure for your notebook:

```python
# ===== CELL 1 =====
# MUST BE FIRST! Import compatibility setup
import setup_tf_compat

# ===== CELL 2 =====
# Standard imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
sns.set_style('whitegrid')

# ===== CELL 3 =====
# TensorFlow imports
import tensorflow as tf
import tensorflow_probability as tfp

print("TF:", tf.__version__)
print("TFP:", tfp.__version__)

# ===== CELL 4 =====
# Data generation
from expr.simulate_salesperson_data_7d_rewards import SalespersonSimulator7D

sim = SalespersonSimulator7D(seed=42)
# ... rest of data generation

# ===== CELL 5 =====
# Train/test split
from expr.train_continuous import split_train_test

train_data, test_data = split_train_test(batched_data, train_ratio=0.8)

# ===== CELL 6 =====
# Model training
from expr.train_continuous import train_model

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

# ===== Continue with analysis cells =====
```

## Alternative: Set Environment Variables Manually

If you prefer not to use `setup_tf_compat.py`, you can set environment variables manually:

```python
# FIRST CELL - Set env vars before any imports
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = 'false'

# SECOND CELL - Now import TensorFlow
import tensorflow as tf
# ... rest of imports
```

## Verification

To verify everything is set up correctly, run:

```python
import tensorflow as tf

# Check Keras version
print(f"Using Keras: {tf.keras.__version__}")

# Should show TF-Keras version (< 3.0), not standalone Keras 3.x
```

If you see Keras 3.x, restart your kernel and ensure `setup_tf_compat` is imported FIRST.

## Troubleshooting

### Still Getting Errors?

1. **Restart kernel** - This is the most common fix
2. **Clear all output** - Kernel → Restart & Clear Output
3. **Check import order** - `setup_tf_compat` must be FIRST
4. **Verify environment variables**:
   ```python
   import os
   print(os.environ.get('TF_USE_LEGACY_KERAS'))  # Should print '1'
   print(os.environ.get('TF_XLA_FLAGS'))  # Should print '--tf_xla_auto_jit=0'
   ```

5. **Check TensorFlow wasn't auto-imported** - Some Jupyter extensions auto-import TensorFlow. If so, disable those extensions or use a fresh kernel.

## Summary

✅ **ALWAYS** import `setup_tf_compat` first
✅ **RESTART** kernel if you've already imported TensorFlow
✅ **VERIFY** environment variables are set before TF import
✅ **USE** latest code from the repository

Following these steps will prevent all common compatibility errors!
