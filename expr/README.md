# Continuous Behavioral Hypernetwork Model

This directory contains a refined implementation of the RNN hypernetwork model for **continuous behavioral features**, specifically designed for modeling salesperson learning dynamics.

## Overview

### Key Differences from Original Model

| Aspect | Original Model | Continuous Model |
|--------|---------------|------------------|
| **Actions** | Discrete integers (0, 1, 2, ...) | Continuous feature vectors |
| **Input** | One-hot encoded actions | Raw continuous features (7D) |
| **Output** | Softmax probability distribution | Linear projection to continuous space |
| **Loss** | Categorical cross-entropy | Mean Squared Error (MSE) |
| **Use Case** | Choice tasks, button presses | Skill development, feature evolution |

## Architecture

```
Behavioral Sequence (features, rewards)
              ↓
    [Bidirectional LSTM Encoder]
              ↓
    Latent Code z (3D) ← MMD loss to match Gaussian
              ↓
    [Hypernetwork: z → RNN weights]
              ↓
    [GRU Decoder with generated weights]
              ↓
    Predicted Features (7D) ← MSE loss vs observed
```

### Components

1. **Encoder** (`enc_continuous.py`, `mmdae_enc_continuous.py`)
   - Bidirectional LSTM processes continuous feature sequences
   - Maps to latent code z using MMD autoencoder
   - Learns disentangled representations of learning strategies

2. **Decoder** (`dec_continuous.py`)
   - Hypernetwork generates RNN weights from z
   - GRU processes sequence with individual-specific weights
   - Outputs continuous feature predictions

3. **Combined Model** (`rnn2rnn_continuous.py`)
   - Integrates encoder and decoder
   - Joint training with weighted loss

## Salesperson Learning Data

### Features (7 dimensions, 0-100 scale)

1. **knowledge**: Product/domain knowledge
2. **style**: Communication style smoothness
3. **clarity**: Explanation clarity
4. **energy**: Enthusiasm/energy level
5. **filler_words**: Inverse of filler word usage (higher = better)
6. **sentence_length**: Optimal sentence length (50 = ideal)
7. **pace**: Speaking pace (50 = ideal moderate pace)

### Latent Behavioral Archetypes

The simulation models different learning patterns through 3D latent codes:
- **Dimension 0**: Aggression (aggressive/fast-talking ↔ calm/consultative)
- **Dimension 1**: Knowledge focus (knowledge-first ↔ relationship-first)
- **Dimension 2**: Adaptability (quick adapter/over-corrector ↔ steady learner)

**Archetypes:**
- `aggressive_closer`: High energy, fast pace, relationship-focused
- `calm_consultant`: Low energy, slow pace, knowledge-focused
- `quick_adapter`: Highly responsive to feedback
- `steady_learner`: Gradual, stable improvement
- `style_focused`: Prioritizes smooth style over speed
- `knowledge_focused`: Prioritizes knowledge over delivery polish

### Learning Dynamics

- **Style-Speed Tradeoff**: Improving style may slow pace (for calm archetypes)
- **Knowledge-Energy Tradeoff**: Learning drains energy temporarily
- **Feedback Sensitivity**: Varies by adaptability dimension
- **Natural Learning Rates**: Differ across features and archetypes

## Usage

### 1. Generate Simulated Data

```python
from expr.simulate_salesperson_data import SalespersonSimulator

sim = SalespersonSimulator(seed=42)
dataset = sim.simulate_dataset(
    n_salespeople=50,
    sessions_per_person=20
)
batched_data = sim.dataset_to_arrays(dataset)
```

### 2. Train Model

```bash
cd src_continuous/expr
python train_continuous.py
```

Or programmatically:

```python
from expr.train_continuous import train_model, split_train_test

train_data, test_data = split_train_test(batched_data, train_ratio=0.8)

sess, model = train_model(
    train_data,
    test_data,
    n_epochs=50,
    batch_size=10,
    learning_rate=0.001,
    beta=1.0,
    latent_size=3
)
```

### 3. Analyze Latent Codes

```python
from expr.train_continuous import analyze_latent_representations

latent_codes = analyze_latent_representations(
    sess, model, test_data, n_samples=10
)
```

### 4. Test Predictions

```python
from expr.train_continuous import test_predictions

test_predictions(sess, model, test_data, person_idx=0)
```

## Model Parameters

### Hyperparameters

- `enc_cells`: Number of LSTM units in encoder (default: 20)
- `dec_cells`: Number of GRU units in decoder (default: 20)
- `latent_size`: Dimensionality of latent space (default: 3)
- `n_T`: Maximum sequence length (default: 20)
- `static_loops`: Use static vs dynamic RNN (default: False)
- `mmd_coef`: Weight for MMD loss (default: 2.0)

### Training Parameters

- `n_epochs`: Number of training epochs (default: 50)
- `batch_size`: Mini-batch size (default: 10)
- `learning_rate`: Adam learning rate (default: 0.001)
- `beta`: Weight for encoder loss vs decoder loss (default: 1.0)

## Expected Outputs

### Training Logs

```
Epoch    0 | Total:  245.32 Dec:  123.45 Enc:  121.87 | Test - Dec:  135.21 Enc:  118.45
Epoch   10 | Total:  112.45 Dec:   56.23 Enc:   56.22 | Test - Dec:   61.34 Enc:   54.12
...
```

### Latent Analysis

```
Latent codes shape: (12, 3)
Latent codes mean: [-0.05  0.12  0.03]
Latent codes std: [0.98 1.02 0.95]

sales_042 (aggressive_closer): True latent: [ 1.5 -0.8  0.5], Learned latent: [ 1.23 -0.65  0.42]
```

### Prediction Example

```
Timestep 0:
  knowledge : True= 45.2, Pred= 46.8
  style     : True= 38.5, Pred= 39.1
  clarity   : True= 42.1, Pred= 41.5
  ...
```

## File Structure

```
src_continuous/
├── model/
│   ├── consts.py                      # Constants
│   ├── model_beh_continuous.py        # Base model for continuous features
│   ├── enc_continuous.py              # Encoder (BiLSTM)
│   ├── mmdae_enc_continuous.py        # MMD autoencoder
│   ├── dec_continuous.py              # Decoder (hypernetwork + GRU)
│   ├── rnn2rnn_continuous.py          # Combined model
│   └── rnn_cell.py                    # Custom GRU cell
├── expr/
│   ├── simulate_salesperson_data.py   # Data simulation
│   └── train_continuous.py            # Training script
├── util/
│   ├── logger.py                      # Logging utilities
│   ├── helper.py                      # Helper functions
│   └── losses.py                      # Loss functions (MMD)
└── README.md                          # This file
```

## Extensions

### Adding New Features

To add/modify features, update:
1. `simulate_salesperson_data.py`: Modify `feature_names` and simulation logic
2. Model will automatically adapt to new `feature_dim`

### Custom Archetypes

Define new archetypes in `SalespersonSimulator.sample_latent_code()`:

```python
elif archetype == 'my_custom_type':
    return np.array([aggression, knowledge_focus, adaptability])
```

### Different Loss Functions

Replace MSE in `dec_continuous.py` with:
- **Gaussian NLL**: For probabilistic predictions
- **Huber Loss**: For robustness to outliers
- **Per-feature weighting**: To prioritize certain features

## Citations

Based on:
> Dezfouli, A., Ashtiani, H., Ghattas, O., Nock, R., Dayan, P., & Ong, C. S. (2019).
> Disentangled behavioral representations.
> Advances in Neural Information Processing Systems, 32.

## License

See LICENSE.txt in the root directory.
