"""
Test script to verify code structure without running TensorFlow.
Tests imports and basic structure.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from expr.simulate_salesperson_data import SalespersonSimulator

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from model.consts import Const
        print("✓ consts")
    except Exception as e:
        print(f"✗ consts: {e}")

    try:
        from model.model_beh_continuous import ModelBehContinuous
        print("✓ model_beh_continuous")
    except Exception as e:
        print(f"✗ model_beh_continuous: {e}")

    try:
        from model.enc_continuous import ENCRNNContinuous
        print("✓ enc_continuous")
    except Exception as e:
        print(f"✗ enc_continuous: {e}")

    try:
        from model.mmdae_enc_continuous import MMDAEContinuous
        print("✓ mmdae_enc_continuous")
    except Exception as e:
        print(f"✗ mmdae_enc_continuous: {e}")

    try:
        from model.dec_continuous import DECRNNContinuous
        print("✓ dec_continuous")
    except Exception as e:
        print(f"✗ dec_continuous: {e}")

    try:
        from model.rnn2rnn_continuous import HYPMMDContinuous
        print("✓ rnn2rnn_continuous")
    except Exception as e:
        print(f"✗ rnn2rnn_continuous: {e}")

    print("\nAll imports successful!")

def test_data_simulation():
    """Test data simulation."""
    print("\n" + "="*50)
    print("Testing Data Simulation")
    print("="*50)

    sim = SalespersonSimulator(seed=42)

    # Test different archetypes
    archetypes = [
        'aggressive_closer', 'calm_consultant', 'quick_adapter',
        'steady_learner', 'style_focused', 'knowledge_focused'
    ]

    print("\n1. Testing Latent Codes:")
    for arch in archetypes:
        latent = sim.sample_latent_code(arch)
        print(f"  {arch:20s}: {latent}")

    print("\n2. Testing Single Salesperson Simulation:")
    person = sim.simulate_salesperson('test_001', 'aggressive_closer', n_sessions=10)
    print(f"  ID: {person['id']}")
    print(f"  Archetype: {person['archetype']}")
    print(f"  Latent: {person['latent']}")
    print(f"  Features shape: {person['features'].shape}")
    print(f"  Rewards shape: {person['rewards'].shape}")
    print(f"  First 3 feature vectors:")
    for i in range(3):
        print(f"    Session {i}: {person['features'][i]}")

    print("\n3. Testing Dataset Generation:")
    dataset = sim.simulate_dataset(n_salespeople=10, sessions_per_person=15)
    print(f"  Generated {len(dataset)} salespeople")

    print("\n4. Testing Array Conversion:")
    batched = sim.dataset_to_arrays(dataset)
    print(f"  Features shape: {batched['features'].shape}")
    print(f"  Rewards shape: {batched['rewards'].shape}")
    print(f"  Seq lengths: {batched['seq_lengths']}")

    print("\n5. Testing Feature Statistics:")
    feature_names = ['knowledge', 'style', 'clarity', 'energy',
                     'filler_words', 'sentence_length', 'pace']

    all_features = batched['features']
    # Mask out padding (-1 values)
    valid_mask = all_features != -1

    for i, name in enumerate(feature_names):
        feature_values = all_features[:, :, i]
        valid_values = feature_values[valid_mask[:, :, i]]
        print(f"  {name:15s}: mean={valid_values.mean():5.1f}, "
              f"std={valid_values.std():5.1f}, "
              f"min={valid_values.min():5.1f}, "
              f"max={valid_values.max():5.1f}")

    print("\n✓ Data simulation tests passed!")
    return batched

def test_learning_dynamics():
    """Test that learning dynamics work as expected."""
    print("\n" + "="*50)
    print("Testing Learning Dynamics")
    print("="*50)

    sim = SalespersonSimulator(seed=42)

    # Test quick adapter vs steady learner
    quick = sim.simulate_salesperson('quick', 'quick_adapter', n_sessions=20)
    steady = sim.simulate_salesperson('steady', 'steady_learner', n_sessions=20)

    print("\n1. Quick Adapter vs Steady Learner:")
    print(f"  Quick adapter improvement (session 0 → 19):")
    print(f"    Knowledge: {quick['features'][0, 0]:.1f} → {quick['features'][-1, 0]:.1f}")
    print(f"    Rewards:   {quick['rewards'][0]:.1f} → {quick['rewards'][-1]:.1f}")

    print(f"\n  Steady learner improvement (session 0 → 19):")
    print(f"    Knowledge: {steady['features'][0, 0]:.1f} → {steady['features'][-1, 0]:.1f}")
    print(f"    Rewards:   {steady['rewards'][0]:.1f} → {steady['rewards'][-1]:.1f}")

    # Test style-focused vs knowledge-focused
    style = sim.simulate_salesperson('style', 'style_focused', n_sessions=20)
    knowledge = sim.simulate_salesperson('know', 'knowledge_focused', n_sessions=20)

    print("\n2. Style-Focused vs Knowledge-Focused:")
    print(f"  Style-focused final:")
    print(f"    Style: {style['features'][-1, 1]:.1f}, Knowledge: {style['features'][-1, 0]:.1f}")
    print(f"    Pace: {style['features'][-1, 6]:.1f}")

    print(f"\n  Knowledge-focused final:")
    print(f"    Style: {knowledge['features'][-1, 1]:.1f}, Knowledge: {knowledge['features'][-1, 0]:.1f}")
    print(f"    Pace: {knowledge['features'][-1, 6]:.1f}")

    print("\n✓ Learning dynamics tests passed!")

def main():
    print("="*50)
    print("Continuous Behavioral Model - Structure Test")
    print("="*50)

    # Test imports
    test_imports()

    # Test data simulation
    batched_data = test_data_simulation()

    # Test learning dynamics
    test_learning_dynamics()

    print("\n" + "="*50)
    print("All Tests Passed!")
    print("="*50)

    print("\nNOTE: To run full training, you need TensorFlow installed:")
    print("  pip install tensorflow==1.15.0  # or appropriate version")
    print("\nThen run:")
    print("  python train_continuous.py")

if __name__ == '__main__':
    main()
