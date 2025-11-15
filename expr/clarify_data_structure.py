"""
Clarification: Data structure with clearer naming.

In the continuous behavioral model:
- "Actions" = continuous behavioral features (what the salesperson exhibits)
- "Rewards" = feedback scores (what the trainer provides)

Each timestep is an (action, reward) pair.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from expr.simulate_salesperson_data import SalespersonSimulator


def demonstrate_action_reward_pairs():
    """Show that data contains action-reward pairs."""

    print("="*70)
    print("SALESPERSON LEARNING: ACTION-REWARD PAIR STRUCTURE")
    print("="*70)

    sim = SalespersonSimulator(seed=42)
    person = sim.simulate_salesperson('sales_001', 'aggressive_closer', n_sessions=10)

    print(f"\nSalesperson: {person['id']}")
    print(f"Archetype: {person['archetype']}")
    print(f"Latent behavioral code: {person['latent']}")

    print("\n" + "="*70)
    print("TIME-ORDERED (ACTION, REWARD) SEQUENCE")
    print("="*70)

    feature_names = ['knowledge', 'style', 'clarity', 'energy',
                     'filler_words', 'sentence_length', 'pace']

    for t in range(10):
        print(f"\n{'─'*70}")
        print(f"SESSION {t}:")
        print(f"{'─'*70}")

        # ACTION: Continuous behavioral features exhibited
        print("  ACTION (behavioral features exhibited):")
        action = person['features'][t]
        for i, name in enumerate(feature_names):
            print(f"    {name:15s}: {action[i]:6.2f}")

        # REWARD: Feedback received
        print(f"\n  REWARD (feedback score): {person['rewards'][t]:6.2f}")

        # Show how action influences next state
        if t < 9:
            next_action = person['features'][t+1]
            changes = next_action - action
            print(f"\n  → Changes in next session:")
            for i, name in enumerate(feature_names):
                if abs(changes[i]) > 0.5:
                    direction = "↑" if changes[i] > 0 else "↓"
                    print(f"    {name:15s}: {direction} {abs(changes[i]):+6.2f}")

    print("\n" + "="*70)
    print("DATA STRUCTURE SUMMARY")
    print("="*70)

    print(f"\nFor each salesperson, we have:")
    print(f"  • Sequence length: {person['seq_length']} sessions")
    print(f"  • Actions (features): shape {person['features'].shape}")
    print(f"    → Each action is a 7D continuous vector")
    print(f"  • Rewards: shape {person['rewards'].shape}")
    print(f"    → Each reward is a scalar feedback score")

    print(f"\nThis gives us {person['seq_length']} (action, reward) pairs:")
    print(f"  (action_0, reward_0), (action_1, reward_1), ..., (action_9, reward_9)")

    print("\n" + "="*70)
    print("WHY 'FEATURES' INSTEAD OF 'ACTIONS'?")
    print("="*70)

    print("""
In reinforcement learning terminology:
  • DISCRETE setting:
      Action = discrete choice (e.g., "press button 2")
      Observation = what you see

  • CONTINUOUS setting:
      Action = continuous vector (e.g., "move arm to [x, y, z]")
      Observation = sensor readings

In our BEHAVIORAL modeling:
  • "Action" = behavioral features EXHIBITED by salesperson
      Not a discrete choice, but a continuous behavioral profile
      Examples: speaking with 45.2 knowledge, 38.5 style, 97.8 pace

  • "Reward" = feedback score from trainer
      Based on observed behavioral features

  • The salesperson doesn't "choose action 3"
  • They exhibit a continuous behavioral pattern
  • Hence "features" = "exhibited actions"

Both terminologies are valid:
  ✓ Call them "actions" → emphasizes agent-environment interaction
  ✓ Call them "features" → emphasizes continuous measurement

KEY POINT: The data structure is IDENTICAL either way!
           We have (action, reward) or (features, reward) pairs.
""")


def compare_discrete_vs_continuous():
    """Compare discrete vs continuous action representation."""

    print("\n" + "="*70)
    print("DISCRETE vs CONTINUOUS ACTIONS")
    print("="*70)

    print("\n📊 DISCRETE MODEL (Original):")
    print("─"*70)
    print("  Data for one timestep:")
    print("    action   = 2              (chose option 2)")
    print("    one_hot  = [0, 0, 1]      (one-hot encoding)")
    print("    reward   = 0.8            (feedback)")
    print("    ")
    print("  Sequence: [(a0=1, r0=0.5), (a1=0, r1=0.7), (a2=2, r2=0.8)]")

    print("\n📈 CONTINUOUS MODEL (Ours):")
    print("─"*70)
    print("  Data for one timestep:")
    print("    action   = [45.2, 38.5, 42.1, ...]  (7D behavioral vector)")
    print("    encoding = [45.2, 38.5, 42.1, ...]  (no encoding needed!)")
    print("    reward   = 58.3                     (feedback)")
    print("    ")
    print("  Sequence: [(a0, r0), (a1, r1), (a2, r2)]")
    print("    where each a_t is a 7D continuous vector")

    print("\n🔑 KEY INSIGHT:")
    print("─"*70)
    print("""
  Both models have (action, reward) pairs!

  Discrete:  action ∈ {0, 1, 2, ..., N}     (finite set)
  Continuous: action ∈ ℝ^D                   (continuous space)

  Our salesperson model uses D=7 dimensional continuous actions.
""")


def show_model_input():
    """Show what actually goes into the model."""

    print("\n" + "="*70)
    print("WHAT THE MODEL RECEIVES")
    print("="*70)

    sim = SalespersonSimulator(seed=42)
    dataset = sim.simulate_dataset(n_salespeople=3, sessions_per_person=5)
    batched = sim.dataset_to_arrays(dataset)

    print("\nBatched data structure:")
    print(f"  features: {batched['features'].shape}")
    print(f"            ↑        ↑         ↑")
    print(f"            │        │         └─ 7 feature dimensions")
    print(f"            │        └─────────── 5 timesteps")
    print(f"            └──────────────────── 3 salespeople")

    print(f"\n  rewards:  {batched['rewards'].shape}")
    print(f"            ↑        ↑")
    print(f"            │        └─ 5 timesteps (one reward per action)")
    print(f"            └────────── 3 salespeople")

    print("\nPer salesperson, per timestep:")
    print("  Input to model:")
    print("    • Previous action (features_{t-1}): [7] continuous values")
    print("    • Previous reward (reward_{t-1}):   [1] scalar value")
    print("    ")
    print("  Output from model:")
    print("    • Predicted action (features_t):    [7] continuous values")
    print("    ")
    print("  Training loss:")
    print("    • MSE between predicted and actual features_t")

    print("\n✅ This IS an action-reward structured sequence!")


if __name__ == '__main__':
    demonstrate_action_reward_pairs()
    compare_discrete_vs_continuous()
    show_model_input()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The simulated data DOES contain action-reward pairs!

  ✓ Actions = 'features' array (continuous behavioral vectors)
  ✓ Rewards = 'rewards' array (feedback scores)
  ✓ Structure = time-ordered sequence of (action_t, reward_t) pairs

The naming "features" vs "actions" is just terminology.
Both refer to the same thing: what the agent (salesperson) exhibits.

The model is correctly structured for continuous behavioral modeling!
""")
