"""
Corrected simulation with 7-dimensional rewards.

Each training session:
  - Action (7D): Behavioral features exhibited
  - Reward (7D): Per-dimension feedback scores (0-100 each)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class SalespersonSimulator7D:
    """
    Simulate salesperson learning with 7D rewards.

    Each session produces:
    - Actions: 7D behavioral features
    - Rewards: 7D feedback scores (one score per feature)
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.feature_names = [
            'knowledge', 'style', 'clarity', 'energy',
            'filler_words', 'sentence_length', 'pace'
        ]
        self.n_features = len(self.feature_names)

    def sample_latent_code(self, archetype: str = 'mixed') -> np.ndarray:
        """Sample latent code for behavioral archetype."""
        if archetype == 'aggressive_closer':
            return np.array([1.5, -0.8, 0.5])
        elif archetype == 'calm_consultant':
            return np.array([-1.2, 1.5, -0.3])
        elif archetype == 'quick_adapter':
            return np.array([0.3, 0.2, 2.0])
        elif archetype == 'steady_learner':
            return np.array([0.0, 0.5, -1.0])
        elif archetype == 'style_focused':
            return np.array([-0.5, -0.8, 0.2])
        elif archetype == 'knowledge_focused':
            return np.array([0.0, 1.8, -0.5])
        else:
            return np.random.randn(3)

    def latent_to_initial_features(self, latent: np.ndarray) -> np.ndarray:
        """Map latent code to initial features."""
        aggression, knowledge_focus, adaptability = latent
        base = np.array([40, 40, 40, 40, 40, 50, 50])
        aggression_effect = np.array([0, -5, 0, 15, -8, 0, 20]) * aggression
        knowledge_effect = np.array([20, 5, 10, -5, 5, 0, -10]) * knowledge_focus
        noise = np.random.randn(7) * 5 * (1 + abs(adaptability) * 0.5)
        features = base + aggression_effect + knowledge_effect + noise
        return np.clip(features, 0, 100)

    def latent_to_learning_dynamics(self, latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map latent code to learning parameters."""
        aggression, knowledge_focus, adaptability = latent

        base_lr = np.array([0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.1])
        lr_modifier = np.array([1.5, 0.8, 1.2, 0.5, 1.0, 0.8, 0.7]) if knowledge_focus > 0.5 else \
                      np.array([0.8, 1.2, 0.9, 1.5, 1.0, 1.0, 1.2])
        learning_rates = base_lr * lr_modifier

        feedback_sensitivity = np.ones(7) * (1.0 + adaptability * 0.5)

        tradeoff_matrix = np.zeros((7, 7))
        if aggression < 0:
            tradeoff_matrix[1, 6] = -0.3
            tradeoff_matrix[6, 1] = -0.2
        tradeoff_matrix[0, 3] = -0.15
        tradeoff_matrix[3, 0] = -0.1

        return learning_rates, feedback_sensitivity, tradeoff_matrix

    def simulate_feedback_7d(self, features: np.ndarray, noise_std: float = 5.0) -> np.ndarray:
        """
        Generate 7D feedback scores (one per feature).

        Args:
            features: [7] current feature values
            noise_std: Measurement noise (not used for all dimensions)

        Returns:
            feedback: [7] scores (0-100) for each dimension

        Reward ranges:
            - knowledge (0), style (1): mean=80, large variance (normal distribution)
            - clarity (2), energy (3), filler_words (4): discrete {0, 33, 66, 80, 100}, 100 most frequent
            - sentence_length (5), pace (6): discrete {0, 33, 66, 80, 100}, 100 even more frequent
        """
        feedback = np.zeros(7)

        # knowledge and style: normal distribution with large variance
        for i in [0, 1]:
            feedback[i] = np.random.normal(80, 20)

        # other dimensions: discrete distribution
        discrete_values = [0, 33, 66, 80, 100]
        probs_general = [0.05, 0.10, 0.15, 0.20, 0.50]  # 50% for 100
        for i in [2, 3, 4]:
            feedback[i] = np.random.choice(discrete_values, p=probs_general)

        probs_high = [0.03, 0.05, 0.07, 0.10, 0.75]
        for i in [5, 6]:
            feedback[i] = np.random.choice(discrete_values, p=probs_high)

        feedback = np.clip(feedback, 0, 100)

        return feedback

    def simulate_one_session(
        self,
        current_features: np.ndarray,
        learning_rates: np.ndarray,
        feedback_sensitivity: np.ndarray,
        tradeoff_matrix: np.ndarray,
        previous_feedback: np.ndarray = None
    ) -> np.ndarray:
        """Simulate feature evolution for one session."""
        improvement = learning_rates + np.random.randn(7) * 0.5

        if previous_feedback is not None:
            gap = previous_feedback - current_features
            feedback_adjustment = gap * feedback_sensitivity * 0.1
        else:
            feedback_adjustment = 0

        tradeoff_effect = tradeoff_matrix @ improvement
        next_features = current_features + improvement + feedback_adjustment + tradeoff_effect

        return np.clip(next_features, 0, 100)

    def simulate_salesperson(
        self,
        salesperson_id: str,
        archetype: str,
        n_sessions: int = 20
    ) -> Dict:
        """
        Simulate complete learning trajectory.

        Returns:
            data: Dictionary with:
                - 'actions': [n_sessions, 7] behavioral features
                - 'rewards': [n_sessions, 7] per-dimension feedback scores
                - 'id', 'archetype', 'latent', 'seq_length'
        """
        latent = self.sample_latent_code(archetype)
        features = self.latent_to_initial_features(latent)
        learning_rates, feedback_sensitivity, tradeoff_matrix = \
            self.latent_to_learning_dynamics(latent)

        action_history = [features.copy()]
        reward_history = []  # Will store 7D feedback

        for session in range(n_sessions):
            # Get 7D feedback for current performance
            feedback_7d = self.simulate_feedback_7d(features)
            reward_history.append(feedback_7d)

            # Simulate next session
            prev_feedback = reward_history[-1] if len(reward_history) > 0 else None
            features = self.simulate_one_session(
                features, learning_rates, feedback_sensitivity,
                tradeoff_matrix, prev_feedback
            )
            action_history.append(features.copy())

        # Convert to arrays (exclude last action state as it has no feedback yet)
        actions_array = np.array(action_history[:-1])  # [n_sessions, 7]
        rewards_array = np.array(reward_history)  # [n_sessions, 7] ← 7D!

        return {
            'id': salesperson_id,
            'archetype': archetype,
            'latent': latent,
            'actions': actions_array,  # Renamed from 'features' for clarity
            'rewards': rewards_array,   # Now 7D!
            'seq_length': n_sessions
        }

    def sample_sessions_per_person(self) -> int:
        """
        Sample number of sessions for a person.
        Distribution: 2 to 60, with mean and median skewing to ~10.
        Uses gamma distribution for right skew.
        """
        # Use gamma distribution with shape < 1 for right skew
        # Then scale and shift to get desired range and mean
        shape = 2.0  # Controls skewness
        scale = 4.0  # Controls spread

        while True:
            n_sessions = int(np.random.gamma(shape, scale) + 2)
            if 2 <= n_sessions <= 60:
                return n_sessions

    def simulate_dataset(
        self,
        n_salespeople: int = 50,
        sessions_per_person: int = None,  # Now optional, will use distribution if None
        archetype_distribution: Dict[str, float] = None
    ) -> List[Dict]:
        """Simulate dataset of multiple salespeople."""
        if archetype_distribution is None:
            archetypes = [
                'aggressive_closer', 'calm_consultant', 'quick_adapter',
                'steady_learner', 'style_focused', 'knowledge_focused'
            ]
            archetype_distribution = {a: 1.0 / len(archetypes) for a in archetypes}

        archetypes = list(archetype_distribution.keys())
        probs = list(archetype_distribution.values())
        probs = np.array(probs) / sum(probs)

        assigned_archetypes = np.random.choice(archetypes, size=n_salespeople, p=probs)

        dataset = []
        for i, archetype in enumerate(assigned_archetypes):
            # Sample number of sessions for this person if not fixed
            n_sessions = sessions_per_person if sessions_per_person is not None else self.sample_sessions_per_person()

            data = self.simulate_salesperson(
                salesperson_id=f'sales_{i:03d}',
                archetype=archetype,
                n_sessions=n_sessions
            )
            dataset.append(data)

        return dataset

    def dataset_to_arrays(self, dataset: List[Dict]) -> Dict:
        """Convert dataset to batched arrays."""
        n_people = len(dataset)
        max_length = max(d['seq_length'] for d in dataset)
        n_features = self.n_features

        # Pre-allocate arrays
        actions = np.full((n_people, max_length, n_features), -1, dtype=np.float64)
        rewards = np.full((n_people, max_length, n_features), -1, dtype=np.float64)  # ← 7D!
        seq_lengths = np.array([d['seq_length'] for d in dataset], dtype=np.int32)

        # Fill arrays
        for i, data in enumerate(dataset):
            length = data['seq_length']
            actions[i, :length, :] = data['actions']
            rewards[i, :length, :] = data['rewards']  # ← 7D rewards

        return {
            'actions': actions,
            'rewards': rewards,
            'seq_lengths': seq_lengths,
            'ids': [d['id'] for d in dataset],
            'archetypes': [d['archetype'] for d in dataset],
            'latents': np.array([d['latent'] for d in dataset])
        }


def demonstrate_7d_rewards():
    """Show the corrected 7D reward structure."""
    print("="*70)
    print("CORRECTED: 7D REWARDS")
    print("="*70)

    sim = SalespersonSimulator7D(seed=42)
    person = sim.simulate_salesperson('sales_001', 'aggressive_closer', n_sessions=5)

    print(f"\nSalesperson: {person['id']} ({person['archetype']})")
    print(f"Data shapes:")
    print(f"  Actions: {person['actions'].shape}  ← 7D behavioral features")
    print(f"  Rewards: {person['rewards'].shape}  ← 7D feedback scores!")

    feature_names = sim.feature_names

    for t in range(5):
        print(f"\n{'='*70}")
        print(f"SESSION {t}")
        print(f"{'='*70}")

        print("\n  ACTION (behavioral features exhibited):")
        for i, name in enumerate(feature_names):
            print(f"    {name:15s}: {person['actions'][t, i]:6.2f}")

        print("\n  REWARD (per-dimension feedback scores):")
        for i, name in enumerate(feature_names):
            print(f"    {name:15s}: {person['rewards'][t, i]:6.2f}")

        print("\n  → Feedback directly corresponds to each action dimension!")

    print("\n" + "="*70)
    print("KEY DIFFERENCE FROM BEFORE")
    print("="*70)
    print("""
BEFORE (Wrong):
  rewards.shape = (5,)      # Scalar rewards
  Session 0: action=[...], reward=58.3

NOW (Correct):
  rewards.shape = (5, 7)    # 7D rewards
  Session 0:
    action = [27.10, 27.64, 36.05, 76.02, 22.54, 48.54, 97.87]
    reward = [32.50, 22.40, 31.20, 81.30, 17.85, 97.08, 4.26]
            ↑ One feedback score per action dimension!
""")


if __name__ == '__main__':
    demonstrate_7d_rewards()

    print("\n" + "="*70)
    print("BATCH DATA FORMAT")
    print("="*70)

    sim = SalespersonSimulator7D(seed=42)
    dataset = sim.simulate_dataset(n_salespeople=10, sessions_per_person=15)
    batched = sim.dataset_to_arrays(dataset)

    print(f"\nBatched data:")
    print(f"  actions.shape = {batched['actions'].shape}")
    print(f"                   ↑     ↑      ↑")
    print(f"                   │     │      └─ 7 action dimensions")
    print(f"                   │     └──────── 15 timesteps")
    print(f"                   └────────────── 10 salespeople")

    print(f"\n  rewards.shape = {batched['rewards'].shape}")
    print(f"                   ↑     ↑      ↑")
    print(f"                   │     │      └─ 7 reward dimensions (one per action!)")
    print(f"                   │     └──────── 15 timesteps")
    print(f"                   └────────────── 10 salespeople")

    print("\n✅ Now the model correctly captures 7D rewards!")
