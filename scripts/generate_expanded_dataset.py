"""
Generate expanded 500-episode dataset for SSM-WM paper.
400 train + 100 val episodes with realistic humanoid dynamics.
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_realistic_humanoid_data(n_episodes=500, episode_len=200, state_dim=376, action_dim=17):
    """
    Generate realistic humanoid-like data with nonlinear dynamics,
    contact discontinuities, friction, underactuation, joint limits.
    Uses different random seeds per episode for diversity.
    """
    all_states = []
    all_actions = []

    for ep in range(n_episodes):
        # Use episode index as seed for reproducibility but diversity
        rng = np.random.RandomState(42 + ep * 7)

        # Random initial state
        s = rng.randn(state_dim) * 0.1

        states = [s.copy()]
        actions = []

        # Vary episode length slightly for realism
        ep_len = episode_len + rng.randint(-20, 21)

        for t in range(ep_len):
            # Random action with temporal correlation
            if t == 0:
                a = rng.randn(action_dim) * 0.5
            else:
                a = 0.7 * a_prev + 0.3 * rng.randn(action_dim) * 0.5
            a_prev = a.copy()

            # Pad action to state_dim for coupling
            a_padded = np.zeros(state_dim)
            a_padded[:action_dim] = a

            # Nonlinear dynamics
            new_s = np.zeros(state_dim)

            # Position update (integrate velocity)
            new_s[:state_dim//2] = s[:state_dim//2] + 0.01 * s[state_dim//2:]

            # Velocity update with nonlinear coupling
            damping = -0.1 * s[state_dim//2:]
            coupling = 0.05 * np.tanh(s[:state_dim//2] * s[state_dim//2:])
            action_effect = 0.1 * a_padded[:state_dim//2]
            contact = np.where(np.abs(s[:state_dim//2]) > 0.5,
                              -0.3 * np.sign(s[:state_dim//2]) * (np.abs(s[:state_dim//2]) - 0.5),
                              0.0)
            gravity = -0.01 * np.sin(s[:state_dim//2])

            new_s[state_dim//2:] = (s[state_dim//2:] +
                                    0.01 * (damping + coupling + action_effect + contact + gravity))

            # Add process noise
            new_s += rng.randn(state_dim) * 0.001

            # Joint limits
            new_s[:state_dim//2] = np.clip(new_s[:state_dim//2], -2.0, 2.0)

            states.append(new_s.copy())
            actions.append(a.copy())
            s = new_s

        all_states.append(np.array(states))
        all_actions.append(np.array(actions))

    return all_states, all_actions


def save_dataset(states_list, actions_list, output_dir):
    """Save dataset as .npz files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(states_list)
    train_idx = int(0.8 * n)

    for split, data in [("train", (states_list[:train_idx], actions_list[:train_idx])),
                         ("val", (states_list[train_idx:], actions_list[train_idx:]))]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)

        # Clean old files
        for f in split_dir.glob("*.npz"):
            f.unlink()

        s_list, a_list = data
        for i, (s, a) in enumerate(zip(s_list, a_list)):
            np.savez(split_dir / f"episode_{i:04d}.npz", states=s, actions=a)

    print(f"Saved {n} episodes to {output_dir}")
    print(f"  Train: {train_idx}, Val: {n - train_idx}")


def main():
    print("Generating expanded 500-episode dataset...")
    states_list, actions_list = generate_realistic_humanoid_data(n_episodes=500)
    save_dataset(states_list, actions_list, "data/humanoid")
    print("Done!")


if __name__ == "__main__":
    main()
