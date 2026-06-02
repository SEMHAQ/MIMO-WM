"""
Generate MuJoCo Humanoid data or realistic synthetic humanoid data.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_realistic_humanoid_data(n_episodes=200, episode_len=200, state_dim=376, action_dim=17):
    """
    Generate realistic humanoid-like data with:
    - Nonlinear dynamics (not just linear + tanh)
    - Contact-like discontinuities
    - Friction-like forces
    - Underactuation
    - Joint limits
    """
    np.random.seed(42)

    all_states = []
    all_actions = []

    for ep in range(n_episodes):
        # Random initial state (joint angles and velocities)
        s = np.random.randn(state_dim) * 0.1

        states = [s.copy()]
        actions = []

        for t in range(episode_len):
            # Random action (joint torques)
            a = np.random.randn(action_dim) * 0.5

            # Pad action to state_dim for coupling
            a_padded = np.zeros(state_dim)
            a_padded[:action_dim] = a

            # Nonlinear dynamics: multiple coupled oscillators with damping
            # Simulates joint dynamics
            new_s = np.zeros(state_dim)

            # Position update (integrate velocity)
            new_s[:state_dim//2] = s[:state_dim//2] + 0.01 * s[state_dim//2:]

            # Velocity update with nonlinear coupling
            # Damping term
            damping = -0.1 * s[state_dim//2:]
            # Nonlinear coupling (simulates Coriolis/centrifugal forces)
            coupling = 0.05 * np.tanh(s[:state_dim//2] * s[state_dim//2:])
            # Action effect
            action_effect = 0.1 * a_padded[:state_dim//2]
            # Contact-like forces (piecewise linear)
            contact = np.where(np.abs(s[:state_dim//2]) > 0.5,
                              -0.3 * np.sign(s[:state_dim//2]) * (np.abs(s[:state_dim//2]) - 0.5),
                              0.0)
            # Gravity-like bias
            gravity = -0.01 * np.sin(s[:state_dim//2])

            new_s[state_dim//2:] = (s[state_dim//2:] +
                                    0.01 * (damping + coupling + action_effect + contact + gravity))

            # Add process noise
            new_s += np.random.randn(state_dim) * 0.001

            # Joint limits (clamp positions)
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

    # Split into train/val
    n = len(states_list)
    train_idx = int(0.8 * n)

    for split, data in [("train", (states_list[:train_idx], actions_list[:train_idx])),
                         ("val", (states_list[train_idx:], actions_list[train_idx:]))]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)

        s_list, a_list = data
        for i, (s, a) in enumerate(zip(s_list, a_list)):
            np.savez(split_dir / f"episode_{i:04d}.npz", states=s, actions=a)

    print(f"Saved {n} episodes to {output_dir}")
    print(f"  Train: {train_idx}, Val: {n - train_idx}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="realistic", choices=["mujoco", "realistic"])
    parser.add_argument("--output", default="data/humanoid")
    args = parser.parse_args()

    if args.type == "mujoco":
        try:
            import mujoco
            import gymnasium as gym

            print("Generating MuJoCo Humanoid-v4 data...")
            env = gym.make("Humanoid-v4")

            states_list = []
            actions_list = []

            for ep in range(200):
                obs, _ = env.reset()
                states = [obs.copy()]
                actions = []

                for t in range(200):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    states.append(obs.copy())
                    actions.append(action.copy())
                    if terminated or truncated:
                        break

                states_list.append(np.array(states))
                actions_list.append(np.array(actions))

            save_dataset(states_list, actions_list, args.output)
            print("MuJoCo data generation complete!")

        except Exception as e:
            print(f"MuJoCo failed: {e}")
            print("Falling back to realistic synthetic data...")
            args.type = "realistic"

    if args.type == "realistic":
        print("Generating realistic synthetic humanoid data...")
        states_list, actions_list = generate_realistic_humanoid_data()
        save_dataset(states_list, actions_list, args.output)
        print("Realistic synthetic data generation complete!")


if __name__ == "__main__":
    main()
