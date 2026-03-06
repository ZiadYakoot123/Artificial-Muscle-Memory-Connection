"""
Artificial Muscle Memory — demonstration entry point.

Run with:
    python main.py

The script trains a reinforcement learning agent on a set of repeating
action sequences and prints a table showing how neural-network inference
calls decrease as muscle memory (the familiarity cache) takes over —
simulating the reduced cognitive load of well-practised motor skills.
"""

import numpy as np

from muscle_memory import MuscleMemoryAgent, SequenceEnvironment, Trainer


def main() -> None:
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Task definition
    # Each inner list is a target sequence of discrete actions (0, 1, 2).
    # The agent must reproduce them one step at a time, episode after episode.
    # ------------------------------------------------------------------
    SEQUENCES = [
        [0, 1, 2, 1, 0],           # Pattern A — 5 steps
        [2, 0, 2, 1],              # Pattern B — 4 steps
        [1, 2, 0, 1, 2, 0],       # Pattern C — 6 steps
    ]
    N_ACTIONS = 3
    N_EPISODES = 2000
    REPORT_INTERVAL = 200

    print("=" * 72)
    print("  Artificial Muscle Memory — Reinforcement Learning Simulation")
    print("=" * 72)
    print(
        f"\nTask   : reproduce {len(SEQUENCES)} action sequences "
        f"({N_ACTIONS} possible actions each)"
    )
    for i, seq in enumerate(SEQUENCES):
        print(f"  Seq {i}: {seq}")
    print()

    # ------------------------------------------------------------------
    # Build environment and agent
    # ------------------------------------------------------------------
    env = SequenceEnvironment(sequences=SEQUENCES, n_actions=N_ACTIONS)

    agent = MuscleMemoryAgent(
        state_size=env.state_size,
        n_actions=N_ACTIONS,
        hidden_size=32,
        lr=0.005,
        familiarity_threshold=10,   # 10 correct executions → cached
        epsilon=0.15,
    )

    trainer = Trainer(env=env, agent=agent)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    stats = trainer.train(n_episodes=N_EPISODES, report_interval=REPORT_INTERVAL)

    # ------------------------------------------------------------------
    # Final report — compare early vs late training windows
    # ------------------------------------------------------------------
    early_window = REPORT_INTERVAL
    late_start = N_EPISODES - REPORT_INTERVAL
    early = stats.window_stats_range(0, early_window)
    late = stats.window_stats_range(late_start, N_EPISODES)

    print("\n" + "=" * 72)
    print("  Summary: Early vs Late Training")
    print("=" * 72)
    _print_phase("Early", f"episodes 1–{early_window}", early)
    _print_phase("Late ", f"episodes {late_start + 1}–{N_EPISODES}", late)

    print("\n" + "=" * 72)
    print("  Muscle Memory Statistics")
    print("=" * 72)
    total_contexts = sum(len(s) for s in SEQUENCES)
    print(f"  Possible context positions : {total_contexts}")
    print(f"  Fully cached positions     : {agent.memory.n_cached}")
    print(f"  Familiarity ratio          : {agent.memory.familiarity_ratio():.1%}")
    print(f"  Overall cache hit rate     : {agent.cache_hit_rate:.1%}")
    print(
        f"\n  ✓ {agent.efficiency_gain:.1%} of all decisions required "
        "no neural-network inference."
    )
    print("=" * 72)


def _print_phase(label: str, description: str, ws: dict) -> None:
    if not ws:
        print(f"\n  {label} ({description}): no data")
        return
    print(f"\n  {label} ({description}):")
    print(f"    Success rate  : {ws['success_rate']:.1%}")
    print(f"    Avg NN calls  : {ws['mean_nn_calls']:.1f}")
    print(f"    Avg cache hits: {ws['mean_cache_hits']:.1f}")
    print(f"    Cache hit rate: {ws['cache_hit_rate']:.1%}")


if __name__ == "__main__":
    main()
