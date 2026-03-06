"""
Artificial Muscle Memory Connection – Demo
==========================================
This script shows how practice strengthens motor connections and how a
network can recall a learnt pattern from a partial cue.

Run with:
    python main.py
"""

import numpy as np

from muscle_memory import MuscleMemoryNetwork, MotorPattern


def separator(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'─' * 3} {title} {'─' * (width - len(title) - 5)}")
    else:
        print("─" * width)


def main() -> None:
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # 1. Create a network and two motor patterns                           #
    # ------------------------------------------------------------------ #
    separator("Setup")
    N_UNITS = 8
    net = MuscleMemoryNetwork(
        n_units=N_UNITS,
        learning_rate=0.15,
        decay_rate=0.02,
        rng=rng,
    )
    print(net)

    walk_pattern = MotorPattern.random(
        n_steps=6, n_units=N_UNITS, sparsity=0.5, rng=rng, name="walk"
    )
    run_pattern = MotorPattern.random(
        n_steps=6, n_units=N_UNITS, sparsity=0.5, rng=rng, name="run"
    )
    print(f"\nWalk pattern  : {walk_pattern}")
    print(f"Run  pattern  : {run_pattern}")

    # ------------------------------------------------------------------ #
    # 2. Practice – observe weight growth                                  #
    # ------------------------------------------------------------------ #
    separator("Practice")

    def mean_weight(network: MuscleMemoryNetwork) -> float:
        return float(network.weights.mean())

    print(f"Mean weight before practice : {mean_weight(net):.6f}")

    for trial in range(1, 6):
        net.practice(walk_pattern)
        net.practice(run_pattern)
        print(
            f"  After {trial:2d} trial(s) – mean weight: {mean_weight(net):.6f}"
        )

    print(f"\nPractice counts : {net.practice_counts}")

    # ------------------------------------------------------------------ #
    # 3. Show the strongest learned connections                            #
    # ------------------------------------------------------------------ #
    separator("Strongest connections (walk + run)")
    for src, dst, w in net.strongest_connections(top_k=5):
        print(f"  unit {src} → unit {dst}  weight = {w:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Recall from a cue (first step of the walk pattern)               #
    # ------------------------------------------------------------------ #
    separator("Recall")
    cue = walk_pattern.activations[0]
    recalled = net.recall(cue, n_steps=walk_pattern.n_steps - 1)
    print(f"Cue (step 0 of walk):\n  {np.round(cue, 3)}")
    print(f"\nRecalled sequence ({recalled.n_steps} steps):")
    for step_idx, row in enumerate(recalled.activations):
        print(f"  step {step_idx}: {np.round(row, 3)}")

    # ------------------------------------------------------------------ #
    # 5. Forgetting                                                        #
    # ------------------------------------------------------------------ #
    separator("Forgetting")
    before = mean_weight(net)
    net.forget(n_steps=50)
    after = mean_weight(net)
    print(f"Mean weight before forgetting : {before:.6f}")
    print(f"Mean weight after 50 steps    : {after:.6f}")
    print(
        f"Weight retained               : {after / before * 100:.1f} %"
    )

    separator()
    print("Demo complete.")


if __name__ == "__main__":
    main()
