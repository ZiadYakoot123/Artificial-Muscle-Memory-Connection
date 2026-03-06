"""
Artificial Muscle Memory Connection – Demo
==========================================
This script shows how practice strengthens motor connections and how a
network can recall a learnt pattern from a partial cue.

Run with:
    python main.py

Optional:
    python main.py
    (saves plots to artifacts/ammc_visuals.png when matplotlib is available)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from muscle_memory import MuscleMemoryNetwork, MotorPattern


def separator(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'─' * 3} {title} {'─' * (width - len(title) - 5)}")
    else:
        print("─" * width)


def ascii_bar_chart(values: list[float], title: str, width: int = 32) -> None:
    """Print a compact ASCII bar chart for quick terminal visualization."""
    if not values:
        return
    vmax = max(values)
    scale = width / vmax if vmax > 0 else 0.0
    print(title)
    for idx, val in enumerate(values):
        bar_len = int(round(val * scale))
        bar = "#" * bar_len
        print(f"  {idx:>2d} | {bar:<{width}} {val:.4f}")


def print_mermaid_diagrams() -> None:
    """Show Mermaid snippets users can paste into Markdown viewers."""
    separator("Mermaid flowchart")
    print("""flowchart TD
    A[Setup] --> B[Practice]
    B --> C[Strongest Connections]
    C --> D[Recall]
    D --> E[Forgetting]
""")

    separator("Mermaid sequenceDiagram")
    print("""sequenceDiagram
    participant User
    participant Net as MuscleMemoryNetwork
    participant Pattern

    User->>Net: practice(walk/run)
    Net->>Pattern: read consecutive activations
    Net->>Net: Hebbian update (W += eta * outer)
    User->>Net: recall(cue, n_steps)
    Net->>Net: propagate with sigmoid(W^T * a_t)
    Net-->>User: recalled MotorPattern
""")


def save_visualizations(
    practice_means: list[float],
    learned_weights: np.ndarray,
    recalled: MotorPattern,
    forgetting_means: list[float],
    output_path: Path,
) -> None:
    """Save a single figure containing the main visual summaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(range(len(practice_means)), practice_means, marker="o", linewidth=2)
    ax.set_title("Practice Trend: Mean Weight")
    ax.set_xlabel("Trial Pair Index")
    ax.set_ylabel("Mean Weight")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    heat = ax.imshow(learned_weights, cmap="viridis", aspect="auto")
    ax.set_title("Learned Connection Weights")
    ax.set_xlabel("Target Unit")
    ax.set_ylabel("Source Unit")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    rec = ax.imshow(recalled.activations, cmap="magma", aspect="auto")
    ax.set_title("Recalled Activations")
    ax.set_xlabel("Motor Unit")
    ax.set_ylabel("Step")
    fig.colorbar(rec, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.plot(range(len(forgetting_means)), forgetting_means, color="tab:red", linewidth=2)
    ax.set_title("Forgetting Curve: Mean Weight")
    ax.set_xlabel("Forget Step")
    ax.set_ylabel("Mean Weight")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
    practice_means = [mean_weight(net)]

    for trial in range(1, 6):
        net.practice(walk_pattern)
        net.practice(run_pattern)
        practice_means.append(mean_weight(net))
        print(
            f"  After {trial:2d} trial(s) – mean weight: {mean_weight(net):.6f}"
        )

    print(f"\nPractice counts : {net.practice_counts}")
    separator("Practice trend (ASCII)")
    ascii_bar_chart(practice_means, title="Mean weight over trial pairs")

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
    learned_weights = net.weights
    before = mean_weight(net)
    forgetting_means = [before]
    for _ in range(50):
        net.forget(n_steps=1)
        forgetting_means.append(mean_weight(net))
    after = mean_weight(net)
    print(f"Mean weight before forgetting : {before:.6f}")
    print(f"Mean weight after 50 steps    : {after:.6f}")
    retained = (after / before * 100.0) if before > 0 else 0.0
    print(f"Weight retained               : {retained:.1f} %")

    separator("Forgetting trend (ASCII)")
    # Show every fifth step to keep terminal output compact.
    sampled = [forgetting_means[idx] for idx in range(0, len(forgetting_means), 5)]
    ascii_bar_chart(sampled, title="Mean weight every 5 forget steps")

    print_mermaid_diagrams()

    output_path = Path("artifacts") / "ammc_visuals.png"
    save_visualizations(
        practice_means=practice_means,
        learned_weights=learned_weights,
        recalled=recalled,
        forgetting_means=forgetting_means,
        output_path=output_path,
    )
    separator("Saved visuals")
    print(f"Saved graph summary to: {output_path}")

    separator()
    print("Demo complete.")


if __name__ == "__main__":
    main()
