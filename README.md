# Artificial Muscle Memory Connection

> **Using the analogy of human muscle memory in AI** — a reinforcement
> learning agent that learns to execute repeated action sequences with
> increasing efficiency and reduced computational overhead.

## Overview

This project simulates *artificial muscle memory*: the phenomenon by which a
biological organism offloads frequently practised motor programmes from
conscious deliberation to fast, automatic execution.

A neural policy network initially computes every decision from scratch (slow,
high cognitive load). As the agent repeatedly executes the same sequences
correctly, it builds a **familiarity cache**. Once a context position is
sufficiently familiar, future decisions are served instantly from the cache —
bypassing neural-network inference entirely.

### Key concepts

| Concept | Biological analogy | Implementation |
|---|---|---|
| **Sequence task** | Learning a piano piece, a typing pattern, or a sports movement | `SequenceEnvironment` — reproduce a target list of actions step by step |
| **Neural policy** | Conscious deliberation | `NeuralPolicy` — two-layer MLP trained with REINFORCE |
| **Familiarity cache** | Muscle memory / automaticity | `MuscleMemory` — counts correct executions; caches action once threshold is reached |
| **Hybrid agent** | Practised motor skill | `MuscleMemoryAgent` — checks cache first; falls back to NN only for unfamiliar contexts |
| **Efficiency metric** | Reduced cognitive load | Cache hit rate — fraction of decisions that required **no** NN inference |

## Project structure

```
.
├── muscle_memory/          # Core package
│   ├── __init__.py
│   ├── environment.py      # SequenceEnvironment (gym-style RL env)
│   ├── memory.py           # MuscleMemory (familiarity cache)
│   ├── agent.py            # NeuralPolicy + MuscleMemoryAgent
│   └── trainer.py          # Trainer + EpisodeStats
├── tests/                  # Unit tests (pytest)
│   ├── test_environment.py
│   ├── test_memory.py
│   ├── test_agent.py
│   └── test_trainer.py
├── main.py                 # Demo entry point
└── requirements.txt
```

## Requirements

- Python 3.9+
- [NumPy](https://numpy.org/) ≥ 1.22

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python main.py
```

Example output (abridged):

```
========================================================================
  Artificial Muscle Memory — Reinforcement Learning Simulation
========================================================================

Task   : reproduce 3 action sequences (3 possible actions each)
  Seq 0: [0, 1, 2, 1, 0]
  Seq 1: [2, 0, 2, 1]
  Seq 2: [1, 2, 0, 1, 2, 0]

Training for 2000 episodes…
 Episode |   Reward |  Success |  NN Calls |  Cache Hits |  Cache Rate
----------------------------------------------------------------------
     200 |     0.85 |     2.5% |       1.5 |         1.1 |       43.1%
     400 |     7.80 |    47.0% |       0.8 |         3.9 |       83.1%
     600 |    13.90 |    99.0% |       0.0 |         5.0 |       99.7%
     800 |    13.95 |   100.0% |       0.0 |         5.0 |      100.0%
    ...
    2000 |    14.06 |   100.0% |       0.0 |         5.1 |      100.0%

  Early (episodes 1–200):   success  2.5%,  cache rate  43.1%
  Late  (episodes 1801–2000): success 100.0%, cache rate 100.0%

  ✓ 95.2% of all decisions required no neural-network inference.
```

The table shows the key result: **NN Calls drop to zero** once muscle memory
is fully established, while success rate climbs to 100 %.

## Running tests

```bash
python -m pytest tests/ -v
```

All 54 tests should pass.

## How it works — step by step

1. **Environment** presents a target sequence (e.g. `[0, 1, 2, 1, 0]`).  
   The agent must output each action in order; a wrong action ends the
   episode with a penalty.

2. **MuscleMemoryAgent.act(state, context)**  
   a. Look up `context = (seq_idx, step)` in `MuscleMemory`.  
   b. **Cache hit** → return cached action immediately (no NN).  
   c. **Cache miss** → run `NeuralPolicy.forward(state)`, sample action.

3. **MuscleMemoryAgent.learn(state, action, reward, context, success)**  
   a. Apply a REINFORCE gradient update to the neural policy.  
   b. If `success=True`, increment the familiarity counter for `(context, action)`.  
   c. Once the counter reaches `familiarity_threshold`, the action is cached.

4. Over many episodes, all sequence positions become familiar →  
   **cache hit rate → 100 %** → zero NN inference calls.

## Configuration

Key parameters in `main.py` / `MuscleMemoryAgent`:

| Parameter | Default | Description |
|---|---|---|
| `familiarity_threshold` | 10 | Correct executions before caching |
| `epsilon` | 0.15 | Exploration rate (ε-greedy) |
| `lr` | 0.005 | Learning rate for REINFORCE |
| `hidden_size` | 32 | Hidden units in the policy MLP |

## Research context

This is **Phase 1** of a multi-phase research series on artificial motor
learning. Future phases may explore:

- Hierarchical sequences and sub-routine caching
- Forgetting / interference when sequences change
- Transfer learning across related sequence families
- Comparison with biologically plausible learning rules

## License

See [LICENSE](LICENSE).
