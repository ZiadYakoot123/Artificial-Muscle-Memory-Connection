# Artificial Muscle Memory Connection - Full Documentation

## 1. Project Overview

Artificial Muscle Memory Connection (AMMC) is a small Python simulation of how repeated practice can strengthen motor connections and how those connections can later be used to recall movement sequences.

Core idea:
- Practice strengthens connections (Hebbian-style update).
- Recall generates future activations from an initial cue.
- Forgetting decays connection strength over time.

The project is implemented with NumPy and includes:
- core library code (`muscle_memory/`)
- a runnable demo (`main.py`)
- test suite (`tests/`)
- visual/documentation tooling (`.github/prompts/`, `artifacts/`)

## 2. Folder and File Structure

```text
Artificial-Muscle-Memory-Connection/
|- .github/
|  |- prompts/
|     |- ammc-visual-analysis.prompt.md
|- artifacts/
|  |- ammc_visuals.png
|- muscle_memory/
|  |- __init__.py
|  |- network.py
|  |- pattern.py
|- tests/
|  |- __init__.py
|  |- test_network.py
|  |- test_pattern.py
|- main.py
|- README.md
|- requirements.txt
|- LICENSE
|- .gitignore
```

Also present locally (environment/build metadata):
- `.git/` (Git metadata)
- `.venv/` (virtual environment)
- `.pytest_cache/` (pytest cache)
- `__pycache__/` folders (Python bytecode cache)

## 3. File-by-File Explanation

### `main.py`
Entry-point demo script that executes the full lifecycle:
1. Setup network and random motor patterns.
2. Practice both patterns over multiple trials.
3. Print strongest learned connections.
4. Recall a sequence from a cue.
5. Apply forgetting and measure retention.
6. Print ASCII charts and Mermaid snippets.
7. Save a 2x2 Matplotlib figure to `artifacts/ammc_visuals.png`.

Key helper functions:
- `separator(title)`
- `ascii_bar_chart(values, title, width=32)`
- `print_mermaid_diagrams()`
- `save_visualizations(...)`

### `muscle_memory/__init__.py`
Package initializer.
- Exports `MuscleMemoryNetwork` and `MotorPattern`.
- Defines `__version__ = "0.1.0"`.

### `muscle_memory/pattern.py`
Defines `MotorPattern`, a sequence container for activation vectors.

Responsibilities:
- Validate activation shape (`2-D`) and value range (`[0, 1]`).
- Store activations and optional name.
- Provide convenience properties (`n_steps`, `n_units`).
- Generate random sparse patterns via `MotorPattern.random(...)`.

### `muscle_memory/network.py`
Defines `MuscleMemoryNetwork`, the learning/recall/forgetting engine.

Responsibilities:
- Initialize and manage a weight matrix `W`.
- Apply Hebbian practice updates.
- Perform recall using sigmoid propagation.
- Apply forgetting decay.
- Report connection strengths and top connections.
- Track per-pattern practice counts.

### `tests/test_pattern.py`
Unit tests for `MotorPattern`:
- construction validation
- boundary/range checks
- shape checks
- random generation behavior
- reproducibility with seeds

### `tests/test_network.py`
Unit tests for `MuscleMemoryNetwork`:
- constructor validation and defaults
- practice effects and clipping
- recall output validity
- forgetting behavior
- connection introspection methods

### `.github/prompts/ammc-visual-analysis.prompt.md`
A reusable Copilot prompt definition.

Purpose:
- Generate analysis with visual-first outputs.
- Enforce a response structure with metrics, Mermaid diagrams, ASCII charts, and plotting code.

### `README.md`
Short run instructions and visual output summary.

### `requirements.txt`
Project dependencies:
- `numpy>=1.21.0`
- `matplotlib>=3.7.0`
- `pytest>=7.0.0`

### `artifacts/ammc_visuals.png`
Generated image from `main.py` containing:
- practice trend line plot
- learned weight heatmap
- recalled activation heatmap
- forgetting curve

## 4. Core Data Model

### Motor Pattern (`MotorPattern`)
A matrix of shape `(n_steps, n_units)` where each row is one time step and each value represents activation intensity for a motor unit.

Constraints:
- 2-D numeric data
- all values in `[0.0, 1.0]`

### Network Weights (`MuscleMemoryNetwork._W`)
A matrix of shape `(n_units, n_units)`.
- `W[i, j]` is strength of connection from source unit `i` to target unit `j`.
- diagonal is forced to `0.0` (no self-connections).
- values clipped to `[0.0, 1.0]` after practice updates.

## 5. Learning, Recall, Forgetting Logic

### Practice (`practice`)
For each consecutive pair of activation rows `(a_t, a_t+1)`:
- Compute outer product: `delta = outer(a_t, a_t+1)`
- Update weights: `W += learning_rate * delta`
- Zero diagonal and clip to `[0,1]`

This models "units that fire together, wire together".

### Recall (`recall`)
Given initial cue vector `a_0`, generate `n_steps` more rows:
- `a_t+1 = sigmoid(W.T @ a_t)`

Returns a new `MotorPattern` named `"recalled"` with `n_steps + 1` rows.

### Forgetting (`forget`)
Apply exponential-like decay:
- `W *= (1 - decay_rate) ** n_steps`

Larger `n_steps` or `decay_rate` means faster forgetting.

## 6. Public API Summary

### `MotorPattern`
- `MotorPattern(activations, name="")`
- `.activations -> np.ndarray`
- `.n_steps -> int`
- `.n_units -> int`
- `MotorPattern.random(n_steps, n_units, sparsity=0.5, rng=None, name="")`

### `MuscleMemoryNetwork`
- `MuscleMemoryNetwork(n_units, learning_rate=0.1, decay_rate=0.01, rng=None)`
- `.n_units -> int`
- `.weights -> np.ndarray` (copy)
- `.practice_counts -> dict[str, int]`
- `.practice(pattern, n_trials=1) -> None`
- `.recall(initial_activation, n_steps) -> MotorPattern`
- `.forget(n_steps=1) -> None`
- `.connection_strength(unit_a, unit_b) -> float`
- `.strongest_connections(top_k=5) -> list[tuple[int, int, float]]`

## 7. Demo Execution Flow (`main.py`)

1. Create RNG with fixed seed (`42`) for deterministic demo behavior.
2. Build network (`n_units=8`, `learning_rate=0.15`, `decay_rate=0.02`).
3. Generate random `walk` and `run` patterns (`6 x 8`, `sparsity=0.5`).
4. Practice both patterns over 5 rounds and track mean weights.
5. Print top 5 strongest connections.
6. Recall from first row of `walk` and print all recalled steps.
7. Apply forgetting for 50 steps and compute retained percentage.
8. Print compact ASCII bars for practice and forgetting trends.
9. Print Mermaid flowchart and sequence diagram snippets.
10. Save combined Matplotlib visualization image.

## 8. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run demo
```bash
python main.py
```

### Run tests
```bash
pytest
```

## 9. Test Coverage Notes

`test_network.py` verifies:
- input parameter validation
- monotonic effects of practice/forgetting
- no self-connections
- clipping behavior
- shape/range guarantees for recall
- ranking behavior in strongest connections

`test_pattern.py` verifies:
- construction guards
- randomness constraints
- reproducibility with fixed seeds
- sparsity and naming behavior

## 10. Visual and Documentation Outputs

When `main.py` runs, you get:
- Terminal narrative output
- ASCII bar charts
- Mermaid snippets ready to paste in Markdown docs
- Saved figure at `artifacts/ammc_visuals.png`

## 11. Design Choices and Limitations

Design choices:
- Keep model small and interpretable.
- Use deterministic seeds in demo/tests for repeatability.
- Keep API minimal for educational clarity.

Current limitations:
- Recall uses dense sigmoid dynamics only (no thresholding or competition).
- No temporal noise model or motor hierarchy.
- No serialization/checkpoint format for trained weights.
- No CLI arguments yet for trial counts, output path, or plotting toggle.

## 12. Suggested Next Improvements

- Add CLI flags (`--trials`, `--forget-steps`, `--output-path`, `--no-plot`).
- Add save/load methods for network state.
- Add more biologically-inspired dynamics (inhibition, noise, saturation controls).
- Add benchmark script for comparing hyperparameter settings.
- Expand README with quick architecture diagram and API examples.
