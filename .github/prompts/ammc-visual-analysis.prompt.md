---
name: "AMMC Visual Analysis"
description: "Analyze Artificial Muscle Memory simulations and return graph-rich visual explanations."
argument-hint: "Scenario, file path, or metric focus (for example: compare walk vs run after 10 trials)"
agent: "agent"
---
Create a visual-first analysis of the Artificial Muscle Memory Connection workflow.

Use slash-command arguments as the analysis target (scenario, file, or metric focus). If no argument is provided, use [main.py](../../main.py), [muscle_memory/network.py](../../muscle_memory/network.py), and [muscle_memory/pattern.py](../../muscle_memory/pattern.py).

Return output in this exact structure:

## 1) Quick Takeaway
- 2 to 4 bullets with the most important findings.

## 2) Metrics Snapshot
Provide a Markdown table with:
- Metric
- Value
- Why it matters

## 3) Visual Representations
Include all of the following:
- A Mermaid `flowchart` showing setup -> practice -> strongest connections -> recall -> forgetting.
- A Mermaid `sequenceDiagram` showing one practice cycle and one recall cycle.
- A compact ASCII bar chart for any trend you discuss (for example, mean weight growth over trials).

## 4) Graph Code (Python)
Provide runnable Python code that generates at least:
- A line chart for trend over practice trials.
- A heatmap (or matrix image) for strongest/learned connections.

Graph code requirements:
- Use `numpy` and `matplotlib` only.
- Use clear titles, axis labels, and legends when relevant.
- If concrete data is unavailable, create clearly labeled mock data that matches the described behavior.

## 5) Interpretation
- 2 bullets on what the visuals imply about learning and recall quality.
- 2 bullets proposing next experiments.

Quality constraints:
- State assumptions explicitly.
- Round displayed numeric values to a sensible precision.
- Avoid filler text.
