# Artificial-Muscle-Memory-Connection
Using analogy for human muscle memory connection in AI 

## Run Demo

```bash
python main.py
```

## Run With Docker

Build the image:

```bash
docker build -t ammc .
```

Run the demo container:

```bash
docker run --rm -v "${PWD}/artifacts:/app/artifacts" ammc
```

If you use Docker Compose:

```bash
docker compose up --build
```

The demo now includes visual representations:
- ASCII bar charts for practice and forgetting trends in terminal output
- Mermaid flowchart and sequence diagram snippets for documentation
- A saved plot image at `artifacts/ammc_visuals.png` containing:
	- practice trend line chart
	- learned-weights heatmap
	- recalled-activation heatmap
	- forgetting curve line chart
