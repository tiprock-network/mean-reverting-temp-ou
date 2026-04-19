# Entropy-Guided Mean-Reverting Temperature for Long-Horizon Agentic Reasoning

This repository implements inference-time temperature control for language models
using a mean-reverting Ornstein-Uhlenbeck (OU) process with entropy feedback and
KL-divergence stabilization.

## Motivation

Long-horizon reasoning tasks are not uniform. Some generation steps are low
uncertainty (where deterministic behavior is preferred), while others are high
uncertainty (where exploration helps avoid local optima).

A single fixed temperature is often suboptimal across these changing conditions.
This project treats temperature as a dynamic stochastic control variable:

- OU dynamics provide controlled exploration with mean reversion.
- Token-level entropy guides when to increase or reduce exploration.
- KL divergence to a reference policy acts as a drift guardrail.

## Core Method

The temperature process is modeled as:

```math
dT_t = \theta(\mu - T_t)dt + \sigma dW_t
```

where:

- `T_t`: current sampling temperature
- `mu`: long-run mean temperature
- `theta`: mean-reversion speed
- `sigma`: diffusion/noise scale

At each token step, entropy and KL are measured from the model logits:

- If entropy is too low, increase exploration pressure (raise `mu`).
- If entropy is too high, reduce exploration pressure (lower `mu`).
- If KL exceeds threshold, pull toward safer distributions by reducing `mu` and
	increasing reversion strength.

This produces adaptive exploration without unbounded drift.

## Repository Structure

- `main.py`: interactive CLI chat loop with adaptive OU temperature control.
- `utils/ou_eulerm.py`: core controllers.
	- `AdaptiveOUInference`: Euler-Maruyama OU update.
	- `AdaptiveOUExact`: exact OU transition update.
- `utils/math_functions.py`: entropy, KL, and OU helper math.
- `utils/benchmarking/benchmark.py`: benchmark runner on ARC-Challenge,
	TruthfulQA-MC1, and MMLU.
- `utils/multi_turn.py`: entropy-vs-temperature and turning-point analysis.
- `utils/simulations/`: standalone simulation scripts.
- `figures/`: generated example plots.

## Reproducible Setup

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install torch transformers datasets tqdm numpy matplotlib pandas colorama
```

Notes:

- GPU is optional, but strongly recommended for larger models.
- Some Hugging Face models require authentication:

```bash
huggingface-cli login
```

## Reproducible Runs

All commands below are run from repository root.

### A) Interactive adaptive OU inference

```bash
python main.py meta-llama/Llama-3.2-1B-Instruct
```

To exit the console, type:

```text
exit
```

### B) Multi-turn entropy/temperature simulation

```bash
python utils/simulations/ou_eu_sampling.py
```

This script produces curves for:

- average entropy by generation step
- average temperature by generation step
- entropy vs temperature trajectory

### C) Entropy turning-point analysis across fixed temperatures

```bash
python utils/multi_turn.py meta-llama/Llama-3.2-1B-Instruct
```

This script estimates a turning point from curvature of log-entropy vs
temperature.

### D) Benchmark on ARC, TruthfulQA, MMLU

```bash
python -m utils.benchmarking.benchmark \
	--model meta-llama/Meta-Llama-3-8B-Instruct \
	--mu 0.5 \
	--T 0.5 \
	--theta 0.2 \
	--sigma 0.05 \
	--max_tokens 50 \
	--n_samples 100 \
	--seed 42 \
	--output results.json
```

Output:

- per-benchmark softmatch accuracy
- overall softmatch accuracy
- detailed per-question predictions in `results.json`

## Slurm Jobs (Babel)

The repository includes one Slurm script per experiment run in `slurm/`:

- `slurm/run_benchmark.slurm`
- `slurm/run_multi_turn.slurm`
- `slurm/run_ou_sampling.slurm`
- `slurm/run_main_once.slurm`

### Submit benchmark

```bash
sbatch slurm/run_benchmark.slurm \
	meta-llama/Meta-Llama-3-8B-Instruct 0.5 0.5 0.2 0.05 50 100 42
```

### Submit multi-turn entropy analysis

```bash
sbatch slurm/run_multi_turn.slurm \
	meta-llama/Llama-3.2-1B-Instruct 120 100
```

### Submit OU sampling simulation

```bash
sbatch slurm/run_ou_sampling.slurm \
	meta-llama/Llama-3.2-1B-Instruct 50 80
```

### Submit one-shot main run

```bash
sbatch slurm/run_main_once.slurm \
	meta-llama/Llama-3.2-1B-Instruct \
	"Explain why adaptive temperature can help long-horizon reasoning."
```

Notes:

- All scripts auto-activate `.venv` (or `venv`) if available.
- Logs are written to `slurm_logs/`.
- Set `OU_ALLOW_CPU_FALLBACK=1` only if you intentionally want CPU fallback.

## Evaluation Focus

This project studies generation behavior under different temperature-control
strategies, including:

- entropy (diversity/uncertainty)
- average temperature trajectory
- verbosity
- instruction adherence for structured outputs
- reasoning stability under long-horizon generation

## Current Status

Implemented:

- adaptive OU generation controller (Euler and exact transition variants)
- entropy and KL feedback loop in inference
- simulation scripts for trajectory analysis
- benchmark pipeline for multiple-choice reasoning datasets

In progress / next:

- additional long-horizon datasets (for stronger error propagation analysis)
- expanded experiment tracking and report-ready result summaries

## Citation

If you use this repository, please cite the project report/work-in-progress from
the authors.