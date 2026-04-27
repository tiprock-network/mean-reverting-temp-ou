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

- `main.py`: Interactive CLI chat loop with adaptive OU temperature control.
- `config.yaml`: Central config for model, dataset, temperature, and experiment parameters.
- `utils/ou_eulerm.py`: Core controllers (OU, Adaptive OU, OU+KL).
- `utils/math_functions.py`: Entropy, KL, and OU helper math.
- `utils/benchmarking/benchmark.py`: Benchmark runner (now supports GSM8K, BBH, Causal Judgement, ARC, TruthfulQA, MMLU).
- `utils/multi_turn.py`: Entropy-vs-temperature and turning-point analysis.
- `utils/simulations/`: Standalone simulation scripts:
    - `ou_plain.py`: Plain OU (no feedback)
    - `ou_adaptive.py`: Adaptive OU (entropy feedback)
    - `turn_fixed.py`: TURN (fixed temperature baseline)
- `figures/`: Generated example plots.
- `outputs/figures/`: Output directory for metrics and plots.

## Workflow: Reproducible Experiments

All experiments are configured via `config.yaml` and results are logged to JSON in `outputs/figures/`.

### 1. Multi-turn entropy/temperature analysis

Run:
```bash
python utils/multi_turn.py meta-llama/Llama-3.2-1B-Instruct
```
This produces:
- Entropy and temperature dynamics plots
- Entropy vs. temperature curve
- Entropy acceleration (elbow) plot

**Purpose:** Use these plots to set the elbow temperature and entropy thresholds in `config.yaml` for all subsequent experiments.

### 2. Plain OU, Adaptive OU, and TURN Baselines

Run (all use config.yaml for parameters):
```bash
python utils/simulations/ou_plain.py
python utils/simulations/ou_adaptive.py
python utils/simulations/turn_fixed.py
```
Each script logs metrics (entropy, temperature, output text) to JSON in `outputs/figures/`.

### 3. Full Benchmarking

Run:
```bash
python -m utils.benchmarking.benchmark --model <model_id> --output outputs/figures/benchmark_results.json
```
This evaluates the selected model and temperature controller on GSM8K, BBH, Causal Judgement, ARC, TruthfulQA, and MMLU (as configured).

### 4. Interactive Demo

```bash
python main.py meta-llama/Llama-3.2-1B-Instruct
```
Type `exit` to quit.

## Multi-Turn Analysis Example

See `Analysis.tex` for a sample writeup and example plots from the multi-turn analysis. This step is critical for calibrating temperature and entropy thresholds for all experiments.

## Supported Datasets and Models

You can switch between:
- Datasets: GSM8K (`openai/gsm8k`), BBH (`BBEH/bbeh`), Causal Judgement (`allenanie/causal_judgment`), ARC, TruthfulQA, MMLU
- Models: meta-llama/Llama-3.2-1B-Instruct, meta-llama/Meta-Llama-3-8B-Instruct, Qwen/Qwen2-7B-Instruct

Edit `config.yaml` to change models, datasets, or experiment parameters.

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