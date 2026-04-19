#!/bin/bash
#SBATCH --job-name=ou_benchmark
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

echo "==== Adaptive OU Benchmark Job ===="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p slurm_logs outputs/results

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
else
  echo "[ERROR] No virtualenv found (.venv/bin/activate or venv/bin/activate) in $PROJECT_DIR"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

if [[ -z "${HF_TOKEN:-}" ]]; then
  _hf_cache="$HOME/.cache/huggingface/token"
  [[ -f "$_hf_cache" ]] && export HF_TOKEN="$(cat "$_hf_cache")" || true
fi

if ! python - <<'PY'
import sys
import torch
available = torch.cuda.is_available()
print(f"Torch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {available}")
sys.exit(0 if available else 1)
PY
then
  if [[ "${OU_ALLOW_CPU_FALLBACK:-0}" == "1" ]]; then
    echo "[WARN] CUDA unavailable; continuing on CPU because OU_ALLOW_CPU_FALLBACK=1"
  else
    echo "[ERROR] CUDA unavailable; set OU_ALLOW_CPU_FALLBACK=1 to force CPU run"
    exit 1
  fi
fi

MODEL="${1:-meta-llama/Meta-Llama-3-8B-Instruct}"
MU="${2:-0.5}"
T_INIT="${3:-0.5}"
THETA="${4:-0.2}"
SIGMA="${5:-0.05}"
MAX_TOKENS="${6:-50}"
N_SAMPLES="${7:-100}"
SEED="${8:-42}"
OUTFILE="${9:-outputs/results/benchmark_${SLURM_JOB_ID:-local}.json}"

echo "Model: $MODEL"
echo "Output: $OUTFILE"

python -m utils.benchmarking.benchmark \
  --model "$MODEL" \
  --mu "$MU" \
  --T "$T_INIT" \
  --theta "$THETA" \
  --sigma "$SIGMA" \
  --max_tokens "$MAX_TOKENS" \
  --n_samples "$N_SAMPLES" \
  --seed "$SEED" \
  --output "$OUTFILE"

echo "Done: $(date)"
