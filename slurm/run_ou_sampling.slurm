#!/bin/bash
#SBATCH --job-name=ou_sampling
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

echo "==== OU Sampling Job ===="
echo "Host: $(hostname)"
echo "Start: $(date)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p slurm_logs outputs/figures

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
else
  echo "[ERROR] No virtualenv found (.venv/bin/activate or venv/bin/activate) in $PROJECT_DIR"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
NUM_SAMPLES="${2:-50}"
MAX_NEW_TOKENS="${3:-80}"
SAVE_DIR="${4:-outputs/figures/ou_sampling_${SLURM_JOB_ID:-local}}"

echo "Model: $MODEL"
echo "Figure dir: $SAVE_DIR"

python utils/simulations/ou_eu_sampling.py \
  --model "$MODEL" \
  --num_samples "$NUM_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --save_dir "$SAVE_DIR"

echo "Done: $(date)"
