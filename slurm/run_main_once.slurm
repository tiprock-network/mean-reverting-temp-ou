#!/bin/bash
#SBATCH --job-name=ou_main_once
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=normal
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

echo "==== Main Interactive (Single Prompt) Job ===="
echo "Host: $(hostname)"
echo "Start: $(date)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

mkdir -p slurm_logs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
else
  echo "[ERROR] No virtualenv found (.venv/bin/activate or venv/bin/activate) in $PROJECT_DIR"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
PROMPT="${2:-Explain why adaptive temperature can help long-horizon reasoning.}"

echo "Model: $MODEL"
echo "Prompt: $PROMPT"

python main.py "$MODEL" <<EOF
$PROMPT
exit
EOF

echo "Done: $(date)"
