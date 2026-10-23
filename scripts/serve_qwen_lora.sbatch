#!/usr/bin/env bash
#SBATCH --job-name=qwen-lora-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=qwen-lora-%j.log

set -euo pipefail
: "${DEPLOY_ROOT:?Set DEPLOY_ROOT to the absolute repository directory before sbatch}"
if [[ -n "${VLLM_VENV:-}" ]]; then
  source "$VLLM_VENV/bin/activate"
fi
exec bash "$DEPLOY_ROOT/scripts/serve_qwen_lora.sh"
