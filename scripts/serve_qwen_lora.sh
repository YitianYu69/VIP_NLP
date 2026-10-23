#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL to the exact training base model}"
: "${ADAPTER_DIR:?Set ADAPTER_DIR to the PEFT adapter directory}"
: "${MAX_LORA_RANK:?Set MAX_LORA_RANK to at least the adapter rank}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/validate_qwen_lora.py" "$ADAPTER_DIR" "$BASE_MODEL" "$MAX_LORA_RANK"

exec vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 --port "${VLLM_PORT:-8000}" \
  --enable-lora --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "qwen-lora=$ADAPTER_DIR"
