#!/usr/bin/env python3
"""Check a PEFT Qwen LoRA before starting a vLLM server."""

import argparse
import json
import sys
from pathlib import Path


def validate(adapter_dir: Path, base_model: str, max_lora_rank: int) -> None:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing {config_path}")
    if not any((adapter_dir / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"Missing adapter weights in {adapter_dir}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {config_path}: {exc}") from exc

    if config.get("peft_type") != "LORA":
        raise ValueError("Adapter config must have peft_type=LORA")
    if config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Adapter config must have task_type=CAUSAL_LM for chat serving")
    trained_base = config.get("base_model_name_or_path")
    if not trained_base or trained_base != base_model:
        raise ValueError(f"Base model mismatch: adapter was trained on {trained_base!r}, requested {base_model!r}")
    rank = config.get("r")
    if type(rank) is not int or rank < 1:
        raise ValueError(f"Invalid adapter rank: {rank!r}")
    if max_lora_rank not in (1, 8, 16, 32, 64, 128, 256, 320, 512):
        raise ValueError(f"Unsupported MAX_LORA_RANK={max_lora_rank}; choose a value accepted by vLLM")
    if max_lora_rank < rank:
        raise ValueError(f"MAX_LORA_RANK={max_lora_rank} is lower than adapter rank {rank}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("base_model")
    parser.add_argument("max_lora_rank", type=int)
    args = parser.parse_args()
    try:
        validate(args.adapter_dir, args.base_model, args.max_lora_rank)
    except ValueError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1
    print("LoRA preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
