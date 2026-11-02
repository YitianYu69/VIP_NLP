#!/usr/bin/env python3
"""Inference for a trained Qwen jargon LoRA adapter."""

from __future__ import annotations

import argparse
import json
from typing import Any, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen_jargon import SYSTEM_PROMPT, make_user_message


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--sentence", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def extract_json_array(text: str) -> List[dict]:
    """Strict first, then recover a JSON array if surrounding text slipped through."""
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        left = text.find("[")
        right = text.rfind("]")
        if left < 0 or right < left:
            raise ValueError(f"Model output is not a JSON array: {text!r}")
        value = json.loads(text[left : right + 1])

    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    for item in value:
        if not isinstance(item, dict) or set(item.keys()) != {"jargon", "translation"}:
            raise ValueError(f"Invalid item schema: {item!r}")
        if not isinstance(item["jargon"], str) or not isinstance(item["translation"], str):
            raise ValueError(f"Invalid field types: {item!r}")
    return value


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support native BF16")

    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.to(device).eval()
    model.config.use_cache = True

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_message(args.sentence)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = extract_json_array(text)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
