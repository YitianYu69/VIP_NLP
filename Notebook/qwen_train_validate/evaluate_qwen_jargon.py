#!/usr/bin/env python3
"""Evaluate an already-trained best Qwen LoRA adapter without retraining.

This is especially useful when training completed successfully but the old metric
stage failed because an optional evaluation package was missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_qwen_jargon import (
    CausalLMCollator,
    JargonSFTDataset,
    compute_generation_metrics,
    evaluate_loss,
    generate_predictions,
    load_best_adapter_for_eval,
    print_generation_metrics,
    save_generation_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True, help="Path to best_adapter")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument(
        "--test-split",
        default=None,
        help="Explicit split file; otherwise uses <adapter parent>/data_splits/<split>.jsonl",
    )
    p.add_argument("--output-dir", default=None, help="Defaults to adapter parent")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--generation-batch-size", type=int, default=8)
    p.add_argument("--generation-max-new-tokens", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bertscore-model", default="roberta-large")
    p.add_argument("--bertscore-batch-size", type=int, default=32)
    p.add_argument("--skip-bertscore", action="store_true")
    return p.parse_args()


def load_test_split(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", lines=True)
    if "targets" not in df.columns or "sentence" not in df.columns:
        raise ValueError(f"{path} must contain sentence and targets columns")

    def parse_targets(value):
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, list):
            return value
        raise TypeError(f"Unexpected targets value: {type(value)}")

    df["targets"] = df["targets"].map(parse_targets)
    if "has_jargon" not in df.columns:
        df["has_jargon"] = df["targets"].map(lambda x: int(bool(x)))
    return df


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support native BF16")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    adapter = Path(args.adapter).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else adapter.parent
    )
    test_path = (
        Path(args.test_split).expanduser().resolve()
        if args.test_split
        else adapter.parent / "data_splits" / f"{args.split}.jsonl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = load_test_split(test_path)
    loader_args = SimpleNamespace(model_name=args.model_name)
    model, tokenizer = load_best_adapter_for_eval(loader_args, adapter, device)

    # Teacher-forced test loss on the selected adapter.
    tokenizer.padding_side = "right"
    dataset = JargonSFTDataset(test_df, tokenizer, args.max_length)
    collator = CausalLMCollator(tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collator,
    )
    test_loss = evaluate_loss(model, loader, device, False, 0, f"{args.split}(best)")

    records = generate_predictions(
        model,
        tokenizer,
        test_df,
        device,
        batch_size=args.generation_batch_size,
        max_input_length=args.max_length,
        max_new_tokens=args.generation_max_new_tokens,
        desc=f"generate {args.split}",
    )
    metrics = compute_generation_metrics(
        records,
        device=device,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        skip_bertscore=args.skip_bertscore,
    )
    metrics[f"{args.split}_loss"] = float(test_loss)
    metrics["best_adapter"] = str(adapter)
    metrics["split"] = args.split
    metrics["split_path"] = str(test_path)
    prediction_path = output_dir / f"{args.split}_predictions.jsonl"
    metric_path = output_dir / f"{args.split}_metrics.json"
    with prediction_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print_generation_metrics(metrics)
    print(f"predictions -> {prediction_path}")
    print(f"metrics     -> {metric_path}")



if __name__ == "__main__":
    main()
