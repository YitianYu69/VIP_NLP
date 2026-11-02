#!/usr/bin/env python3
"""Qwen LoRA SFT for joint jargon extraction + jargon translation/definition.

Launch on one node with e.g.:
  torchrun --standalone --nproc_per_node=4 train_qwen_jargon.py \
      --final-data ../../Data/final_jargon_dataset.csv \
      --new-data ../../Data/new_jargon_data.csv \
      --output-dir ../../Model/qwen2.5-1.5b-jargon

The training objective is causal-LM SFT. The prompt tokens are masked with -100,
so loss is computed only on the assistant JSON response.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


SYSTEM_PROMPT = """You are a jargon extraction and translation assistant.
Given one English sentence, identify every domain-specific jargon term in the sentence and explain each term in plain English in the context of that sentence.
Return ONLY a valid JSON array.
Each element must contain exactly two string fields: \"jargon\" and \"translation\".
The \"jargon\" value must copy the jargon span from the input sentence.
If the sentence contains no domain-specific jargon, return []."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--final-data", default="../../Data/final_jargon_dataset.csv")
    p.add_argument("--new-data", default="../../Data/new_jargon_data.csv")
    p.add_argument("--output-dir", default="../../Model/qwen2.5-1.5b-jargon")

    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--train-batch-size", type=int, default=4,
                   help="Per-GPU microbatch size.")
    p.add_argument("--eval-batch-size", type=int, default=4,
                   help="Per-GPU validation microbatch size.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile. Compilation is enabled by default.")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--gradient-checkpointing", action="store_true")

    # Generation-based task metrics. By default the validation split is generated
    # every epoch so training logs include paper-style task metrics in addition to
    # teacher-forced loss. The untouched test split is still used only after model
    # selection.
    p.add_argument("--generation-batch-size", type=int, default=8)
    p.add_argument("--generation-max-new-tokens", type=int, default=256)
    p.add_argument("--bertscore-model", default="roberta-large")
    p.add_argument("--bertscore-batch-size", type=int, default=32)
    p.add_argument("--skip-bertscore", action="store_true",
                   help="Skip BERTScore in validation/test generation metrics.")
    p.add_argument(
        "--validation-generation-every",
        type=int,
        default=1,
        help=(
            "Generate on the validation split every N epochs and log the full benchmark "
            "suite (extraction/detection + definition metrics). Default: 1. Set 0 to disable."
        ),
    )
    return p.parse_args()


# def ddp_setup() -> tuple[int, int, int, bool]:
#     distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
#     if distributed:
#         dist.init_process_group(backend="nccl", init_method="env://")
#         rank = int(os.environ["RANK"])
#         local_rank = int(os.environ["LOCAL_RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#     else:
#         rank = 0
#         local_rank = 0
#         world_size = 1

#     torch.cuda.set_device(local_rank)
#     return rank, local_rank, world_size, distributed
def ddp_setup() -> tuple[int, int, int, bool]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, distributed


def cleanup(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_rank0(rank: int) -> bool:
    return rank == 0


def normalize_sentence(x: str) -> str:
    # Old data sometimes contains extra wrapping quotes.
    return str(x).strip().strip('"').strip()


def is_positive_row(row: pd.Series) -> bool:
    discipline = str(row.get("discipline", "")).strip().lower()
    translation = str(row.get("translation", "")).strip().lower()
    jargon = str(row.get("jargon", "")).strip()
    if not jargon or jargon.lower() == "nan":
        return False
    return discipline != "no discipline" and translation not in {"", "nan", "no translation"}


def load_and_group(final_path: str, new_path: str) -> pd.DataFrame:
    """Merge both CSVs and create ONE example per sentence.

    Grouping happens before splitting so repeated multi-jargon sentences cannot leak
    across train/validation/test sets.
    """
    frames = []
    for path in (final_path, new_path):
        path = os.path.expanduser(path)
        if path and os.path.exists(path):
            frame = pd.read_csv(path)
            required = {"sentence", "jargon", "translation", "discipline"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            frames.append(frame)
        elif path == final_path:
            raise FileNotFoundError(path)

    raw = pd.concat(frames, ignore_index=True)
    raw["sentence"] = raw["sentence"].map(normalize_sentence)

    records: List[dict] = []
    for sentence, group in raw.groupby("sentence", sort=False):
        pairs: List[dict] = []
        seen = set()
        for _, row in group.iterrows():
            if not is_positive_row(row):
                continue
            jargon = str(row["jargon"]).strip()
            translation = str(row["translation"]).strip()
            key = (jargon.casefold(), translation.casefold())
            if key not in seen:
                seen.add(key)
                pairs.append({"jargon": jargon, "translation": translation})
        records.append(
            {
                "sentence": sentence,
                "targets": pairs,
                "has_jargon": int(bool(pairs)),
            }
        )

    return pd.DataFrame(records)


def split_dataframe(df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=df["has_jargon"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=temp_df["has_jargon"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def make_user_message(sentence: str) -> str:
    return f"Sentence:\n{sentence}"


def target_to_json(targets: Sequence[dict]) -> str:
    # Deterministic compact JSON helps the model learn structure with less token noise.
    return json.dumps(list(targets), ensure_ascii=False, separators=(",", ":"))


class JargonSFTDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.rows: List[dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for row in dataframe.itertuples(index=False):
            self.rows.append(self._encode(row.sentence, row.targets))

    def __len__(self) -> int:
        return len(self.rows)

    def _build_prompt_ids(self, sentence: str) -> List[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(sentence)},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return list(ids)

    def _encode(self, sentence: str, targets: Sequence[dict]) -> dict:
        target_text = target_to_json(targets)
        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
        )["input_ids"]
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        target_ids = list(target_ids) + [self.tokenizer.eos_token_id]

        prompt_ids = self._build_prompt_ids(sentence)

        # Preserve all assistant target tokens. If a future dataset contains a very
        # long sentence, truncate only the sentence and rebuild the chat prompt.
        if len(prompt_ids) + len(target_ids) > self.max_length:
            sentence_ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"]
            lo, hi = 0, len(sentence_ids)
            best_prompt = None
            while lo <= hi:
                mid = (lo + hi) // 2
                shortened = self.tokenizer.decode(sentence_ids[:mid], skip_special_tokens=True)
                candidate = self._build_prompt_ids(shortened)
                if len(candidate) + len(target_ids) <= self.max_length:
                    best_prompt = candidate
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_prompt is None:
                raise ValueError(
                    "max_length is too small to fit the fixed prompt plus target. "
                    f"Increase --max-length above {len(target_ids)}."
                )
            prompt_ids = best_prompt

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        bs = len(features)
        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            n = len(feature["input_ids"])
            input_ids[i, :n] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(feature["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    split_dir = output_dir / "data_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        out = frame.copy()
        out["targets"] = out["targets"].map(target_to_json)
        out.to_json(split_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)


def build_model_and_tokenizer(args: argparse.Namespace, device: torch.device, rank: int):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.to(device)

    if rank == 0:
        model.print_trainable_parameters()

    # PyTorch's current guidance is to compile the inner module and then wrap it in DDP.
    # nn.Module.compile() compiles __call__ in-place, keeping the PEFT object's save API.
    if not args.no_compile:
        model.compile(mode=args.compile_mode, dynamic=True)

    return model, tokenizer


def make_loaders(
    splits: Dict[str, pd.DataFrame],
    tokenizer,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    distributed: bool,
):
    datasets = {
        name: JargonSFTDataset(frame, tokenizer, args.max_length)
        for name, frame in splits.items()
    }
    collator = CausalLMCollator(tokenizer.pad_token_id)

    samplers = {}
    for name in datasets:
        samplers[name] = (
            DistributedSampler(
                datasets[name],
                num_replicas=world_size,
                rank=rank,
                shuffle=(name == "train"),
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )

    common = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collator,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.train_batch_size,
            sampler=samplers["train"],
            shuffle=(samplers["train"] is None),
            **common,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=args.eval_batch_size,
            sampler=samplers["valid"],
            shuffle=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            sampler=samplers["test"],
            shuffle=False,
            **common,
        ),
    }
    return loaders, samplers


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_accum: int,
    max_grad_norm: float,
    distributed: bool,
    rank: int,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_num = torch.zeros((), device=device, dtype=torch.float64)
    token_den = torch.zeros((), device=device, dtype=torch.float64)
    progress = tqdm(loader, disable=(rank != 0), desc=f"train epoch {epoch}")

    num_batches = len(loader)
    for step, batch in enumerate(progress, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        valid_tokens = (batch["labels"] != -100).sum()
        is_update_step = (step % grad_accum == 0) or (step == num_batches)

        # The final accumulation group may contain fewer than grad_accum batches.
        # Normalize by the actual group size so that its gradient scale matches
        # all complete accumulation groups.
        group_start = ((step - 1) // grad_accum) * grad_accum + 1
        group_end = min(group_start + grad_accum - 1, num_batches)
        current_accum_size = group_end - group_start + 1

        sync_context = nullcontext()
        if distributed and not is_update_step:
            sync_context = model.no_sync()

        with sync_context:
            # BF16 AMP does not require GradScaler.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch, use_cache=False)
                raw_loss = outputs.loss
                loss = raw_loss / current_accum_size
            loss.backward()

        loss_num += raw_loss.detach().double() * valid_tokens.double()
        token_den += valid_tokens.double()

        if is_update_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if rank == 0:
            progress.set_postfix(loss=f"{raw_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    reduce_sum(loss_num, distributed)
    reduce_sum(token_den, distributed)
    return (loss_num / token_den.clamp_min(1)).item()


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device, distributed: bool, rank: int, name: str) -> float:
    model.eval()
    loss_num = torch.zeros((), device=device, dtype=torch.float64)
    token_den = torch.zeros((), device=device, dtype=torch.float64)
    progress = tqdm(loader, disable=(rank != 0), desc=name)

    for batch in progress:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        valid_tokens = (batch["labels"] != -100).sum()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch, use_cache=False)
        loss_num += outputs.loss.detach().double() * valid_tokens.double()
        token_den += valid_tokens.double()

    reduce_sum(loss_num, distributed)
    reduce_sum(token_den, distributed)
    return (loss_num / token_den.clamp_min(1)).item()


def save_adapter(model, tokenizer, output_path: Path, metadata: dict) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    inner = model.module if isinstance(model, DDP) else model
    inner.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    with open(output_path / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def broadcast_bool(value: bool, device: torch.device, distributed: bool) -> bool:
    if not distributed:
        return value
    t = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(t.item())


def normalize_jargon(text: str) -> str:
    """Normalization used only for scoring exact jargon-span matches."""
    return " ".join(str(text).strip().casefold().split())


def parse_generated_json(text: str) -> Tuple[bool, List[dict]]:
    """Parse the model response and enforce the exact output schema.

    Returns (is_valid_json_schema, parsed_pairs). Invalid outputs are represented
    as an empty prediction for downstream extraction metrics, while their raw text
    is still preserved in test_predictions.jsonl for error analysis.
    """
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Measure strict JSON adherence, but recover an embedded array for useful
        # downstream scoring if the model added a tiny amount of surrounding text.
        left = text.find("[")
        right = text.rfind("]")
        if left < 0 or right < left:
            return False, []
        try:
            value = json.loads(text[left : right + 1])
        except json.JSONDecodeError:
            return False, []
        strictly_valid = False
    else:
        strictly_valid = True

    if not isinstance(value, list):
        return False, []

    pairs: List[dict] = []
    for item in value:
        if not isinstance(item, dict) or set(item.keys()) != {"jargon", "translation"}:
            return False, []
        if not isinstance(item["jargon"], str) or not isinstance(item["translation"], str):
            return False, []
        pairs.append({"jargon": item["jargon"], "translation": item["translation"]})
    return strictly_valid, pairs


def load_best_adapter_for_eval(
    args: argparse.Namespace,
    best_dir: Path,
    device: torch.device,
):
    """Reload the selected adapter into a fresh, uncompiled inference model."""
    tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, best_dir, is_trainable=False)
    model.to(device).eval()
    model.config.use_cache = True
    return model, tokenizer


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    desc: str = "generate",
) -> List[dict]:
    """Greedy-generate structured jargon outputs for a dataframe on rank 0."""
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    records: List[dict] = []

    rows = list(dataframe.itertuples(index=False))
    progress = tqdm(range(0, len(rows), batch_size), desc=desc)
    for start in progress:
        batch_rows = rows[start : start + batch_size]
        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(row.sentence)},
            ]
            for row in batch_rows
        ]
        prompts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conv in conversations
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        input_width = inputs["input_ids"].shape[1]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = outputs[:, input_width:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, raw_text in zip(batch_rows, decoded):
            json_valid, pred_pairs = parse_generated_json(raw_text)
            records.append(
                {
                    "sentence": row.sentence,
                    "gold": list(row.targets),
                    "prediction": pred_pairs,
                    "raw_prediction": raw_text.strip(),
                    "json_valid": json_valid,
                }
            )

    tokenizer.padding_side = old_padding_side
    return records


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _warn_metric(metric: str, exc: Exception) -> None:
    print(f"[metrics] {metric} unavailable: {type(exc).__name__}: {exc}")


def _normalized_text(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def compute_extraction_metrics(records: List[dict]) -> dict:
    """Benchmark-style jargon extraction and sentence-level detection metrics.

    The primary extraction score is exact normalized jargon-span matching, which is
    directly analogous to entity/terminology extraction P/R/F1 used by jargon
    extraction benchmarks. Sentence detection metrics evaluate whether the model
    correctly decides that a sentence contains at least one jargon term.
    """
    tp = fp = fn = 0
    exact_sets = 0
    json_valid = 0
    negative_total = 0
    negative_correct = 0
    positive_total = 0
    positive_exact = 0
    macro_f1_positive: List[float] = []

    det_tp = det_fp = det_fn = det_tn = 0
    gold_jargon_total = 0
    pred_jargon_total = 0

    for record in records:
        json_valid += int(record["json_valid"])
        gold_pairs = record["gold"]
        pred_pairs = record["prediction"]

        gold_set = {normalize_jargon(pair["jargon"]) for pair in gold_pairs if str(pair["jargon"]).strip()}
        pred_set = {normalize_jargon(pair["jargon"]) for pair in pred_pairs if str(pair["jargon"]).strip()}

        inter = gold_set & pred_set
        tp_i = len(inter)
        fp_i = len(pred_set - gold_set)
        fn_i = len(gold_set - pred_set)
        tp += tp_i
        fp += fp_i
        fn += fn_i
        gold_jargon_total += len(gold_set)
        pred_jargon_total += len(pred_set)

        exact = pred_set == gold_set
        exact_sets += int(exact)

        gold_positive = bool(gold_set)
        pred_positive = bool(pred_set)
        if gold_positive and pred_positive:
            det_tp += 1
        elif (not gold_positive) and pred_positive:
            det_fp += 1
        elif gold_positive and (not pred_positive):
            det_fn += 1
        else:
            det_tn += 1

        if not gold_positive:
            negative_total += 1
            negative_correct += int(not pred_positive)
        else:
            positive_total += 1
            positive_exact += int(exact)
            p_i = _safe_div(tp_i, tp_i + fp_i)
            r_i = _safe_div(tp_i, tp_i + fn_i)
            macro_f1_positive.append(_safe_div(2 * p_i * r_i, p_i + r_i))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    det_precision = _safe_div(det_tp, det_tp + det_fp)
    det_recall = _safe_div(det_tp, det_tp + det_fn)
    det_f1 = _safe_div(2 * det_precision * det_recall, det_precision + det_recall)
    det_accuracy = _safe_div(det_tp + det_tn, det_tp + det_fp + det_fn + det_tn)
    specificity = _safe_div(det_tn, det_tn + det_fp)

    return {
        "num_sentences": len(records),
        "json_validity": _safe_div(json_valid, len(records)),
        # Exact entity/span-style jargon extraction benchmark.
        "jargon_exact_precision": precision,
        "jargon_exact_recall": recall,
        "jargon_exact_f1": f1,
        # Backward-compatible names from the previous evaluator.
        "jargon_micro_precision": precision,
        "jargon_micro_recall": recall,
        "jargon_micro_f1": f1,
        "jargon_set_exact_match": _safe_div(exact_sets, len(records)),
        "positive_sentence_exact_match": _safe_div(positive_exact, positive_total),
        "positive_sentence_macro_f1": (
            float(np.mean(macro_f1_positive)) if macro_f1_positive else 0.0
        ),
        # Sentence-level jargon/no-jargon benchmark.
        "jargon_detection_accuracy": det_accuracy,
        "jargon_detection_precision": det_precision,
        "jargon_detection_recall": det_recall,
        "jargon_detection_f1": det_f1,
        "jargon_detection_specificity": specificity,
        "no_jargon_accuracy": _safe_div(negative_correct, negative_total),
        "no_jargon_false_positive_rate": _safe_div(
            negative_total - negative_correct, negative_total
        ),
        "num_no_jargon_sentences": negative_total,
        "num_positive_sentences": positive_total,
        "gold_jargon_terms": gold_jargon_total,
        "predicted_jargon_terms": pred_jargon_total,
    }


def compute_generation_metrics(
    records: List[dict],
    device: torch.device,
    bertscore_model: str,
    bertscore_batch_size: int,
    skip_bertscore: bool,
) -> dict:
    """Compute extraction metrics plus definition-generation benchmark metrics.

    Definition metrics are evaluated only for correctly extracted jargon terms.
    This avoids pairing an explanation for a hallucinated span with an unrelated
    gold definition. `translation_pair_coverage` therefore remains essential when
    interpreting the definition scores.

    The benchmark suite mirrors metrics commonly reported in definition-generation
    work: BLEU, NIST, ROUGE-L, METEOR, SacreBLEU, BERTScore-F1, and exact match.
    chrF++ is retained as an additional robust lexical metric.
    """
    metrics = compute_extraction_metrics(records)

    translation_refs: List[str] = []
    translation_hyps: List[str] = []

    for record in records:
        gold_map = {}
        for pair in record["gold"]:
            key = normalize_jargon(pair["jargon"])
            gold_map.setdefault(key, str(pair["translation"]).strip())
        pred_map = {}
        for pair in record["prediction"]:
            key = normalize_jargon(pair["jargon"])
            pred_map.setdefault(key, str(pair["translation"]).strip())

        for key in sorted(set(gold_map) & set(pred_map)):
            translation_refs.append(gold_map[key])
            translation_hyps.append(pred_map[key])

    gold_jargon_total = int(metrics["gold_jargon_terms"])
    metrics["matched_translation_pairs"] = len(translation_refs)
    metrics["translation_pair_coverage"] = _safe_div(
        len(translation_refs), gold_jargon_total
    )

    definition_keys = [
        "definition_exact_match",
        "definition_bleu",
        "definition_nist",
        "definition_rougeL_f1",
        "definition_meteor",
        "definition_sacrebleu",
        "definition_chrf2",
        "definition_bertscore_precision",
        "definition_bertscore_recall",
        "definition_bertscore_f1",
    ]
    for key in definition_keys:
        metrics[key] = None

    if not translation_refs:
        return metrics

    # Exact definition match after conservative whitespace/case normalization.
    metrics["definition_exact_match"] = float(np.mean([
        _normalized_text(ref) == _normalized_text(hyp)
        for ref, hyp in zip(translation_refs, translation_hyps)
    ]))

    # NLTK: conventional BLEU + NIST + METEOR. All are optional at runtime;
    # a missing package/resource produces a warning instead of discarding every
    # already-computed extraction metric.
    try:
        from nltk.translate.bleu_score import corpus_bleu
        from nltk.translate.nist_score import corpus_nist
        from nltk.translate.meteor_score import meteor_score

        refs_tok = [[ref.split()] for ref in translation_refs]
        hyps_tok = [hyp.split() for hyp in translation_hyps]
        metrics["definition_bleu"] = float(corpus_bleu(refs_tok, hyps_tok))

        # NIST may reject n-gram orders unsupported by a short evaluation corpus.
        # Use the highest valid order, normally the standard n=5.
        for n in range(5, 0, -1):
            try:
                metrics["definition_nist"] = float(corpus_nist(refs_tok, hyps_tok, n=n))
                metrics["definition_nist_order"] = n
                break
            except (ValueError, ZeroDivisionError):
                continue

        try:
            meteor_values = [
                meteor_score([ref.split()], hyp.split())
                for ref, hyp in zip(translation_refs, translation_hyps)
            ]
            metrics["definition_meteor"] = float(np.mean(meteor_values))
        except LookupError as exc:
            _warn_metric(
                "METEOR (install NLTK WordNet: python -m nltk.downloader wordnet omw-1.4)",
                exc,
            )
    except Exception as exc:
        _warn_metric("BLEU/NIST/METEOR (pip install nltk)", exc)

    # ROUGE-L.
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, hyp)["rougeL"].fmeasure
            for ref, hyp in zip(translation_refs, translation_hyps)
        ]
        metrics["definition_rougeL_f1"] = float(np.mean(rouge_scores))
    except Exception as exc:
        _warn_metric("ROUGE-L (pip install rouge-score)", exc)

    # SacreBLEU and chrF++.
    try:
        import sacrebleu
        metrics["definition_sacrebleu"] = float(
            sacrebleu.corpus_bleu(translation_hyps, [translation_refs]).score
        )
        metrics["definition_chrf2"] = float(
            sacrebleu.corpus_chrf(
                translation_hyps,
                [translation_refs],
                word_order=2,
            ).score
        )
    except Exception as exc:
        _warn_metric("SacreBLEU/chrF++ (pip install sacrebleu)", exc)

    # BERTScore captures semantic equivalence when a definition is paraphrased.
    if not skip_bertscore:
        try:
            from bert_score import score as bert_score
            b_p, b_r, b_f1 = bert_score(
                translation_hyps,
                translation_refs,
                model_type=bertscore_model,
                lang="en",
                batch_size=bertscore_batch_size,
                device=str(device),
                verbose=True,
                rescale_with_baseline=False,
            )
            metrics["definition_bertscore_precision"] = float(b_p.mean().item())
            metrics["definition_bertscore_recall"] = float(b_r.mean().item())
            metrics["definition_bertscore_f1"] = float(b_f1.mean().item())
        except Exception as exc:
            _warn_metric("BERTScore (pip install bert-score)", exc)

    # Backward-compatible aliases used by the previous version.
    metrics["rougeL_f1"] = metrics["definition_rougeL_f1"]
    metrics["meteor"] = metrics["definition_meteor"]
    metrics["sacrebleu"] = metrics["definition_sacrebleu"]
    metrics["chrf2"] = metrics["definition_chrf2"]
    metrics["bertscore_precision"] = metrics["definition_bertscore_precision"]
    metrics["bertscore_recall"] = metrics["definition_bertscore_recall"]
    metrics["bertscore_f1"] = metrics["definition_bertscore_f1"]
    return metrics

def save_generation_report(records: List[dict], metrics: dict, output_dir: Path) -> None:
    pred_path = output_dir / "test_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metric_path = output_dir / "test_metrics.json"
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def append_validation_metrics(output_dir: Path, epoch: int, train_loss: float, val_loss: float, metrics: dict) -> None:
    """Append one epoch of validation metrics to a JSONL training history."""
    row = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        **metrics,
    }
    path = output_dir / "validation_metrics.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_validation_metrics(epoch: int, metrics: dict) -> None:
    """Compact but comprehensive per-epoch benchmark log."""
    def fmt(key: str) -> str:
        value = metrics.get(key)
        return "NA" if value is None else f"{float(value):.4f}"

    print(
        f"val_metrics epoch={epoch} | "
        f"jargon_P={fmt('jargon_exact_precision')} "
        f"jargon_R={fmt('jargon_exact_recall')} "
        f"jargon_F1={fmt('jargon_exact_f1')} "
        f"set_EM={fmt('jargon_set_exact_match')} "
        f"det_acc={fmt('jargon_detection_accuracy')} "
        f"det_F1={fmt('jargon_detection_f1')} "
        f"specificity={fmt('jargon_detection_specificity')} "
        f"no_jargon_acc={fmt('no_jargon_accuracy')} "
        f"FP_rate={fmt('no_jargon_false_positive_rate')} "
        f"JSON={fmt('json_validity')}"
    )
    print(
        f"val_definition epoch={epoch} | "
        f"coverage={fmt('translation_pair_coverage')} "
        f"BERTScore_F1={fmt('definition_bertscore_f1')} "
        f"ROUGE-L={fmt('definition_rougeL_f1')} "
        f"BLEU={fmt('definition_bleu')} "
        f"NIST={fmt('definition_nist')} "
        f"SacreBLEU={fmt('definition_sacrebleu')} "
        f"METEOR={fmt('definition_meteor')} "
        f"chrF++={fmt('definition_chrf2')} "
        f"def_EM={fmt('definition_exact_match')}"
    )


def print_generation_metrics(metrics: dict) -> None:
    print("\n=== Best-checkpoint benchmark metrics on test split ===")
    ordered = [
        "test_loss",
        "json_validity",
        "jargon_exact_precision",
        "jargon_exact_recall",
        "jargon_exact_f1",
        "jargon_set_exact_match",
        "positive_sentence_exact_match",
        "positive_sentence_macro_f1",
        "jargon_detection_accuracy",
        "jargon_detection_precision",
        "jargon_detection_recall",
        "jargon_detection_f1",
        "jargon_detection_specificity",
        "no_jargon_accuracy",
        "no_jargon_false_positive_rate",
        "translation_pair_coverage",
        "definition_bertscore_f1",
        "definition_rougeL_f1",
        "definition_bleu",
        "definition_nist",
        "definition_sacrebleu",
        "definition_meteor",
        "definition_exact_match",
        "definition_chrf2",
    ]
    for key in ordered:
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"{key:36s}: {value:.6f}")
        else:
            print(f"{key:36s}: {value}")

def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, distributed = ddp_setup()
    device = torch.device("cuda", local_rank)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training pipeline.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support native BF16 AMP.")

        set_seed(args.seed)
        torch.set_float32_matmul_precision("high")

        output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
        if is_rank0(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            # This script always starts a fresh training history. Reusing an output
            # directory should not silently mix metrics from an older experiment.
            history_path = output_dir / "validation_metrics.jsonl"
            if history_path.exists():
                history_path.unlink()

        grouped = load_and_group(args.final_data, args.new_data)
        splits = split_dataframe(grouped, args.seed)

        if is_rank0(rank):
            save_splits(splits, output_dir)
            print(
                f"examples={len(grouped)} | train={len(splits['train'])} "
                f"valid={len(splits['valid'])} test={len(splits['test'])} | "
                f"no-jargon={(grouped.has_jargon == 0).sum()}"
            )
        if distributed:
            dist.barrier()

        model, tokenizer = build_model_and_tokenizer(args, device, rank)
        loaders, samplers = make_loaders(
            splits, tokenizer, args, rank, world_size, distributed
        )

        if distributed:
            # Compile the inner PEFT module first, then apply DDP.
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )

        updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
        total_steps = updates_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val = float("inf")
        bad_epochs = 0
        best_dir = output_dir / "best_adapter"

        for epoch in range(1, args.epochs + 1):
            if distributed:
                samplers["train"].set_epoch(epoch)

            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                scheduler,
                device,
                args.gradient_accumulation_steps,
                args.max_grad_norm,
                distributed,
                rank,
                epoch,
            )
            val_loss = evaluate_loss(
                model, loaders["valid"], device, distributed, rank, "valid"
            )

            if is_rank0(rank):
                print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            # Paper-style validation monitoring during training. In addition to
            # teacher-forced val_loss, rank 0 autoregressively generates the entire
            # validation split and computes the same extraction/detection and
            # definition-generation benchmark suite used for the final test set.
            val_task_metrics = None
            if (
                args.validation_generation_every > 0
                and epoch % args.validation_generation_every == 0
            ):
                if is_rank0(rank):
                    gen_model = model.module if isinstance(model, DDP) else model
                    val_records = generate_predictions(
                        gen_model,
                        tokenizer,
                        splits["valid"],
                        device,
                        batch_size=args.generation_batch_size,
                        max_input_length=args.max_length,
                        max_new_tokens=args.generation_max_new_tokens,
                        desc=f"generate valid e{epoch}",
                    )
                    val_task_metrics = compute_generation_metrics(
                        val_records,
                        device=device,
                        bertscore_model=args.bertscore_model,
                        bertscore_batch_size=args.bertscore_batch_size,
                        skip_bertscore=args.skip_bertscore,
                    )
                    print_validation_metrics(epoch, val_task_metrics)
                    append_validation_metrics(
                        output_dir, epoch, train_loss, val_loss, val_task_metrics
                    )
                if distributed:
                    dist.barrier()

            stop = False
            if is_rank0(rank):
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                    save_adapter(
                        model,
                        tokenizer,
                        best_dir,
                        {
                            "base_model": args.model_name,
                            "best_epoch": epoch,
                            "best_val_loss": best_val,
                            "bf16_amp": True,
                            "torch_compile": not args.no_compile,
                            "world_size": world_size,
                            "lora_r": args.lora_r,
                            "lora_alpha": args.lora_alpha,
                            "lora_dropout": args.lora_dropout,
                        },
                    )
                    print(f"saved best LoRA adapter -> {best_dir}")
                else:
                    bad_epochs += 1
                    stop = bad_epochs >= args.patience
                    print(f"early-stop patience: {bad_epochs}/{args.patience}")

            stop = broadcast_bool(stop, device, distributed)
            if distributed:
                dist.barrier()
            if stop:
                break

        # Model selection is complete. Discard the final training state and reload
        # the saved best adapter before touching the test split. This guarantees
        # that both test loss and generation metrics correspond to the checkpoint
        # selected by validation loss rather than the last epoch.
        if distributed:
            dist.barrier()
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()

        best_model, eval_tokenizer = load_best_adapter_for_eval(
            args, best_dir, device
        )
        test_loss = evaluate_loss(
            best_model, loaders["test"], device, distributed, rank, "test(best)"
        )

        if is_rank0(rank):
            records = generate_predictions(
                best_model,
                eval_tokenizer,
                splits["test"],
                device,
                batch_size=args.generation_batch_size,
                max_input_length=args.max_length,
                max_new_tokens=args.generation_max_new_tokens,
                desc="generate test",
            )
            metrics = compute_generation_metrics(
                records,
                device=device,
                bertscore_model=args.bertscore_model,
                bertscore_batch_size=args.bertscore_batch_size,
                skip_bertscore=args.skip_bertscore,
            )
            metrics["test_loss"] = float(test_loss)
            metrics["best_val_loss"] = float(best_val)
            metrics["best_adapter"] = str(best_dir)
            save_generation_report(records, metrics, output_dir)

            # Also place the final evaluation summary next to the adapter itself.
            with open(best_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            print_generation_metrics(metrics)
            print(f"predictions -> {output_dir / 'test_predictions.jsonl'}")
            print(f"metrics     -> {output_dir / 'test_metrics.json'}")
            print(f"best adapter: {best_dir}")

        # Keep non-zero ranks alive while rank 0 performs autoregressive generation
        # and BERTScore, then tear the process group down together.
        if distributed:
            dist.barrier()

    finally:
        cleanup(distributed)


if __name__ == "__main__":
    main()
