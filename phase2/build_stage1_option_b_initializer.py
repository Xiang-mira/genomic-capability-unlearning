"""Build a minimal TAR Option B classification-CE initializer checkpoint.

This runner is intentionally small-scope:
1. supervised elicitation on one formal target task with a pooled classifier;
2. classification-CE ascent on the same target while preserving retain LM CE;
3. save the adapted base-model tensors as a Stage 1 initializer artifact.

The first supported use case is host-only formal Stage 1 work. Once a usable
CINI disjoint source exists, the same runner can be pointed at that manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint, read_manifest
from phase2.checkpoint_io import save_checkpoint, set_trainable_by_suffixes, snapshot_state
from phase2.eval_benchmarks import BenchmarkRecord, apply_checkpoint, read_benchmark_manifest
from phase2.lora_utils import PooledEvoClassifier, encode_labels
from phase2.run_metadata import build_run_metadata, write_metadata
from phase2.utils import get_trainable_params, iterate_batches, language_model_loss, tokenize_batch

csv.field_size_limit(sys.maxsize)


def parse_tasks(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def split_records(records: Iterable[BenchmarkRecord]) -> dict[str, list[BenchmarkRecord]]:
    buckets = {"train": [], "val": [], "test": []}
    for record in records:
        split = "val" if record.split in {"dev", "valid", "validation"} else record.split
        if split in buckets:
            buckets[split].append(record)
    return buckets


def deterministic_label_subset(
    records: list[BenchmarkRecord],
    max_rows: int,
    seed: int,
    salt: str,
) -> list[BenchmarkRecord]:
    if max_rows <= 0 or len(records) <= max_rows:
        return list(records)
    rng = random.Random(f"{seed}:{salt}")
    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    selected: list[BenchmarkRecord] = []
    labels = sorted(grouped)
    per_label_cap = max(1, max_rows // max(1, len(labels)))
    for label in labels:
        bucket = list(grouped[label])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_label_cap])
    remaining = [record for record in records if record not in selected]
    if len(selected) < max_rows and remaining:
        rng.shuffle(remaining)
        selected.extend(remaining[: max_rows - len(selected)])
    return sorted(selected[:max_rows], key=lambda row: (row.split, row.record_id, row.sequence[:32]))


def summarize_labels(records: list[BenchmarkRecord]) -> dict[str, int]:
    return dict(Counter(record.label for record in records))


def load_target_splits(args: argparse.Namespace) -> dict[str, list[BenchmarkRecord]]:
    records = read_benchmark_manifest(
        args.benchmark_manifest,
        benchmark_scope="task",
        task_filter={args.target_task},
        requested_split_type=args.split_type,
    )
    splits = split_records(records)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(f"Target task {args.target_task} must contain train/val/test rows")
    splits["train"] = deterministic_label_subset(splits["train"], args.target_train_max_rows, args.seed, "target_train")
    splits["val"] = deterministic_label_subset(splits["val"], args.target_val_max_rows, args.seed, "target_val")
    splits["test"] = deterministic_label_subset(splits["test"], args.target_test_max_rows, args.seed, "target_test")
    return splits


def load_retain_train_records(args: argparse.Namespace):
    records = [record for record in read_manifest(args.retain_csv) if record.split == "train"]
    if args.retain_max_rows > 0 and len(records) > args.retain_max_rows:
        rng = random.Random(f"{args.seed}:retain")
        records = list(records)
        rng.shuffle(records)
        records = sorted(records[: args.retain_max_rows], key=lambda row: row.record_id)
    if not records:
        raise ValueError(f"No retain train rows found in {args.retain_csv}")
    return records


def score_logits(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    pred = probs.argmax(axis=1)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    if probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
    return metrics


def evaluate_classifier(
    model: PooledEvoClassifier,
    records: list[BenchmarkRecord],
    encoded_labels: np.ndarray,
    tokenizer: CharLevelTokenizer,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    model.eval()
    logits_all = []
    labels_all = []
    total_loss = 0.0
    total_weight = 0
    with torch.no_grad():
        for start in range(0, len(records), args.eval_batch_size):
            batch = records[start : start + args.eval_batch_size]
            labels = encoded_labels[start : start + len(batch)]
            ids, mask = tokenize_batch([record.sequence for record in batch], tokenizer, args.max_length, args.device)
            targets = torch.tensor(labels, dtype=torch.long, device=args.device)
            logits = model(ids, mask)
            loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            total_loss += float(loss.item()) * len(batch)
            total_weight += len(batch)
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(labels)
    merged_logits = np.concatenate(logits_all, axis=0)
    merged_labels = np.concatenate(labels_all, axis=0)
    return total_loss / max(1, total_weight), score_logits(merged_labels, merged_logits)


def target_step(
    model: PooledEvoClassifier,
    batch: list[BenchmarkRecord],
    labels: np.ndarray,
    tokenizer: CharLevelTokenizer,
    args: argparse.Namespace,
) -> torch.Tensor:
    ids, mask = tokenize_batch([record.sequence for record in batch], tokenizer, args.max_length, args.device)
    targets = torch.tensor(labels, dtype=torch.long, device=args.device)
    logits = model(ids, mask)
    return F.cross_entropy(logits.float(), targets, reduction="mean")


def retain_step(
    base_model,
    batch,
    tokenizer: CharLevelTokenizer,
    args: argparse.Namespace,
) -> torch.Tensor:
    ids, mask = tokenize_batch([record.sequence for record in batch], tokenizer, args.max_length, args.device)
    logits, _ = base_model(ids, padding_mask=mask)
    return language_model_loss(logits, ids, mask)


def build_initializer_metadata(
    args: argparse.Namespace,
    *,
    weights_path: Path,
    save_result,
    layers: list[int],
    selected_names: set[str],
    target_splits: dict[str, list[BenchmarkRecord]],
    retain_records: list,
    final_val_metrics: dict[str, float],
    final_test_metrics: dict[str, float],
) -> dict[str, object]:
    extra = {
        "method": "tar_option_b_initializer",
        "target_task": args.target_task,
        "split_type": args.split_type,
        "benchmark_manifest": args.benchmark_manifest,
        "retain_csv": args.retain_csv,
        "layers": layers,
        "trainable_suffixes": args.trainable_suffixes,
        "checkpoint_policy": args.checkpoint_policy,
        "selected_tensor_count": len(selected_names),
        "weights_path": str(weights_path),
        "save_result": {
            "saved": save_result.saved,
            "tensor_count": save_result.tensor_count,
            "checkpoint_policy": save_result.checkpoint_policy,
        },
        "elicitation_steps": args.elicitation_steps,
        "ascent_steps": args.ascent_steps,
        "alpha_target": args.alpha_target,
        "alpha_retain": args.alpha_retain,
        "target_counts": {split: len(rows) for split, rows in target_splits.items()},
        "target_label_counts": {split: summarize_labels(rows) for split, rows in target_splits.items()},
        "retain_train_rows": len(retain_records),
        "val_metrics_after_ascent": final_val_metrics,
        "test_metrics_after_ascent": final_test_metrics,
        "readout_disruption_flag": "readout_disruption",
    }
    return build_run_metadata(
        args=args,
        source_checkpoint=args.model_dir,
        init_checkpoint=args.init_ckpt or "",
        output_checkpoint=str(weights_path) if save_result.saved else "",
        data_paths=[args.benchmark_manifest, args.retain_csv],
        trainable_modules=[args.trainable_suffixes],
        trainable_tensor_names=sorted(selected_names),
        trainable_param_count=len(selected_names),
        loss_layers=layers,
        seed=args.seed,
        checkpoint_policy=args.checkpoint_policy,
        extra=extra,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--target-task", default="hvue_human_host_tropism")
    parser.add_argument("--split-type", default="cluster_disjoint")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--target-train-max-rows", type=int, default=256)
    parser.add_argument("--target-val-max-rows", type=int, default=128)
    parser.add_argument("--target-test-max-rows", type=int, default=128)
    parser.add_argument("--retain-max-rows", type=int, default=256)
    parser.add_argument("--elicitation-steps", type=int, default=20)
    parser.add_argument("--ascent-steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--elicitation-lr", type=float, default=1e-5)
    parser.add_argument("--ascent-lr", type=float, default=1e-5)
    parser.add_argument("--alpha-target", type=float, default=1.0)
    parser.add_argument("--alpha-retain", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="5-9")
    parser.add_argument("--trainable-suffixes", default="all")
    parser.add_argument("--checkpoint-policy", choices=["selected_modules", "delta", "full"], default="selected_modules")
    parser.add_argument("--init-ckpt", default="")
    parser.add_argument("--out-dir", default="data/phase2/stage1_option_b_initializer/hostonly")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    target_splits = load_target_splits(args)
    retain_records = load_retain_train_records(args)
    print(
        f"[option-b] target rows train/val/test="
        f"{len(target_splits['train'])}/{len(target_splits['val'])}/{len(target_splits['test'])} "
        f"labels train={summarize_labels(target_splits['train'])}"
    )
    print(f"[option-b] retain train rows={len(retain_records)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.safetensors"
    meta_path = out_dir / "meta.json"
    summary_path = out_dir / "summary.json"
    log_path = out_dir / "train_log.jsonl"

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if args.init_ckpt:
        apply_checkpoint(model, args.init_ckpt)

    layers = []
    for part in args.layers.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    selected_names = set_trainable_by_suffixes(model, layers, args.trainable_suffixes)
    init_state = snapshot_state(model, selected_names if args.checkpoint_policy == "delta" else None)

    hidden_dim = int(model.blocks[0].pre_norm.scale.shape[0])
    label_encoding = encode_labels(record.label for record in target_splits["train"] + target_splits["val"] + target_splits["test"])[1]
    if label_encoding is None or label_encoding.num_classes < 2:
        raise ValueError("Option B target task must have at least two labels")
    model_wrapper = PooledEvoClassifier(model, hidden_dim, label_encoding.num_classes, "classification").to(args.device)
    tokenizer = CharLevelTokenizer(args.max_length)

    def encode_target(records: list[BenchmarkRecord]) -> np.ndarray:
        return np.array([label_encoding.label_to_id[record.label] for record in records], dtype=np.int64)

    y_train = encode_target(target_splits["train"])
    y_val = encode_target(target_splits["val"])
    y_test = encode_target(target_splits["test"])

    phase1_params = list(get_trainable_params(model)) + [param for param in model_wrapper.head.parameters() if param.requires_grad]
    phase1_optimizer = torch.optim.AdamW(phase1_params, lr=args.elicitation_lr)
    phase2_optimizer = None
    best_val_auroc = -float("inf")
    best_state = None
    started = time.time()

    def maybe_log(payload: dict[str, object]) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")

    try:
        rng = random.Random(args.seed)
        train_indices = list(range(len(target_splits["train"])))
        for step in range(1, args.elicitation_steps + 1):
            rng.shuffle(train_indices)
            batch_idx = train_indices[: min(args.train_batch_size, len(train_indices))]
            batch = [target_splits["train"][idx] for idx in batch_idx]
            batch_labels = y_train[batch_idx]
            phase1_optimizer.zero_grad(set_to_none=True)
            loss = target_step(model_wrapper, batch, batch_labels, tokenizer, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phase1_params, args.grad_clip)
            phase1_optimizer.step()
            if step % args.eval_every == 0 or step == args.elicitation_steps:
                val_loss, val_metrics = evaluate_classifier(model_wrapper, target_splits["val"], y_val, tokenizer, args)
                test_loss, test_metrics = evaluate_classifier(model_wrapper, target_splits["test"], y_test, tokenizer, args)
                val_auroc = float(val_metrics.get("auroc", val_metrics.get("accuracy", float("nan"))))
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_state = {
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in selected_names},
                        "head": {k: v.detach().cpu().clone() for k, v in model_wrapper.head.state_dict().items()},
                        "step": step,
                    }
                maybe_log(
                    {
                        "phase": "elicitation",
                        "step": step,
                        "train_loss": float(loss.item()),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "test_loss": test_loss,
                        "test_metrics": test_metrics,
                        "elapsed_sec": time.time() - started,
                    }
                )
                print(f"[option-b] elicitation step={step} train_loss={loss.item():.4f} val_auroc={val_metrics.get('auroc')}")

        if best_state is not None:
            current = model.state_dict()
            for key, value in best_state["model"].items():
                current[key].copy_(value.to(current[key].dtype).to(current[key].device))
            model_wrapper.head.load_state_dict(best_state["head"])
            print(f"[option-b] restored elicitation best_step={best_state['step']} val_auroc={best_val_auroc}")

        for param in model_wrapper.head.parameters():
            param.requires_grad_(False)
        phase2_params = list(get_trainable_params(model))
        phase2_optimizer = torch.optim.AdamW(phase2_params, lr=args.ascent_lr)
        retain_rng = random.Random(f"{args.seed}:retain")
        target_rng = random.Random(f"{args.seed}:target")

        for step in range(1, args.ascent_steps + 1):
            target_batch = next(iterate_batches(target_splits["train"], args.train_batch_size, shuffle=True, rng=target_rng))
            retain_batch = next(iterate_batches(retain_records, args.train_batch_size, shuffle=True, rng=retain_rng))
            target_batch_labels = encode_target(target_batch)
            phase2_optimizer.zero_grad(set_to_none=True)
            target_ce = target_step(model_wrapper, target_batch, target_batch_labels, tokenizer, args)
            retain_ce = retain_step(model, retain_batch, tokenizer, args)
            loss = (-args.alpha_target * target_ce) + (args.alpha_retain * retain_ce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phase2_params, args.grad_clip)
            phase2_optimizer.step()
            if step % args.eval_every == 0 or step == args.ascent_steps:
                val_loss, val_metrics = evaluate_classifier(model_wrapper, target_splits["val"], y_val, tokenizer, args)
                test_loss, test_metrics = evaluate_classifier(model_wrapper, target_splits["test"], y_test, tokenizer, args)
                maybe_log(
                    {
                        "phase": "ascent",
                        "step": step,
                        "loss": float(loss.item()),
                        "target_ce": float(target_ce.item()),
                        "retain_ce": float(retain_ce.item()),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "test_loss": test_loss,
                        "test_metrics": test_metrics,
                        "elapsed_sec": time.time() - started,
                    }
                )
                print(
                    f"[option-b] ascent step={step} loss={loss.item():.4f} "
                    f"target_ce={target_ce.item():.4f} retain_ce={retain_ce.item():.4f} "
                    f"val_auroc={val_metrics.get('auroc')}"
                )

        save_result = save_checkpoint(
            model,
            str(weights_path),
            policy=args.checkpoint_policy,
            layers=layers,
            suffixes=args.trainable_suffixes,
            init_state=init_state if args.checkpoint_policy == "delta" else None,
            metadata={
                "method_family": "tar_option_b_initializer",
                "target_task": args.target_task,
                "split_type": args.split_type,
                "elicitation_steps": args.elicitation_steps,
                "ascent_steps": args.ascent_steps,
                "trainable_suffixes": args.trainable_suffixes,
            },
        )
        final_val_loss, final_val_metrics = evaluate_classifier(model_wrapper, target_splits["val"], y_val, tokenizer, args)
        final_test_loss, final_test_metrics = evaluate_classifier(model_wrapper, target_splits["test"], y_test, tokenizer, args)
        meta = build_initializer_metadata(
            args,
            weights_path=weights_path,
            save_result=save_result,
            layers=layers,
            selected_names=selected_names,
            target_splits=target_splits,
            retain_records=retain_records,
            final_val_metrics=final_val_metrics,
            final_test_metrics=final_test_metrics,
        )
        write_metadata(meta_path, meta)
        summary_path.write_text(
            json.dumps(
                {
                    "val_loss": final_val_loss,
                    "val_metrics": final_val_metrics,
                    "test_loss": final_test_loss,
                    "test_metrics": final_test_metrics,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[option-b] wrote {weights_path}")
        print(f"[option-b] wrote {meta_path}")
    finally:
        model_wrapper.close()
        del model_wrapper, phase1_optimizer, phase2_optimizer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
