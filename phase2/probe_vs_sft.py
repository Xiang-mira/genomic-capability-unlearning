"""Compare frozen linear probes with supervised fine-tuning.

This script uses the same train/val/test rows for both adaptation modes. Frozen
probe results reuse the same representation extraction and logistic-regression
logic as eval_benchmarks.py. SFT trains the selected Evo checkpoint end-to-end
with a mean-pooled classification head and validation-based early stopping.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint
from phase2.eval_benchmarks_probe_legacy import (
    apply_checkpoint,
    BenchmarkRecord,
    fit_task_layers,
    get_features_for_task,
    labels_to_int,
    parse_layers,
    read_benchmark_manifest,
)
from phase2.utils import tokenize_batch


csv.field_size_limit(sys.maxsize)

DEFAULT_TASKS = [
    "hvue_human_host_tropism",
    "hvue_human_virus_pathogenicity_cini",
    "gue_prom_300_all",
    "virobench_all_taxon_genus",
]


@dataclass
class TaskData:
    records: list
    labels: np.ndarray
    splits: np.ndarray
    num_classes: int


class MeanPoolClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim: int, num_classes: int, layer_idx: int):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.layer_idx = layer_idx
        self.num_layers = len(base_model.blocks)
        self._mask = None
        self._pooled = None
        self._hook = base_model.blocks[layer_idx].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.layer_idx + 1 < self.num_layers:
            hidden = self.base_model.blocks[self.layer_idx + 1].pre_norm(hidden)
        else:
            hidden = self.base_model.norm(hidden)
        mask = self._mask
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        self._pooled = (hidden.float() * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._mask = mask
        self._pooled = None
        _ = self.base_model(input_ids, padding_mask=mask)
        if self._pooled is None:
            raise RuntimeError(f"No hidden state captured for SFT layer {self.layer_idx}")
        return self.classifier(self._pooled)

    def remove_hook(self) -> None:
        self._hook.remove()


def maybe_limit(records: list, max_per_split: int, seed: int) -> list:
    if max_per_split <= 0:
        return records
    rng = random.Random(seed)
    limited = []
    buckets: dict[tuple[str, str], list] = {}
    for record in records:
        buckets.setdefault((record.split, record.label), []).append(record)
    for bucket in buckets.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        limited.extend(bucket[:max_per_split])
    return sorted(limited, key=lambda r: (r.split, r.record_id, r.sequence[:32]))


def load_task_data(manifest: str, task: str, max_per_split: int, seed: int) -> TaskData:
    records = [record for record in read_benchmark_manifest(manifest) if record.task == task]
    records = maybe_limit(records, max_per_split, seed)
    if not records:
        raise ValueError(f"No rows found for task={task}")
    labels = labels_to_int([record.label for record in records])
    splits = np.array([record.split for record in records])
    if not {"train", "test"}.issubset(set(splits.tolist())):
        raise ValueError(f"Task {task} must contain at least train and test splits")
    num_classes = int(len(np.unique(labels)))
    if num_classes < 2:
        raise ValueError(f"Task {task} has fewer than two classes")
    return TaskData(records=records, labels=labels, splits=splits, num_classes=num_classes)


def load_controlled_task_data(path: str, max_per_split: int, seed: int) -> TaskData:
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"sequence", "label", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Controlled split CSV missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            split = str(row.get("split", "")).lower()
            if split not in {"train", "val", "test"}:
                continue
            sequence = str(row.get("sequence", "")).upper()
            label = str(row.get("label", "")).strip()
            if not sequence or not label:
                continue
            records.append(
                BenchmarkRecord(
                    benchmark=row.get("benchmark", "host_tropism_controlled") or "host_tropism_controlled",
                    task=row.get("task", "host_tropism_controlled") or "host_tropism_controlled",
                    split=split,
                    sequence=sequence,
                    label=label,
                    family=row.get("family", row.get("group_value", "")),
                    group=row.get("group", "host_tropism_controlled"),
                    record_id=row.get("id", row.get("record_id", f"controlled|{idx}")),
                )
            )
    records = maybe_limit(records, max_per_split, seed)
    if not records:
        raise ValueError(f"No usable rows found in controlled split CSV {path}")
    labels = labels_to_int([record.label for record in records])
    splits = np.array([record.split for record in records])
    if not {"train", "test"}.issubset(set(splits.tolist())):
        raise ValueError(f"Controlled split CSV {path} must contain at least train and test rows")
    num_classes = int(len(np.unique(labels)))
    if num_classes < 2:
        raise ValueError(f"Controlled split CSV {path} has fewer than two classes")
    return TaskData(records=records, labels=labels, splits=splits, num_classes=num_classes)


def score_predictions(y_true: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    scores = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    if logits.shape[1] == 2 and len(np.unique(y_true)) == 2:
        scores["auroc"] = float(roc_auc_score(y_true, logits[:, 1]))
    elif logits.shape[1] > 2 and len(np.unique(y_true)) == logits.shape[1]:
        scores["macro_auroc"] = float(roc_auc_score(y_true, logits, multi_class="ovr", average="macro"))
    return scores


def selected_score(scores: dict) -> float:
    for key in ("auroc", "macro_auroc", "accuracy"):
        if key in scores:
            return float(scores[key])
    return float("nan")


def build_model(args, ckpt_path: Optional[str], num_classes: int):
    base = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if ckpt_path:
        apply_checkpoint(base, ckpt_path)
    hidden_dim = base.blocks[0].pre_norm.scale.shape[0]
    model = MeanPoolClassifier(base, hidden_dim, num_classes, layer_idx=args.sft_layer).to(args.device)
    return model


def iter_batches(indices: np.ndarray, batch_size: int, rng: random.Random):
    shuffled = list(indices.tolist())
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def evaluate_sft(model, records: list, labels: np.ndarray, indices: np.ndarray, tokenizer, args) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            batch_records = [records[int(i)] for i in batch_idx]
            ids, mask = tokenize_batch([record.sequence for record in batch_records], tokenizer, args.max_length, args.device)
            logits = model(ids, mask)
            all_logits.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            all_labels.append(labels[batch_idx])
    return score_predictions(np.concatenate(all_labels), np.concatenate(all_logits))


def run_sft(args, task_data: TaskData, ckpt_name: str, ckpt_path: Optional[str], seed: int) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = CharLevelTokenizer(512)
    model = build_model(args, ckpt_path, task_data.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_idx = np.where(task_data.splits == "train")[0]
    val_idx = np.where(task_data.splits == "val")[0]
    test_idx = np.where(task_data.splits == "test")[0]
    if len(val_idx) == 0:
        val_idx = test_idx

    best_val = -math.inf
    best_step = 0
    best_test_scores = None
    stale = 0
    started = time.time()
    step = 0
    while step < args.sft_steps:
        for batch_idx in iter_batches(train_idx, args.batch_size, rng):
            step += 1
            model.train()
            batch_records = [task_data.records[int(i)] for i in batch_idx]
            ids, mask = tokenize_batch([record.sequence for record in batch_records], tokenizer, args.max_length, args.device)
            y = torch.tensor(task_data.labels[batch_idx], dtype=torch.long, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = criterion(logits.float(), y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.eval_every == 0 or step == args.sft_steps:
                val_scores = evaluate_sft(model, task_data.records, task_data.labels, val_idx, tokenizer, args)
                val_score = selected_score(val_scores)
                if val_score > best_val:
                    best_val = val_score
                    best_step = step
                    best_test_scores = evaluate_sft(model, task_data.records, task_data.labels, test_idx, tokenizer, args)
                    stale = 0
                else:
                    stale += 1
                if stale >= args.patience:
                    step = args.sft_steps
                    break
            if step >= args.sft_steps:
                break

    if best_test_scores is None:
        best_test_scores = evaluate_sft(model, task_data.records, task_data.labels, test_idx, tokenizer, args)
    model.remove_hook()
    return {
        "checkpoint": ckpt_name,
        "adaptation": "sft",
        "seed": seed,
        "best_validation_step": best_step,
        "selection_score": best_val,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "elapsed_sec": time.time() - started,
        **best_test_scores,
    }


def run_probe(args, task_data: TaskData, task: str, ckpt_name: str, ckpt_path: Optional[str]) -> list[dict]:
    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if ckpt_path:
        apply_checkpoint(model, ckpt_path)
    model.eval()
    tokenizer = CharLevelTokenizer(512)
    layers = parse_layers(args.layers)
    features = get_features_for_task(
        model=model,
        task_records=task_data.records,
        tokenizer=tokenizer,
        layers=layers,
        batch_size=args.feature_batch_size,
        auto_batch_size=args.auto_batch_size,
        max_length=args.max_length,
        device=args.device,
        task_name=task,
        progress_every=args.progress_every,
        ckpt_id=ckpt_name,
        feature_cache_dir=args.feature_cache_dir,
        feature_cache_compression=args.feature_cache_compression,
        feature_cache_write=not args.no_feature_cache_write,
    )
    rows = []
    for layer, result in fit_task_layers(
        pending_layers=layers,
        features_by_layer=features,
        labels=task_data.labels,
        splits=task_data.splits,
        probe_jobs=args.probe_jobs,
        cpu_threads=args.cpu_threads,
        solver=args.probe_solver,
    ):
        if result is None:
            continue
        rows.append({
            "checkpoint": ckpt_name,
            "adaptation": "frozen_probe",
            "seed": "",
            "layer": layer,
            "best_validation_step": "",
            **result,
        })
    return rows


def parse_checkpoints(spec: str) -> list[tuple[str, Optional[str]]]:
    checkpoints: list[tuple[str, Optional[str]]] = [("base", None)]
    if not spec.strip():
        return checkpoints
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).parent.name
        checkpoints.append((name.strip(), path.strip()))
    return checkpoints


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metric_from_result_row(row: dict) -> Optional[float]:
    for key in ("auroc", "macro_auroc", "accuracy"):
        if row.get(key, "") != "":
            value = float(row[key])
            if not np.isnan(value):
                return value
    return None


def rank_consistency(probe_delta: np.ndarray, sft_delta: np.ndarray) -> Optional[float]:
    if len(probe_delta) < 2:
        return None
    total = 0
    agree = 0
    for i in range(len(probe_delta)):
        for j in range(i + 1, len(probe_delta)):
            probe_order = np.sign(probe_delta[i] - probe_delta[j])
            sft_order = np.sign(sft_delta[i] - sft_delta[j])
            if probe_order == 0 or sft_order == 0:
                continue
            total += 1
            if probe_order == sft_order:
                agree += 1
    return float(agree / total) if total else None


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    rx = average_ranks(x[mask])
    ry = average_ranks(y[mask])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def write_correlation(path: Path, rows: list[dict]) -> None:
    by_key: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        metric = metric_from_result_row(row)
        if metric is None:
            continue
        group = by_key.setdefault((row["task"], row["checkpoint"]), {"probe": [], "sft": []})
        if row["adaptation"] == "frozen_probe":
            group["probe"].append(metric)
        elif row["adaptation"] == "sft":
            group["sft"].append(metric)
    pairs = []
    for (task, checkpoint), values in by_key.items():
        if values["probe"] and values["sft"]:
            pairs.append({
                "task": task,
                "checkpoint": checkpoint,
                "probe_mean": float(np.mean(values["probe"])),
                "sft_mean": float(np.mean(values["sft"])),
            })
    if len(pairs) >= 2:
        probe = np.array([row["probe_mean"] for row in pairs])
        sft = np.array([row["sft_mean"] for row in pairs])
        pearson = float(np.corrcoef(probe, sft)[0, 1])
        spearman = spearman_corr(probe, sft)
    else:
        pearson = None
        spearman = None

    by_task = {}
    for task in sorted({row["task"] for row in pairs}):
        task_pairs = [row for row in pairs if row["task"] == task]
        if len(task_pairs) < 2:
            by_task[task] = {
                "pearson": None,
                "spearman": None,
                "rank_consistency": None,
                "probe_degradation_predicts_sft_degradation": None,
                "pairs": task_pairs,
            }
            continue
        probe = np.array([row["probe_mean"] for row in task_pairs])
        sft = np.array([row["sft_mean"] for row in task_pairs])
        task_pearson = float(np.corrcoef(probe, sft)[0, 1])
        task_spearman = spearman_corr(probe, sft)
        base_pair = next((row for row in task_pairs if row["checkpoint"] == "base"), None)
        if base_pair:
            probe_delta = np.array([base_pair["probe_mean"] - row["probe_mean"] for row in task_pairs])
            sft_delta = np.array([base_pair["sft_mean"] - row["sft_mean"] for row in task_pairs])
            degradation_corr = (
                float(np.corrcoef(probe_delta, sft_delta)[0, 1])
                if len(task_pairs) >= 2 and np.std(probe_delta) > 0 and np.std(sft_delta) > 0
                else None
            )
            predicts = (
                bool(degradation_corr > 0)
                if degradation_corr is not None
                else None
            )
            rank_value = rank_consistency(probe_delta, sft_delta)
        else:
            degradation_corr = None
            predicts = None
            rank_value = None
        by_task[task] = {
            "pearson": task_pearson,
            "spearman": task_spearman,
            "rank_consistency": rank_value,
            "probe_degradation_sft_degradation_pearson": degradation_corr,
            "probe_degradation_predicts_sft_degradation": predicts,
            "pairs": task_pairs,
        }

    payload = {
        "scientific_claim": "frozen probe metrics are valid only if they predict downstream supervised fine-tuning behavior",
        "pearson": pearson,
        "spearman": spearman,
        "by_task": by_task,
        "pairs": pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument(
        "--controlled-split-csv",
        default="",
        help="Optional host-tropism controlled split CSV with sequence,label,split columns.",
    )
    parser.add_argument(
        "--controlled-task-name",
        default="host_tropism_controlled",
        help="Task name to use for --controlled-split-csv.",
    )
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--checkpoints", default="", help="Comma-separated name=weights.safetensors entries. Base is always included.")
    parser.add_argument("--out-dir", default="data/phase2/virobench_diagnostics/probe_vs_sft")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="3-9")
    parser.add_argument(
        "--sft-layer",
        type=int,
        default=9,
        help="Layer representation used by the supervised fine-tuning classifier head.",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-per-split", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=0)
    parser.add_argument("--auto-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sft-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--probe-jobs", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--probe-solver", choices=["auto", "lbfgs", "saga"], default="auto")
    parser.add_argument("--progress-every", type=int, default=4096)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--feature-cache-compression", choices=["compressed", "none"], default="compressed")
    parser.add_argument("--no-feature-cache-write", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--sft-only", action="store_true")
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    checkpoints = parse_checkpoints(args.checkpoints)
    rows = []
    task_inputs = [(task, None) for task in tasks]
    if args.controlled_split_csv:
        task_inputs.append((args.controlled_task_name, args.controlled_split_csv))
    for task, controlled_csv in task_inputs:
        if controlled_csv:
            task_data = load_controlled_task_data(controlled_csv, args.max_per_split, min(seeds))
        else:
            task_data = load_task_data(args.benchmark_manifest, task, args.max_per_split, min(seeds))
        for ckpt_name, ckpt_path in checkpoints:
            if not args.sft_only:
                for row in run_probe(args, task_data, task, ckpt_name, ckpt_path):
                    rows.append({"task": task, **row})
            if not args.probe_only:
                for seed in seeds:
                    row = run_sft(args, task_data, ckpt_name, ckpt_path, seed)
                    rows.append({"task": task, "layer": "", **row})
            write_rows(Path(args.out_dir) / "probe_vs_sft_results.csv", rows)
            write_correlation(Path(args.out_dir) / "probe_sft_correlation.json", rows)
    print(f"[probe-vs-sft] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
