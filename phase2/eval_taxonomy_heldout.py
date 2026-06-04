"""
Taxonomy-held-out probe evaluation for Phase 2.

This script keeps benchmark engineering narrow: Host Tropism is evaluated from
the existing local manifest with taxonomy columns, while CINI is evaluated only
when the provided local input already contains usable taxonomy/group metadata.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint
from phase2.eval_benchmarks import (
    apply_checkpoint,
    append_rows,
    extract_features_for_layers,
    fit_task_layers,
    labels_to_int,
    parse_layers,
    summarize,
    tune_runtime,
)


TAXONOMY_COLUMNS = ["family", "genus", "species", "virus_tax_id", "accession", "id"]
CINI_TASK = "hvue_human_virus_pathogenicity_cini"


@dataclass
class TaxonomyRecord:
    benchmark: str
    task: str
    sequence: str
    label: str
    group_value: str
    record_id: str
    original_split: str = ""
    split: str = ""


def normalize_label(label: object) -> Optional[str]:
    value = str(label).strip()
    if not value or value.lower() in {"nan", "na", "none", "null"}:
        return None
    try:
        numeric = float(value)
    except ValueError:
        return value
    if not np.isfinite(numeric):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g")


def clean_sequence(seq: object) -> str:
    seq = str(seq).upper()
    return "".join(ch for ch in seq if ch in {"A", "C", "G", "T", "N"})


def read_csv_rows(path: str) -> Tuple[List[dict], List[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def resolve_group_key(fieldnames: Iterable[str], requested: str) -> Optional[str]:
    fields = set(fieldnames)
    if requested != "auto":
        return requested if requested in fields else None
    for key in TAXONOMY_COLUMNS:
        if key in fields:
            return key
    return None


def load_host_tropism_records(path: str, group_key: str) -> Tuple[List[TaxonomyRecord], str]:
    rows, fieldnames = read_csv_rows(path)
    selected_key = resolve_group_key(fieldnames, group_key)
    if selected_key is None:
        raise ValueError(
            f"No usable taxonomy group key in {path}. Requested={group_key}; "
            f"available={fieldnames}"
        )

    records: List[TaxonomyRecord] = []
    for idx, row in enumerate(rows):
        label = normalize_label(row.get("label", ""))
        seq = clean_sequence(row.get("sequence", ""))
        group_value = str(row.get(selected_key, "")).strip()
        if not label or not seq or not group_value:
            continue
        records.append(
            TaxonomyRecord(
                benchmark="host_tropism",
                task="host_tropism_taxonomy_heldout",
                sequence=seq,
                label=label,
                group_value=group_value,
                record_id=row.get("id") or row.get("accession") or f"host_tropism|{idx}",
                original_split=str(row.get("split", "")).lower(),
            )
        )
    return records, selected_key


def load_cini_records(path: str, group_key: str) -> Tuple[List[TaxonomyRecord], Optional[str], Optional[str]]:
    rows, fieldnames = read_csv_rows(path)
    selected_key = resolve_group_key(fieldnames, group_key)
    if selected_key is None:
        return [], None, (
            "CINI taxonomy-held-out skipped: local input does not contain taxonomy "
            f"columns. Available columns: {fieldnames}"
        )

    filtered = rows
    if "task" in fieldnames:
        filtered = [row for row in rows if row.get("task") == CINI_TASK]
    if not filtered:
        return [], selected_key, f"CINI taxonomy-held-out skipped: no rows for task={CINI_TASK}"

    records: List[TaxonomyRecord] = []
    for idx, row in enumerate(filtered):
        label = normalize_label(row.get("label", ""))
        seq = clean_sequence(row.get("sequence", ""))
        group_value = str(row.get(selected_key, "")).strip()
        if not label or not seq or not group_value:
            continue
        records.append(
            TaxonomyRecord(
                benchmark=row.get("benchmark", "hvue") or "hvue",
                task=row.get("task", CINI_TASK) or CINI_TASK,
                sequence=seq,
                label=label,
                group_value=group_value,
                record_id=row.get("id") or f"cini|{idx}",
                original_split=str(row.get("split", "")).lower(),
            )
        )
    if not records:
        return [], selected_key, (
            f"CINI taxonomy-held-out skipped: group key {selected_key!r} exists but "
            "all rows have missing sequence, label, or group values."
        )
    unique_groups = {record.group_value for record in records}
    if len(unique_groups) < 3:
        return [], selected_key, (
            f"CINI taxonomy-held-out skipped: group key {selected_key!r} has only "
            f"{len(unique_groups)} unique group value(s), so train/val/test group-held-out "
            "evaluation is not meaningful."
        )
    return records, selected_key, None


def count_by_split_label(records: List[TaxonomyRecord]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        counts[record.split][record.label] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def group_label_counts(records: List[TaxonomyRecord]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        counts[record.group_value][record.label] += 1
    return counts


def choose_group_splits(
    records: List[TaxonomyRecord],
    val_frac: float,
    test_frac: float,
    seed: int,
    max_eval_groups: int,
) -> Dict[str, str]:
    groups = group_label_counts(records)
    group_names = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    group_names.sort(key=lambda name: sum(groups[name].values()), reverse=True)

    total = len(records)
    targets = {
        "test": max(1, int(round(total * test_frac))),
        "val": max(1, int(round(total * val_frac))),
        "train": total,
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    assignment: Dict[str, str] = {}

    for group in group_names:
        size = sum(groups[group].values())
        if split_counts["test"] < targets["test"]:
            split = "test"
        elif split_counts["val"] < targets["val"]:
            split = "val"
        else:
            split = "train"
        assignment[group] = split
        split_counts[split] += size

    # If a split is single-class, move one compatible group from train when possible.
    for split in ("val", "test"):
        labels = Counter()
        for group, group_split in assignment.items():
            if group_split == split:
                labels.update(groups[group])
        if len(labels) >= 2:
            continue
        present = set(labels)
        needed = [label for label in sorted({r.label for r in records}) if label not in present]
        for label in needed:
            candidates = [
                group
                for group in group_names
                if assignment[group] == "train" and groups[group].get(label, 0) > 0
            ]
            if candidates:
                assignment[candidates[-1]] = split

    if max_eval_groups > 0:
        eval_groups = [g for g, s in assignment.items() if s in {"val", "test"}]
        if len(eval_groups) > max_eval_groups:
            keep = set(eval_groups[:max_eval_groups])
            for group in eval_groups:
                if group not in keep:
                    assignment[group] = "train"
    return assignment


def apply_group_splits(records: List[TaxonomyRecord], assignment: Dict[str, str]) -> None:
    for record in records:
        record.split = assignment[record.group_value]


def apply_random_splits(
    records: List[TaxonomyRecord],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    by_label: Dict[str, List[TaxonomyRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    for label_records in by_label.values():
        rng.shuffle(label_records)
        n_total = len(label_records)
        n_test = max(1, int(round(n_total * test_frac)))
        n_val = max(1, int(round(n_total * val_frac)))
        for idx, record in enumerate(label_records):
            if idx < n_test:
                record.split = "test"
            elif idx < n_test + n_val:
                record.split = "val"
            else:
                record.split = "train"


def validate_records(records: List[TaxonomyRecord], min_per_split_class: int) -> List[str]:
    issues = []
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        label_counts = Counter(record.label for record in split_records)
        if len(label_counts) < 2:
            issues.append(f"{split} split has fewer than two labels: {dict(label_counts)}")
        for label, count in label_counts.items():
            if count < min_per_split_class:
                issues.append(
                    f"{split} split label={label} has {count} rows, below minimum {min_per_split_class}"
                )
    return issues


def write_skip(out_dir: str, dataset: str, reason: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "taxonomy_heldout_summary.json")
    splits_path = os.path.join(out_dir, "taxonomy_heldout_splits.json")
    payload = {
        "status": "skipped",
        "dataset": dataset,
        "reason": reason,
        "groups": {},
        "tasks": {},
        "benchmarks": {},
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(splits_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[tax] skipped: {reason}")
    print(f"[tax] wrote {summary_path}")


def write_split_summary(
    path: str,
    records: List[TaxonomyRecord],
    dataset: str,
    group_key: str,
    split_mode: str,
    seed: int,
    validation_issues: List[str],
) -> None:
    groups = group_label_counts(records)
    group_splits: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record.group_value].add(record.split)
    payload = {
        "status": "complete" if not validation_issues else "warning",
        "dataset": dataset,
        "group_key": group_key,
        "split_mode": split_mode,
        "seed": seed,
        "n_records": len(records),
        "n_groups": len(groups),
        "split_label_counts": count_by_split_label(records),
        "validation_issues": validation_issues,
        "groups": {
            group: {
                "split": sorted(group_splits[group])[0]
                if len(group_splits[group]) == 1
                else "mixed",
                "n": int(sum(counts.values())),
                "label_counts": dict(counts),
            }
            for group, counts in sorted(groups.items())
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def metric_from_row(row: dict) -> Optional[float]:
    for key in ("auroc", "macro_auroc", "accuracy"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def write_taxonomy_summary(path: str, rows: List[dict], extra: Dict[str, object]) -> None:
    payload = summarize(rows)
    metrics = [metric_from_row(row) for row in rows]
    metrics = [value for value in metrics if value is not None]
    payload["taxonomy_heldout"] = {
        **extra,
        "mean_score": float(np.mean(metrics)) if metrics else None,
        "n_task_layers": len(metrics),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def evaluate_records(args, records: List[TaxonomyRecord], group_key: str, validation_issues: List[str]) -> None:
    layers = parse_layers(args.layers)
    out_dir = args.out_dir or (
        os.path.dirname(args.ckpt) if args.ckpt else os.path.join("data/phase2/taxonomy_heldout", "base")
    )
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "taxonomy_heldout.csv")
    summary_path = os.path.join(out_dir, "taxonomy_heldout_summary.json")
    split_path = os.path.join(out_dir, "taxonomy_heldout_splits.json")
    if os.path.exists(results_path):
        os.remove(results_path)

    task_records = sorted(records, key=lambda record: (record.task, record.record_id))
    write_split_summary(
        split_path,
        task_records,
        args.dataset,
        group_key,
        args.split_mode,
        args.seed,
        validation_issues,
    )

    tune_runtime(args.device, args.cpu_threads)
    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if args.ckpt:
        apply_checkpoint(model, args.ckpt)
    else:
        print("[tax] evaluating base model without unlearning checkpoint")
    model.eval()
    tokenizer = CharLevelTokenizer(512)

    rows: List[dict] = []
    tasks = defaultdict(list)
    for record in task_records:
        tasks[record.task].append(record)

    started = time.time()
    for task_idx, (task, task_group_records) in enumerate(sorted(tasks.items()), start=1):
        sequences = [record.sequence for record in task_group_records]
        labels = labels_to_int(record.label for record in task_group_records)
        splits = np.array([record.split for record in task_group_records])
        pending_layers = layers
        print(
            f"[tax] task {task_idx}/{len(tasks)} extracting task={task} "
            f"n={len(task_group_records)} group_key={group_key} layers={pending_layers}"
        )
        features_by_layer = extract_features_for_layers(
            model=model,
            sequences=sequences,
            tokenizer=tokenizer,
            layers=pending_layers,
            batch_size=args.batch_size,
            auto_batch_size=args.auto_batch_size,
            max_length=args.max_length,
            device=args.device,
            task_name=task,
            progress_every=args.progress_every,
        )
        layer_results = fit_task_layers(
            pending_layers=pending_layers,
            features_by_layer=features_by_layer,
            labels=labels,
            splits=splits,
            probe_jobs=args.probe_jobs,
            cpu_threads=args.cpu_threads,
            solver=args.probe_solver,
        )
        task_rows: List[dict] = []
        for layer, result in layer_results:
            if result is None:
                print(f"[tax] skipped task={task} layer={layer}: insufficient splits/classes")
                continue
            row = {
                "benchmark": task_group_records[0].benchmark,
                "task": task,
                "group": "taxonomy_heldout",
                "layer": layer,
                **result,
            }
            rows.append(row)
            task_rows.append(row)
            metric = metric_from_row(row)
            print(f"  layer {layer:>2}: score={metric:.4f} acc={row['accuracy']:.4f}")
        append_rows(results_path, task_rows)
        write_taxonomy_summary(
            summary_path,
            rows,
            {
                "status": "complete" if not validation_issues else "warning",
                "dataset": args.dataset,
                "group_key": group_key,
                "split_mode": args.split_mode,
                "validation_issues": validation_issues,
                "elapsed_sec": time.time() - started,
            },
        )
        del features_by_layer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_taxonomy_summary(
        summary_path,
        rows,
        {
            "status": "complete" if not validation_issues else "warning",
            "dataset": args.dataset,
            "group_key": group_key,
            "split_mode": args.split_mode,
            "validation_issues": validation_issues,
            "elapsed_sec": time.time() - started,
        },
    )
    print(f"[tax] wrote taxonomy-held-out results to {results_path}")
    print(f"[tax] wrote taxonomy-held-out summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["host_tropism", "cini"], default="host_tropism")
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument("--cini-input", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--group-key", default="auto")
    parser.add_argument(
        "--split-mode",
        choices=["taxonomy", "random"],
        default="taxonomy",
        help="taxonomy holds out whole taxonomy groups; random stratifies rows by label.",
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--auto-batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--probe-jobs", type=int, default=7)
    parser.add_argument("--probe-solver", choices=["auto", "lbfgs", "saga"], default="auto")
    parser.add_argument("--progress-every", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layers", default="3-9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--min-per-split-class", type=int, default=1)
    parser.add_argument(
        "--max-eval-groups",
        type=int,
        default=0,
        help="Optional smoke-test cap on total val+test groups; 0 keeps all groups.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (
        os.path.dirname(args.ckpt) if args.ckpt else os.path.join("data/phase2/taxonomy_heldout", "base")
    )
    if args.dataset == "host_tropism":
        records, group_key = load_host_tropism_records(args.manifest, args.group_key)
    else:
        records, group_key, skip_reason = load_cini_records(args.cini_input, args.group_key)
        if skip_reason:
            write_skip(out_dir, args.dataset, skip_reason)
            return
        assert group_key is not None

    if not records:
        write_skip(out_dir, args.dataset, "No usable records after filtering.")
        return

    if args.split_mode == "taxonomy":
        assignment = choose_group_splits(
            records=records,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            max_eval_groups=args.max_eval_groups,
        )
        apply_group_splits(records, assignment)
    else:
        apply_random_splits(records, args.val_frac, args.test_frac, args.seed)
    validation_issues = validate_records(records, args.min_per_split_class)
    if validation_issues:
        print("[tax] split validation warnings:")
        for issue in validation_issues:
            print(f"  - {issue}")
    evaluate_records(args, records, group_key, validation_issues)


if __name__ == "__main__":
    main()
