"""Evaluate k-mer composition baselines on benchmark-style manifests."""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


csv.field_size_limit(sys.maxsize)

FIELDNAMES = [
    "benchmark",
    "task",
    "split_type",
    "group",
    "baseline",
    "kmer_min",
    "kmer_max",
    "kmer_binary",
    "max_length",
    "best_c",
    "selection_metric",
    "selection_score",
    "n_train",
    "n_val",
    "n_test",
    "accuracy",
    "f1",
    "auroc",
    "auprc",
]


@dataclass
class Record:
    benchmark: str
    task: str
    split_type: str
    group: str
    split: str
    sequence: str
    label: str


def parse_task_filter(spec: str) -> Optional[set[str]]:
    if not spec:
        return None
    return {part.strip() for part in spec.split(",") if part.strip()}


def clean_sequence(value: object, max_length: int) -> str:
    seq = str(value or "").upper()
    seq = "".join(ch for ch in seq if ch in {"A", "C", "G", "T", "N"})
    if max_length > 0:
        seq = seq[:max_length]
    return seq


def normalize_split(value: object) -> str:
    split = str(value or "").lower()
    if split in {"dev", "valid", "validation"}:
        return "val"
    return split


def normalize_split_type(value: object) -> str:
    split_type = str(value or "").strip().lower()
    if not split_type:
        return "random"
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type


def infer_group(task: str, benchmark: str, fallback: str) -> str:
    if fallback:
        return fallback
    if benchmark == "hvue" or task.startswith("hvue_"):
        return "hvue_forget"
    if benchmark == "host_tropism_hiyata" or task == "host_tropism_hiyata":
        return "host_tropism_adaptation"
    return "unspecified"


def read_manifest(args) -> list[Record]:
    task_filter = parse_task_filter(args.task_filter)
    records: list[Record] = []
    with open(args.benchmark_manifest, newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "sequence", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for row in reader:
            benchmark = row.get("benchmark") or args.default_benchmark
            task = row.get("task") or args.default_task
            if task_filter is not None and task not in task_filter:
                continue
            split = normalize_split(row.get("split"))
            label = str(row.get("label", "")).strip()
            seq = clean_sequence(row.get("sequence", ""), args.max_length)
            if split not in {"train", "val", "test"} or not label or not seq:
                continue
            split_type = normalize_split_type(row.get("split_type"))
            group = infer_group(task, benchmark, row.get("group", "") or args.default_group)
            records.append(Record(benchmark, task, split_type, group, split, seq, label))
    return records


def encode_labels(labels: Iterable[str]) -> tuple[np.ndarray, Dict[str, int]]:
    labels = list(labels)
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([label_to_id[label] for label in labels], dtype=np.int64), label_to_id


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auroc": None,
        "auprc": None,
    }
    observed = np.unique(y_true)
    try:
        if prob.shape[1] == 2 and len(observed) == 2:
            positive = prob[:, 1]
            metrics["auroc"] = float(roc_auc_score(y_true, positive))
            metrics["auprc"] = float(average_precision_score(y_true, positive))
        elif prob.shape[1] > 2 and len(observed) == prob.shape[1]:
            metrics["auroc"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return metrics


def selection_score(metrics: dict) -> tuple[str, float]:
    for key in ("auroc", "f1", "accuracy"):
        value = metrics.get(key)
        if value is not None and math.isfinite(float(value)):
            return key, float(value)
    return "accuracy", float("-inf")


def fit_task(task: str, split_type: str, records: list[Record], args) -> Optional[dict]:
    splits = np.array([record.split for record in records])
    y, _label_to_id = encode_labels(record.label for record in records)
    if len(np.unique(y)) < 2:
        print(f"[kmer] skip task={task} split_type={split_type}: fewer than two classes")
        return None
    masks = {name: splits == name for name in ("train", "val", "test")}
    for name, mask in masks.items():
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            print(f"[kmer] skip task={task} split_type={split_type}: split {name} missing rows or classes")
            return None

    sequences = [record.sequence for record in records]
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(args.kmer_min, args.kmer_max),
        lowercase=False,
        binary=args.kmer_binary,
    )
    x = vectorizer.fit_transform(sequences)

    best_clf = None
    best_c = None
    best_metric_name = ""
    best_score = -float("inf")
    for c in [float(part) for part in args.c_grid.split(",") if part.strip()]:
        clf = LogisticRegression(
            C=c,
            solver=args.solver,
            max_iter=args.max_iter,
            class_weight="balanced",
            n_jobs=args.n_jobs,
        )
        clf.fit(x[masks["train"]], y[masks["train"]])
        val_prob = clf.predict_proba(x[masks["val"]])
        val_metrics = compute_metrics(y[masks["val"]], val_prob)
        metric_name, score = selection_score(val_metrics)
        if score > best_score:
            best_clf = clf
            best_c = c
            best_metric_name = metric_name
            best_score = score

    assert best_clf is not None
    test_prob = best_clf.predict_proba(x[masks["test"]])
    test_metrics = compute_metrics(y[masks["test"]], test_prob)
    first = records[0]
    return {
        "benchmark": first.benchmark,
        "task": task,
        "split_type": split_type,
        "group": first.group,
        "baseline": "kmer",
        "kmer_min": args.kmer_min,
        "kmer_max": args.kmer_max,
        "kmer_binary": args.kmer_binary,
        "max_length": args.max_length,
        "best_c": best_c,
        "selection_metric": best_metric_name,
        "selection_score": best_score,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        **test_metrics,
    }


def write_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--default-benchmark", default="host_tropism_hiyata")
    parser.add_argument("--default-task", default="host_tropism_hiyata")
    parser.add_argument("--default-group", default="host_tropism_adaptation")
    parser.add_argument("--out-csv", default="data/phase2/kmer_baselines/kmer_metrics.csv")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--kmer-min", type=int, default=1)
    parser.add_argument("--kmer-max", type=int, default=4)
    parser.add_argument("--kmer-binary", action="store_true")
    parser.add_argument("--c-grid", default="0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--solver", choices=["lbfgs", "saga"], default="lbfgs")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    records = read_manifest(args)
    by_task: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        by_task[(record.task, record.split_type)].append(record)
    rows = []
    for (task, split_type), task_records in sorted(by_task.items()):
        row = fit_task(task, split_type, task_records, args)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No k-mer baseline rows were produced")
    write_rows(args.out_csv, rows)
    summary_path = args.summary_json or os.path.splitext(args.out_csv)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"[kmer] wrote {args.out_csv}")
    print(f"[kmer] wrote {summary_path}")


if __name__ == "__main__":
    main()
