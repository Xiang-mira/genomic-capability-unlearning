"""
Build a deterministic task/split/label-stratified pilot benchmark manifest.

The sampler keeps small tasks intact and caps larger tasks per split/label so a
fixed pilot manifest can be reused for paired checkpoint comparisons.
"""
import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


Bucket = Tuple[str, str, str]


def stable_score(seed: int, row_id: str, row_number: int) -> int:
    payload = f"{seed}|{row_id}|{row_number}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cap_for_split(split: str, args) -> int:
    split = split.lower()
    if split == "train":
        return args.train_per_label
    if split == "val":
        return args.val_per_label
    if split == "test":
        return args.test_per_label
    return args.other_split_per_label


def count_manifest(path: Path) -> tuple[list[str], Counter, Counter]:
    task_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {"benchmark", "task", "split", "sequence", "label"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")
        for row in reader:
            task = row["task"]
            split = row["split"].lower()
            label = str(row["label"])
            task_counts[task] += 1
            bucket_counts[(task, split, label)] += 1
    return fieldnames, task_counts, bucket_counts


def select_large_task_rows(path: Path, task_counts: Counter, args) -> set[int]:
    heaps: Dict[Bucket, list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader):
            task = row["task"]
            if task_counts[task] <= args.keep_all_task_rows:
                continue
            split = row["split"].lower()
            label = str(row["label"])
            cap = cap_for_split(split, args)
            if cap <= 0:
                continue
            row_id = row.get("id") or f"{task}|{split}|{label}|{row_number}"
            score = stable_score(args.seed, row_id, row_number)
            heap = heaps[(task, split, label)]
            entry = (-score, row_number)
            if len(heap) < cap:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    selected: set[int] = set()
    for heap in heaps.values():
        selected.update(row_number for _neg_score, row_number in heap)
    return selected


def write_sampled_manifest(
    in_path: Path,
    out_path: Path,
    fieldnames: Iterable[str],
    task_counts: Counter,
    selected_large_rows: set[int],
    args,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_manifest": str(in_path),
        "output_manifest": str(out_path),
        "seed": args.seed,
        "keep_all_task_rows": args.keep_all_task_rows,
        "caps_per_label": {
            "train": args.train_per_label,
            "val": args.val_per_label,
            "test": args.test_per_label,
            "other": args.other_split_per_label,
        },
        "rows_in": int(sum(task_counts.values())),
        "rows_out": 0,
        "task_counts": {},
        "task_split_label_counts": {},
    }
    task_out: Counter = Counter()
    bucket_out: Counter = Counter()

    with in_path.open(newline="") as src, out_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=list(fieldnames))
        writer.writeheader()
        for row_number, row in enumerate(reader):
            task = row["task"]
            keep = task_counts[task] <= args.keep_all_task_rows or row_number in selected_large_rows
            if not keep:
                continue
            writer.writerow(row)
            split = row["split"].lower()
            label = str(row["label"])
            task_out[task] += 1
            bucket_out[(task, split, label)] += 1

    summary["rows_out"] = int(sum(task_out.values()))
    summary["task_counts"] = dict(sorted((task, int(count)) for task, count in task_out.items()))
    nested: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (task, split, label), count in sorted(bucket_out.items()):
        nested[task][split][label] = int(count)
    summary["task_split_label_counts"] = {
        task: {split: labels for split, labels in splits.items()}
        for task, splits in sorted(nested.items())
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--output-manifest", default="data/benchmarks/hvue_gue_pilot_manifest.csv")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-all-task-rows", type=int, default=6000)
    parser.add_argument("--train-per-label", type=int, default=2000)
    parser.add_argument("--val-per-label", type=int, default=500)
    parser.add_argument("--test-per-label", type=int, default=1500)
    parser.add_argument("--other-split-per-label", type=int, default=500)
    args = parser.parse_args()

    in_path = Path(args.input_manifest)
    out_path = Path(args.output_manifest)
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")

    fieldnames, task_counts, _bucket_counts = count_manifest(in_path)
    selected_large_rows = select_large_task_rows(in_path, task_counts, args)
    summary = write_sampled_manifest(in_path, out_path, fieldnames, task_counts, selected_large_rows, args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"[subsample] wrote {out_path} rows={summary['rows_out']}/{summary['rows_in']} "
        f"tasks={len(summary['task_counts'])} seed={args.seed}"
    )
    print(f"[subsample] wrote {summary_path}")


if __name__ == "__main__":
    main()
