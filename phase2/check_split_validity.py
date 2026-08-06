"""Validate random/disjoint benchmark splits before they are used downstream.

The checker enforces the new Stage 0 rule:
1. every split must have acceptable label balance;
2. disjoint splits should be harder than random splits under shortcut baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata


def safe_float(value: Any) -> float | None:
    if value in (None, "", "NA", "null"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def infer_split_type(row: dict[str, str], path: Path) -> str:
    for key in ("split_type", "split_name", "eval_split", "name"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    stem = path.stem.lower()
    if "cluster" in stem or "disjoint" in stem:
        return "cluster_disjoint"
    return "random"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_manifest(path: Path, min_test_fraction: float) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task", "").strip()
            split_type = (row.get("split_type") or "random").strip() or "random"
            label = row.get("label", "").strip()
            split = (row.get("split") or "").strip()
            if not task or split != "test" or not label:
                continue
            counts[(task, split_type)][label] += 1

    rows = []
    for (task, split_type), label_counts in sorted(counts.items()):
        total = sum(label_counts.values())
        majority = max(label_counts.values()) if label_counts else 0
        minority = min(label_counts.values()) if label_counts else 0
        minority_fraction = 0.0 if total == 0 else minority / total
        rows.append(
            {
                "task": task,
                "split_type": split_type,
                "test_rows": total,
                "n_labels": len(label_counts),
                "label_counts": json.dumps(dict(sorted(label_counts.items())), sort_keys=True),
                "minority_fraction": minority_fraction,
                "label_balance_ok": len(label_counts) >= 2 and minority_fraction >= min_test_fraction,
                "validity_note": "" if len(label_counts) >= 2 and minority_fraction >= min_test_fraction else "label_balance_fail",
            }
        )
    return rows


def summarize_baselines(paths: list[Path], min_drop: float) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = (row.get("task") or "").strip()
                if not task:
                    continue
                split_type = infer_split_type(row, path)
                score = safe_float(row.get("auroc") or row.get("macro_auroc") or row.get("selection_score"))
                if score is None:
                    continue
                current = per_task[task].get(split_type)
                if current is None or score > current:
                    per_task[task][split_type] = score

    rows = []
    for task, scores in sorted(per_task.items()):
        random_score = scores.get("random")
        disjoint_score = scores.get("cluster_disjoint")
        drop = None if random_score is None or disjoint_score is None else random_score - disjoint_score
        rows.append(
            {
                "task": task,
                "random_score": random_score,
                "cluster_disjoint_score": disjoint_score,
                "random_minus_disjoint": drop,
                "disjoint_harder_ok": drop is not None and drop >= min_drop,
                "validity_note": ""
                if drop is not None and drop >= min_drop
                else "baseline_gap_too_small" if drop is not None else "missing_random_or_disjoint_baseline",
            }
        )
    return rows, per_task


def merge_rows(
    manifest_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_map = {row["task"]: row for row in baseline_rows}
    merged = []
    for row in manifest_rows:
        merged_row = dict(row)
        baseline = baseline_map.get(row["task"], {})
        merged_row["random_score"] = baseline.get("random_score", "")
        merged_row["cluster_disjoint_score"] = baseline.get("cluster_disjoint_score", "")
        merged_row["random_minus_disjoint"] = baseline.get("random_minus_disjoint", "")
        merged_row["disjoint_harder_ok"] = baseline.get("disjoint_harder_ok", "")
        merged_row["overall_ok"] = bool(row["label_balance_ok"]) and (
            row["split_type"] != "cluster_disjoint" or baseline.get("disjoint_harder_ok") is True
        )
        note_parts = [part for part in [row.get("validity_note", ""), baseline.get("validity_note", "")] if part]
        merged_row["overall_note"] = ";".join(dict.fromkeys(note_parts))
        merged.append(merged_row)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-csv", action="append", default=[])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--min-test-minority-fraction", type=float, default=0.2)
    parser.add_argument("--min-random-minus-disjoint", type=float, default=0.02)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    baseline_paths = [Path(path) for path in args.baseline_csv]
    manifest_rows = summarize_manifest(manifest_path, args.min_test_minority_fraction)
    baseline_rows, per_task_scores = summarize_baselines(baseline_paths, args.min_random_minus_disjoint)
    merged_rows = merge_rows(manifest_rows, baseline_rows)

    fieldnames = [
        "task",
        "split_type",
        "test_rows",
        "n_labels",
        "label_counts",
        "minority_fraction",
        "label_balance_ok",
        "random_score",
        "cluster_disjoint_score",
        "random_minus_disjoint",
        "disjoint_harder_ok",
        "overall_ok",
        "overall_note",
    ]
    out_csv = Path(args.out_csv)
    write_csv(out_csv, merged_rows, fieldnames)

    summary = {
        "manifest": str(manifest_path),
        "baseline_csvs": [str(path) for path in baseline_paths],
        "rows": merged_rows,
        "tasks_with_baselines": per_task_scores,
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_csv.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_metadata(
        out_csv.with_name(f"{out_csv.stem}_metadata.json"),
        build_run_metadata(
            args=args,
            data_paths=[args.manifest, *args.baseline_csv],
            extra={
                "phase": "check_split_validity",
                "out_csv": str(out_csv),
                "summary_json": str(summary_path),
                "row_count": len(merged_rows),
                "manifest_tasks": sorted({row["task"] for row in merged_rows}),
            },
        ),
    )
    print(f"[split-validity] wrote {out_csv}")
    print(f"[split-validity] wrote {summary_path}")


if __name__ == "__main__":
    main()
