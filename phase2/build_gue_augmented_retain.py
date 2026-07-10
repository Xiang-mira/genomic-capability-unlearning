"""
Build a GUE-augmented Phase 2 retain CSV.

Starting from a base retain CSV, sample deterministic train rows from selected
GUE tasks and append them as unlearning retain rows (`label=0`). Original GUE
task labels are preserved in the generated id/source metadata.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def stable_score(seed: int, row_id: str) -> int:
    payload = f"{seed}|{row_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def strip_task_prefix(name: str) -> str:
    task = name.strip()
    return task[4:] if task.startswith("gue_") else task


def task_match_key(name: str) -> str:
    return strip_task_prefix(name).lower()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-retain-csv", default="data/phase2/coronaviridae_splits/retain.csv")
    parser.add_argument("--benchmark-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--out-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--summary-json", default="data/phase2/splits/retain_with_gue_summary.json")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["prom_300_notata", "splice_reconstructed", "human_tf_1", "mouse_1", "emp_H3"],
        help="Requested GUE task names. Accepts either bare task names or gue_* ids.",
    )
    parser.add_argument("--per-label", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_path = Path(args.base_retain_csv)
    benchmark_path = Path(args.benchmark_manifest)
    out_path = Path(args.out_csv)
    summary_path = Path(args.summary_json)

    base_rows = read_csv(base_path)
    manifest_rows = read_csv(benchmark_path)
    requested = [strip_task_prefix(task) for task in args.tasks]
    requested_by_key = {task_match_key(task): task for task in requested}
    heldout_sequences = {
        row["sequence"]
        for row in manifest_rows
        if row.get("benchmark") == "gue" and row.get("split") in {"val", "test"}
    }

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    excluded_heldout_duplicates = 0
    for row in manifest_rows:
        if row.get("benchmark") != "gue":
            continue
        if row.get("group") != "gue_retain":
            continue
        if row.get("split") != "train":
            continue
        if row["sequence"] in heldout_sequences:
            excluded_heldout_duplicates += 1
            continue
        task_key = task_match_key(row["task"])
        if task_key not in requested_by_key:
            continue
        grouped[requested_by_key[task_key]][str(row["label"])].append(row)

    sampled_rows: list[dict] = []
    summary_tasks: dict[str, dict] = {}
    for task in requested:
        label_buckets = grouped.get(task, {})
        if not label_buckets:
            raise ValueError(f"No train GUE rows found for task={task!r} in {benchmark_path}")
        task_summary = {
            "available_train_rows": sum(len(rows) for rows in label_buckets.values()),
            "sampled_original_label_counts": {},
            "sampled_rows": 0,
        }
        for label, rows in sorted(label_buckets.items()):
            ranked = sorted(rows, key=lambda row: stable_score(args.seed, row["id"]))
            chosen = ranked[: min(args.per_label, len(ranked))]
            task_summary["sampled_original_label_counts"][label] = len(chosen)
            task_summary["sampled_rows"] += len(chosen)
            for row in chosen:
                rownum = row["id"].rsplit("|", 1)[-1]
                sampled_rows.append(
                    {
                        "id": f"gue|{task}|train|{rownum}|orig_label={label}",
                        "label": "0",
                        "split": "train",
                        "sequence": row["sequence"],
                        "source": f"GUE:{task}:orig_label={label}",
                        "length": len(row["sequence"]),
                    }
                )
        summary_tasks[task] = task_summary

    combined = list(base_rows)
    seen_ids = {row["id"] for row in combined}
    duplicate_added_ids = []
    for row in sampled_rows:
        if row["id"] in seen_ids:
            duplicate_added_ids.append(row["id"])
            continue
        seen_ids.add(row["id"])
        combined.append(row)

    fieldnames = ["id", "label", "split", "sequence", "source", "length"]
    write_csv(out_path, combined, fieldnames)

    summary = {
        "base_retain_csv": str(base_path),
        "benchmark_manifest": str(benchmark_path),
        "output_csv": str(out_path),
        "seed": args.seed,
        "per_label_requested": args.per_label,
        "selected_gue_tasks": requested,
        "base_retain_rows": len(base_rows),
        "added_gue_rows": len(sampled_rows) - len(duplicate_added_ids),
        "new_retain_rows": len(combined),
        "duplicate_added_ids": duplicate_added_ids[:20],
        "excluded_train_rows_matching_gue_val_or_test_sequence": excluded_heldout_duplicates,
        "gue_rows_by_task": summary_tasks,
        "notes": [
            "GUE rows are sampled from train split only.",
            "Train rows whose sequence exactly matches any GUE val/test sequence are excluded.",
            "All injected GUE rows use label=0 because this is the unlearning retain label.",
            "Original GUE labels are preserved in id and source metadata.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[retain-gue] base_rows={len(base_rows)} added_gue_rows={summary['added_gue_rows']} new_rows={len(combined)}")
    for task in requested:
        task_summary = summary_tasks[task]
        print(
            f"[retain-gue] task={task} available={task_summary['available_train_rows']} "
            f"sampled={task_summary['sampled_original_label_counts']}"
        )
    print(f"[retain-gue] wrote {out_path}")
    print(f"[retain-gue] wrote {summary_path}")


if __name__ == "__main__":
    main()
