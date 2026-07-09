"""
Audit a Phase 2 retain CSV before RMU/GD sweeps.

The current retain strategy is expected to preserve both:
  - non-GUE viral retain rows
  - GUE-sourced retain rows injected for downstream retention coverage

This script verifies that the retain file still contains both populations and
that rows were not accidentally re-appended.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_gue_task(source: str) -> str | None:
    if not source.startswith("GUE:"):
        return None
    payload = source[len("GUE:") :]
    task, _sep, _rest = payload.partition(":orig_label=")
    return task or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"Retain CSV not found: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"id", "label", "split", "sequence", "source", "length"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Retain CSV missing required columns: {sorted(missing)}")

        rows = list(reader)

    id_counts = Counter(row["id"] for row in rows)
    duplicate_ids = sorted(row_id for row_id, count in id_counts.items() if count > 1)
    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    source_prefix_counts = Counter()
    gue_rows_by_task = Counter()
    gue_rows = 0
    non_gue_rows = 0
    invalid_label_rows = 0

    for row in rows:
        source = row["source"]
        task = parse_gue_task(source)
        if task is not None:
            gue_rows += 1
            gue_rows_by_task[task] += 1
            source_prefix_counts["GUE"] += 1
        else:
            non_gue_rows += 1
            prefix = source.split(":", 1)[0] if ":" in source else source
            source_prefix_counts[prefix] += 1
        if row["label"] != "0":
            invalid_label_rows += 1

    summary = {
        "retain_csv": str(path),
        "total_rows": len(rows),
        "unique_ids": len(id_counts),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids_preview": duplicate_ids[:20],
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "gue_rows": gue_rows,
        "non_gue_rows": non_gue_rows,
        "gue_rows_by_task": dict(sorted(gue_rows_by_task.items())),
        "source_prefix_counts": dict(source_prefix_counts.most_common(20)),
        "invalid_label_rows": invalid_label_rows,
    }

    summary_json = args.summary_json or str(path.with_name("retain_audit.json"))
    out_path = Path(summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[retain-audit] csv={path}")
    print(
        f"[retain-audit] rows={len(rows)} unique_ids={len(id_counts)} "
        f"gue_rows={gue_rows} non_gue_rows={non_gue_rows}"
    )
    print(f"[retain-audit] labels={dict(sorted(label_counts.items()))}")
    print(f"[retain-audit] splits={dict(sorted(split_counts.items()))}")
    print(
        "[retain-audit] top_gue_tasks="
        + ", ".join(f"{task}:{count}" for task, count in gue_rows_by_task.most_common(10))
    )
    print(f"[retain-audit] wrote {out_path}")

    failures = []
    if not rows:
        failures.append("retain CSV is empty")
    if gue_rows == 0:
        failures.append("retain CSV contains no GUE rows")
    if non_gue_rows == 0:
        failures.append("retain CSV contains no non-GUE retain rows")
    if duplicate_ids:
        failures.append(f"retain CSV contains duplicate ids ({len(duplicate_ids)})")
    if invalid_label_rows:
        failures.append(f"retain CSV contains {invalid_label_rows} non-zero labels")

    if failures:
        for failure in failures:
            print(f"[retain-audit] ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
