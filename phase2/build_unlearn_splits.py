"""
Build forget / retain splits for Phase 2 unlearning.

By default this reuses a binary manifest:
  - label=1 -> forget
  - label=0 -> retain

Optionally, extra manifests can contribute additional `label=1` examples to the
forget side only. The default merged build uses the host-tropism manifest as
the primary split source and appends Coronaviridae positives onto the forget
side, matching the current Phase 2 selective-unlearning dataset.

Train/val/test are taken from each manifest's existing split column so
unlearning training uses only `split == train` and we hold out val/test for
evaluation (consistent with the Phase 1 probes).
"""
import argparse
import csv
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List

def write_csv(path: str, records, fieldnames) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r.record_id, "label": r.label, "split": r.split,
                "sequence": r.sequence, "source": r.source, "length": r.length,
            })


@dataclass
class ManifestRecord:
    record_id: str
    label: int
    split: str
    sequence: str
    source: str
    length: int


@dataclass
class BenchmarkRow:
    benchmark: str
    task: str
    split: str
    sequence: str
    label: int
    group: str
    row_id: str


def read_manifest(path: str) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ManifestRecord(
                    record_id=row["id"],
                    label=int(row["label"]),
                    split=row["split"],
                    sequence=row["sequence"],
                    source=row.get("source", ""),
                    length=int(row.get("length", len(row["sequence"]))),
                )
            )
    return records


def read_benchmark_rows(path: str) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"benchmark", "task", "split", "sequence", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Benchmark manifest missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader):
            rows.append(
                BenchmarkRow(
                    benchmark=row["benchmark"],
                    task=row["task"],
                    split=row["split"],
                    sequence=row["sequence"],
                    label=int(row["label"]),
                    group=row.get("group", ""),
                    row_id=row.get("id") or f"bench|{row['task']}|{row['split']}|{row_number}",
                )
            )
    return rows


def dedupe_records(records: Iterable) -> List:
    deduped = []
    seen_ids = set()
    for record in records:
        if record.record_id in seen_ids:
            continue
        seen_ids.add(record.record_id)
        deduped.append(record)
    return deduped


def summarize_counts(records, label_name: str) -> None:
    counts = defaultdict(int)
    for record in records:
        counts[record.split] += 1
    summary = "  ".join(f"{split}={counts[split]}" for split in ["train", "val", "test"])
    print(f"  {label_name}: total={len(records)}  {summary}")


def stable_score(seed: int, group: str, row_id: str) -> int:
    payload = f"{seed}|{group}|{row_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sample_benchmark_forget(
    benchmark_manifest: str,
    groups: List[str],
    split: str,
    label: int,
    samples_per_group: int,
    seed: int,
) -> List[ManifestRecord]:
    benchmark_rows = read_benchmark_rows(benchmark_manifest)
    normalized_groups = {group.strip() for group in groups if group.strip()}
    grouped_rows = defaultdict(list)

    for row in benchmark_rows:
        if row.benchmark.lower() != "hvue":
            continue
        if row.group not in normalized_groups:
            continue
        if row.split.lower() != split.lower():
            continue
        if row.label != label:
            continue
        grouped_rows[row.group].append(row)

    sampled: List[ManifestRecord] = []
    for group in sorted(normalized_groups):
        candidates = grouped_rows.get(group, [])
        if not candidates:
            print(f"[splits] warning: no HVUE benchmark rows found for group={group}")
            continue
        by_task = defaultdict(list)
        for row in candidates:
            by_task[row.task].append(row)
        task_names = sorted(by_task)
        base_quota = samples_per_group // max(len(task_names), 1)
        remainder = samples_per_group % max(len(task_names), 1)
        chosen = []
        for task_index, task in enumerate(task_names):
            task_rows = by_task[task]
            task_rows.sort(key=lambda row: stable_score(seed, f"{group}|{task}", row.row_id))
            task_quota = base_quota + (1 if task_index < remainder else 0)
            chosen.extend(task_rows[: min(task_quota, len(task_rows))])
        print(
            f"[splits] HVUE sampled group={group} split={split} label={label} "
            f"picked={len(chosen)}/{len(candidates)} tasks={len(task_names)} from {benchmark_manifest}"
        )
        for row in chosen:
            sampled.append(
                ManifestRecord(
                    record_id=f"{row.row_id}|unlearn_forget_hvue",
                    label=1,
                    split=row.split,
                    sequence=row.sequence,
                    source=f"HVUE:{row.task}:group={row.group}:orig_label={row.label}",
                    length=len(row.sequence),
                )
            )
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/host_tropism/manifest.csv")
    parser.add_argument(
        "--extra-forget-manifest",
        action="append",
        default=["data/family_targets/coronaviridae/manifest.csv"],
        help="Additional manifest(s) whose label=1 rows are appended to forget.csv only.",
    )
    parser.add_argument(
        "--sample-forget-benchmark-manifest",
        default="",
        help="Optional benchmark manifest to sample extra HVUE forget rows from.",
    )
    parser.add_argument(
        "--sample-forget-groups",
        default="",
        help="Comma-separated HVUE benchmark groups to sample from, e.g. primary_forget,secondary_forget.",
    )
    parser.add_argument(
        "--sample-forget-split",
        default="train",
        help="Benchmark split to sample for forget augmentation. Default keeps benchmark val/test untouched.",
    )
    parser.add_argument(
        "--sample-forget-label",
        type=int,
        default=1,
        help="Benchmark label value to treat as forget-positive when sampling. Default: 1.",
    )
    parser.add_argument(
        "--sample-forget-per-group",
        type=int,
        default=0,
        help="How many HVUE rows to sample per selected group. Disabled at 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/phase2/splits")
    args = parser.parse_args()

    base_records = read_manifest(args.manifest)
    forget = [r for r in base_records if r.label == 1]
    retain = [r for r in base_records if r.label == 0]

    print(f"[splits] base manifest: {args.manifest}")
    summarize_counts(forget, "base_forget")
    summarize_counts(retain, "base_retain")

    for extra_manifest in args.extra_forget_manifest:
        extra_records = read_manifest(extra_manifest)
        extra_forget = [r for r in extra_records if r.label == 1]
        print(f"[splits] extra forget manifest: {extra_manifest}")
        summarize_counts(extra_forget, "extra_forget")
        forget.extend(extra_forget)

    if args.sample_forget_benchmark_manifest and args.sample_forget_per_group > 0:
        groups = [group.strip() for group in args.sample_forget_groups.split(",") if group.strip()]
        sampled_forget = sample_benchmark_forget(
            benchmark_manifest=args.sample_forget_benchmark_manifest,
            groups=groups,
            split=args.sample_forget_split,
            label=args.sample_forget_label,
            samples_per_group=args.sample_forget_per_group,
            seed=args.seed,
        )
        summarize_counts(sampled_forget, "sampled_hvue_forget")
        forget.extend(sampled_forget)

    forget = dedupe_records(forget)
    retain = dedupe_records(retain)

    fieldnames = ["id", "label", "split", "sequence", "source", "length"]
    write_csv(os.path.join(args.out_dir, "forget.csv"), forget, fieldnames)
    write_csv(os.path.join(args.out_dir, "retain.csv"), retain, fieldnames)

    # Summary
    counts = defaultdict(lambda: defaultdict(int))
    for r in forget:
        counts[r.split]["forget"] += 1
    for r in retain:
        side = "retain"
        counts[r.split][side] += 1
    print(f"Wrote forget ({len(forget)}) and retain ({len(retain)}) to {args.out_dir}")
    for split in ["train", "val", "test"]:
        print(f"  {split}: forget={counts[split]['forget']}  retain={counts[split]['retain']}")


if __name__ == "__main__":
    main()
