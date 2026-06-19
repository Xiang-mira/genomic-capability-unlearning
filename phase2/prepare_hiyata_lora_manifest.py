"""Build a canonical LoRA evaluation manifest for Hiyata host tropism."""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["benchmark", "task", "split", "sequence", "label", "group", "id"]
DEFAULT_BENCHMARK = "host_tropism_hiyata"
DEFAULT_TASK = "host_tropism_hiyata"
DEFAULT_GROUP = "host_tropism_adaptation"


def clean_sequence(value: object, max_length: int) -> str:
    seq = str(value or "").upper()
    seq = "".join(ch for ch in seq if ch in {"A", "C", "G", "T", "N"})
    if max_length > 0:
        seq = seq[:max_length]
    return seq


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def normalize_split(split: str) -> str:
    split = str(split or "").lower()
    if split in {"dev", "valid", "validation"}:
        return "val"
    return split


def read_source_rows(path: Path, max_length: int, max_rows: int) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "label", "split", "sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")
        for row in reader:
            label = str(row.get("label", "")).strip()
            split = normalize_split(row.get("split", ""))
            seq = clean_sequence(row.get("sequence", ""), max_length)
            if split not in {"train", "val", "test"} or label not in {"0", "1"} or not seq:
                continue
            rows.append(
                {
                    "benchmark": DEFAULT_BENCHMARK,
                    "task": DEFAULT_TASK,
                    "split": split,
                    "sequence": seq,
                    "label": label,
                    "group": DEFAULT_GROUP,
                    "id": row.get("id") or row.get("accession") or f"hiyata|{len(rows)}",
                }
            )
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def split_train_val(rows: list[dict], val_fraction: float, seed: int) -> list[dict]:
    existing_val = [row for row in rows if row["split"] == "val"]
    if existing_val:
        return rows

    rng = random.Random(seed)
    train_by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if row["split"] == "train":
            train_by_label[row["label"]].append(idx)

    val_indices: set[int] = set()
    for label, indices in train_by_label.items():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)
        val_indices.update(shuffled[:n_val])

    out = []
    for idx, row in enumerate(rows):
        row = dict(row)
        if idx in val_indices:
            row["split"] = "val"
        out.append(row)
    return out


def count_nested(rows: Iterable[dict]) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["split"]][row["label"]] += 1
    return {split: dict(labels) for split, labels in sorted(counts.items())}


def validate_splits(rows: list[dict]) -> None:
    counts = count_nested(rows)
    for split in ("train", "val", "test"):
        if split not in counts:
            raise ValueError(f"Missing split after derivation: {split}")
        if len(counts[split]) < 2:
            raise ValueError(f"Split {split} does not contain both labels: {counts[split]}")


def read_hvue_host_hashes(path: Optional[Path], max_length: int) -> set[str]:
    if path is None or not path.exists():
        return set()
    hashes = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task", "")
            if task != "hvue_human_host_tropism":
                continue
            seq = clean_sequence(row.get("sequence", ""), max_length)
            if seq:
                hashes.add(seq_hash(seq))
    return hashes


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_audit(
    path: Path,
    rows: list[dict],
    input_path: Path,
    hvue_manifest: Optional[Path],
    hvue_hashes: set[str],
    args,
) -> None:
    hashes = {seq_hash(row["sequence"]) for row in rows}
    overlap = hashes & hvue_hashes
    payload = {
        "input_manifest": str(input_path),
        "output_manifest": str(args.out_manifest),
        "benchmark": DEFAULT_BENCHMARK,
        "task": DEFAULT_TASK,
        "group": DEFAULT_GROUP,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "max_length": args.max_length,
        "rows": len(rows),
        "split_label_counts": count_nested(rows),
        "unique_sequence_hashes": len(hashes),
        "hvue_manifest": str(hvue_manifest) if hvue_manifest else "",
        "hvue_host_unique_sequence_hashes": len(hvue_hashes),
        "hiyata_hvue_exact_sequence_overlap": len(overlap),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", default="data/host_tropism_hiyata/manifest_no_gemini.csv")
    parser.add_argument("--out-manifest", default="data/host_tropism_hiyata/eval_manifest_lora.csv")
    parser.add_argument("--audit-json", default="data/host_tropism_hiyata/eval_manifest_lora_audit.json")
    parser.add_argument("--hvue-manifest", default="data/benchmarks/final_fast_eval_manifest.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input_manifest)
    rows = read_source_rows(input_path, args.max_length, args.max_rows)
    rows = split_train_val(rows, args.val_fraction, args.seed)
    validate_splits(rows)

    out_manifest = Path(args.out_manifest)
    write_manifest(out_manifest, rows)

    hvue_path = Path(args.hvue_manifest) if args.hvue_manifest else None
    hvue_hashes = read_hvue_host_hashes(hvue_path, args.max_length)
    write_audit(Path(args.audit_json), rows, input_path, hvue_path, hvue_hashes, args)
    print(f"[hiyata-lora] wrote {out_manifest} rows={len(rows)}")
    print(f"[hiyata-lora] wrote {args.audit_json}")


if __name__ == "__main__":
    main()
