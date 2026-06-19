"""Prepare hiyata/Virus-Host-Genomes for host-tropism target validation.

The output manifest follows the local host-tropism schema but preserves the
family/genus metadata needed for real taxonomy-controlled splits.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

csv.field_size_limit(sys.maxsize)


DEFAULT_DATASET = "hiyata/Virus-Host-Genomes"
OUTPUT_COLUMNS = [
    "id",
    "label",
    "split",
    "sequence",
    "source",
    "length",
    "accession",
    "virus_tax_id",
    "virus_name",
    "host_tax_id",
    "host_name",
    "host_common_name",
    "family",
    "genus",
    "host",
    "standardized_host",
    "host_category",
    "zoonotic",
    "processing_method",
    "gemini_annotated",
    "is_segmented",
    "segment_label",
]


def clean_sequence(seq: object) -> str:
    text = str(seq or "").upper()
    return "".join(ch for ch in text if ch in {"A", "C", "G", "T", "N"})


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na"}:
        return ""
    return text


def normalize_bool(value: object) -> str:
    text = normalize_text(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    return normalize_text(value)


def host_to_label(row: dict) -> Optional[str]:
    candidates = [
        normalize_text(row.get("host")),
        normalize_text(row.get("standardized_host")),
        normalize_text(row.get("host_category")),
    ]
    text = " ".join(candidates).lower()
    if "non-human" in text or "nonhuman" in text:
        return "0"
    if candidates[0].lower() in {"non-human", "nonhuman", "animal", "other"}:
        return "0"
    if re.search(r"\bhuman\b|homo sapiens", text):
        return "1"
    return None


def load_hf_dataset(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Use the project environment: "
            "/home/teacher1/miniconda3/envs/UT-p1/bin/python"
        ) from exc
    return load_dataset(dataset_name)


def iter_dataset_rows(dataset_name: str, local_files: list[str]) -> Iterable[tuple[str, dict]]:
    if local_files:
        for file_path in local_files:
            path = Path(file_path)
            split = path.stem
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix in {".csv", ".tsv"}:
                df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
            elif path.suffix in {".jsonl", ".json"}:
                df = pd.read_json(path, lines=path.suffix == ".jsonl")
            else:
                raise ValueError(f"Unsupported input file type: {path}")
            for row in df.to_dict(orient="records"):
                yield split, row
        return

    ds = load_hf_dataset(dataset_name)
    for split, table in ds.items():
        for row in table:
            yield split, row


def convert_row(split: str, row: dict, idx: int) -> Optional[dict]:
    seq = clean_sequence(row.get("sequence"))
    label = host_to_label(row)
    if not seq or label is None:
        return None
    accession = normalize_text(row.get("accession") or row.get("id"))
    virus_name = normalize_text(row.get("virus_name") or row.get("source"))
    family = normalize_text(row.get("family"))
    genus = normalize_text(row.get("genus"))
    host = normalize_text(row.get("host"))
    standardized_host = normalize_text(row.get("standardized_host"))
    host_category = normalize_text(row.get("host_category"))
    record_id = accession or f"hiyata|{split}|{idx}"
    return {
        "id": record_id,
        "label": label,
        "split": split.lower(),
        "sequence": seq,
        "source": virus_name,
        "length": len(seq),
        "accession": accession,
        "virus_tax_id": normalize_text(row.get("virus_tax_id") or row.get("tax_id")),
        "virus_name": virus_name,
        "host_tax_id": normalize_text(row.get("host_tax_id")),
        "host_name": standardized_host or host,
        "host_common_name": host_category,
        "family": family,
        "genus": genus,
        "host": host,
        "standardized_host": standardized_host,
        "host_category": host_category,
        "zoonotic": normalize_bool(row.get("zoonotic")),
        "processing_method": normalize_text(row.get("processing_method")),
        "gemini_annotated": normalize_bool(row.get("gemini_annotated")),
        "is_segmented": normalize_bool(row.get("is_segmented")),
        "segment_label": normalize_text(row.get("segment_label")),
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def summarize(rows: list[dict], skipped: int) -> dict:
    def counts(key: str) -> dict:
        out = {}
        for row in rows:
            value = row.get(key, "")
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items(), key=lambda item: (-item[1], item[0]))[:50])

    return {
        "dataset": DEFAULT_DATASET,
        "n_rows": len(rows),
        "n_skipped": skipped,
        "label_counts": counts("label"),
        "split_counts": counts("split"),
        "family_counts_top50": counts("family"),
        "genus_counts_top50": counts("genus"),
        "gemini_annotated_counts": counts("gemini_annotated"),
        "processing_method_counts": counts("processing_method"),
        "columns": OUTPUT_COLUMNS,
        "target_validity_role": (
            "Primary external host-tropism dataset for family/genus/homology/"
            "within-family controlled split validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--local-files",
        nargs="*",
        default=[],
        help="Optional local parquet/csv/jsonl files. If omitted, load from Hugging Face.",
    )
    parser.add_argument("--out", default="data/host_tropism_hiyata/manifest.csv")
    parser.add_argument("--summary-out", default="data/host_tropism_hiyata/summary.json")
    parser.add_argument("--exclude-gemini", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test cap after filtering.")
    args = parser.parse_args()

    rows = []
    skipped = 0
    for idx, (split, raw_row) in enumerate(iter_dataset_rows(args.dataset, args.local_files)):
        converted = convert_row(split, raw_row, idx)
        if converted is None:
            skipped += 1
            continue
        if args.exclude_gemini and converted["gemini_annotated"] == "true":
            skipped += 1
            continue
        rows.append(converted)
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break

    if not rows:
        raise RuntimeError("No usable hiyata host-tropism rows after filtering.")
    write_manifest(Path(args.out), rows)
    summary = summarize(rows, skipped)
    summary["source_dataset"] = args.dataset
    summary["exclude_gemini"] = args.exclude_gemini
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"[hiyata] wrote {args.out} rows={len(rows)} skipped={skipped}")
    print(f"[hiyata] wrote {args.summary_out}")


if __name__ == "__main__":
    main()
