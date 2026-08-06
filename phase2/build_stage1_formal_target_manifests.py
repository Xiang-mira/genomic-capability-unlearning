"""Build Stage 1 formal-target manifests with explicit split semantics.

This utility normalizes task-specific manifest sources into eval_benchmarks.py
format and records which formal targets still lack a usable split source.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.audit_stage1_target_sources import build_report as build_source_audit_report
from phase2.run_metadata import build_run_metadata, write_metadata


def normalize_split_type(value: str) -> str:
    split_type = str(value or "").strip().lower()
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type or "random"


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    label_counts = {split: dict(Counter(row["label"] for row in rows if row["split"] == split)) for split in split_counts}
    return {
        "n_rows": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": label_counts,
    }


def load_rows(
    path: Path,
    *,
    split_column: str,
    split_type: str,
    default_group: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"benchmark", "task", "sequence", "label"}
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        if split_column not in fields:
            raise ValueError(f"{path} missing split column: {split_column}")
        for row in reader:
            split = (row.get(split_column) or "").strip().lower()
            if split not in {"train", "val", "test"}:
                continue
            sequence = row.get("sequence", "")
            label = str(row.get("label", "")).strip()
            if not sequence or not label:
                continue
            rows.append(
                {
                    "benchmark": row.get("benchmark", ""),
                    "task": row.get("task", ""),
                    "split": split,
                    "split_type": normalize_split_type(split_type),
                    "sequence": sequence,
                    "label": label,
                    "family": row.get("family", ""),
                    "group": row.get("group", "") or default_group,
                    "id": row.get("id", ""),
                }
            )
    if not rows:
        raise ValueError(f"{path} produced no usable rows")
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["benchmark", "task", "split", "split_type", "sequence", "label", "family", "group", "id"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_formal_target_outputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {"manifests": {}, "missing_targets": []}
    merged_rows: list[dict[str, str]] = []

    host_rows = load_rows(
        Path(args.host_formal_manifest),
        split_column=args.host_split_column,
        split_type=args.host_split_type,
        default_group=args.host_group,
    )
    host_out = out_dir / "hvue_human_host_tropism_cluster_disjoint.csv"
    write_manifest(host_out, host_rows)
    outputs["manifests"]["hvue_human_host_tropism"] = {
        "path": str(host_out),
        "source_path": args.host_formal_manifest,
        "split_column": args.host_split_column,
        "split_type": normalize_split_type(args.host_split_type),
        **summarize_rows(host_rows),
    }
    merged_rows.extend(host_rows)

    if args.cini_formal_manifest:
        cini_rows = load_rows(
            Path(args.cini_formal_manifest),
            split_column=args.cini_split_column,
            split_type=args.cini_split_type,
            default_group=args.cini_group,
        )
        cini_out = out_dir / "hvue_human_virus_pathogenicity_cini_cluster_disjoint.csv"
        write_manifest(cini_out, cini_rows)
        outputs["manifests"]["hvue_human_virus_pathogenicity_cini"] = {
            "path": str(cini_out),
            "source_path": args.cini_formal_manifest,
            "split_column": args.cini_split_column,
            "split_type": normalize_split_type(args.cini_split_type),
            **summarize_rows(cini_rows),
        }
        merged_rows.extend(cini_rows)
    else:
        outputs["missing_targets"].append(
            {
                "task": "hvue_human_virus_pathogenicity_cini",
                "reason": "No task-specific formal manifest with explicit disjoint split source was provided",
            }
        )

    merged_out = out_dir / "stage1_formal_targets_available_manifest.csv"
    write_manifest(merged_out, merged_rows)
    outputs["merged_manifest"] = str(merged_out)
    outputs["merged_summary"] = summarize_rows(merged_rows)
    outputs["source_audit"] = build_source_audit_report(args)

    report_path = out_dir / "stage1_formal_target_manifest_report.json"
    report_path.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    write_metadata(
        out_dir / "stage1_formal_target_manifest_metadata.json",
        build_run_metadata(
            args=args,
            data_paths=[
                args.host_formal_manifest,
                args.cini_formal_manifest,
                args.cini_unified_manifest,
                args.cini_raw_dir,
            ],
            extra={
                "phase": "build_stage1_formal_target_manifests",
                "merged_manifest": str(merged_out),
                "report_path": str(report_path),
                "available_manifest_tasks": sorted(outputs["manifests"].keys()),
                "missing_target_tasks": [row["task"] for row in outputs["missing_targets"]],
            },
        ),
    )
    return outputs, merged_out, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/phase2/stage1_formal_target_manifests")
    parser.add_argument(
        "--host-formal-manifest",
        default="data/phase2/audits/task7s_clean_gate_20260715/candidates/matched_all_pairs/formal_task_manifests/hvue_human_host_tropism.csv",
    )
    parser.add_argument("--host-task", default="hvue_human_host_tropism")
    parser.add_argument("--host-split-column", default="similarity_split")
    parser.add_argument("--host-split-type", default="cluster_disjoint")
    parser.add_argument("--host-group", default="primary_forget")
    parser.add_argument("--cini-formal-manifest", default="")
    parser.add_argument("--cini-task", default="hvue_human_virus_pathogenicity_cini")
    parser.add_argument("--cini-split-column", default="split")
    parser.add_argument("--cini-split-type", default="cluster_disjoint")
    parser.add_argument("--cini-group", default="primary_forget")
    parser.add_argument("--cini-raw-dir", default="data/benchmarks/raw/hvue/Pathogenecity/CINI")
    parser.add_argument("--cini-unified-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    args = parser.parse_args()

    _outputs, merged_out, report_path = build_formal_target_outputs(args)

    print(f"[stage1-manifest] wrote {merged_out}")
    print(f"[stage1-manifest] wrote {report_path}")


if __name__ == "__main__":
    main()
