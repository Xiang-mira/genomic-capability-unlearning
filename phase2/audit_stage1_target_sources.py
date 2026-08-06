"""Audit Stage 1 formal-target source availability for host tropism and CINI.

The goal is to make the current dual-target blocker executable: host tropism
has an explicit disjoint source in-repo, while CINI currently does not.
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

from phase2.run_metadata import build_run_metadata, write_metadata

TAXONOMY_COLUMNS = [
    "family",
    "genus",
    "species",
    "virus_tax_id",
    "virus_name",
    "source",
    "accession",
]


def normalize_split_type(value: str) -> str:
    split_type = str(value or "").strip().lower()
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type or "random"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def inspect_manifest_source(
    path: Path,
    *,
    task_filter: str,
    split_column: str,
    split_type: str,
) -> dict[str, Any]:
    rows, fieldnames = read_rows(path)
    filtered = [row for row in rows if not task_filter or row.get("task") == task_filter]
    taxonomy_columns_present = [col for col in TAXONOMY_COLUMNS if col in fieldnames]
    family_values = Counter((row.get("family") or "").strip() for row in filtered if row.get("family") is not None)
    split_values = Counter((row.get(split_column) or "").strip().lower() for row in filtered if split_column in row)
    explicit_split_rows = sum(count for key, count in split_values.items() if key in {"train", "val", "test"})
    return {
        "path": str(path),
        "task_filter": task_filter,
        "fieldnames": fieldnames,
        "n_rows_total": len(rows),
        "n_rows_task": len(filtered),
        "taxonomy_columns_present": taxonomy_columns_present,
        "family_values": dict(family_values),
        "split_column": split_column,
        "split_type": normalize_split_type(split_type),
        "split_values": dict(split_values),
        "explicit_split_rows": explicit_split_rows,
    }


def inspect_raw_split_dir(path: Path) -> dict[str, Any]:
    split_info: dict[str, Any] = {}
    all_columns: set[str] = set()
    total_rows = 0
    for split_name in ("train", "dev", "test"):
        csv_path = path / f"{split_name}.csv"
        if not csv_path.exists():
            split_info[split_name] = {"exists": False}
            continue
        rows, fieldnames = read_rows(csv_path)
        total_rows += len(rows)
        all_columns.update(fieldnames)
        split_info[split_name] = {
            "exists": True,
            "rows": len(rows),
            "fieldnames": fieldnames,
        }
    taxonomy_columns_present = [col for col in TAXONOMY_COLUMNS if col in all_columns]
    return {
        "path": str(path),
        "splits": split_info,
        "all_columns": sorted(all_columns),
        "taxonomy_columns_present": taxonomy_columns_present,
        "total_rows": total_rows,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    host = inspect_manifest_source(
        Path(args.host_formal_manifest),
        task_filter=args.host_task,
        split_column=args.host_split_column,
        split_type=args.host_split_type,
    )
    host["status"] = "available" if host["explicit_split_rows"] else "blocked"
    host["reason"] = (
        "Task-specific formal manifest contains an explicit split column that can be normalized "
        "to cluster_disjoint rows."
        if host["status"] == "available"
        else "Host source is missing usable explicit split rows."
    )

    cini_raw = inspect_raw_split_dir(Path(args.cini_raw_dir))
    cini_raw["status"] = "blocked"
    cini_raw["reason"] = (
        "Raw CINI CSVs expose only sequence/label, so they cannot support explicit disjoint split "
        "construction without external taxonomy or accession metadata."
    )

    cini_unified = inspect_manifest_source(
        Path(args.cini_unified_manifest),
        task_filter=args.cini_task,
        split_column="split",
        split_type="random",
    )
    cini_unified["status"] = "blocked"
    cini_unified["reason"] = (
        "Unified manifest preserves only benchmark/task/split/sequence/label/family/group/id; "
        "for CINI the family value is mixed and no taxonomy columns remain, so this is not enough "
        "to derive a validated cluster_disjoint split."
    )

    return {
        "host_formal_source": host,
        "cini_sources": {
            "raw_split_dir": cini_raw,
            "unified_manifest": cini_unified,
        },
        "overall_status": "host_only_ready",
        "recommended_next_step": (
            "Use the host formal source for current Stage 1 smoke, and treat CINI as blocked until "
            "a task-specific manifest with explicit disjoint semantics or external taxonomy metadata is added."
        ),
    }


def write_audit_outputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    report = build_report(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_metadata(
        out_path.with_name(f"{out_path.stem}_metadata.json"),
        build_run_metadata(
            args=args,
            data_paths=[args.host_formal_manifest, args.cini_raw_dir, args.cini_unified_manifest],
            extra={
                "phase": "audit_stage1_target_sources",
                "out_json": str(out_path),
                "overall_status": report["overall_status"],
                "host_status": report["host_formal_source"]["status"],
                "cini_raw_status": report["cini_sources"]["raw_split_dir"]["status"],
                "cini_unified_status": report["cini_sources"]["unified_manifest"]["status"],
            },
        ),
    )
    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="data/phase2/stage1_formal_target_manifests/stage1_target_source_audit.json")
    parser.add_argument(
        "--host-formal-manifest",
        default="data/phase2/audits/task7s_clean_gate_20260715/candidates/matched_all_pairs/formal_task_manifests/hvue_human_host_tropism.csv",
    )
    parser.add_argument("--host-task", default="hvue_human_host_tropism")
    parser.add_argument("--host-split-column", default="similarity_split")
    parser.add_argument("--host-split-type", default="cluster_disjoint")
    parser.add_argument("--cini-raw-dir", default="data/benchmarks/raw/hvue/Pathogenecity/CINI")
    parser.add_argument("--cini-unified-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--cini-task", default="hvue_human_virus_pathogenicity_cini")
    args = parser.parse_args()

    _report, out_path = write_audit_outputs(args)
    print(f"[stage1-source-audit] wrote {out_path}")


if __name__ == "__main__":
    main()
