"""Expanded ProteinGym ESM2-150M top-20 screening workflow.

This controller reuses the completed 48h ProteinGym qualification artifacts as
the frozen candidate source, then screens only the remaining top-20 assays with
a staged protocol.  The implementation deliberately delegates ProteinGym data
validation, public evolutionary score handling, split logic, baselines, ESM2
feature caching, and LoRA execution to ``proteingym_esm2_qualification``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import phase2.proteingym_esm2_qualification as base


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/phase2/protein_48h_esm2_qualification"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/protein_top20_esm2_expanded_qualification"
DEFAULT_SMOKE_ROOT = PROJECT_ROOT / "data/phase2/protein_top20_esm2_expanded_qualification_smoke"
DEFAULT_LOG = PROJECT_ROOT / "logs/protein_top20_esm2_expanded_qualification.log"

EXCLUDED_COMPLETED_ASSAYS = {
    "CCDB_ECOLI_Adkar_2012",
    "GAL4_YEAST_Kitzman_2015",
    "MET_HUMAN_Estevam_2023",
}

SCREENING_REPRESENTATIONS = ("mutation_position", "wt_mutant_position_difference")
CONFIRMATION_EXTRA_REPRESENTATIONS = ("local_window", "whole_sequence_mean")
SPLIT_TYPES = ("random", "position_heldout")

TOP20_INVENTORY_FIELDS = [
    "source_candidate_rank",
    "assay_id",
    "scheduled_for_screening",
    "excluded_previous_completed",
    "source_quality_score",
    "source_exclusion_reason",
    "records_loaded",
    "unique_positions",
    "sequence_length",
    "valid_single_substitution_count",
    "data_quality_score",
    "top20_status",
    "reused_previous_artifact_root",
    "dms_path",
    "wild_type_sequence",
]

HEADROOM_FIELDS = [
    "source_candidate_rank",
    "assay_id",
    "records",
    "unique_positions",
    "strongest_simple_baseline",
    "strongest_simple_val_spearman",
    "strongest_evolutionary_baseline",
    "strongest_evolutionary_val_spearman",
    "strongest_overall_non_PLM_baseline",
    "strongest_overall_val_spearman",
    "best_esm2_method",
    "best_esm2_readout",
    "best_esm2_representation",
    "validation_spearman",
    "validation_excess",
    "validation_bootstrap_ci_low",
    "validation_bootstrap_ci_high",
    "random_split_best_validation_excess",
    "test_excess_after_freeze",
    "advancement_status",
    "rejection_reason",
    "prediction_path",
]

QUALIFICATION_FIELDS = [
    "assay_id",
    "stage",
    "status",
    "selected_method",
    "selected_readout",
    "selected_representation",
    "validation_excess",
    "test_excess",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "lora_seed_count",
    "lora_mean_excess",
    "lora_std_excess",
    "lora_worst_seed_excess",
    "labels",
    "reason",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def top20_registry_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "protein_top20_registry.json"


def top20_status_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "protein_top20_status.json"


def load_top20_registry(args: argparse.Namespace) -> dict[str, Any]:
    path = top20_registry_path(args)
    if path.exists():
        return base.read_json(path)
    return {
        "workflow": {
            "name": "protein_top20_esm2_expanded_qualification",
            "created_at": now_utc(),
            "out_root": str(Path(args.out_root)),
            "source_root": str(Path(args.source_root)),
            "excluded_completed_assays": sorted(EXCLUDED_COMPLETED_ASSAYS),
            "max_concurrent_gpu_jobs": 1,
            "formal": bool(args.formal),
            "mock_esm2": bool(args.mock_esm2),
        },
        "tasks": [],
    }


def save_top20_registry(args: argparse.Namespace, registry: dict[str, Any]) -> None:
    registry.setdefault("workflow", {})["updated_at"] = now_utc()
    base.write_json(top20_registry_path(args), registry, overwrite=True)


def update_top20_task(
    args: argparse.Namespace,
    registry: dict[str, Any],
    task_id: str,
    *,
    stage: str,
    status: str,
    **extra: Any,
) -> None:
    if status not in base.STATES:
        raise ValueError(f"Invalid registry status: {status}")
    tasks = registry.setdefault("tasks", [])
    row = next((task for task in tasks if task.get("task_id") == task_id), None)
    if row is None:
        row = {
            "task_id": task_id,
            "stage": stage,
            "pid": os.getpid(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", str(args.cuda_visible_devices)),
            "attempts": 0,
            "created_at": now_utc(),
        }
        tasks.append(row)
    row["stage"] = stage
    row["status"] = status
    row["updated_at"] = now_utc()
    if status == "running":
        row["attempts"] = int(row.get("attempts", 0)) + 1
        row["started_at"] = now_utc()
    if status in {"complete", "valid", "invalid", "failed", "skipped"}:
        row["completed_at"] = now_utc()
    row.update(extra)
    save_top20_registry(args, registry)
    base.write_json(
        top20_status_path(args),
        {
            "updated_at": now_utc(),
            "status": status,
            "stage": stage,
            "task_id": task_id,
            "registry_path": str(top20_registry_path(args)),
            **extra,
        },
        overwrite=True,
    )


def run_top20_stage(
    args: argparse.Namespace,
    registry: dict[str, Any],
    task_id: str,
    stage: str,
    output_path: Path,
    func: Any,
    *,
    validator: Any | None = None,
    skippable: bool = True,
) -> Any:
    if args.resume and skippable and base.stage_done(output_path, validator):
        update_top20_task(args, registry, task_id, stage=stage, status="skipped", reason="resume_output_valid", output_path=str(output_path))
        base.append_log(args, f"skip {stage}: output already valid at {output_path}")
        return None
    last_exc = None
    for attempt in range(int(args.max_retries) + 1):
        update_top20_task(args, registry, task_id, stage=stage, status="running", output_path=str(output_path), retry_attempt=attempt)
        try:
            result = func()
            if validator is not None and not validator(output_path):
                raise RuntimeError(f"Validation failed for stage {stage}: {output_path}")
            update_top20_task(args, registry, task_id, stage=stage, status="complete", output_path=str(output_path))
            base.append_log(args, f"complete {stage}: {output_path}")
            return result
        except Exception as exc:  # noqa: PERF203 - retry evidence belongs in the registry.
            last_exc = exc
            update_top20_task(
                args,
                registry,
                task_id,
                stage=stage,
                status="failed",
                output_path=str(output_path),
                exception_type=type(exc).__name__,
                exception=str(exc),
                retry_attempt=attempt,
            )
            base.append_log(args, f"failed {stage} attempt={attempt}: {type(exc).__name__}: {exc}")
            if attempt >= int(args.max_retries):
                break
    raise RuntimeError(f"Stage {stage} failed after retries") from last_exc


def candidate_ranking_path(args: argparse.Namespace) -> Path:
    return Path(args.source_root) / "protein_48h_candidate_ranking.csv"


def load_frozen_top20_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = candidate_ranking_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Frozen candidate ranking not found: {path}")
    rows = list(csv.DictReader(path.open()))
    top20 = [row for row in rows if 0 < base.numeric(row.get("candidate_rank"), 999999.0) <= 20]
    if len(top20) < 20:
        top20 = sorted(rows, key=lambda row: (base.numeric(row.get("candidate_rank"), 999999.0), row.get("assay_id", "")))[:20]
    return sorted(top20, key=lambda row: (base.numeric(row.get("candidate_rank"), 999999.0), row.get("assay_id", "")))


def scheduled_top20_assay_ids(args: argparse.Namespace) -> list[str]:
    return [row["assay_id"] for row in load_frozen_top20_rows(args) if row.get("assay_id") not in EXCLUDED_COMPLETED_ASSAYS]


def alias_48h_to_top20(out_root: Path, src_name: str, dst_name: str) -> None:
    src = out_root / src_name
    dst = out_root / dst_name
    if src.exists():
        dst.write_bytes(src.read_bytes())


def write_top20_protocol(args: argparse.Namespace) -> dict[str, Any]:
    payload = base.frozen_protocol_payload(args)
    payload.update(
        {
            "workflow": "ProteinGym-ESM2-150M expanded top-20 qualification",
            "source_root": str(Path(args.source_root)),
            "excluded_completed_assays": sorted(EXCLUDED_COMPLETED_ASSAYS),
            "candidate_set_rule": "frozen source ranking candidate_rank <= 20; first three completed assays are linked and excluded from new computation",
            "stage_order": [
                "proteingym_data_validation",
                "top20_inventory",
                "position_heldout_and_random_split_creation",
                "evolutionary_baseline_resources",
                "baseline_evaluation",
                "resume_validation",
                "esm2_low_cost_screening",
                "headroom_ranking",
                "full_frozen_confirmation",
                "qualification_gate",
                "fresh_lora_calibration_and_confirmation",
                "final_aggregation",
            ],
            "esm2_screening": {
                "model": args.esm2_model,
                "zero_shot": True,
                "representations": list(SCREENING_REPRESENTATIONS),
                "readouts": list(base.ESM2_READOUTS),
                "selection_split": "val",
                "test_set_used_for_candidate_selection": False,
            },
            "frozen_confirmation": {
                "max_promoted_assays": args.max_confirmation_assays,
                "extra_representations": list(CONFIRMATION_EXTRA_REPRESENTATIONS),
                "max_lora_assays": args.max_lora_assays,
                "test_set_used_after_method_freeze": True,
            },
        }
    )
    payload["protocol_hash"] = base.stable_hash(payload)
    base.write_json(Path(args.out_root) / "protein_top20_frozen_protocol.json", payload, overwrite=True)
    return payload


def top20_inventory(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    top20_rows = load_frozen_top20_rows(args)
    assay_inputs = {assay.assay_id: assay for assay in base.build_assay_inputs(args)}
    inventory_rows: list[dict[str, Any]] = []
    records_payload: dict[str, list[dict[str, Any]]] = {}
    scheduled_ids = []
    missing_ids = []

    for source_row in top20_rows:
        assay_id = source_row["assay_id"]
        excluded = assay_id in EXCLUDED_COMPLETED_ASSAYS
        scheduled = not excluded
        inv: dict[str, Any] = {
            "source_candidate_rank": int(base.numeric(source_row.get("candidate_rank"), 0)),
            "assay_id": assay_id,
            "scheduled_for_screening": scheduled,
            "excluded_previous_completed": excluded,
            "source_quality_score": source_row.get("data_quality_score", ""),
            "source_exclusion_reason": source_row.get("exclusion_reason", ""),
            "records_loaded": "",
            "unique_positions": "",
            "sequence_length": source_row.get("sequence_length", ""),
            "valid_single_substitution_count": source_row.get("valid_single_substitution_count", ""),
            "data_quality_score": source_row.get("data_quality_score", ""),
            "top20_status": "linked_to_completed_48h_run" if excluded else "scheduled",
            "reused_previous_artifact_root": str(Path(args.source_root)) if excluded else "",
            "dms_path": "",
            "wild_type_sequence": "",
        }
        if scheduled:
            assay = assay_inputs.get(assay_id)
            if assay is None:
                inv["top20_status"] = "missing_dms_file"
                missing_ids.append(assay_id)
            else:
                row, records = base.assay_inventory_row(args, assay)
                inv.update(
                    {
                        "records_loaded": len(records),
                        "unique_positions": len({record.position for record in records}),
                        "sequence_length": row.get("sequence_length", ""),
                        "valid_single_substitution_count": row.get("valid_single_substitution_count", ""),
                        "data_quality_score": row.get("data_quality_score", source_row.get("data_quality_score", "")),
                        "top20_status": "scheduled_with_records" if records else "no_valid_single_substitution_records",
                        "dms_path": row.get("dms_path", ""),
                        "wild_type_sequence": row.get("wild_type_sequence", ""),
                    }
                )
                records_payload[assay_id] = [
                    {
                        "assay_id": record.assay_id,
                        "sample_id": record.sample_id,
                        "mutation": record.mutation,
                        "wt": record.wt,
                        "position": record.position,
                        "mut": record.mut,
                        "score": record.score,
                        "mutated_sequence": record.mutated_sequence,
                        "wt_sequence": record.wt_sequence,
                    }
                    for record in records
                ]
                if records:
                    scheduled_ids.append(assay_id)
        inventory_rows.append(inv)

    base.write_csv(out_root / "protein_top20_candidate_inventory.csv", inventory_rows, TOP20_INVENTORY_FIELDS, overwrite=True)
    base.write_json(out_root / "protein_top20_valid_records.json", records_payload, overwrite=True)
    base.write_json(out_root / "protein_48h_valid_records.json", records_payload, overwrite=True)
    payload = {
        "created_at": now_utc(),
        "source_ranking": str(candidate_ranking_path(args)),
        "top20_assays": [row["assay_id"] for row in top20_rows],
        "excluded_completed_assays": sorted(EXCLUDED_COMPLETED_ASSAYS),
        "scheduled_assays": scheduled_ids,
        "missing_scheduled_assays": missing_ids,
        "records_path": str(out_root / "protein_top20_valid_records.json"),
    }
    return payload


def create_top20_splits(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    records_by_assay = base.load_valid_records(out_root)
    scheduled_ids = scheduled_top20_assay_ids(args)
    all_entries: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for assay_id in scheduled_ids:
        records = records_by_assay.get(assay_id, [])
        if not records:
            continue
        audits[assay_id] = {}
        for split_type in SPLIT_TYPES:
            entries = base.split_records(records, split_type, args)
            all_entries.extend(entries)
            audits[assay_id][split_type] = base.split_audit_for_entries(entries, records[0].wt_sequence)
    payload = {
        "created_at": now_utc(),
        "split_seed": args.split_seed,
        "primary_split": "position_heldout",
        "pilot_assay_ids": [assay_id for assay_id in scheduled_ids if assay_id in records_by_assay],
        "excluded_completed_assays": sorted(EXCLUDED_COMPLETED_ASSAYS),
        "entries": all_entries,
    }
    payload["manifest_hash"] = base.stable_hash(payload)
    audit_payload = {"created_at": now_utc(), "assays": audits, "manifest_hash": payload["manifest_hash"]}
    base.write_json(out_root / "protein_top20_split_manifest.json", payload, overwrite=True)
    base.write_json(out_root / "protein_top20_split_audit.json", audit_payload, overwrite=True)
    base.write_json(out_root / "protein_48h_split_manifest.json", payload, overwrite=True)
    base.write_json(out_root / "protein_48h_split_audit.json", audit_payload, overwrite=True)
    return payload


def source_evolutionary_report(args: argparse.Namespace) -> dict[str, Any]:
    return base.read_json(Path(args.source_root) / "protein_48h_evolutionary_baseline_report.json")


def official_public_score_archive(args: argparse.Namespace) -> tuple[dict[str, Any], str, int, str, list[dict[str, Any]]]:
    unavailable: list[dict[str, Any]] = []
    record: dict[str, Any] = {"id": "", "doi": "", "metadata": {"version": ""}, "files": []}
    archive_url = ""
    archive_size = 0
    archive_checksum = ""
    if args.auto_download_proteingym and not args.skip_proteingym_download:
        try:
            record = base.fetch_proteingym_record(args.proteingym_source_api, retries=max(1, args.download_retries))
            archive_info = base.zenodo_file_map(record).get(base.PROTEINGYM_PUBLIC_SCORE_ARCHIVE, {})
            archive_url = archive_info.get("links", {}).get("self", "")
            archive_size = int(archive_info.get("size") or 0)
            archive_checksum = archive_info.get("checksum", "")
        except Exception as exc:
            unavailable.append(
                {
                    "resource": "zenodo_record_metadata",
                    "status": "TRANSIENT_NOT_AVAILABLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if not archive_url:
        previous = source_evolutionary_report(args)
        archive = previous.get("archive", {})
        archive_url = str(archive.get("source_url", ""))
        archive_size = int(archive.get("size_bytes") or 0)
        archive_checksum = str(archive.get("checksum", ""))
        if previous:
            record = {
                "id": previous.get("dataset_revision", ""),
                "doi": previous.get("source_doi", ""),
                "metadata": {"version": previous.get("dataset_version", "")},
                "files": [],
            }
            unavailable.append(
                {
                    "resource": "zenodo_record_metadata",
                    "status": "REUSED_PREVIOUS_SOURCE_METADATA",
                    "reason": f"using source-root evolutionary report from {Path(args.source_root)}",
                }
            )
    return record, archive_url, archive_size, archive_checksum, unavailable


def score_archive_entries(args: argparse.Namespace, archive_url: str, archive_size: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not archive_url or not archive_size:
        return {}, [
            {
                "resource": base.PROTEINGYM_PUBLIC_SCORE_ARCHIVE,
                "status": "NOT_AVAILABLE",
                "reason": "official ProteinGym public-score archive URL or size was unavailable",
            }
        ]
    try:
        entries = base.remote_zip_central_directory(archive_url, archive_size, retries=max(5, args.download_retries))
        return entries, []
    except Exception as exc:
        return {}, [
            {
                "resource": base.PROTEINGYM_PUBLIC_SCORE_ARCHIVE,
                "status": "TRANSIENT_NOT_AVAILABLE",
                "reason": f"central-directory read failed: {type(exc).__name__}: {exc}",
            }
        ]


def find_score_member(entries: Mapping[str, Mapping[str, Any]], assay_id: str) -> Mapping[str, Any] | None:
    member_name = f"{assay_id}.csv"
    entry = entries.get(member_name)
    if entry is not None:
        return entry
    matches = [row for name, row in entries.items() if Path(name).name == member_name]
    return matches[0] if matches else None


def run_top20_evolutionary_resources(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    report_path = out_root / "protein_48h_evolutionary_baseline_report.json"
    records_by_assay = base.load_valid_records(out_root)
    split_payload = base.read_json(out_root / "protein_top20_split_manifest.json")
    assay_ids = list(split_payload.get("pilot_assay_ids", []))
    record, archive_url, archive_size, archive_checksum, unavailable = official_public_score_archive(args)
    entries, entry_unavailable = score_archive_entries(args, archive_url, archive_size)
    unavailable.extend(entry_unavailable)

    reused: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    per_assay: dict[str, Any] = {}
    for assay_id in assay_ids:
        records = records_by_assay.get(assay_id, [])
        existing_path, existing_validation = base.find_existing_valid_public_score(args, assay_id, records)
        if existing_path and existing_validation:
            reused.append({"assay_id": assay_id, **existing_validation})
            per_assay[assay_id] = {"status": "valid", "source": "reused_existing_file", "validation": existing_validation}
            continue

        entry = find_score_member(entries, assay_id)
        if entry is None:
            reason = f"member {assay_id}.csv was not found in {base.PROTEINGYM_PUBLIC_SCORE_ARCHIVE}"
            unavailable.append({"assay_id": assay_id, "status": "NOT_AVAILABLE", "reason": reason})
            per_assay[assay_id] = {"status": "NOT_AVAILABLE", "reason": reason}
            continue

        target_path = base.public_score_target_path(args, assay_id)
        try:
            extraction = base.extract_zip_member_by_range(archive_url, entry, target_path, retries=max(8, args.download_retries))
            validation = base.validate_evolutionary_prediction_file(target_path, assay_id, records)
            if validation.get("status") != "valid":
                target_path.unlink(missing_ok=True)
                reason = str(validation.get("reason", "extracted_file_invalid"))
                unavailable.append({"assay_id": assay_id, "status": "NOT_AVAILABLE", "reason": reason, "validation": validation})
                per_assay[assay_id] = {"status": "NOT_AVAILABLE", "reason": reason, "validation": validation}
                continue
            extracted.append({"assay_id": assay_id, **extraction, "validation": validation})
            per_assay[assay_id] = {"status": "valid", "source": "official_public_score_archive", "validation": validation}
        except Exception as exc:
            reason = f"official_public_score_extraction_failed:{type(exc).__name__}: {exc}"
            unavailable.append({"assay_id": assay_id, "status": "NOT_AVAILABLE", "reason": reason})
            per_assay[assay_id] = {"status": "NOT_AVAILABLE", "reason": reason}

    retained = [
        {
            "assay_id": assay_id,
            "path": row["validation"]["path"],
            "sha256": row["validation"].get("sha256", ""),
            "size_bytes": row["validation"].get("size_bytes", ""),
            "method_counts": row["validation"].get("method_counts", {}),
        }
        for assay_id, row in per_assay.items()
        if row.get("status") == "valid"
    ]
    valid_count = sum(1 for row in per_assay.values() if row.get("status") == "valid")
    payload = {
        "created_at": now_utc(),
        "official_source": f"https://zenodo.org/records/{record.get('id', base.PROTEINGYM_ZENODO_RECORD_ID)}" if record.get("id") else "https://zenodo.org/records/15293562",
        "source_api": args.proteingym_source_api,
        "source_doi": record.get("doi", ""),
        "dataset_version": record.get("metadata", {}).get("version", ""),
        "dataset_revision": str(record.get("id", "")),
        "archive": {
            "key": base.PROTEINGYM_PUBLIC_SCORE_ARCHIVE,
            "source_url": archive_url,
            "size_bytes": archive_size,
            "checksum": archive_checksum,
            "retained_complete_archive": False,
            "extraction_strategy": "HTTP byte-range extraction of selected assay CSV members only; per-assay failures are recorded as NOT_AVAILABLE",
        },
        "pilot_assays": assay_ids,
        "allowed_evolutionary_columns": list(base.EVOLUTIONARY_BASELINE_COLUMNS),
        "status": "valid" if valid_count == len(assay_ids) else ("partial" if valid_count else "NOT_AVAILABLE"),
        "valid_assay_count": valid_count,
        "per_assay": per_assay,
        "downloaded_and_retained_files": retained,
        "reused_files": reused,
        "extracted_files": extracted,
        "removed_temporary_files": [],
        "unavailable_optional_resources": unavailable,
        "storage_policy": {
            "canonical_public_prediction_dir": str(Path(args.public_predictions_dir)),
            "complete_public_score_archive_retained": False,
            "complete_assay_files_copied_to_run_dirs": False,
        },
    }
    base.write_json(report_path, payload, overwrite=True)
    base.write_evolutionary_resource_report_md(out_root / "protein_48h_evolutionary_baseline_report.md", payload)
    alias_48h_to_top20(out_root, "protein_48h_evolutionary_baseline_report.json", "protein_top20_evolutionary_baseline_report.json")
    alias_48h_to_top20(out_root, "protein_48h_evolutionary_baseline_report.md", "protein_top20_evolutionary_baseline_report.md")
    return payload


def run_top20_baselines(args: argparse.Namespace) -> dict[str, Any]:
    payload = evaluate_top20_baselines(args)
    alias_48h_to_top20(Path(args.out_root), "protein_48h_baseline_results.csv", "protein_top20_baseline_results.csv")
    alias_48h_to_top20(Path(args.out_root), "protein_48h_baseline_report.md", "protein_top20_baseline_report.md")
    alias_48h_to_top20(Path(args.out_root), "protein_48h_evolutionary_baseline_report.json", "protein_top20_evolutionary_baseline_report.json")
    alias_48h_to_top20(Path(args.out_root), "protein_48h_evolutionary_baseline_report.md", "protein_top20_evolutionary_baseline_report.md")
    return payload


def evaluate_top20_baselines(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    split_payload = base.read_json(out_root / "protein_top20_split_manifest.json")
    resource_report = base.read_json(out_root / "protein_top20_evolutionary_baseline_report.json")
    records_by_assay = base.load_valid_records(out_root)
    split_entries = split_payload.get("entries", [])
    rows: list[dict[str, Any]] = []
    pred_root = out_root / "protein_48h_baseline_predictions"
    for assay_id in split_payload.get("pilot_assay_ids", []):
        records_all = records_by_assay.get(assay_id, [])
        for split_type in SPLIT_TYPES:
            paired = base.rows_for_split(records_all, split_entries, split_type)
            if not paired:
                continue
            records, y, splits = base.split_arrays(paired)
            for baseline_name in [
                "wt_residue",
                "mutant_residue",
                "wt_plus_mutant_residue",
                "mutation_position",
                "normalized_position",
                "residue_pair_plus_position",
                "blosum_substitution",
                "charge_change",
                "hydrophobicity_change",
                "residue_volume_change",
            ]:
                meta, pred = base.fit_ridge_baseline(records, y, splits, baseline_name)
                pred_path = pred_root / assay_id / split_type / f"{baseline_name}.csv"
                base.write_prediction_csv(pred_path, records, splits, y, pred, baseline_name, overwrite=True)
                rows.append(base.metric_row(assay_id, split_type, baseline_name, "simple", "complete", records, y, splits, pred, meta, pred_path))
            for baseline_name, key_func in [
                ("site_specific_mean", lambda record: record.position),
                ("site_independent_pair_mean", lambda record: (record.wt, record.mut)),
            ]:
                meta, pred = base.mean_lookup_baseline(records, y, splits, key_func)
                pred_path = pred_root / assay_id / split_type / f"{baseline_name}.csv"
                base.write_prediction_csv(pred_path, records, splits, y, pred, baseline_name, overwrite=True)
                rows.append(base.metric_row(assay_id, split_type, baseline_name, "simple", "complete", records, y, splits, pred, meta, pred_path))
            public_rows = base.load_public_prediction_table(args, assay_id, records)
            if public_rows:
                for baseline_name, source, raw_pred in public_rows:
                    finite = np.isfinite(raw_pred)
                    pred = np.where(finite, raw_pred, np.nanmean(raw_pred[finite]))
                    sign = base.direction_sign_from_validation(y[splits == "val"], pred[splits == "val"])
                    pred = base.apply_direction(pred, sign)
                    pred_path = pred_root / assay_id / split_type / f"{base.normalize_id(baseline_name)}.csv"
                    base.write_prediction_csv(pred_path, records, splits, y, pred, baseline_name, overwrite=True)
                    rows.append(base.metric_row(assay_id, split_type, baseline_name, "evolutionary", "complete", records, y, splits, pred, {"direction_sign": sign, "selected_alpha": source}, pred_path))
            else:
                per_assay = resource_report.get("per_assay", {}).get(assay_id, {})
                reason = per_assay.get("reason") or "no matching MSA/evolutionary public prediction table was found"
                rows.append(
                    {
                        "assay_id": assay_id,
                        "split_type": split_type,
                        "baseline": "strongest_available_msa_or_evolutionary_prediction",
                        "baseline_family": "evolutionary",
                        "status": "NOT_AVAILABLE",
                        "selection_split": "val",
                        "selected_alpha": "",
                        "direction_sign": "",
                        "n_train": int((splits == "train").sum()),
                        "n_val": int((splits == "val").sum()),
                        "n_test": int((splits == "test").sum()),
                        "val_spearman": "",
                        "test_spearman": "",
                        "val_mse": "",
                        "test_mse": "",
                        "mse_improvement_over_global_mean": "",
                        "prediction_path": "",
                        "is_strongest_available_non_plm": False,
                        "not_available_reason": reason,
                    }
                )
    strongest = base.strongest_baseline_rows(rows)
    for row in rows:
        key = (row.get("assay_id"), row.get("split_type"))
        if key in strongest and row.get("baseline") == strongest[key].get("baseline"):
            row["is_strongest_available_non_plm"] = True
    baseline_path = out_root / "protein_48h_baseline_results.csv"
    if baseline_path.exists():
        base.snapshot_existing_artifact(out_root, baseline_path, "baseline_results", "top20_baseline_recomputed_after_resumable_evolutionary_resource_validation")
    base.write_csv(baseline_path, rows, base.BASELINE_FIELDS, overwrite=True)
    base.write_baseline_report(out_root, rows)
    base.write_evolutionary_baseline_summary(args, rows, resource_report)
    return {"rows": rows, "strongest": strongest}


def load_top20_records(out_root: Path) -> dict[str, list[base.VariantRecord]]:
    return base.load_valid_records(out_root)


def baseline_strength(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_root) / "protein_top20_baseline_results.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    return base.baseline_strength_payload(rows)


def baseline_complete_assay_ids(args: argparse.Namespace) -> list[str]:
    strength = baseline_strength(args)
    audits = base.read_json(Path(args.out_root) / "protein_top20_split_audit.json").get("assays", {})
    result = []
    for assay_id in scheduled_top20_assay_ids(args):
        pos = strength.get(assay_id, {}).get("position_heldout", {})
        if pos.get("evolutionary_status") == "complete" and audits.get(assay_id, {}).get("position_heldout", {}).get("status") == "valid":
            result.append(assay_id)
    return result


def method_filename(rep_type: str, readout: str) -> str:
    if rep_type == "masked_log_odds":
        return "zero_shot.csv"
    return f"{rep_type}_{readout}.csv"


def top20_prediction_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "protein_top20_esm2_predictions"


def top20_expected_prediction_specs(
    args: argparse.Namespace,
    assay_id: str,
    split_type: str,
    representations: Sequence[str],
) -> list[dict[str, Any]]:
    pred_root = top20_prediction_root(args)
    specs = [
        {
            "method": "zero_shot",
            "readout": "wild_type_relative_masked_log_odds",
            "representation_type": "masked_log_odds",
            "path": pred_root / assay_id / split_type / "zero_shot.csv",
            "expected_method": "zero_shot",
            "cache_key": base.stable_hash(["mock", assay_id, split_type, "zero"]) if args.mock_esm2 else "",
        }
    ]
    for rep_type in representations:
        for readout in base.ESM2_READOUTS:
            specs.append(
                {
                    "method": "frozen_representation",
                    "readout": readout,
                    "representation_type": rep_type,
                    "path": pred_root / assay_id / split_type / method_filename(rep_type, readout),
                    "expected_method": f"{rep_type}_{readout}",
                    "cache_key": "",
                }
            )
    return specs


def validate_top20_resume_state(args: argparse.Namespace, registry: dict[str, Any]) -> dict[str, Any]:
    out_root = Path(args.out_root)
    records_by_assay = load_top20_records(out_root)
    split_payload = base.read_json(out_root / "protein_top20_split_manifest.json")
    split_hash = split_payload.get("manifest_hash", base.stable_hash(split_payload))
    assay_ids = baseline_complete_assay_ids(args)
    promoted_existing = []
    ranking_path = out_root / "protein_top20_candidate_headroom_ranking.csv"
    if ranking_path.exists():
        promoted_existing = [
            row["assay_id"]
            for row in csv.DictReader(ranking_path.open())
            if row.get("advancement_status") in {"PROMOTED_TO_FULL_CONFIRMATION", "PROMOTED_TO_LORA"}
        ][: args.max_confirmation_assays]
    rep_by_assay = {assay_id: list(SCREENING_REPRESENTATIONS) for assay_id in assay_ids}
    for assay_id in promoted_existing:
        rep_by_assay[assay_id] = [*SCREENING_REPRESENTATIONS, *CONFIRMATION_EXTRA_REPRESENTATIONS]

    valid_predictions: list[dict[str, Any]] = []
    invalid_predictions: list[dict[str, Any]] = []
    missing_predictions: list[dict[str, Any]] = []
    valid_caches: list[dict[str, Any]] = []
    invalid_caches: list[dict[str, Any]] = []
    missing_caches: list[dict[str, Any]] = []
    for assay_id, representations in rep_by_assay.items():
        records_all = records_by_assay.get(assay_id, [])
        for split_type in SPLIT_TYPES:
            paired = base.rows_for_split(records_all, split_payload.get("entries", []), split_type)
            if not paired:
                continue
            records, _y, splits = base.split_arrays(paired)
            for spec in top20_expected_prediction_specs(args, assay_id, split_type, representations):
                validation = base.validate_prediction_csv(Path(spec["path"]), records, splits, expected_method=spec["expected_method"])
                row = {"assay_id": assay_id, "split_type": split_type, **{k: v for k, v in spec.items() if k != "path"}, **validation}
                if validation.get("status") == "valid":
                    valid_predictions.append(row)
                elif validation.get("status") == "missing":
                    missing_predictions.append(row)
                else:
                    invalid_predictions.append(row)
            for rep_type in representations:
                cache_key = base.esm2_cache_key(args, assay_id, split_type, rep_type, records, split_hash)
                cache_path = out_root / "caches/esm2_frozen" / f"{cache_key}.npz"
                validation = base.validate_feature_cache(cache_path, records)
                row = {"assay_id": assay_id, "split_type": split_type, "representation_type": rep_type, "cache_key": cache_key, **validation}
                if validation.get("status") == "valid":
                    valid_caches.append(row)
                elif validation.get("status") == "missing":
                    missing_caches.append(row)
                else:
                    invalid_caches.append(row)

    stale = []
    for task in registry.get("tasks", []):
        if task.get("status") == "running" and not base.process_alive(task.get("pid")):
            stale.append(dict(task))
            task["status"] = "failed"
            task["completed_at"] = now_utc()
            task["stale_running_reconciled_at"] = now_utc()
            task["stale_running_reconciliation_reason"] = "recorded PID is no longer active during top20 resume validation"
    if stale:
        registry.setdefault("stale_running_reconciliations", []).extend(stale)
        save_top20_registry(args, registry)

    payload = {
        "created_at": now_utc(),
        "out_root": str(out_root),
        "status": "valid" if not invalid_predictions and not invalid_caches else "requires_rerun",
        "baseline_complete_assays_considered": assay_ids,
        "expected_esm2_prediction_outputs": len(valid_predictions) + len(invalid_predictions) + len(missing_predictions),
        "valid_reused_esm2_prediction_outputs": len(valid_predictions),
        "invalid_esm2_prediction_outputs": len(invalid_predictions),
        "missing_esm2_prediction_outputs": len(missing_predictions),
        "valid_reused_frozen_caches": len(valid_caches),
        "invalid_frozen_caches": len(invalid_caches),
        "missing_frozen_caches": len(missing_caches),
        "valid_reused_artifacts": valid_predictions[:500],
        "invalid_or_incomplete_artifacts": [*invalid_predictions, *invalid_caches][:500],
        "outputs_scheduled_for_rerun": [*invalid_predictions, *missing_predictions, *invalid_caches, *missing_caches][:500],
        "cache_hits": valid_caches[:500],
        "registry_corrections": stale,
        "completed_previous_assays_not_recomputed": sorted(EXCLUDED_COMPLETED_ASSAYS),
    }
    base.write_json(out_root / "protein_top20_resume_validation.json", payload, overwrite=True)
    return payload


def compute_esm2_rows(
    args: argparse.Namespace,
    assay_ids: Sequence[str],
    representations: Sequence[str],
) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    records_by_assay = load_top20_records(out_root)
    split_payload = base.read_json(out_root / "protein_top20_split_manifest.json")
    split_hash = split_payload.get("manifest_hash", base.stable_hash(split_payload))
    strongest = base.baseline_lookup(out_root)
    rows: list[dict[str, Any]] = []
    pred_root = top20_prediction_root(args)
    model_hash = base.model_hash_for_args(args)
    pred_root.mkdir(parents=True, exist_ok=True)

    for assay_id in assay_ids:
        records_all = records_by_assay.get(assay_id, [])
        for split_type in SPLIT_TYPES:
            paired = base.rows_for_split(records_all, split_payload.get("entries", []), split_type)
            if not paired:
                continue
            records, y, splits = base.split_arrays(paired)
            base_row = strongest.get((assay_id, split_type))
            try:
                pred_path = pred_root / assay_id / split_type / "zero_shot.csv"
                zero_pred, validation = base.load_valid_prediction_values(pred_path, records, splits, expected_method="zero_shot")
                if zero_pred is None:
                    if validation.get("status") == "invalid":
                        base.quarantine_invalid_artifact(args, pred_path, str(validation.get("reason", "invalid_zero_shot_prediction")))
                    if args.mock_esm2:
                        zero_pred = np.array([base.mock_signal(record) for record in records], dtype=np.float64)
                    else:
                        zero_pred = base.run_zero_shot_esm2(args, records, splits)
                    sign = base.direction_sign_from_validation(y[splits == "val"], zero_pred[splits == "val"])
                    zero_pred = base.apply_direction(zero_pred, sign)
                    base.write_prediction_csv(pred_path, records, splits, y, zero_pred, "zero_shot", overwrite=True)
                rows.append(
                    base.esm2_metric_row(
                        args,
                        assay_id,
                        split_type,
                        "zero_shot",
                        "wild_type_relative_masked_log_odds",
                        records,
                        y,
                        splits,
                        zero_pred,
                        {"selected_alpha": "", "representation_type": "masked_log_odds"},
                        pred_path,
                        base.stable_hash([model_hash, assay_id, split_hash, split_type, "zero"]),
                        base_row,
                    )
                )
            except Exception as exc:
                rows.append(base.unavailable_esm2_row(args, assay_id, split_type, "zero_shot", str(exc), base_row))

            for rep_type in representations:
                cache_key = base.esm2_cache_key(args, assay_id, split_type, rep_type, records, split_hash)
                cached_features: np.ndarray | None = None
                for readout in base.ESM2_READOUTS:
                    method = f"{rep_type}_{readout}"
                    pred_path = pred_root / assay_id / split_type / method_filename(rep_type, readout)
                    pred, validation = base.load_valid_prediction_values(pred_path, records, splits, expected_method=method)
                    if pred is None:
                        if validation.get("status") == "invalid":
                            base.quarantine_invalid_artifact(args, pred_path, str(validation.get("reason", "invalid_esm2_prediction")))
                        try:
                            if cached_features is None:
                                if args.mock_esm2:
                                    cached_features = base.mock_features(records, rep_type)
                                    cache_path = out_root / "caches/esm2_frozen" / f"{cache_key}.npz"
                                    if base.validate_feature_cache(cache_path, records).get("status") != "valid":
                                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                                        tmp_cache = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                                        with tmp_cache.open("wb") as handle:
                                            np.savez_compressed(handle, features=cached_features.astype(np.float32), sample_ids=np.array([r.sample_id for r in records]))
                                        os.replace(tmp_cache, cache_path)
                                else:
                                    cached_features = base.extract_real_esm2_features(args, records, rep_type, cache_key)
                            meta, pred = base.fit_readout_features(cached_features, y, splits, readout)
                            meta["representation_type"] = rep_type
                            base.write_prediction_csv(pred_path, records, splits, y, pred, method, overwrite=True)
                        except Exception as exc:
                            rows.append(base.unavailable_esm2_row(args, assay_id, split_type, f"frozen_{rep_type}", str(exc), base_row))
                            continue
                    else:
                        meta = {"selected_alpha": "", "representation_type": rep_type}
                    rows.append(
                        base.esm2_metric_row(
                            args,
                            assay_id,
                            split_type,
                            "frozen_representation",
                            readout,
                            records,
                            y,
                            splits,
                            pred,
                            meta,
                            pred_path,
                            cache_key,
                            base_row,
                        )
                    )
    return rows


def run_low_cost_screening(args: argparse.Namespace) -> dict[str, Any]:
    assay_ids = baseline_complete_assay_ids(args)
    rows = compute_esm2_rows(args, assay_ids, SCREENING_REPRESENTATIONS)
    path = Path(args.out_root) / "protein_top20_esm2_screening_metrics.csv"
    base.write_csv(path, rows, base.ESM2_FIELDS, overwrite=True)
    return {"rows": rows, "assays": assay_ids}


def best_row(rows: Sequence[Mapping[str, Any]], key: Any) -> Mapping[str, Any]:
    return max(rows, key=key, default={})


def make_headroom_ranking(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    rows = list(csv.DictReader((out_root / "protein_top20_esm2_screening_metrics.csv").open())) if (out_root / "protein_top20_esm2_screening_metrics.csv").exists() else []
    baseline_rows = list(csv.DictReader((out_root / "protein_top20_baseline_results.csv").open())) if (out_root / "protein_top20_baseline_results.csv").exists() else []
    strength = base.baseline_strength_payload(baseline_rows)
    simple = base.strongest_rows_by_family(baseline_rows, "simple")
    evo = base.strongest_rows_by_family(baseline_rows, "evolutionary")
    overall = base.strongest_baseline_rows(baseline_rows)
    audits = base.read_json(out_root / "protein_top20_split_audit.json").get("assays", {})
    inventory = {row["assay_id"]: row for row in csv.DictReader((out_root / "protein_top20_candidate_inventory.csv").open())}

    ranking_rows = []
    for assay_id in scheduled_top20_assay_ids(args):
        inv = inventory.get(assay_id, {})
        pos_rows = [row for row in rows if row.get("assay_id") == assay_id and row.get("split_type") == "position_heldout" and row.get("status") == "complete"]
        random_rows = [row for row in rows if row.get("assay_id") == assay_id and row.get("split_type") == "random" and row.get("status") == "complete"]
        best_pos = best_row(pos_rows, key=lambda row: (base.numeric(row.get("val_excess"), -999.0), base.numeric(row.get("val_spearman"), -999.0)))
        best_random = best_row(random_rows, key=lambda row: base.numeric(row.get("val_excess"), -999.0))
        val_excess = base.numeric(best_pos.get("val_excess"), -999.0)
        ci_low = base.numeric(best_pos.get("position_bootstrap_ci_low"), float("nan"))
        pos_audit = audits.get(assay_id, {}).get("position_heldout", {})
        evo_status = strength.get(assay_id, {}).get("position_heldout", {}).get("evolutionary_status", "missing")
        rejection_reasons = []
        promoted = True
        if pos_audit.get("status") != "valid":
            promoted = False
            rejection_reasons.append("invalid_position_heldout_split")
        if evo_status != "complete":
            promoted = False
            rejection_reasons.append("evolutionary_baseline_incomplete")
        if not best_pos:
            promoted = False
            rejection_reasons.append("no_complete_esm2_screening_result")
        if val_excess < args.frozen_excess_threshold:
            promoted = False
            rejection_reasons.append(f"validation_excess_below_{args.frozen_excess_threshold:g}")
        if math.isfinite(ci_low) and ci_low <= 0:
            promoted = False
            rejection_reasons.append("validation_bootstrap_ci_not_positive")
        if base.numeric(best_random.get("val_excess"), -999.0) > 0 and val_excess <= 0:
            promoted = False
            rejection_reasons.append("random_split_only_signal")
        ranking_rows.append(
            {
                "source_candidate_rank": inv.get("source_candidate_rank", ""),
                "assay_id": assay_id,
                "records": inv.get("records_loaded", ""),
                "unique_positions": inv.get("unique_positions", ""),
                "strongest_simple_baseline": simple.get((assay_id, "position_heldout"), {}).get("baseline", ""),
                "strongest_simple_val_spearman": simple.get((assay_id, "position_heldout"), {}).get("val_spearman", ""),
                "strongest_evolutionary_baseline": evo.get((assay_id, "position_heldout"), {}).get("baseline", "NOT_AVAILABLE"),
                "strongest_evolutionary_val_spearman": evo.get((assay_id, "position_heldout"), {}).get("val_spearman", ""),
                "strongest_overall_non_PLM_baseline": overall.get((assay_id, "position_heldout"), {}).get("baseline", ""),
                "strongest_overall_val_spearman": overall.get((assay_id, "position_heldout"), {}).get("val_spearman", ""),
                "best_esm2_method": best_pos.get("method", ""),
                "best_esm2_readout": best_pos.get("readout", ""),
                "best_esm2_representation": best_pos.get("representation_type", ""),
                "validation_spearman": best_pos.get("val_spearman", ""),
                "validation_excess": best_pos.get("val_excess", ""),
                "validation_bootstrap_ci_low": best_pos.get("position_bootstrap_ci_low", ""),
                "validation_bootstrap_ci_high": best_pos.get("position_bootstrap_ci_high", ""),
                "random_split_best_validation_excess": best_random.get("val_excess", ""),
                "test_excess_after_freeze": "",
                "advancement_status": "PROMOTED_TO_FULL_CONFIRMATION" if promoted else "REJECTED_AT_SCREENING",
                "rejection_reason": "; ".join(dict.fromkeys(rejection_reasons)),
                "prediction_path": best_pos.get("prediction_path", ""),
            }
        )
    ranking_rows = sorted(
        ranking_rows,
        key=lambda row: (
            row["advancement_status"] != "PROMOTED_TO_FULL_CONFIRMATION",
            -base.numeric(row.get("validation_excess"), -999.0),
            -base.numeric(row.get("validation_spearman"), -999.0),
            base.numeric(row.get("source_candidate_rank"), 999999.0),
        ),
    )
    promoted = [row["assay_id"] for row in ranking_rows if row["advancement_status"] == "PROMOTED_TO_FULL_CONFIRMATION"][: args.max_confirmation_assays]
    for row in ranking_rows:
        if row["advancement_status"] == "PROMOTED_TO_FULL_CONFIRMATION" and row["assay_id"] not in promoted:
            row["advancement_status"] = "REJECTED_AT_SCREENING"
            row["rejection_reason"] = "outside_top_five_validation_headroom_limit"
    payload = {
        "created_at": now_utc(),
        "selection_split": "val",
        "test_used_for_candidate_selection": False,
        "promotion_threshold": args.frozen_excess_threshold,
        "max_promoted_to_confirmation": args.max_confirmation_assays,
        "promoted_to_confirmation": promoted,
        "ranking_rows": ranking_rows,
    }
    base.write_csv(out_root / "protein_top20_candidate_headroom_ranking.csv", ranking_rows, HEADROOM_FIELDS, overwrite=True)
    return payload


def run_frozen_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    ranking_path = out_root / "protein_top20_candidate_headroom_ranking.csv"
    promoted = [
        row["assay_id"]
        for row in csv.DictReader(ranking_path.open())
        if row.get("advancement_status") == "PROMOTED_TO_FULL_CONFIRMATION"
    ][: args.max_confirmation_assays] if ranking_path.exists() else []
    if not promoted:
        base.write_csv(out_root / "protein_top20_frozen_confirmation.csv", [], base.ESM2_FIELDS, overwrite=True)
        return {"rows": [], "promoted_to_lora": []}
    rows = compute_esm2_rows(args, promoted, [*SCREENING_REPRESENTATIONS, *CONFIRMATION_EXTRA_REPRESENTATIONS])
    base.write_csv(out_root / "protein_top20_frozen_confirmation.csv", rows, base.ESM2_FIELDS, overwrite=True)
    return {"rows": rows, "promoted_to_lora": select_lora_candidates(args, rows)}


def select_lora_candidates(args: argparse.Namespace, confirmation_rows: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    out_root = Path(args.out_root)
    rows = list(confirmation_rows or [])
    if not rows and (out_root / "protein_top20_frozen_confirmation.csv").exists():
        rows = list(csv.DictReader((out_root / "protein_top20_frozen_confirmation.csv").open()))
    screening_rank = {row["assay_id"]: row for row in csv.DictReader((out_root / "protein_top20_candidate_headroom_ranking.csv").open())} if (out_root / "protein_top20_candidate_headroom_ranking.csv").exists() else {}
    strength = baseline_strength(args)
    strongest = base.baseline_lookup(out_root)
    audits = base.read_json(out_root / "protein_top20_split_audit.json").get("assays", {})
    decisions = []
    for assay_id in [row["assay_id"] for row in screening_rank.values() if row.get("advancement_status") == "PROMOTED_TO_FULL_CONFIRMATION"]:
        pos_rows = [row for row in rows if row.get("assay_id") == assay_id and row.get("split_type") == "position_heldout" and row.get("status") == "complete"]
        random_rows = [row for row in rows if row.get("assay_id") == assay_id and row.get("split_type") == "random" and row.get("status") == "complete"]
        best_pos = best_row(pos_rows, key=lambda row: (base.numeric(row.get("val_excess"), -999.0), base.numeric(row.get("val_spearman"), -999.0)))
        best_random = best_row(random_rows, key=lambda row: base.numeric(row.get("val_excess"), -999.0))
        val_excess = base.numeric(best_pos.get("val_excess"), float("nan"))
        test_excess = base.numeric(best_pos.get("test_excess"), float("nan"))
        val_ci_low = base.numeric(best_pos.get("position_bootstrap_ci_low"), float("nan"))
        test_ci_low, test_ci_high = (None, None)
        strong = strongest.get((assay_id, "position_heldout"))
        if best_pos.get("prediction_path") and strong and strong.get("prediction_path"):
            test_ci_low, test_ci_high = base.bootstrap_delta_by_position(
                best_pos["prediction_path"],
                strong["prediction_path"],
                "test",
                args.n_bootstrap,
                args.bootstrap_seed,
            )
        ci_low = base.numeric(test_ci_low, val_ci_low)
        reasons = []
        pass_gate = True
        if audits.get(assay_id, {}).get("position_heldout", {}).get("status") != "valid":
            pass_gate = False
            reasons.append("invalid_position_heldout_split")
        if strength.get(assay_id, {}).get("position_heldout", {}).get("evolutionary_status") != "complete":
            pass_gate = False
            reasons.append("evolutionary_baseline_incomplete")
        if not best_pos:
            pass_gate = False
            reasons.append("no_complete_frozen_confirmation_result")
        if not (math.isfinite(val_excess) and math.isfinite(test_excess) and val_excess > 0 and test_excess > 0):
            pass_gate = False
            reasons.append("validation_and_test_excess_not_same_positive_direction")
        if not (math.isfinite(test_excess) and test_excess >= args.frozen_excess_threshold):
            pass_gate = False
            reasons.append(f"test_excess_below_{args.frozen_excess_threshold:g}")
        if math.isfinite(ci_low) and ci_low <= 0:
            pass_gate = False
            reasons.append("bootstrap_ci_not_positive")
        if base.numeric(best_random.get("val_excess"), -999.0) > 0 and val_excess <= 0:
            pass_gate = False
            reasons.append("random_split_only_signal")
        decisions.append(
            {
                "assay_id": assay_id,
                "advance_to_lora": pass_gate,
                "selected_method": best_pos.get("method", ""),
                "selected_readout": best_pos.get("readout", ""),
                "selected_representation": best_pos.get("representation_type", ""),
                "val_excess": best_pos.get("val_excess", ""),
                "test_excess": best_pos.get("test_excess", ""),
                "validation_ci_low": best_pos.get("position_bootstrap_ci_low", ""),
                "validation_ci_high": best_pos.get("position_bootstrap_ci_high", ""),
                "test_ci_low": test_ci_low if test_ci_low is not None else "",
                "test_ci_high": test_ci_high if test_ci_high is not None else "",
                "ci_low": test_ci_low if test_ci_low is not None else best_pos.get("position_bootstrap_ci_low", ""),
                "ci_high": test_ci_high if test_ci_high is not None else best_pos.get("position_bootstrap_ci_high", ""),
                "reason": "; ".join(dict.fromkeys(reasons)),
                "prediction_path": best_pos.get("prediction_path", ""),
            }
        )
    decisions = sorted(
        decisions,
        key=lambda row: (
            not row.get("advance_to_lora"),
            -base.numeric(row.get("val_excess"), -999.0),
            -base.numeric(row.get("test_excess"), -999.0),
            row["assay_id"],
        ),
    )
    promoted = [row["assay_id"] for row in decisions if row.get("advance_to_lora")][: args.max_lora_assays]
    gate_payload = {
        "created_at": now_utc(),
        "selection_split": "val",
        "test_used_only_after_method_freeze": True,
        "max_lora_assays": args.max_lora_assays,
        "lora_candidate_order": promoted,
        "decisions": decisions,
    }
    base.write_json(out_root / "protein_top20_qualification_gate.json", gate_payload, overwrite=True)
    base.write_json(out_root / "protein_48h_advancement_gate.json", {"created_at": now_utc(), "lora_candidate_order": promoted, "decisions": []}, overwrite=True)

    ranking_path = out_root / "protein_top20_candidate_headroom_ranking.csv"
    if ranking_path.exists():
        ranking_rows = list(csv.DictReader(ranking_path.open()))
        by_decision = {row["assay_id"]: row for row in decisions}
        for row in ranking_rows:
            decision = by_decision.get(row["assay_id"])
            if not decision:
                continue
            row["test_excess_after_freeze"] = decision.get("test_excess", "")
            row["advancement_status"] = "PROMOTED_TO_LORA" if row["assay_id"] in promoted else "REJECTED_AT_CONFIRMATION"
            row["rejection_reason"] = "" if row["assay_id"] in promoted else decision.get("reason", "")
        base.write_csv(ranking_path, ranking_rows, HEADROOM_FIELDS, overwrite=True)
    return promoted


def run_top20_lora(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    gate = base.read_json(out_root / "protein_top20_qualification_gate.json")
    candidates = gate.get("lora_candidate_order", [])[: args.max_lora_assays]
    rows: list[dict[str, Any]] = []
    qualifications = []
    attempted = []
    if not candidates:
        base.write_csv(out_root / "protein_top20_lora_metrics.csv", rows, base.LORA_FIELDS, overwrite=True)
        base.write_csv(out_root / "protein_48h_lora_metrics.csv", rows, base.LORA_FIELDS, overwrite=True)
        return {"rows": rows, "attempted_assays": [], "qualifications": []}
    for assay_id in candidates:
        attempted.append(assay_id)
        if args.mock_esm2 or args.mock_lora:
            assay_rows = base.mock_lora_rows(args, assay_id)
        else:
            assay_rows = []
            calibration = []
            for rank in (8, 16):
                for lr in (1e-5, 5e-5):
                    row = base.train_real_lora_run(args, assay_id, rank, lr, args.calibration_seed, "calibration")
                    assay_rows.append(row)
                    calibration.append(row)
            best = max(calibration, key=lambda row: base.numeric(row.get("val_spearman"), -999.0))
            for seed in [int(part) for part in str(args.formal_seeds).split(",") if part.strip()]:
                assay_rows.append(base.train_real_lora_run(args, assay_id, int(best["rank"]), float(best["learning_rate"]), seed, "formal"))
        rows.extend(assay_rows)
        gate_decision = {"labels": ["FROZEN_PLM_SIGNAL"]}
        qualifications.append(base.lora_complete_qualification(args, assay_id, assay_rows, gate_decision))
    base.write_csv(out_root / "protein_top20_lora_metrics.csv", rows, base.LORA_FIELDS, overwrite=True)
    base.write_csv(out_root / "protein_48h_lora_metrics.csv", rows, base.LORA_FIELDS, overwrite=True)
    return {"rows": rows, "attempted_assays": attempted, "qualifications": qualifications}


def write_qualification_evidence(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    lora_rows = list(csv.DictReader((out_root / "protein_top20_lora_metrics.csv").open())) if (out_root / "protein_top20_lora_metrics.csv").exists() else []
    gate = base.read_json(out_root / "protein_top20_qualification_gate.json")
    qualifications = []
    for assay_id in gate.get("lora_candidate_order", []):
        qualifications.append(base.lora_complete_qualification(args, assay_id, lora_rows, {"labels": ["FROZEN_PLM_SIGNAL"]}))
    qualified = [row["assay_id"] for row in qualifications if row.get("qualified") and "PRELIMINARILY_QUALIFIED" in row.get("labels", [])]
    rows = []
    for decision in gate.get("decisions", []):
        evidence = next((row for row in qualifications if row["assay_id"] == decision["assay_id"]), {})
        rows.append(
            {
                "assay_id": decision["assay_id"],
                "stage": "lora" if decision["assay_id"] in gate.get("lora_candidate_order", []) else "frozen_confirmation",
                "status": "PRELIMINARILY_QUALIFIED" if decision["assay_id"] in qualified else "REJECTED",
                "selected_method": decision.get("selected_method", ""),
                "selected_readout": decision.get("selected_readout", ""),
                "selected_representation": decision.get("selected_representation", ""),
                "validation_excess": decision.get("val_excess", ""),
                "test_excess": decision.get("test_excess", ""),
                "bootstrap_ci_low": decision.get("test_ci_low", decision.get("ci_low", "")),
                "bootstrap_ci_high": decision.get("test_ci_high", decision.get("ci_high", "")),
                "lora_seed_count": evidence.get("formal_seed_count", ""),
                "lora_mean_excess": evidence.get("mean_excess", ""),
                "lora_std_excess": evidence.get("std_excess", ""),
                "lora_worst_seed_excess": evidence.get("worst_seed_excess", ""),
                "labels": json.dumps(evidence.get("labels", [])),
                "reason": evidence.get("reasons", decision.get("reason", "")),
            }
        )
    payload = {
        "created_at": now_utc(),
        "preliminarily_qualified": qualified,
        "lora_qualifications": qualifications,
        "gate": gate,
        "evidence_rows": rows,
    }
    base.write_json(out_root / "protein_top20_qualification_evidence.json", payload, overwrite=True)
    base.write_csv(out_root / "protein_top20_qualification_evidence.csv", rows, QUALIFICATION_FIELDS, overwrite=True)
    return payload


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def previous_prediction_window_hours(source_root: Path) -> float | None:
    pred_root = source_root / "protein_48h_esm2_pilot_predictions"
    paths = sorted(pred_root.rglob("*.csv")) if pred_root.exists() else []
    if len(paths) < 2:
        return None
    mtimes = [path.stat().st_mtime for path in paths]
    span = (max(mtimes) - min(mtimes)) / 3600.0
    return span if span > 0 else None


def token_weight_from_inventory(rows: Sequence[Mapping[str, Any]], assay_ids: set[str]) -> float:
    total = 0.0
    for row in rows:
        if row.get("assay_id") in assay_ids:
            total += base.numeric(row.get("valid_single_substitution_count"), 0.0) * base.numeric(row.get("sequence_length"), 0.0)
    return total


def estimate_runtime(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    source_root = Path(args.source_root)
    top20_rows = load_frozen_top20_rows(args) if candidate_ranking_path(args).exists() else []
    scheduled = set(scheduled_top20_assay_ids(args)) if top20_rows else set()
    previous = EXCLUDED_COMPLETED_ASSAYS
    prev_weight = token_weight_from_inventory(top20_rows, previous) or 1.0
    new_weight = token_weight_from_inventory(top20_rows, scheduled)
    previous_hours = previous_prediction_window_hours(source_root) or 0.824
    screening_factor = 5.0 / 9.0
    screening_hours = previous_hours * (new_weight / prev_weight) * screening_factor

    headroom = out_root / "protein_top20_candidate_headroom_ranking.csv"
    confirmation_promoted = 0
    confirmation_weight = 0.0
    if headroom.exists():
        promoted = [
            row["assay_id"]
            for row in csv.DictReader(headroom.open())
            if row.get("advancement_status") in {"PROMOTED_TO_FULL_CONFIRMATION", "PROMOTED_TO_LORA"}
        ][: args.max_confirmation_assays]
        confirmation_promoted = len(promoted)
        confirmation_weight = token_weight_from_inventory(top20_rows, set(promoted))
    else:
        confirmation_promoted = args.max_confirmation_assays
        largest = sorted(
            [row for row in top20_rows if row.get("assay_id") in scheduled],
            key=lambda row: base.numeric(row.get("valid_single_substitution_count"), 0.0) * base.numeric(row.get("sequence_length"), 0.0),
            reverse=True,
        )[: args.max_confirmation_assays]
        confirmation_weight = token_weight_from_inventory(largest, {row["assay_id"] for row in largest})
    confirmation_hours = previous_hours * (confirmation_weight / prev_weight) * (4.0 / 9.0)

    lora_gate = out_root / "protein_top20_qualification_gate.json"
    if lora_gate.exists():
        lora_assays = len(base.read_json(lora_gate).get("lora_candidate_order", []))
    else:
        lora_assays = args.max_lora_assays
    one_lora_assay_hours = 0.52
    baseline_hours = 1.5 if not args.mock_esm2 else 0.05
    completed_screening_rows = 0
    if (out_root / "protein_top20_esm2_screening_metrics.csv").exists():
        completed_screening_rows = sum(1 for row in csv.DictReader((out_root / "protein_top20_esm2_screening_metrics.csv").open()) if row.get("status") == "complete")
    total_screening_rows = max(1, len(scheduled) * len(SPLIT_TYPES) * (1 + len(SCREENING_REPRESENTATIONS) * len(base.ESM2_READOUTS)))
    screening_remaining_fraction = max(0.0, 1.0 - completed_screening_rows / total_screening_rows)

    best_case = baseline_hours + screening_hours * screening_remaining_fraction
    expected = best_case + min(confirmation_hours, previous_hours * (new_weight / prev_weight) * 0.25) + one_lora_assay_hours * min(max(lora_assays, 1), 1)
    worst = best_case + confirmation_hours + one_lora_assay_hours * max(lora_assays, args.max_lora_assays)
    return {
        "created_at": now_utc(),
        "basis": {
            "previous_prediction_window_hours": previous_hours,
            "previous_completed_assays": sorted(previous),
            "previous_token_weight": prev_weight,
            "new_scheduled_token_weight": new_weight,
            "screening_methods_per_split": 5,
            "full_methods_per_split_reference": 9,
            "completed_screening_rows": completed_screening_rows,
            "total_screening_rows": total_screening_rows,
            "confirmation_promoted_assays_assumed_or_observed": confirmation_promoted,
            "lora_assays_assumed_or_observed": lora_assays,
        },
        "best_case_hours": round(best_case, 2),
        "expected_hours": round(expected, 2),
        "worst_case_hours": round(worst, 2),
        "scenario_notes": {
            "best_case": "no assay advances beyond low-cost ESM2 screening",
            "expected": "one assay reaches LoRA after limited frozen confirmation",
            "worst_case": "five assays reach full frozen confirmation and two assays reach LoRA",
        },
    }


def write_progress_report(args: argparse.Namespace, registry: dict[str, Any], stage: str) -> dict[str, Any]:
    out_root = Path(args.out_root)
    eta = estimate_runtime(args)
    inventory = list(csv.DictReader((out_root / "protein_top20_candidate_inventory.csv").open())) if (out_root / "protein_top20_candidate_inventory.csv").exists() else []
    payload = {
        "created_at": now_utc(),
        "status": stage,
        "current_stage": stage,
        "out_root": str(out_root),
        "source_root": str(Path(args.source_root)),
        "scheduled_assay_count": sum(1 for row in inventory if str(row.get("scheduled_for_screening", "")).lower() == "true"),
        "excluded_completed_assays": sorted(EXCLUDED_COMPLETED_ASSAYS),
        "eta": eta,
        "registry_path": str(top20_registry_path(args)),
        "tasks": registry.get("tasks", []),
    }
    base.write_json(out_root / "protein_top20_progress_report.json", payload, overwrite=True)
    return payload


def final_report(args: argparse.Namespace, registry: dict[str, Any]) -> dict[str, Any]:
    out_root = Path(args.out_root)
    inventory = list(csv.DictReader((out_root / "protein_top20_candidate_inventory.csv").open())) if (out_root / "protein_top20_candidate_inventory.csv").exists() else []
    headroom = list(csv.DictReader((out_root / "protein_top20_candidate_headroom_ranking.csv").open())) if (out_root / "protein_top20_candidate_headroom_ranking.csv").exists() else []
    qualification = base.read_json(out_root / "protein_top20_qualification_evidence.json")
    baseline_strengths = baseline_strength(args)
    scheduled = [row["assay_id"] for row in inventory if str(row.get("scheduled_for_screening", "")).lower() == "true"]
    baseline_incomplete = [
        assay_id
        for assay_id in scheduled
        if baseline_strengths.get(assay_id, {}).get("position_heldout", {}).get("evolutionary_status") != "complete"
    ]
    promoted_confirmation = [row["assay_id"] for row in headroom if row.get("advancement_status") in {"PROMOTED_TO_FULL_CONFIRMATION", "PROMOTED_TO_LORA"}]
    promoted_lora = base.read_json(out_root / "protein_top20_qualification_gate.json").get("lora_candidate_order", [])
    rejection_reasons = {
        row["assay_id"]: row.get("rejection_reason", "")
        for row in headroom
        if row.get("advancement_status", "").startswith("REJECTED")
    }
    qualified = qualification.get("preliminarily_qualified", [])
    payload = {
        "created_at": now_utc(),
        "workflow_status": "complete",
        "formal": bool(args.formal),
        "out_root": str(out_root),
        "source_root": str(Path(args.source_root)),
        "number_candidates_evaluated": len(scheduled),
        "number_rejected_incomplete_baselines_or_invalid_splits": len(baseline_incomplete),
        "number_promoted_to_full_frozen_confirmation": len(promoted_confirmation),
        "number_promoted_to_lora": len(promoted_lora),
        "qualified_assay_ids": qualified,
        "rejection_reasons": rejection_reasons,
        "reused_artifacts": {
            "previous_completed_assays": sorted(EXCLUDED_COMPLETED_ASSAYS),
            "previous_output_root": str(Path(args.source_root)),
            "full_assay_files_copied_from_previous_root": False,
        },
        "newly_computed_artifacts": {
            "candidate_inventory": str(out_root / "protein_top20_candidate_inventory.csv"),
            "baseline_results": str(out_root / "protein_top20_baseline_results.csv"),
            "esm2_screening_metrics": str(out_root / "protein_top20_esm2_screening_metrics.csv"),
            "frozen_confirmation": str(out_root / "protein_top20_frozen_confirmation.csv"),
            "lora_metrics": str(out_root / "protein_top20_lora_metrics.csv"),
        },
        "eta": estimate_runtime(args),
        "final_scientific_conclusion": (
            "At least one credible ProteinGym-ESM2-150M benchmark was identified."
            if qualified
            else "No screened assay showed a stable ESM2-specific advantage after strong evolutionary-baseline, position-held-out, bootstrap, and multi-seed controls."
        ),
        "registry_path": str(top20_registry_path(args)),
    }
    base.write_json(out_root / "protein_top20_final_report.json", payload, overwrite=True)
    write_final_report_md(out_root / "protein_top20_final_report.md", payload)
    write_progress_report(args, registry, "complete")
    return payload


def write_final_report_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# ProteinGym ESM2-150M Top-20 Expansion Report",
        "",
        f"- Workflow status: `{payload.get('workflow_status')}`.",
        f"- Candidates evaluated: `{payload.get('number_candidates_evaluated')}`.",
        f"- Incomplete baselines or invalid splits: `{payload.get('number_rejected_incomplete_baselines_or_invalid_splits')}`.",
        f"- Promoted to full frozen confirmation: `{payload.get('number_promoted_to_full_frozen_confirmation')}`.",
        f"- Promoted to LoRA: `{payload.get('number_promoted_to_lora')}`.",
        f"- Qualified assays: `{', '.join(payload.get('qualified_assay_ids', [])) if payload.get('qualified_assay_ids') else 'none'}`.",
        "",
        "## Conclusion",
        "",
        str(payload.get("final_scientific_conclusion", "")),
        "",
        "## Rejection Reasons",
        "",
    ]
    for assay_id, reason in sorted(payload.get("rejection_reasons", {}).items()):
        lines.append(f"- `{assay_id}`: {reason or 'not promoted by validation-only gate'}")
    path.write_text("\n".join(lines) + "\n")


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return base.shutil.disk_usage(path).free / (1024**3)


def require_storage(args: argparse.Namespace) -> None:
    free = free_disk_gb(Path(args.out_root))
    if free < args.stop_on_low_disk_gb:
        raise RuntimeError(f"low disk before top20 workflow: free={free:.2f}G threshold={args.stop_on_low_disk_gb:.2f}G")


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    require_storage(args)
    registry = load_top20_registry(args)
    write_progress_report(args, registry, "starting")

    run_top20_stage(
        args,
        registry,
        "stage0_protocol",
        "frozen_protocol",
        Path(args.out_root) / "protein_top20_frozen_protocol.json",
        lambda: write_top20_protocol(args),
        validator=base.json_object,
        skippable=False,
    )
    run_top20_stage(
        args,
        registry,
        "stage0_data",
        "proteingym_data_validation",
        Path(args.out_root) / "protein_48h_proteingym_download_report.json",
        lambda: base.ensure_proteingym_data(args),
        validator=base.json_object,
    )
    inventory = run_top20_stage(
        args,
        registry,
        "stage1_inventory",
        "top20_inventory",
        Path(args.out_root) / "protein_top20_candidate_inventory.csv",
        lambda: top20_inventory(args),
        validator=base.csv_nonempty_or_header,
    )
    if inventory is None:
        inventory = {
            "scheduled_assays": [
                row["assay_id"]
                for row in csv.DictReader((Path(args.out_root) / "protein_top20_candidate_inventory.csv").open())
                if str(row.get("scheduled_for_screening", "")).lower() == "true"
            ]
        }
    write_progress_report(args, registry, "top20_inventory")
    run_top20_stage(
        args,
        registry,
        "stage2_splits",
        "split_creation",
        Path(args.out_root) / "protein_top20_split_audit.json",
        lambda: create_top20_splits(args),
        validator=base.json_object,
    )
    run_top20_stage(
        args,
        registry,
        "stage2_evolutionary_resources",
        "evolutionary_baseline_resources",
        Path(args.out_root) / "protein_top20_evolutionary_baseline_report.json",
        lambda: run_top20_evolutionary_resources(args),
        validator=base.json_object,
        skippable=False,
    )
    run_top20_stage(
        args,
        registry,
        "stage2_baselines",
        "baseline_evaluation",
        Path(args.out_root) / "protein_top20_baseline_results.csv",
        lambda: run_top20_baselines(args),
        validator=base.csv_nonempty_or_header,
        skippable=False,
    )
    run_top20_stage(
        args,
        registry,
        "stage2_resume_validation",
        "resume_validation",
        Path(args.out_root) / "protein_top20_resume_validation.json",
        lambda: validate_top20_resume_state(args, registry),
        validator=base.json_object,
        skippable=False,
    )
    write_progress_report(args, registry, "baseline_complete")
    screening = run_top20_stage(
        args,
        registry,
        "stage3_esm2_screening",
        "esm2_low_cost_screening",
        Path(args.out_root) / "protein_top20_esm2_screening_metrics.csv",
        lambda: run_low_cost_screening(args),
        validator=base.csv_nonempty_or_header,
        skippable=False,
    )
    write_progress_report(args, registry, "esm2_low_cost_screening")
    headroom = run_top20_stage(
        args,
        registry,
        "stage3_headroom",
        "headroom_ranking",
        Path(args.out_root) / "protein_top20_candidate_headroom_ranking.csv",
        lambda: make_headroom_ranking(args),
        validator=base.csv_nonempty_or_header,
        skippable=False,
    )
    promoted = (headroom or {}).get("promoted_to_confirmation", [])
    if not promoted:
        base.write_csv(Path(args.out_root) / "protein_top20_frozen_confirmation.csv", [], base.ESM2_FIELDS, overwrite=True)
        base.write_json(Path(args.out_root) / "protein_top20_qualification_gate.json", {"created_at": now_utc(), "lora_candidate_order": [], "decisions": [], "reason": "no validation-positive baseline-complete assay promoted"}, overwrite=True)
        base.write_csv(Path(args.out_root) / "protein_top20_lora_metrics.csv", [], base.LORA_FIELDS, overwrite=True)
        base.write_csv(Path(args.out_root) / "protein_48h_lora_metrics.csv", [], base.LORA_FIELDS, overwrite=True)
        update_top20_task(args, registry, "stage4_frozen_confirmation_skipped", stage="full_frozen_confirmation", status="skipped", reason="no assay passed validation-only low-cost screening gate")
        update_top20_task(args, registry, "stage5_lora_skipped", stage="fresh_lora_qualification", status="skipped", reason="no assay passed validation-only low-cost screening gate")
    else:
        run_top20_stage(
            args,
            registry,
            "stage4_frozen_confirmation",
            "full_frozen_confirmation",
            Path(args.out_root) / "protein_top20_frozen_confirmation.csv",
            lambda: run_frozen_confirmation(args),
            validator=base.csv_nonempty_or_header,
            skippable=False,
        )
        select_lora_candidates(args)
        if base.read_json(Path(args.out_root) / "protein_top20_qualification_gate.json").get("lora_candidate_order"):
            run_top20_stage(
                args,
                registry,
                "stage5_lora",
                "fresh_lora_qualification",
                Path(args.out_root) / "protein_top20_lora_metrics.csv",
                lambda: run_top20_lora(args),
                validator=base.csv_nonempty_or_header,
                skippable=False,
            )
        else:
            update_top20_task(args, registry, "stage5_lora_skipped", stage="fresh_lora_qualification", status="skipped", reason="no assay passed frozen confirmation gate")
            base.write_csv(Path(args.out_root) / "protein_top20_lora_metrics.csv", [], base.LORA_FIELDS, overwrite=True)
            base.write_csv(Path(args.out_root) / "protein_48h_lora_metrics.csv", [], base.LORA_FIELDS, overwrite=True)
    run_top20_stage(
        args,
        registry,
        "stage6_qualification_evidence",
        "qualification_evidence",
        Path(args.out_root) / "protein_top20_qualification_evidence.json",
        lambda: write_qualification_evidence(args),
        validator=base.json_object,
        skippable=False,
    )
    summary = final_report(args, registry)
    update_top20_task(args, registry, "stage_final_report", stage="final_aggregation", status="complete", output_path=str(Path(args.out_root) / "protein_top20_final_report.json"))
    write_progress_report(args, registry, "complete")
    return summary


def make_top20_smoke_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    dms_dir = root / "fixture/DMS_substitutions"
    metadata_path = root / "fixture/DMS_substitutions.csv"
    pred_dir = root / "fixture/public_predictions/ProteinGym_zero_shot_substitution_scores"
    msa_dir = root / "fixture/MSA_files"
    source_root = root / "fixture/source_48h"
    dms_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260805)
    metadata_rows = []
    ranking_rows = []
    completed = ["CCDB_ECOLI_Adkar_2012", "GAL4_YEAST_Kitzman_2015", "MET_HUMAN_Estevam_2023"]
    for rank, assay_id in enumerate(completed, start=1):
        ranking_rows.append({"candidate_rank": rank, "assay_id": assay_id, "valid_single_substitution_count": 100, "sequence_length": 50, "data_quality_score": 20 - rank, "exclusion_reason": ""})
    for offset, assay_id in enumerate(["TOP20_SIGNAL_A", "TOP20_SIGNAL_B"], start=4):
        wt = "".join(rng.choice(base.AA) for _ in range(90 + offset))
        rows = []
        for pos in range(3, 78):
            wt_res = wt[pos - 1]
            for jump in (3, 7):
                mut = base.AA[(base.AA.index(wt_res) + jump) % len(base.AA)]
                chars = list(wt)
                chars[pos - 1] = mut
                signal = 0.07 * sum((base.AA.index(ch) + 1) * (idx + 1) for idx, ch in enumerate(wt[max(0, pos - 4) : pos - 1] + wt[pos : pos + 3]))
                score = signal + 0.5 * (base.AA.index(mut) - base.AA.index(wt_res))
                mutation = f"{wt_res}{pos}{mut}"
                rows.append({"mutant": mutation, "mutated_sequence": "".join(chars), "DMS_score": score})
        base.write_csv(dms_dir / f"{assay_id}.csv", rows, ["mutant", "mutated_sequence", "DMS_score"], overwrite=True)
        metadata_rows.append({"DMS_id": assay_id, "DMS_filename": f"{assay_id}.csv", "target_name": assay_id, "selection_type": "activity", "target_seq": wt})
        with (pred_dir / f"{assay_id}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mutant", "EVmutation"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"mutant": row["mutant"], "EVmutation": float(row["DMS_score"]) * (0.65 if assay_id.endswith("_A") else 1.0)})
        ranking_rows.append({"candidate_rank": offset, "assay_id": assay_id, "valid_single_substitution_count": len(rows), "sequence_length": len(wt), "data_quality_score": 20 - offset, "exclusion_reason": ""})
    base.write_csv(metadata_path, metadata_rows, ["DMS_id", "DMS_filename", "target_name", "selection_type", "target_seq"], overwrite=True)
    base.write_csv(
        source_root / "protein_48h_candidate_ranking.csv",
        ranking_rows,
        ["candidate_rank", "assay_id", "valid_single_substitution_count", "sequence_length", "data_quality_score", "exclusion_reason"],
        overwrite=True,
    )
    return dms_dir, metadata_path, pred_dir.parent, msa_dir, source_root


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    dms_dir, metadata, pred_dir, msa_dir, source_root = make_top20_smoke_fixture(Path(args.out_root))
    args.dms_dir = str(dms_dir)
    args.metadata_csv = str(metadata)
    args.public_predictions_dir = str(pred_dir)
    args.msa_dir = str(msa_dir)
    args.source_root = str(source_root)
    args.mock_esm2 = True
    args.mock_lora = True
    args.formal = False
    args.skip_proteingym_download = True
    args.min_valid_samples = min(args.min_valid_samples, 20)
    args.n_bootstrap = min(args.n_bootstrap, 100)
    args.max_retries = 0
    return run_workflow(args)


def launch(args: argparse.Namespace) -> int:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        "-u",
        "phase2/proteingym_esm2_top20_expansion.py",
        "run",
        "--out-root",
        str(args.out_root),
        "--source-root",
        str(args.source_root),
        "--dms-dir",
        str(args.dms_dir),
        "--metadata-csv",
        str(args.metadata_csv),
        "--public-predictions-dir",
        str(args.public_predictions_dir),
        "--msa-dir",
        str(args.msa_dir),
        "--esm2-model",
        str(args.esm2_model),
        "--device",
        str(args.device),
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--proteingym-source-api",
        str(args.proteingym_source_api),
        "--resume",
        "--formal",
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if not args.auto_download_proteingym:
        cmd.append("--no-auto-download-proteingym")
    if args.skip_proteingym_download:
        cmd.append("--skip-proteingym-download")
    with Path(args.log_file).open("a") as log:
        log.write(f"[{now_utc()}] launcher cmd={' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    payload = {
        "created_at": now_utc(),
        "pid": proc.pid,
        "cmd": cmd,
        "log_file": str(args.log_file),
        "out_root": str(args.out_root),
        "scheduled_assays": scheduled_top20_assay_ids(args),
        "eta": estimate_runtime(args),
    }
    base.write_json(Path(args.out_root) / "protein_top20_launcher_status.json", payload, overwrite=True)
    print(proc.pid)
    return 0


def add_args(parser: argparse.ArgumentParser) -> None:
    base.add_common_args(parser)
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_file=str(DEFAULT_LOG), max_static_candidates=20, max_pilot_assays=20)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--max-confirmation-assays", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    add_args(run)
    smoke = sub.add_parser("smoke")
    add_args(smoke)
    smoke.set_defaults(out_root=str(DEFAULT_SMOKE_ROOT), log_file=str(PROJECT_ROOT / "logs/protein_top20_esm2_expanded_qualification_smoke.log"), mock_esm2=True, mock_lora=True)
    launch_parser = sub.add_parser("launch")
    add_args(launch_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "launch":
        raise SystemExit(launch(args))
    else:
        run_workflow(args)


if __name__ == "__main__":
    main()
