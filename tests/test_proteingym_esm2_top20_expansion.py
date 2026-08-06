from __future__ import annotations

import csv
import json
from pathlib import Path

import phase2.proteingym_esm2_top20_expansion as top20


DELIVERABLES = [
    "protein_top20_candidate_inventory.csv",
    "protein_top20_split_audit.json",
    "protein_top20_baseline_results.csv",
    "protein_top20_esm2_screening_metrics.csv",
    "protein_top20_candidate_headroom_ranking.csv",
    "protein_top20_frozen_confirmation.csv",
    "protein_top20_lora_metrics.csv",
    "protein_top20_qualification_evidence.json",
    "protein_top20_registry.json",
    "protein_top20_progress_report.json",
    "protein_top20_final_report.json",
    "protein_top20_final_report.md",
]


def parse_args(parts: list[str]):
    return top20.build_parser().parse_args(parts)


def smoke_args(tmp_path: Path):
    return parse_args(
        [
            "smoke",
            "--out-root",
            str(tmp_path / "top20_smoke"),
            "--log-file",
            str(tmp_path / "top20.log"),
            "--device",
            "cpu",
            "--stop-on-low-disk-gb",
            "1",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "10",
            "--max-retries",
            "0",
        ]
    )


def test_top20_smoke_excludes_completed_assays_and_writes_deliverables(tmp_path: Path) -> None:
    args = smoke_args(tmp_path)
    summary = top20.run_smoke(args)
    out_root = Path(args.out_root)

    for name in DELIVERABLES:
        assert (out_root / name).exists(), name

    inventory = list(csv.DictReader((out_root / "protein_top20_candidate_inventory.csv").open()))
    scheduled = {row["assay_id"] for row in inventory if row["scheduled_for_screening"] == "true"}
    excluded = {row["assay_id"] for row in inventory if row["excluded_previous_completed"] == "true"}
    assert scheduled == {"TOP20_SIGNAL_A", "TOP20_SIGNAL_B"}
    assert excluded == top20.EXCLUDED_COMPLETED_ASSAYS
    assert summary["number_candidates_evaluated"] == 2

    registry = json.loads((out_root / "protein_top20_registry.json").read_text())
    assert registry["workflow"]["max_concurrent_gpu_jobs"] == 1
    assert all(task["status"] in {"complete", "skipped"} for task in registry["tasks"])


def test_top20_resume_validation_detects_partial_and_reruns_only_invalid_prediction(tmp_path: Path) -> None:
    args = smoke_args(tmp_path)
    top20.run_smoke(args)
    out_root = Path(args.out_root)
    pred_path = out_root / "protein_top20_esm2_predictions/TOP20_SIGNAL_A/random/zero_shot.csv"
    valid_path = out_root / "protein_top20_esm2_predictions/TOP20_SIGNAL_B/random/zero_shot.csv"
    valid_mtime = valid_path.stat().st_mtime_ns
    pred_path.write_text("partial\n")

    registry = top20.load_top20_registry(args)
    validation = top20.validate_top20_resume_state(args, registry)
    assert validation["invalid_esm2_prediction_outputs"] == 1

    top20.run_low_cost_screening(args)

    assert list(csv.DictReader(pred_path.open()))
    assert (out_root / "resume_invalid_artifacts").exists()
    assert valid_path.stat().st_mtime_ns == valid_mtime


def test_top20_lora_gate_requires_test_excess_and_bootstrap_support(tmp_path: Path) -> None:
    args = smoke_args(tmp_path)
    top20.run_smoke(args)
    out_root = Path(args.out_root)
    screening_rows = list(csv.DictReader((out_root / "protein_top20_esm2_screening_metrics.csv").open()))

    headroom_rows = list(csv.DictReader((out_root / "protein_top20_candidate_headroom_ranking.csv").open()))
    for row in headroom_rows:
        row["advancement_status"] = "PROMOTED_TO_FULL_CONFIRMATION"
        row["rejection_reason"] = ""
    top20.base.write_csv(out_root / "protein_top20_candidate_headroom_ranking.csv", headroom_rows, top20.HEADROOM_FIELDS, overwrite=True)

    edited = []
    for row in screening_rows:
        row = dict(row)
        if row["assay_id"] == "TOP20_SIGNAL_A" and row["split_type"] == "position_heldout":
            row["val_excess"] = "0.08"
            row["test_excess"] = "-0.01"
            row["position_bootstrap_ci_low"] = "0.02"
        if row["assay_id"] == "TOP20_SIGNAL_B" and row["split_type"] == "position_heldout":
            row["val_excess"] = "0.08"
            row["test_excess"] = "0.08"
            row["position_bootstrap_ci_low"] = "0.02"
        edited.append(row)

    promoted = top20.select_lora_candidates(args, edited)

    gate = json.loads((out_root / "protein_top20_qualification_gate.json").read_text())
    reasons = {row["assay_id"]: row["reason"] for row in gate["decisions"]}
    assert "TOP20_SIGNAL_A" not in promoted
    assert "validation_and_test_excess_not_same_positive_direction" in reasons["TOP20_SIGNAL_A"]

