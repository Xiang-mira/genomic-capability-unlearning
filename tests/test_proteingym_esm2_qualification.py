from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import phase2.proteingym_esm2_qualification as workflow
from phase2.proteingym_esm2_qualification import (
    build_parser,
    download_report_files_intact,
    ensure_proteingym_data,
    lora_complete_qualification,
    make_smoke_fixture,
    md5_file,
    run_smoke,
    run_workflow,
    validate_resume_state,
)


DELIVERABLES = [
    "protein_48h_candidate_inventory.csv",
    "protein_48h_candidate_ranking.csv",
    "protein_48h_frozen_protocol.json",
    "protein_48h_split_manifest.json",
    "protein_48h_split_audit.json",
    "protein_48h_baseline_results.csv",
    "protein_48h_baseline_report.md",
    "protein_48h_esm2_pilot_metrics.csv",
    "protein_48h_lora_metrics.csv",
    "protein_48h_registry.json",
    "protein_48h_artifact_audit.json",
    "protein_48h_summary_report.md",
    "protein_48h_summary_report.json",
]


def parse_args(parts: list[str]):
    return build_parser().parse_args(parts)


def test_smoke_workflow_writes_all_requested_deliverables_and_reaches_lora(tmp_path: Path) -> None:
    out_root = tmp_path / "smoke"
    args = parse_args(
        [
            "smoke",
            "--out-root",
            str(out_root),
            "--log-file",
            str(tmp_path / "smoke.log"),
            "--device",
            "cpu",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "20",
            "--stop-on-low-disk-gb",
            "1",
        ]
    )

    summary = run_smoke(args)

    for name in DELIVERABLES:
        assert (out_root / name).exists(), name
    assert summary["answers"]["9_any_assay_advanced_to_lora"] is True
    assert summary["answers"]["12_preliminarily_qualified_pair_found"] is True
    registry = json.loads((out_root / "protein_48h_registry.json").read_text())
    assert registry["workflow"]["max_concurrent_gpu_jobs"] == 1
    assert all(task["status"] in {"complete", "skipped"} for task in registry["tasks"])


def test_position_heldout_split_has_no_position_or_duplicate_mutation_overlap(tmp_path: Path) -> None:
    out_root = tmp_path / "smoke"
    args = parse_args(
        [
            "smoke",
            "--out-root",
            str(out_root),
            "--log-file",
            str(tmp_path / "smoke.log"),
            "--device",
            "cpu",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "10",
            "--stop-on-low-disk-gb",
            "1",
        ]
    )
    run_smoke(args)

    audit = json.loads((out_root / "protein_48h_split_audit.json").read_text())
    position_audit = audit["assays"]["SMOKE_SIGNAL"]["position_heldout"]
    assert position_audit["status"] == "valid"
    assert position_audit["position_overlap"] == {"train_test": [], "train_val": [], "val_test": []}
    assert position_audit["duplicate_mutation_overlap"] == {"train_test": [], "train_val": [], "val_test": []}


def test_protocol_and_gate_freeze_validation_only_selection_rules(tmp_path: Path) -> None:
    out_root = tmp_path / "smoke"
    args = parse_args(
        [
            "smoke",
            "--out-root",
            str(out_root),
            "--log-file",
            str(tmp_path / "smoke.log"),
            "--device",
            "cpu",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "10",
            "--stop-on-low-disk-gb",
            "1",
        ]
    )
    run_smoke(args)

    protocol = json.loads((out_root / "protein_48h_frozen_protocol.json").read_text())
    gate = json.loads((out_root / "protein_48h_advancement_gate.json").read_text())
    assert protocol["baseline_selection"]["test_set_used_for_selection"] is False
    assert protocol["advancement_gate"]["uses_test_for_advancement"] is False
    assert gate["test_used_for_advancement"] is False
    assert gate["lora_candidate_order"] == ["SMOKE_SIGNAL"]


def test_missing_formal_inputs_complete_negative_not_available_package(tmp_path: Path) -> None:
    out_root = tmp_path / "formal_missing"
    args = parse_args(
        [
            "run",
            "--formal",
            "--out-root",
            str(out_root),
            "--dms-dir",
            str(tmp_path / "missing_dms"),
            "--metadata-csv",
            str(tmp_path / "missing_metadata.csv"),
            "--public-predictions-dir",
            str(tmp_path / "missing_predictions"),
            "--msa-dir",
            str(tmp_path / "missing_msa"),
            "--log-file",
            str(tmp_path / "formal.log"),
            "--device",
            "cpu",
            "--local-files-only",
            "--no-auto-download-proteingym",
            "--stop-on-low-disk-gb",
            "1",
            "--max-retries",
            "0",
        ]
    )

    summary = run_workflow(args)

    assert summary["answers"]["1_static_candidate_count"] == 0
    assert summary["answers"]["9_any_assay_advanced_to_lora"] is False
    assert summary["answers"]["12_preliminarily_qualified_pair_found"] is False
    assert summary["input_status"]["status"] == "NOT_AVAILABLE"
    assert "WORKFLOW_INPUTS" in summary["answers"]["13_failure_reasons"]
    with (out_root / "protein_48h_lora_metrics.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert "No candidate task-model pair" in summary["permitted_conclusion"]


def test_downloader_fetches_required_zip_and_metadata_from_authoritative_record(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    wt = "ACDEFGHIKLMNPQRSTVWY"
    assay_csv = source / "ASSAY_A.csv"
    assay_csv.write_text(
        "mutant,mutated_sequence,DMS_score\n"
        f"A1C,CCDEFGHIKLMNPQRSTVWY,1.0\n"
        f"C2D,ADDEFGHIKLMNPQRSTVWY,2.0\n"
        f"D3E,ACEEFGHIKLMNPQRSTVWY,3.0\n"
    )
    metadata_csv = source / "DMS_substitutions.csv"
    metadata_csv.write_text(
        "DMS_id,DMS_filename,target_name,selection_type,target_seq\n"
        f"ASSAY_A,ASSAY_A.csv,synthetic,activity,{wt}\n"
    )
    archive = source / "DMS_ProteinGym_substitutions.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(assay_csv, arcname="DMS_ProteinGym_substitutions/ASSAY_A.csv")
    api = source / "record.json"
    api.write_text(
        json.dumps(
            {
                "id": "local-record",
                "doi": "10.0000/local",
                "metadata": {"version": "local"},
                "files": [
                    {
                        "key": "DMS_substitutions.csv",
                        "size": metadata_csv.stat().st_size,
                        "checksum": f"md5:{md5_file(metadata_csv)}",
                        "links": {"self": metadata_csv.resolve().as_uri()},
                    },
                    {
                        "key": "DMS_ProteinGym_substitutions.zip",
                        "size": archive.stat().st_size,
                        "checksum": f"md5:{md5_file(archive)}",
                        "links": {"self": archive.resolve().as_uri()},
                    },
                ],
            }
        )
    )
    out_root = tmp_path / "out"
    args = parse_args(
        [
            "download-data",
            "--out-root",
            str(out_root),
            "--dms-dir",
            str(tmp_path / "canonical" / "DMS_substitutions"),
            "--metadata-csv",
            str(tmp_path / "canonical" / "DMS_substitutions.csv"),
            "--proteingym-source-api",
            api.resolve().as_uri(),
            "--log-file",
            str(tmp_path / "download.log"),
            "--stop-on-low-disk-gb",
            "1",
        ]
    )

    report = ensure_proteingym_data(args)

    assert report["validation_status"] == "valid"
    assert report["discovered_assays"] == 1
    assert report["valid_single_substitution_records"] == 3
    assert (tmp_path / "canonical" / "DMS_substitutions" / "ASSAY_A.csv").exists()
    assert not (tmp_path / "canonical" / "download_cache" / "DMS_ProteinGym_substitutions.zip").exists()
    assert report["removed_temporary_files"]
    assert download_report_files_intact(out_root / "protein_48h_proteingym_download_report.json", args)

    (tmp_path / "canonical" / "DMS_substitutions" / "ASSAY_A.csv").write_text("not,the,same\n")
    assert not download_report_files_intact(out_root / "protein_48h_proteingym_download_report.json", args)


def test_evolutionary_baseline_integration_selects_strongest_overall(tmp_path: Path) -> None:
    out_root = tmp_path / "evo"
    dms_dir, metadata, pred_dir, msa_dir = make_smoke_fixture(out_root)
    source_rows = list(csv.DictReader((dms_dir / "SMOKE_SIGNAL.csv").open()))
    with (pred_dir / "SMOKE_SIGNAL.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mutant", "EVmutation", "ESM2_150M"])
        writer.writeheader()
        for row in source_rows:
            writer.writerow({"mutant": row["mutant"], "EVmutation": row["DMS_score"], "ESM2_150M": row["DMS_score"]})
    args = parse_args(
        [
            "run",
            "--out-root",
            str(out_root),
            "--dms-dir",
            str(dms_dir),
            "--metadata-csv",
            str(metadata),
            "--public-predictions-dir",
            str(pred_dir),
            "--msa-dir",
            str(msa_dir),
            "--log-file",
            str(tmp_path / "evo.log"),
            "--device",
            "cpu",
            "--mock-esm2",
            "--mock-lora",
            "--skip-proteingym-download",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "20",
            "--stop-on-low-disk-gb",
            "1",
            "--max-retries",
            "0",
        ]
    )

    run_workflow(args)

    rows = list(csv.DictReader((out_root / "protein_48h_baseline_results.csv").open()))
    evo_rows = [row for row in rows if row["baseline_family"] == "evolutionary" and row["status"] == "complete"]
    assert {row["baseline"] for row in evo_rows} == {"public_evolutionary:EVmutation"}
    assert all("ESM2" not in row["baseline"] for row in rows)
    strongest = [
        row
        for row in rows
        if row["assay_id"] == "SMOKE_SIGNAL"
        and row["split_type"] == "position_heldout"
        and row["is_strongest_available_non_plm"] == "true"
    ]
    assert strongest
    assert strongest[0]["baseline_family"] == "evolutionary"
    report = json.loads((out_root / "protein_48h_evolutionary_baseline_report.json").read_text())
    assert report["baseline_strength_by_assay"]["SMOKE_SIGNAL"]["position_heldout"]["strong_baseline_evidence"] == "complete"


def test_real_esm2_resume_reuses_valid_predictions_after_baseline_update(tmp_path: Path, monkeypatch) -> None:
    out_root = tmp_path / "reuse"
    args = parse_args(
        [
            "smoke",
            "--out-root",
            str(out_root),
            "--log-file",
            str(tmp_path / "reuse.log"),
            "--device",
            "cpu",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "10",
            "--stop-on-low-disk-gb",
            "1",
        ]
    )
    run_smoke(args)
    pred_path = out_root / "protein_48h_esm2_pilot_predictions/SMOKE_SIGNAL/position_heldout/zero_shot.csv"
    mtime = pred_path.stat().st_mtime_ns

    def fail_zero(*_args, **_kwargs):
        raise AssertionError("zero-shot inference should not run when a valid prediction CSV exists")

    def fail_features(*_args, **_kwargs):
        raise AssertionError("frozen feature extraction should not run when valid readout predictions exist")

    monkeypatch.setattr(workflow, "run_zero_shot_esm2", fail_zero)
    monkeypatch.setattr(workflow, "extract_real_esm2_features", fail_features)
    args.command = "run"
    args.resume = True
    args.mock_esm2 = False
    args.mock_lora = True
    run_workflow(args)

    assert pred_path.stat().st_mtime_ns == mtime
    validation = json.loads((out_root / "protein_48h_resume_validation.json").read_text())
    assert validation["valid_reused_esm2_prediction_outputs"] == 18


def test_resume_validation_reconciles_stale_running_and_repairs_partial_prediction(tmp_path: Path) -> None:
    out_root = tmp_path / "partial"
    args = parse_args(
        [
            "smoke",
            "--out-root",
            str(out_root),
            "--log-file",
            str(tmp_path / "partial.log"),
            "--device",
            "cpu",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "10",
            "--stop-on-low-disk-gb",
            "1",
        ]
    )
    run_smoke(args)
    pred_path = out_root / "protein_48h_esm2_pilot_predictions/SMOKE_SIGNAL/random/zero_shot.csv"
    pred_path.write_text("partial\n")
    registry = json.loads((out_root / "protein_48h_registry.json").read_text())
    registry["tasks"].append({"task_id": "stale_task", "stage": "esm2_pilot", "status": "running", "pid": 987654321})
    (out_root / "protein_48h_registry.json").write_text(json.dumps(registry))

    registry = workflow.load_registry(args)
    validation = validate_resume_state(args, registry)
    assert validation["invalid_esm2_prediction_outputs"] == 1
    assert validation["registry_corrections"]

    args.command = "run"
    args.resume = True
    run_workflow(args)

    assert list(csv.DictReader(pred_path.open()))
    assert (out_root / "resume_invalid_artifacts").exists()


def test_lora_fallback_requires_complete_preliminary_qualification(tmp_path: Path) -> None:
    out_root = tmp_path / "gate"
    args = parse_args(
        [
            "smoke",
            "--out-root",
            str(out_root),
            "--log-file",
            str(tmp_path / "gate.log"),
            "--device",
            "cpu",
            "--min-valid-samples",
            "20",
            "--n-bootstrap",
            "10",
            "--stop-on-low-disk-gb",
            "1",
        ]
    )
    run_smoke(args)
    rows = [
        {
            "assay_id": "SMOKE_SIGNAL",
            "stage": "formal",
            "seed": seed,
            "status": "complete",
            "test_excess": "0.05",
            "position_bootstrap_ci_low": "-0.01",
            "position_bootstrap_ci_high": "0.09",
            "checkpoint_selection_evidence": json.dumps({"selection_metric": "val_spearman", "adapter_path": "adapter", "head_path": "head"}),
        }
        for seed in (42, 43, 44)
    ]
    args.mock_esm2 = False
    args.mock_lora = False
    evidence = lora_complete_qualification(args, "SMOKE_SIGNAL", rows, {"labels": ["FROZEN_PLM_SIGNAL"]})
    assert evidence["qualified"] is False
    assert "PRELIMINARILY_QUALIFIED" not in evidence["labels"]

    for row in rows:
        row["position_bootstrap_ci_low"] = "0.01"
    evidence = lora_complete_qualification(args, "SMOKE_SIGNAL", rows, {"labels": ["FROZEN_PLM_SIGNAL"]})
    assert evidence["qualified"] is True
    assert "PRELIMINARILY_QUALIFIED" in evidence["labels"]
