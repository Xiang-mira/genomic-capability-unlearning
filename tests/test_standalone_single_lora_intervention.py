import csv
import json
from pathlib import Path

import pytest

from phase2.standalone_single_lora_intervention import (
    BASE_RECOVERY_EVIDENCE_PATH,
    run,
)

# `run()` reads the Stage 1 base-recovery evidence table from a fixed repository
# path. That table is a large generated artifact under the git-ignored data/
# tree, so it is absent in a fresh clone. Skip rather than fail there; see
# README.md (Data contracts) for how to regenerate it.
pytestmark = pytest.mark.skipif(
    not BASE_RECOVERY_EVIDENCE_PATH.exists(),
    reason=f"requires generated artifact {BASE_RECOVERY_EVIDENCE_PATH}",
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_args(tmp_path: Path):
    args = type("Args", (), {})()
    args.out_dir = tmp_path / "out"
    args.metrics_csv = tmp_path / "confirmatory_adapter_metrics.csv"
    args.effective_update_stats_csv = tmp_path / "confirmatory_effective_update_statistics.csv"
    args.merge_equivalence_csv = tmp_path / "confirmatory_adapter_merge_equivalence_by_module.csv"
    args.base_module_norms_csv = tmp_path / "base_module_norms.csv"
    args.strengths = (0.25, 0.5, 0.75)
    return args


def seed_fixture_files(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "confirmatory_adapter_metrics.csv",
        [
            "run_id",
            "rank",
            "learning_rate",
            "seed",
            "status",
            "best_step",
            "validation_auroc",
            "validation_mcc",
            "test_auroc",
            "test_mcc",
            "selected_threshold",
            "adapter_path",
            "adapter_sha256",
            "validation_prediction_path",
            "test_prediction_path",
        ],
        [
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed42",
                "rank": 16,
                "learning_rate": 5e-5,
                "seed": 42,
                "status": "complete",
                "best_step": 200,
                "validation_auroc": 0.85,
                "validation_mcc": 0.55,
                "test_auroc": 0.97,
                "test_mcc": 0.92,
                "selected_threshold": 0.55,
                "adapter_path": "adapter42.pt",
                "adapter_sha256": "hash42",
                "validation_prediction_path": "val42.csv",
                "test_prediction_path": "test42.csv",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed43",
                "rank": 16,
                "learning_rate": 5e-5,
                "seed": 43,
                "status": "complete",
                "best_step": 200,
                "validation_auroc": 0.88,
                "validation_mcc": 0.60,
                "test_auroc": 0.86,
                "test_mcc": 0.60,
                "selected_threshold": 0.52,
                "adapter_path": "adapter43.pt",
                "adapter_sha256": "hash43",
                "validation_prediction_path": "val43.csv",
                "test_prediction_path": "test43.csv",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed44",
                "rank": 16,
                "learning_rate": 5e-5,
                "seed": 44,
                "status": "complete",
                "best_step": 0,
                "validation_auroc": 0.90,
                "validation_mcc": 0.62,
                "test_auroc": 0.99,
                "test_mcc": 0.98,
                "selected_threshold": 0.51,
                "adapter_path": "adapter44.pt",
                "adapter_sha256": "hash44",
                "validation_prediction_path": "val44.csv",
                "test_prediction_path": "test44.csv",
            },
            {
                "run_id": "fresh_lora_base_r32_lr5e-5_seed45",
                "rank": 32,
                "learning_rate": 5e-5,
                "seed": 45,
                "status": "complete",
                "best_step": 200,
                "validation_auroc": 0.95,
                "validation_mcc": 0.70,
                "test_auroc": 0.95,
                "test_mcc": 0.70,
                "selected_threshold": 0.50,
                "adapter_path": "adapter45.pt",
                "adapter_sha256": "hash45",
                "validation_prediction_path": "val45.csv",
                "test_prediction_path": "test45.csv",
            },
        ],
    )
    write_csv(
        tmp_path / "confirmatory_effective_update_statistics.csv",
        [
            "run_id",
            "module",
            "layer",
            "module_short_name",
            "frobenius_norm",
            "spectral_norm",
            "effective_rank_99pct",
            "top_singular_values",
        ],
        [
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed42",
                "module": "blocks.5.mlp.l3",
                "layer": 5,
                "module_short_name": "mlp.l3",
                "frobenius_norm": 2.0,
                "spectral_norm": 1.7,
                "effective_rank_99pct": 4,
                "top_singular_values": "1.7;0.6",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed42",
                "module": "blocks.6.mlp.l3",
                "layer": 6,
                "module_short_name": "mlp.l3",
                "frobenius_norm": 1.0,
                "spectral_norm": 0.8,
                "effective_rank_99pct": 3,
                "top_singular_values": "0.8;0.3",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed43",
                "module": "blocks.5.mlp.l3",
                "layer": 5,
                "module_short_name": "mlp.l3",
                "frobenius_norm": 2.2,
                "spectral_norm": 1.9,
                "effective_rank_99pct": 4,
                "top_singular_values": "1.9;0.5",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed43",
                "module": "blocks.6.mlp.l3",
                "layer": 6,
                "module_short_name": "mlp.l3",
                "frobenius_norm": 1.2,
                "spectral_norm": 0.9,
                "effective_rank_99pct": 3,
                "top_singular_values": "0.9;0.2",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed44",
                "module": "blocks.5.mlp.l3",
                "layer": 5,
                "module_short_name": "mlp.l3",
                "frobenius_norm": 2.1,
                "spectral_norm": 1.8,
                "effective_rank_99pct": 4,
                "top_singular_values": "1.8;0.4",
            },
            {
                "run_id": "fresh_lora_base_r16_lr5e-5_seed44",
                "module": "blocks.6.mlp.l3",
                "layer": 6,
                "module_short_name": "mlp.l3",
                "frobenius_norm": 1.1,
                "spectral_norm": 0.85,
                "effective_rank_99pct": 3,
                "top_singular_values": "0.85;0.2",
            },
        ],
    )
    write_csv(
        tmp_path / "confirmatory_adapter_merge_equivalence_by_module.csv",
        ["run_id", "module", "status"],
        [
            {"run_id": "fresh_lora_base_r16_lr5e-5_seed42", "module": "blocks.5.mlp.l3", "status": "pass"},
            {"run_id": "fresh_lora_base_r16_lr5e-5_seed42", "module": "blocks.6.mlp.l3", "status": "pass"},
            {"run_id": "fresh_lora_base_r16_lr5e-5_seed43", "module": "blocks.5.mlp.l3", "status": "pass"},
            {"run_id": "fresh_lora_base_r16_lr5e-5_seed43", "module": "blocks.6.mlp.l3", "status": "pass"},
            {"run_id": "fresh_lora_base_r16_lr5e-5_seed44", "module": "blocks.5.mlp.l3", "status": "pass"},
            {"run_id": "fresh_lora_base_r16_lr5e-5_seed44", "module": "blocks.6.mlp.l3", "status": "pass"},
        ],
    )
    write_csv(
        tmp_path / "base_module_norms.csv",
        ["module", "weight_frobenius_norm"],
        [
            {"module": "blocks.5.mlp.l3", "weight_frobenius_norm": 20.0},
            {"module": "blocks.6.mlp.l3", "weight_frobenius_norm": 10.0},
        ],
    )


def test_run_selects_validation_only_source_and_writes_deliverables(tmp_path: Path) -> None:
    seed_fixture_files(tmp_path)
    args = make_args(tmp_path)

    run(args)

    out_dir = args.out_dir
    selection = json.loads((out_dir / "standalone_source_adapter_selection.json").read_text())
    assert selection["selected_source_adapter"]["run_id"] == "fresh_lora_base_r16_lr5e-5_seed43"
    assert selection["selection_rule"]["forbidden_inputs"] == [
        "test AUROC",
        "test MCC",
        "test-set thresholds",
        "test predictions",
    ]
    selection_md = (out_dir / "standalone_source_adapter_selection.md").read_text()
    assert "validation-only" in selection_md
    assert "Test performance was recorded for reporting only and was not used for selection" in selection_md

    registry = json.loads((out_dir / "standalone_candidate_registry.json").read_text())
    assert registry["status"] == "planned_candidate_checkpoints_not_materialized"
    assert len(registry["arms"]) == 10
    assert {row["arm_type"] for row in registry["arms"]} == {
        "base",
        "source_subspace_intervention",
        "random_subspace_control",
        "random_layer_control",
    }

    prefilter = json.loads((out_dir / "standalone_prefilter_report.json").read_text())
    assert prefilter["status"] == "planned_prefilter_not_run"
    assert prefilter["prefilter_policy"]["retain_ppl_max_fractional_change"] == 0.10

    fresh = json.loads((out_dir / "standalone_fresh_lora_evaluation.json").read_text())
    assert fresh["status"] == "planned_fresh_lora_evaluations_not_run"
    assert fresh["full_finetune_policy"]["selection_target"].startswith("strongest retain-safe source-subspace intervention")
    assert fresh["evaluation_families"][1]["learning_rate"] == 5e-5
    assert fresh["evaluation_families"][1]["predefined_alternative_configuration"] == "frozen_rank32_lr5e-5"

    required = [
        "standalone_source_adapter_selection.json",
        "standalone_source_adapter_selection.md",
        "standalone_candidate_registry.json",
        "standalone_intervention_norm_audit.csv",
        "standalone_intervention_report.md",
        "standalone_prefilter_metrics.csv",
        "standalone_prefilter_report.md",
        "standalone_prefilter_report.json",
        "standalone_fresh_lora_registry.json",
        "standalone_fresh_lora_adaptation_curves.csv",
        "standalone_fresh_lora_evaluation.md",
        "standalone_fresh_lora_evaluation.json",
    ]
    for name in required:
        assert (out_dir / name).exists(), name


def test_norm_audit_uses_source_modules_and_strength_calibration(tmp_path: Path) -> None:
    seed_fixture_files(tmp_path)
    args = make_args(tmp_path)

    run(args)

    rows = list(csv.DictReader((args.out_dir / "standalone_intervention_norm_audit.csv").open()))
    source_rows = [row for row in rows if row["arm_id"] == "source_subspace_intervention_eta1"]
    assert len(source_rows) == 2
    module_to_rho = {row["module"]: float(row["relative_perturbation_rho"]) for row in source_rows}
    assert module_to_rho["blocks.5.mlp.l3"] == 0.25 * 2.2 / 20.0
    assert module_to_rho["blocks.6.mlp.l3"] == 0.25 * 1.2 / 10.0
