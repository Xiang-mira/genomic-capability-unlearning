from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def write_variant_spec(path: Path, option_a_attacked: str = "", option_b_attacked: str = "") -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "variant_id": "option_a_base",
                    "initializer_label": "none",
                    "k0_ckpt": "",
                    "attacked_ckpt": option_a_attacked,
                    "attacked_ckpt_by_recipe": {},
                },
                {
                    "variant_id": "option_b_classification_ce",
                    "initializer_label": "classification_ce",
                    "k0_ckpt": str(path.parent / "option_b_k0.safetensors"),
                    "attacked_ckpt": option_b_attacked,
                    "attacked_ckpt_by_recipe": {},
                    "recipe_ids": ["k0_no_attack", "lora_r8_lr1e5_l5l9"],
                    "readout_disruption_flag": "readout_disruption",
                },
            ]
        )
    )


def test_stage2_ablation_planner_reports_shared_k0_only(tmp_path: Path) -> None:
    spec_path = tmp_path / "variants.json"
    (tmp_path / "option_b_k0.safetensors").write_text("stub")
    write_variant_spec(spec_path)

    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant_id",
                "recipe_id",
                "auroc",
                "metric_excess_over_kmer",
                "checkpoint",
                "readout_disruption_flag",
                "result_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant_id": "option_a_base",
                "recipe_id": "k0_no_attack",
                "auroc": "0.57",
                "metric_excess_over_kmer": "",
                "checkpoint": "base",
                "readout_disruption_flag": "",
                "result_path": "a.csv",
            }
        )
        writer.writerow(
            {
                "variant_id": "option_b_classification_ce",
                "recipe_id": "k0_no_attack",
                "auroc": "0.60",
                "metric_excess_over_kmer": "",
                "checkpoint": "b.safetensors",
                "readout_disruption_flag": "readout_disruption",
                "result_path": "b.csv",
            }
        )

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "phase2/plan_stage2_initializer_ablation.py",
            "--variant-spec-json",
            str(spec_path),
            "--existing-summary-csv",
            str(summary_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    report = json.loads((out_dir / "stage2_initializer_ablation_report.json").read_text())
    reduced = json.loads((out_dir / "stage2_initializer_ablation_variants.json").read_text())
    plan = json.loads((out_dir / "stage2_initializer_ablation_plan.json").read_text())
    metadata = json.loads((out_dir / "stage2_initializer_ablation_metadata.json").read_text())

    assert report["shared_runnable_recipe_ids"] == ["k0_no_attack"]
    assert report["status"] == "ready_for_execution"
    option_b_row = next(row for row in reduced if row["variant_id"] == "option_b_classification_ce")
    assert option_b_row["recipe_ids"] == ["k0_no_attack"]
    assert plan["suggested_tar_smoke_command"]
    assert len(report["current_comparison_rows"]) == 2
    assert metadata["phase"] == "plan_stage2_initializer_ablation"
    assert metadata["shared_runnable_recipe_ids"] == ["k0_no_attack"]


def test_stage2_ablation_planner_upgrades_when_shared_attacked_recipe_exists(tmp_path: Path) -> None:
    spec_path = tmp_path / "variants.json"
    (tmp_path / "option_b_k0.safetensors").write_text("stub")
    option_a_attacked = tmp_path / "option_a_attack.safetensors"
    option_a_attacked.write_text("stub")
    attacked = tmp_path / "option_b_attack.safetensors"
    attacked.write_text("stub")
    write_variant_spec(spec_path, option_a_attacked=str(option_a_attacked), option_b_attacked=str(attacked))

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "phase2/plan_stage2_initializer_ablation.py",
            "--variant-spec-json",
            str(spec_path),
            "--existing-summary-csv",
            str(tmp_path / "missing_summary.csv"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    report = json.loads((out_dir / "stage2_initializer_ablation_report.json").read_text())
    metadata = json.loads((out_dir / "stage2_initializer_ablation_metadata.json").read_text())
    assert report["shared_runnable_recipe_ids"] == ["k0_no_attack", "lora_r8_lr1e5_l5l9"]
    assert report["status"] == "ready_for_execution"
    assert metadata["status"] == "ready_for_execution"
