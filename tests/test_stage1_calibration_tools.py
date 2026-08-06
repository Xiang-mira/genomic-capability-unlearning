from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase2.stage1_calibration_tools import (
    DEFAULT_TASK,
    extract_manifest_sha256,
    run_write_exploratory_comparison_report,
    run_write_exploratory_attack_config,
    run_write_input_fairness_report,
    run_write_registry,
)


def test_extract_manifest_sha256_accepts_relative_metadata_key(tmp_path) -> None:
    manifest = tmp_path / "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("task\n")
    metadata = {
        "data_hashes": {
            "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv": "abc123"
        }
    }

    assert extract_manifest_sha256(metadata, manifest) == "abc123"


def test_run_write_registry_marks_failed_run_and_preserves_manifest_hash(tmp_path) -> None:
    project_root = tmp_path
    out_dir = project_root / "results/seed_43"
    out_dir.mkdir(parents=True)
    metadata_path = out_dir / "eval_benchmarks_metadata.json"
    progress_path = out_dir / "eval_benchmarks_progress.json"
    summary_path = out_dir / "eval_benchmarks_summary.json"
    results_path = out_dir / "eval_benchmarks.csv"
    manifest = project_root / "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("task\n")
    metadata_path.write_text(
        json.dumps(
            {
                "commit_hash": "deadbeef",
                "git_dirty": True,
                "config_hash": "cfg123",
                "data_hashes": {
                    "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv": "manifest123"
                },
            }
        )
    )
    progress_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "exit_reason": "UnboundLocalError: cannot access local variable 'row'",
            }
        )
    )
    summary_path.write_text("{}\n")
    results_path.write_text("task,auroc,mcc\n")

    plan_path = project_root / "plan.json"
    plan_path.write_text(
        json.dumps(
            [
                {
                    "name": "fresh_lora_base_r8_lr1e-5_seed43",
                    "checkpoint": "base",
                    "rank": 8,
                    "lr": 1e-5,
                    "seed": 43,
                    "cmd": [
                        "python",
                        "-u",
                        "phase2/eval_benchmarks.py",
                        "--out-dir",
                        str(out_dir.relative_to(project_root)),
                    ],
                }
            ]
        )
    )

    out_csv = project_root / "registry.csv"
    out_json = project_root / "registry.json"
    args = argparse.Namespace(
        plan_json=plan_path,
        out_csv=out_csv,
        out_json=out_json,
        task=DEFAULT_TASK,
        formal_manifest=manifest,
        formal_kmer_baseline=project_root / "kmer.csv",
    )

    import phase2.stage1_calibration_tools as tools

    original_root = tools.PROJECT_ROOT
    tools.PROJECT_ROOT = project_root
    try:
        run_write_registry(args)
    finally:
        tools.PROJECT_ROOT = original_root

    with out_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["run_status"] == "failed"
    assert rows[0]["manifest_sha256"] == "manifest123"
    assert "UnboundLocalError" in rows[0]["notes"]


def test_write_input_fairness_report_marks_0849_as_matched_input(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "task,split_type,id,split,label,sequence\n"
        "hvue_human_host_tropism,cluster_disjoint,a,train,0," + "A" * 900 + "\n"
        "hvue_human_host_tropism,cluster_disjoint,b,val,1," + "C" * 1000 + "\n"
        "hvue_human_host_tropism,cluster_disjoint,c,test,0," + "G" * 950 + "\n"
    )
    out_json = tmp_path / "fairness.json"
    out_md = tmp_path / "fairness.md"
    args = argparse.Namespace(
        formal_manifest=manifest,
        task=DEFAULT_TASK,
        split_type="cluster_disjoint",
        out_json=out_json,
        out_md=out_md,
    )

    run_write_input_fairness_report(args)

    payload = json.loads(out_json.read_text())
    assert payload["kmer_0849_matches_lora_input_budget"] is True
    assert payload["matched_input_baseline"] == "0.8496470934222136"
    assert payload["lora_truncated_fraction"] == 1.0


def test_write_exploratory_attack_config_chooses_best_validation_mean(tmp_path) -> None:
    selection_json = tmp_path / "selection.json"
    selection_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "rank": 16,
                        "lr": 5e-5,
                        "dev_validation_mean": 0.86,
                        "dev_validation_std": 0.001,
                        "status": "dev_not_positive",
                        "confirmation_seed44_auroc_excess": -0.001,
                    },
                    {
                        "rank": 32,
                        "lr": 5e-5,
                        "dev_validation_mean": 0.84,
                        "dev_validation_std": 0.01,
                        "status": "dev_not_positive",
                        "confirmation_seed44_auroc_excess": 0.0,
                    },
                ]
            }
        )
    )
    out_json = tmp_path / "exploratory.json"
    args = argparse.Namespace(selection_json=selection_json, out_json=out_json)

    run_write_exploratory_attack_config(args)

    payload = json.loads(out_json.read_text())
    assert payload["status"] == "exploratory_only"
    assert payload["selected_rank"] == 16
    assert payload["selected_learning_rate"] == 5e-5
    assert payload["formal_strong_baseline_calibration"] == "failed"


def test_write_exploratory_comparison_report_writes_partial_summary(tmp_path) -> None:
    registry_json = tmp_path / "registry.json"
    result_dir = tmp_path / "fresh_lora" / "base" / "rank_16" / "lr_5e-5" / "seed_42"
    (result_dir / "logs").mkdir(parents=True)
    (result_dir / "eval_benchmarks.csv").write_text(
        "benchmark,task,group,model_name,checkpoint,seed,problem_type,n_train,n_val,n_val_full,n_val_early_stop,n_test,n_test_eval,train_loss,val_loss,validation_metric,metric_for_best,best_step,best_checkpoint,exported_attack_checkpoint,accuracy,f1,mcc,auroc,auprc,mse,rmse,r2,pearson,lora_rank,lora_alpha,lora_dropout,train_batch_size,eval_batch_size,checkpoint_retained,lora_modules,trainable_params,total_params,split_type,kmer_baseline_score,metric_excess_over_kmer,attack_recipe_id,post_attack_fresh_head_score,readout_disruption_flag,test_loss\n"
        "hvue,hvue_human_host_tropism,hvue_forget,Evo,base,42,classification,1,1,1,1,1,1,0.1,0.2,0.86,auroc,1200,, ,0.8,0.8,0.6,0.88,0.85,,,,,16,32,0.0,1,1,False,10,10,100,cluster_disjoint,0.8496470934,0.0303529066,,, ,0.1\n"
    )
    (result_dir / "logs" / "hvue_human_host_tropism.jsonl").write_text(
        json.dumps({"step": 1200, "elapsed_sec": 500.0, "selection_value": 0.86}) + "\n"
    )
    registry_json.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "checkpoint": "base",
                        "run_id": "run1",
                        "seed": 42,
                        "rank": 16,
                        "lr": 5e-5,
                        "run_status": "completed_pending_canonical_baseline",
                        "results_path": str(result_dir / "eval_benchmarks.csv"),
                    }
                ]
            }
        )
    )
    fairness_json = tmp_path / "fairness.json"
    fairness_json.write_text(json.dumps({"matched_input_baseline": "0.8496470934222136"}))
    canonical_json = tmp_path / "canonical.json"
    canonical_json.write_text(json.dumps({"test_auroc": 0.8930006862072345, "test_mcc": 0.6452985196139533}))
    exploratory_json = tmp_path / "exploratory.json"
    exploratory_json.write_text(json.dumps({"selected_rank": 16, "selected_learning_rate": 5e-5}))
    out_csv = tmp_path / "comparison.csv"
    out_json = tmp_path / "comparison.json"
    out_md = tmp_path / "comparison.md"
    args = argparse.Namespace(
        registry_json=registry_json,
        fairness_json=fairness_json,
        canonical_json=canonical_json,
        exploratory_json=exploratory_json,
        out_csv=out_csv,
        out_json=out_json,
        out_md=out_md,
    )

    run_write_exploratory_comparison_report(args)

    payload = json.loads(out_json.read_text())
    assert payload["status"] == "partial_running"
    assert payload["checkpoint_summary"][0]["checkpoint"] == "base"
