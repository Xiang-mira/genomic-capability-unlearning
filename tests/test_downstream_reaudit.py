from __future__ import annotations

import argparse
import json
from phase2.downstream_reaudit import (
    RETAIN_GROUPS,
    TARGET_GROUPS,
    classify_decision,
    collect_scores,
    command_for_eval,
    summarize_role_delta,
    worst_task_delta,
    aggregate,
)


def test_target_drop_uses_base_minus_checkpoint() -> None:
    base = {
        (42, "hvue_forget", "target_a"): 0.90,
        (42, "gue_retain", "retain_a"): 0.80,
    }
    checkpoint = {
        (42, "hvue_forget", "target_a"): 0.70,
        (42, "gue_retain", "retain_a"): 0.79,
    }
    target = summarize_role_delta(checkpoint, base, TARGET_GROUPS, "target")
    retain = summarize_role_delta(checkpoint, base, RETAIN_GROUPS, "retain")

    assert round(target["mean_delta"], 6) == 0.20
    assert round(retain["mean_delta"], 6) == -0.01


def test_decision_requires_random_adjusted_drop() -> None:
    target = {"mean_delta": 0.08, "ci_low": 0.01}
    retain = {"mean_delta": 0.0, "ci_low": 0.0}

    decision = classify_decision(
        "method",
        "gd_random_control",
        target,
        retain,
        worst_retain=0.0,
        random_adjusted=0.01,
        random_adjusted_ci_low=0.001,
    )

    assert decision == "target_drop_not_stronger_than_random"


def test_decision_rejects_catastrophic_retain_task() -> None:
    target = {"mean_delta": 0.08, "ci_low": 0.01}
    retain = {"mean_delta": -0.005, "ci_low": -0.02}

    decision = classify_decision(
        "method",
        "gd_random_control",
        target,
        retain,
        worst_retain=-0.06,
        random_adjusted=0.03,
        random_adjusted_ci_low=0.01,
    )

    assert decision == "target_drop_with_retain_damage"


def test_worst_task_delta_is_task_mean_not_single_row() -> None:
    base = {
        (42, "gue_retain", "retain_a"): 0.80,
        (43, "gue_retain", "retain_a"): 0.82,
        (42, "gue_retain", "retain_b"): 0.70,
        (43, "gue_retain", "retain_b"): 0.70,
    }
    checkpoint = {
        (42, "gue_retain", "retain_a"): 0.75,
        (43, "gue_retain", "retain_a"): 0.79,
        (42, "gue_retain", "retain_b"): 0.71,
        (43, "gue_retain", "retain_b"): 0.72,
    }

    assert round(worst_task_delta(checkpoint, base, RETAIN_GROUPS), 6) == -0.04


def test_collect_scores_reads_completed_seed_csv(tmp_path) -> None:
    result_dir = tmp_path / "global_host_tropism" / "base" / "seed_42"
    result_dir.mkdir(parents=True)
    (result_dir / "eval_benchmarks.csv").write_text(
        "benchmark,task,group,seed,auroc,accuracy,metric_for_best\n"
        "hvue,target_a,hvue_forget,42,0.9,0.8,auroc\n"
    )

    scores = collect_scores(tmp_path, "global_host_tropism", "base", [42, 43])

    assert scores == {(42, "hvue_forget", "target_a"): 0.9}


def test_aggregate_writes_provenance_metadata(tmp_path) -> None:
    result_dir = tmp_path / "global_host_tropism"
    for checkpoint in ["base", "gd_random_control", "projection_rank32", "gd_localized", "rmu_joint"]:
        seed_dir = result_dir / checkpoint / "seed_42"
        seed_dir.mkdir(parents=True)
        if checkpoint == "base":
            auroc = "0.90"
        elif checkpoint == "gd_random_control":
            auroc = "0.85"
        else:
            auroc = "0.80"
        (seed_dir / "eval_benchmarks.csv").write_text(
            "benchmark,task,group,seed,auroc,accuracy,metric_for_best\n"
            f"hvue,target_a,hvue_forget,42,{auroc},0.8,auroc\n"
            "gue,retain_a,gue_retain,42,0.80,0.8,auroc\n"
        )

    args = argparse.Namespace(
        project_root=str(tmp_path),
        out_dir=".",
        seeds="42",
        cohort="global_host_tropism",
        bootstrap_samples=10,
    )
    aggregate(args)

    payload = json.loads((tmp_path / "downstream_reaudit_aggregate_metadata.json").read_text())
    assert payload["phase"] == "downstream_reaudit_aggregate"
    assert payload["cohort_filter"] == "global_host_tropism"
    assert payload["decision_row_count"] >= 1
    assert payload["selection_rule_version"] == "downstream_reaudit_v1"
    assert payload["result_manifest"]["path"].endswith("downstream_reaudit_eval_manifest.csv")
    assert payload["metric_thresholds"]["random_adjusted_drop_min_auroc"] == 0.02
    assert payload["random_control_source"]["global_host_tropism"] == "gd_random_control"
    assert "git_diff_sha256" in payload
    assert "input_result_files" in payload


def test_command_for_eval_keeps_validation_limit_when_resume_is_enabled(tmp_path) -> None:
    args = argparse.Namespace(
        python_bin="python",
        out_dir=str(tmp_path / "out"),
        device="cuda:0",
        cpu_threads=8,
        train_batch_size=4,
        eval_batch_size=16,
        max_length=512,
        epochs=3,
        max_steps=600,
        eval_every=200,
        validation_max_rows=1234,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    command = command_for_eval(
        args=args,
        cohort="global_host_tropism",
        checkpoint="projection_rank32",
        weights="weights.safetensors",
        seed=42,
        benchmark_manifest=tmp_path / "manifest.csv",
    )

    assert "--resume" in command
    idx = command.index("--validation-max-rows")
    assert command[idx + 1] == "1234"
