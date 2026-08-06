from __future__ import annotations

import argparse
import json
import time

from phase2.run_downstream_reaudit_triage import (
    CheckpointSpec,
    aggregate_outputs,
    build_eval_cmd,
    write_run_metadata,
)


def test_write_run_metadata_writes_triage_provenance(tmp_path) -> None:
    args = argparse.Namespace(out_dir=str(tmp_path), python_bin="python")

    write_run_metadata(
        tmp_path,
        args,
        phase="triage_complete",
        confirmation_candidates=["projection_rank32"],
        seed44_ran=False,
    )

    payload = json.loads((tmp_path / "triage_metadata.json").read_text())
    assert payload["phase"] == "triage_complete"
    assert payload["confirmation_candidates"] == ["projection_rank32"]
    assert payload["seed44_ran"] is False
    assert payload["selection_rule_version"] == "light_downstream_triage_v1"
    assert payload["random_control_source"] == "gd_random_control"
    assert "git_diff_sha256" in payload


def test_build_eval_cmd_keeps_validation_limit_when_resume_is_enabled(tmp_path) -> None:
    command = build_eval_cmd(
        python_bin="python",
        manifest=tmp_path / "manifest.csv",
        out_dir=tmp_path,
        checkpoint=CheckpointSpec(
            name="projection_rank32",
            weights="data/phase2/checkpoints_projection_adaptive_rank32/weights.safetensors",
        ),
        seed=42,
        tasks=["hvue_human_host_tropism"],
    )

    assert "--resume" in command
    idx = command.index("--validation-max-rows")
    assert command[idx + 1] == "1000"


def test_aggregate_outputs_writes_light_report_provenance(tmp_path) -> None:
    for checkpoint in ["base", "gd_random_control", "projection_rank32"]:
        seed_dir = tmp_path / "global_host_tropism" / checkpoint / "seed_42"
        seed_dir.mkdir(parents=True)
        auroc = {"base": "0.90", "gd_random_control": "0.85", "projection_rank32": "0.80"}[checkpoint]
        (seed_dir / "eval_benchmarks.csv").write_text(
            "benchmark,task,group,seed,auroc,mcc,f1,accuracy,auprc\n"
            f"hvue,hvue_human_host_tropism,hvue_forget,42,{auroc},0.0,0.0,0.8,0.0\n"
            "gue,gue_mouse_3,gue_retain,42,0.70,0.0,0.0,0.8,0.0\n"
        )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("task\nhvue_human_host_tropism\n")

    from phase2 import run_downstream_reaudit_triage as triage

    original_manifest = triage.DEFAULT_MANIFEST
    triage.DEFAULT_MANIFEST = manifest
    try:
        aggregate_outputs(
            tmp_path,
            confirmation_candidates=["projection_rank32"],
            seed44_ran=False,
            started_at=time.time() - 60,
        )
    finally:
        triage.DEFAULT_MANIFEST = original_manifest

    payload = json.loads((tmp_path / "light_downstream_reaudit_metadata.json").read_text())
    assert payload["phase"] == "light_downstream_reaudit_aggregate"
    assert payload["selection_rule_version"] == "light_downstream_triage_v1"
    assert payload["result_manifest"]["sha256"]
    assert payload["metric_thresholds"]["retain_mean_delta_max"] == 0.02
    assert payload["random_control_source"] == "gd_random_control"
    assert payload["included_confirmation_checkpoints"] == [
        "base",
        "gd_random_control",
        "projection_rank32",
    ]
