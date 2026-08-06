from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase2.run_stage1_option_b_sweep import build_command, load_configs, summarize_runs, write_sweep_metadata


def make_args(**overrides):
    defaults = dict(
        python_bin="python",
        config_json="",
        preview_json="preview.json",
        summary_csv="summary.csv",
        execute=False,
        benchmark_manifest="manifest.csv",
        target_task="hvue_human_host_tropism",
        split_type="cluster_disjoint",
        retain_csv="retain.csv",
        out_root="runs",
        target_train_max_rows=256,
        target_val_max_rows=128,
        target_test_max_rows=128,
        retain_max_rows=256,
        elicitation_steps=20,
        ascent_steps=20,
        eval_every=5,
        train_batch_size=4,
        eval_batch_size=8,
        alpha_target=1.0,
        alpha_retain=1.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_configs_uses_defaults_when_no_json() -> None:
    configs = load_configs(make_args())
    assert [cfg["config_id"] for cfg in configs] == ["smoke_2x2", "formal_20x20", "retain_heavy_20x20"]


def test_build_command_uses_config_overrides() -> None:
    args = make_args()
    cmd = build_command(
        args,
        {
            "config_id": "custom",
            "elicitation_steps": 7,
            "ascent_steps": 9,
            "alpha_retain": 2.5,
        },
    )
    assert "--elicitation-steps" in cmd
    assert "7" in cmd
    assert "--ascent-steps" in cmd
    assert "9" in cmd
    assert "--alpha-retain" in cmd
    assert "2.5" in cmd


def test_summarize_runs_reads_meta_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    run_dir = out_root / "formal_20x20"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "target_task": "hvue_human_host_tropism",
                "split_type": "cluster_disjoint",
                "elicitation_steps": 20,
                "ascent_steps": 20,
                "alpha_target": 1.0,
                "alpha_retain": 2.0,
                "retain_train_rows": 256,
                "selected_tensor_count": 65,
                "val_metrics_after_ascent": {"auroc": 0.45},
                "test_metrics_after_ascent": {"auroc": 0.42},
                "readout_disruption_flag": "readout_disruption",
                "weights_path": "weights.safetensors",
            }
        )
    )
    rows = summarize_runs(out_root, [{"config_id": "formal_20x20"}])
    assert rows[0]["status"] == "completed"
    assert rows[0]["test_auroc_after_ascent"] == 0.42


def test_write_sweep_metadata_records_configs(tmp_path: Path) -> None:
    args = make_args(
        benchmark_manifest=str(tmp_path / "manifest.csv"),
        retain_csv=str(tmp_path / "retain.csv"),
        config_json="",
        execute=False,
    )
    Path(args.benchmark_manifest).write_text("task\n")
    Path(args.retain_csv).write_text("sequence\n")
    preview_path = tmp_path / "preview.json"
    preview_path.write_text("[]\n")
    summary_path = tmp_path / "summary.csv"
    summary_path.write_text("config_id,status\n")

    metadata_path = write_sweep_metadata(
        args,
        preview_path=preview_path,
        summary_path=summary_path,
        configs=[{"config_id": "formal_20x20"}],
        rows=[{"config_id": "formal_20x20", "status": "completed"}],
    )

    payload = json.loads(metadata_path.read_text())
    assert payload["phase"] == "run_stage1_option_b_sweep"
    assert payload["config_ids"] == ["formal_20x20"]
    assert payload["completed_count"] == 1
