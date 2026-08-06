from __future__ import annotations

import argparse
from pathlib import Path

from phase2.capability_probe_metadata import (
    build_capability_probe_run_metadata,
    build_identity_capability_summary_metadata,
    capability_probe_task_name,
)


def test_capability_probe_task_name_infers_task_variant() -> None:
    assert capability_probe_task_name("data/task7_capability_probe_20260713") == "task7_capability_probe"
    assert capability_probe_task_name("data/task5b_capability_reaudit_20260713") == "task5b_capability_reaudit"
    assert capability_probe_task_name("data/other_probe_run") == "capability_probe_eval"


def test_build_capability_probe_run_metadata_passes_probe_context(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.capability_probe_metadata.build_run_metadata", fake_build_run_metadata)
    args = argparse.Namespace(
        model_dir="./evo-1-8k-base",
        dataset_manifest=str(tmp_path / "dataset.csv"),
        dataset_audit=str(tmp_path / "audit.json"),
        checkpoint_manifest=str(tmp_path / "manifest.json"),
        config_path="configs/evo-1-8k-base_inference.yml",
        split_column="split",
        save_feature_cache=True,
        export_predictions=True,
        prediction_models="hidden_only_model,raw_hidden_joint_model",
        prediction_output="predictions.csv",
    )

    payload = build_capability_probe_run_metadata(
        args=args,
        out_dir=tmp_path / "task5b_capability_reaudit_20260713",
        signature={"script_version": "abc"},
        dataset_lineage={"row_order_hash": "rowhash"},
        checkpoints=[
            {"checkpoint_name": "base", "checkpoint_path": "", "checkpoint_exists": True},
            {"checkpoint_name": "missing", "checkpoint_path": "missing.safetensors", "checkpoint_exists": False},
        ],
        layers=[0, 1, 2],
        seeds=[42, 43],
        c_grid=[0.01, 0.1],
        feature_entries=[{"task": "toy", "layer": 0}],
        metric_row_count=12,
        prediction_row_count=4,
        dry_run=False,
    )

    assert payload["phase"] == "eval_capability_probe"
    assert payload["task"] == "task5b_capability_reaudit"
    assert payload["metric_row_count"] == 12
    assert payload["prediction_row_count"] == 4
    assert payload["missing_checkpoint_names"] == ["missing"]
    assert captured["source_checkpoint"] == "./evo-1-8k-base"
    assert captured["loss_layers"] == [0, 1, 2]
    assert "configs/evo-1-8k-base_inference.yml" in captured["data_paths"]


def test_build_identity_capability_summary_metadata_tracks_summary_inputs(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.capability_probe_metadata.build_run_metadata", fake_build_run_metadata)
    args = argparse.Namespace(
        metrics=str(tmp_path / "capability_probe_metrics.csv"),
        dataset_audit=str(tmp_path / "capability_dataset_audit.json"),
        task5a_summary=str(tmp_path / "task5a_summary.json"),
        task7_calibration=str(tmp_path / "identity_capability_calibration.json"),
    )

    payload = build_identity_capability_summary_metadata(
        args=args,
        out_dir=tmp_path / "task7_capability_probe_20260713",
        signature={"config_hash": "cfg"},
        phase="task7_identity_capability_summary",
        extra={"metric_row_count": 24, "formal_success_allowed": False},
    )

    assert payload["phase"] == "task7_identity_capability_summary"
    assert payload["task"] == "task7_capability_probe"
    assert payload["metric_row_count"] == 24
    assert captured["source_checkpoint"] == "capability_probe_summary"
    assert captured["data_paths"] == [
        str(tmp_path / "capability_probe_metrics.csv"),
        str(tmp_path / "capability_dataset_audit.json"),
        str(tmp_path / "task5a_summary.json"),
        str(tmp_path / "identity_capability_calibration.json"),
    ]
