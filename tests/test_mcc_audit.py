from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from phase2.mcc_audit import (
    PREDICTION_FIELDS,
    compute_metrics,
    compute_thresholds,
    load_prediction_groups,
    load_prediction_shards,
    metric_values,
    paired_bootstrap,
)


FIELDS = [
    "sample_id",
    "task",
    "split",
    "split_seed",
    "checkpoint_name",
    "method_family",
    "checkpoint_path",
    "probe_protocol",
    "probe_type",
    "probe_seed",
    "model_name",
    "layer",
    "label",
    "score",
    "score_type",
    "source_artifact",
    "manifest_hash",
]


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            full = {field: "" for field in FIELDS}
            full.update(row)
            writer.writerow(full)


def make_row(sample: str, split: str, checkpoint: str, label: int, score: float) -> dict[str, object]:
    return {
        "sample_id": sample,
        "task": "toy",
        "split": split,
        "checkpoint_name": checkpoint,
        "method_family": "base" if checkpoint == "base" else "method",
        "probe_protocol": "fresh_probe_from_feature_cache",
        "probe_type": "capability_hidden_only",
        "probe_seed": 42,
        "model_name": "hidden_only_model",
        "layer": 0,
        "label": label,
        "score": score,
        "score_type": "predict_proba_positive",
    }


def test_metric_values_confusion_specificity_balanced_accuracy() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    metrics = metric_values(labels, scores, 0.5)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["mcc"] == 0.0


def test_metric_values_warns_single_class_and_reverse_score() -> None:
    single = metric_values(np.array([1, 1]), np.array([0.2, 0.8]), 0.5)
    assert "single_class_split" in single["warnings"]
    reversed_scores = metric_values(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]), 0.5)
    assert "reverse_score_orientation_possible" in reversed_scores["warnings"]


def test_compute_metrics_warns_missing_split(tmp_path: Path) -> None:
    rows = [
        make_row("v0", "val", "base", 0, 0.1),
        make_row("v1", "val", "base", 1, 0.9),
    ]
    path = tmp_path / "pred.csv"
    write_predictions(path, rows)
    thresholds = compute_thresholds(load_prediction_groups(path))
    metrics = compute_metrics(path, thresholds)
    test_metric = [row for row in metrics if row["split"] == "test"][0]
    assert test_metric["warnings"] == "empty_split"


def test_validation_threshold_does_not_use_test_label(tmp_path: Path) -> None:
    rows = []
    for checkpoint in ("base", "method"):
        rows.extend(
            [
                make_row("v0", "val", checkpoint, 0, 0.1),
                make_row("v1", "val", checkpoint, 1, 0.9),
                make_row("t0", "test", checkpoint, 1, 0.1),
                make_row("t1", "test", checkpoint, 0, 0.9),
            ]
        )
    path = tmp_path / "pred.csv"
    write_predictions(path, rows)
    thresholds = compute_thresholds(load_prediction_groups(path))
    method_val_threshold = [
        row for row in thresholds if row["checkpoint_name"] == "method" and row["threshold_rule"] == "validation_mcc_max"
    ][0]
    assert 0.1 < method_val_threshold["threshold"] < 0.9


def test_base_fixed_threshold_uses_base_for_modified_checkpoint(tmp_path: Path) -> None:
    rows = [
        make_row("v0", "val", "base", 0, 0.1),
        make_row("v1", "val", "base", 1, 0.9),
        make_row("v0", "val", "method", 0, 0.4),
        make_row("v1", "val", "method", 1, 0.6),
    ]
    path = tmp_path / "pred.csv"
    write_predictions(path, rows)
    thresholds = compute_thresholds(load_prediction_groups(path))
    base_fixed = [
        row for row in thresholds if row["checkpoint_name"] == "base" and row["threshold_rule"] == "base_validation_mcc_max"
    ][0]["threshold"]
    method_fixed = [
        row for row in thresholds if row["checkpoint_name"] == "method" and row["threshold_rule"] == "base_validation_mcc_max"
    ][0]["threshold"]
    assert method_fixed == base_fixed


def test_paired_bootstrap_requires_sample_alignment(tmp_path: Path) -> None:
    rows = []
    for checkpoint, scores in {
        "base": [0.1, 0.9, 0.2, 0.8],
        "gd_random_control": [0.2, 0.8, 0.3, 0.7],
        "method": [0.8, 0.2, 0.7, 0.3],
    }.items():
        for idx, (label, score) in enumerate(zip([0, 1, 0, 1], scores)):
            rows.append(make_row(f"s{idx}", "val", checkpoint, label, score))
            rows.append(make_row(f"s{idx}", "test", checkpoint, label, score))
    path = tmp_path / "pred.csv"
    write_predictions(path, rows)
    thresholds = compute_thresholds(load_prediction_groups(path))
    boot = paired_bootstrap(path, thresholds, n_bootstrap=20, seed=1)
    assert boot
    assert {row["status"] for row in boot} == {"ok"}


def test_compute_metrics_reports_random_adjusted_delta(tmp_path: Path) -> None:
    rows = []
    for checkpoint, scores in {
        "base": [0.1, 0.9, 0.2, 0.8],
        "gd_random_control": [0.2, 0.8, 0.3, 0.7],
        "method": [0.8, 0.2, 0.7, 0.3],
    }.items():
        for idx, (label, score) in enumerate(zip([0, 1, 0, 1], scores)):
            rows.append(make_row(f"s{idx}", "val", checkpoint, label, score))
            rows.append(make_row(f"s{idx}", "test", checkpoint, label, score))
    path = tmp_path / "pred.csv"
    write_predictions(path, rows)
    thresholds = compute_thresholds(load_prediction_groups(path))
    metrics = compute_metrics(path, thresholds)
    method_test = [
        row
        for row in metrics
        if row["checkpoint_name"] == "method" and row["split"] == "test" and row["threshold_rule"] == "fixed_0_5"
    ][0]
    assert method_test["method_mcc_minus_base_mcc"] == -2.0
    assert method_test["method_delta_mcc_minus_random_delta_mcc"] == -2.0


def test_capability_prediction_schema_matches_audit_schema() -> None:
    from phase2.eval_capability_probe import PREDICTION_FIELDS as CAPABILITY_PREDICTION_FIELDS

    assert CAPABILITY_PREDICTION_FIELDS == PREDICTION_FIELDS


def test_load_prediction_shards_validates_schema(tmp_path: Path) -> None:
    shard = tmp_path / "eval_predictions.csv"
    write_predictions(shard, [make_row("s0", "test", "method", 1, 0.9), make_row("s0", "test", "method", 1, 0.9)])
    rows = load_prediction_shards(str(shard))
    assert len(rows) == 1
    assert set(rows[0]) == set(PREDICTION_FIELDS)

    bad = tmp_path / "bad.csv"
    bad.write_text("sample_id,score\ns0,0.9\n")
    try:
        load_prediction_shards(str(bad))
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("expected schema validation failure")


def test_task5a_runner_base_command_uses_base_checkpoint(tmp_path: Path) -> None:
    import argparse
    import json
    from unittest.mock import patch

    from phase2.run_task5a_identity_reaudit import CheckpointSpec, run_one, TASK5A_PROTOCOL

    args = argparse.Namespace(
        out_root=str(tmp_path),
        batch_size=4,
        resume=False,
        dry_run=False,
        checkpoint_format="auto",
        internal_target_config="targets.json",
        forget_csv="forget.csv",
        retain_csv="retain.csv",
        model_dir="model",
        config_path="config.yml",
        device="cuda:0",
        max_length=512,
        layers="0-15",
        max_eval=400,
        fresh_c_grid="0.001",
        fresh_max_iter=100,
        probe_seeds="42",
        n_bootstrap=0,
        fresh_gate_threshold=0.6,
        seed=42,
        export_predictions=True,
    )
    spec = CheckpointSpec("base", "base", "", "base_reference")
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        code = run_one(spec, args)
    command = run.call_args.args[0]
    assert code == 2
    assert "--base-checkpoint" in command
    assert "--export-predictions" in command
    assert "base" in command
    meta = json.loads((tmp_path / "base" / "meta.json").read_text())
    assert meta["task"] == "task5a_identity_reaudit"
    assert meta["checkpoint_name"] == "base"
    assert meta["protocol"]["batch_size"] == args.batch_size
    assert meta["protocol"]["target_config"] == TASK5A_PROTOCOL["target_config"]
    assert meta["source_checkpoint"] == args.model_dir
    assert meta["seed"] == args.seed
    assert meta["runtime_environment"]
    assert str(Path(args.forget_csv)) in meta["data_hashes"]
