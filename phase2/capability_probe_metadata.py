"""Shared provenance helpers for Task 7 / Task 5B capability-probe runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from phase2.run_metadata import build_run_metadata


def capability_probe_task_name(out_dir: str | Path) -> str:
    path = str(out_dir)
    if "task5b" in path:
        return "task5b_capability_reaudit"
    if "task7" in path:
        return "task7_capability_probe"
    return "capability_probe_eval"


def build_capability_probe_run_metadata(
    *,
    args: argparse.Namespace,
    out_dir: str | Path,
    signature: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    seeds: Sequence[int],
    c_grid: Sequence[float],
    feature_entries: Sequence[Mapping[str, Any]],
    metric_row_count: int,
    prediction_row_count: int,
    dry_run: bool,
) -> dict[str, Any]:
    task = capability_probe_task_name(out_dir)
    checkpoint_paths = [str(item.get("checkpoint_path", "")) for item in checkpoints if item.get("checkpoint_path")]
    checkpoint_names = [str(item.get("checkpoint_name", "")) for item in checkpoints]
    missing_checkpoints = sorted(
        str(item.get("checkpoint_name", ""))
        for item in checkpoints
        if item.get("checkpoint_path") and not item.get("checkpoint_exists", False)
    )
    return build_run_metadata(
        args=args,
        source_checkpoint=args.model_dir,
        data_paths=[
            args.dataset_manifest,
            args.dataset_audit,
            args.checkpoint_manifest,
            args.config_path,
            *checkpoint_paths,
        ],
        loss_layers=layers,
        seed=None,
        extra={
            "phase": "eval_capability_probe",
            "task": task,
            "out_dir": str(out_dir),
            "split_column": args.split_column,
            "layers": list(layers),
            "seeds": list(seeds),
            "c_grid": list(c_grid),
            "save_feature_cache": args.save_feature_cache,
            "export_predictions": args.export_predictions,
            "prediction_models": args.prediction_models,
            "prediction_output": args.prediction_output,
            "dataset_lineage": dict(dataset_lineage),
            "run_signature": dict(signature),
            "checkpoint_names": checkpoint_names,
            "missing_checkpoint_names": missing_checkpoints,
            "checkpoint_count": len(checkpoint_names),
            "feature_entry_count": len(feature_entries),
            "metric_row_count": metric_row_count,
            "prediction_row_count": prediction_row_count,
            "dry_run": dry_run,
        },
    )


def build_identity_capability_summary_metadata(
    *,
    args: argparse.Namespace,
    out_dir: str | Path,
    signature: Mapping[str, Any],
    phase: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    data_paths = [args.metrics, args.dataset_audit, args.task5a_summary]
    task7_calibration = getattr(args, "task7_calibration", "")
    if task7_calibration:
        data_paths.append(task7_calibration)
    return build_run_metadata(
        args=args,
        source_checkpoint="capability_probe_summary",
        data_paths=data_paths,
        extra={
            "phase": phase,
            "task": capability_probe_task_name(out_dir),
            "out_dir": str(out_dir),
            "run_signature": dict(signature),
            **dict(extra),
        },
    )
