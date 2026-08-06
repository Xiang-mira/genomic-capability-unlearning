"""Evaluate Task 7/5B capability probes from hidden representations.

For every capability task, checkpoint, layer, and seed, this compares shortcut
baselines against hidden-representation and joint models. The probe is
diagnostic: it reports confounding and hidden incremental information rather
than claiming selective-unlearning success.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.eval_unlearn import apply_checkpoint, extract_features_for_layers, parse_layers
from phase2.probe_validity_audit import (
    EXCLUDED_METADATA_FIELDS,
    EXCLUDED_METADATA_PREFIXES,
    build_kmer_features,
    build_raw_features,
    build_raw_plus_metadata_features,
    fit_eval_logistic,
    safe_auroc,
    safe_log_loss,
    sequence_stats,
)
from phase2.build_capability_probe_dataset import family_matrix
from phase2.capability_probe_metadata import build_capability_probe_run_metadata, capability_probe_task_name
from phase2.run_task5a_identity_reaudit import TASK3_CONTEXT
from phase2.run_metadata import file_sha256, git_info, stable_hash, write_metadata


csv.field_size_limit(sys.maxsize)


METRIC_FIELDS = [
    "task",
    "capability_role",
    "checkpoint_name",
    "source_checkpoint_name",
    "method_family",
    "checkpoint_path",
    "checkpoint_exists",
    "layer",
    "seed",
    "model_name",
    "comparator_model",
    "status",
    "n_train",
    "n_val",
    "n_test",
    "best_c",
    "train_auroc",
    "val_auroc",
    "test_auroc",
    "train_log_loss",
    "val_log_loss",
    "test_log_loss",
    "test_separability",
    "hidden_incremental_auroc",
    "hidden_incremental_log_loss",
    "deviance_improvement",
    "family_only_baseline_performance",
    "metadata_only_baseline_performance",
    "raw_only_baseline_performance",
    "kmer_baseline_performance",
    "raw_plus_metadata_baseline_performance",
    "raw_plus_kmer_baseline_performance",
    "raw_plus_kmer_plus_metadata_baseline_performance",
    "family_label_capability_label_correlation",
    "group_heldout_status",
    "group_heldout_generalization_if_feasible",
    "split_column",
    "manifest_hash",
]

PREDICTION_FIELDS = [
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


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_sequence(values: list[str]) -> str:
    return stable_hash(values)


def probe_signature(
    *,
    dataset_path: Path,
    dataset_audit_path: Path,
    checkpoint_manifest: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    git = git_info()
    smoke_config = {
        "split_column": args.split_column,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seeds": args.seeds,
        "c_grid": args.c_grid,
        "device": args.device,
        "model_dir": args.model_dir,
        "config_path": args.config_path,
        "checkpoint_format": args.checkpoint_format,
        "save_feature_cache": args.save_feature_cache,
    }
    script_paths = [
        "phase2/eval_capability_probe.py",
        "phase2/probe_validity_audit.py",
        "phase2/build_capability_probe_dataset.py",
        "phase2/summarize_identity_capability_calibration.py",
    ]
    script_hashes = {
        path: file_sha256(path)
        for path in script_paths
        if Path(path).exists()
    }
    return {
        "task": "capability_probe_eval",
        "git_commit_hash": git.get("commit_hash", ""),
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_hash": file_sha256(dataset_path),
        "dataset_audit": str(dataset_audit_path),
        "dataset_audit_hash": file_sha256(dataset_audit_path) if dataset_audit_path.exists() else "missing",
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_manifest_hash": file_sha256(checkpoint_manifest),
        "smoke_config": smoke_config,
        "smoke_config_hash": stable_hash(smoke_config),
        "feature_exclusion_hash": stable_hash(
            {
                "fields": sorted(EXCLUDED_METADATA_FIELDS),
                "prefixes": list(EXCLUDED_METADATA_PREFIXES),
            }
        ),
        "script_hashes": script_hashes,
        "script_version": stable_hash(script_hashes),
    }


def apply_split_column(rows: list[dict[str, str]], split_column: str) -> list[dict[str, str]]:
    if split_column == "split":
        return rows
    patched = []
    for row in rows:
        if split_column not in row:
            raise KeyError(f"split column missing from dataset manifest: {split_column}")
        copy = dict(row)
        copy["split"] = row.get(split_column, "")
        patched.append(copy)
    return patched


def parse_checkpoint_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint manifest missing: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        payload = read_json(path)
        rows = payload.get("checkpoints", payload if isinstance(payload, list) else [])
    result = []
    for row in rows:
        name = row.get("checkpoint_name") or row.get("name")
        if not name:
            continue
        ckpt_path = row.get("checkpoint_path") or row.get("ckpt") or ""
        result.append(
            {
                "checkpoint_name": name,
                "source_checkpoint_name": row.get("source_checkpoint_name") or name,
                "method_family": row.get("method_family") or ("base" if name == "base" else "unknown"),
                "checkpoint_path": ckpt_path,
                "checkpoint_exists": True if not ckpt_path else Path(ckpt_path).exists(),
                "source_selection_role": row.get("source_selection_role", ""),
            }
        )
    if not any(row["checkpoint_name"] == "base" for row in result):
        result.insert(
            0,
            {
                "checkpoint_name": "base",
                "source_checkpoint_name": "base",
                "method_family": "base",
                "checkpoint_path": "",
                "checkpoint_exists": True,
                "source_selection_role": "base_reference",
            },
        )
    return result


def write_checkpoint_manifest_copy(out_dir: Path, checkpoints: list[dict[str, Any]], source_path: Path) -> None:
    write_json(
        out_dir / "checkpoint_manifest_used.json",
        {
            "created_at": now(),
            "source_path": str(source_path),
            "checkpoints": checkpoints,
        },
    )


def write_probe_run_metadata(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    signature: dict[str, Any],
    dataset_lineage: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    layers: list[int],
    seeds: list[int],
    c_grid: list[float],
    feature_entries: list[dict[str, Any]],
    metric_row_count: int,
    prediction_row_count: int,
    dry_run: bool,
) -> None:
    write_metadata(
        out_dir / "meta.json",
        build_capability_probe_run_metadata(
            args=args,
            out_dir=out_dir,
            signature=signature,
            dataset_lineage=dataset_lineage,
            checkpoints=checkpoints,
            layers=layers,
            seeds=seeds,
            c_grid=c_grid,
            feature_entries=feature_entries,
            metric_row_count=metric_row_count,
            prediction_row_count=prediction_row_count,
            dry_run=dry_run,
        ),
    )


def rows_by_task(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    return dict(grouped)


def labels_and_splits(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    splits = np.array([row["split"] for row in rows])
    return labels, splits


def value(payload: dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    if raw in (None, ""):
        return None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def model_result_row(
    result: dict[str, Any],
    *,
    task: str,
    capability_role: str,
    checkpoint: dict[str, Any],
    layer: int,
    seed: int,
    model_name: str,
    comparator_model: str,
    shortcut_baselines: dict[str, dict[str, Any]],
    family_corr: float | None,
    group_status: str,
    group_result: str,
) -> dict[str, Any]:
    shortcut_model_names = (
        "raw_only_model",
        "metadata_only_model",
        "raw_plus_metadata_model",
        "family_only_model",
        "kmer_only_model",
        "raw_plus_kmer_model",
        "raw_plus_kmer_plus_metadata_model",
    )
    baseline_aurocs = [
        value(shortcut_baselines[name], "test_auroc")
        for name in shortcut_model_names
        if name in shortcut_baselines
    ]
    shortcut_best = max([x for x in baseline_aurocs if x is not None] or [float("nan")])
    comparator = shortcut_baselines.get(comparator_model, {})
    if model_name == "hidden_only_model" or comparator_model == "shortcut_best":
        comparator_auroc = shortcut_best if math.isfinite(shortcut_best) else None
        comparator_log_loss = min(
            [
                value(shortcut_baselines[name], "test_log_loss")
                for name in shortcut_model_names
                if name in shortcut_baselines and value(shortcut_baselines[name], "test_log_loss") is not None
            ]
            or [None]
        )
    else:
        comparator_auroc = value(comparator, "test_auroc")
        comparator_log_loss = value(comparator, "test_log_loss")

    test_auroc = value(result, "test_auroc")
    test_log_loss = value(result, "test_log_loss")
    n_test = int(result.get("n_test", 0) or 0)
    incremental_auroc = test_auroc - comparator_auroc if test_auroc is not None and comparator_auroc is not None else None
    incremental_log_loss = (
        comparator_log_loss - test_log_loss
        if test_log_loss is not None and comparator_log_loss is not None
        else None
    )
    deviance = 2.0 * n_test * incremental_log_loss if incremental_log_loss is not None else None
    return {
        "task": task,
        "capability_role": capability_role,
        "checkpoint_name": checkpoint["checkpoint_name"],
        "source_checkpoint_name": checkpoint["source_checkpoint_name"],
        "method_family": checkpoint["method_family"],
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_exists": checkpoint["checkpoint_exists"],
        "layer": layer,
        "seed": seed,
        "model_name": model_name,
        "comparator_model": comparator_model,
        "status": result.get("status", "unknown"),
        "n_train": result.get("n_train"),
        "n_val": result.get("n_val"),
        "n_test": result.get("n_test"),
        "best_c": result.get("best_c"),
        "train_auroc": result.get("train_auroc"),
        "val_auroc": result.get("val_auroc"),
        "test_auroc": result.get("test_auroc"),
        "train_log_loss": result.get("train_log_loss"),
        "val_log_loss": result.get("val_log_loss"),
        "test_log_loss": result.get("test_log_loss"),
        "test_separability": result.get("test_separability"),
        "hidden_incremental_auroc": incremental_auroc,
        "hidden_incremental_log_loss": incremental_log_loss,
        "deviance_improvement": deviance,
        "family_only_baseline_performance": value(shortcut_baselines["family_only_model"], "test_auroc"),
        "metadata_only_baseline_performance": value(shortcut_baselines.get("metadata_only_model", {}), "test_auroc"),
        "raw_only_baseline_performance": value(shortcut_baselines["raw_only_model"], "test_auroc"),
        "kmer_baseline_performance": value(shortcut_baselines["kmer_only_model"], "test_auroc"),
        "raw_plus_metadata_baseline_performance": value(shortcut_baselines.get("raw_plus_metadata_model", {}), "test_auroc"),
        "raw_plus_kmer_baseline_performance": value(shortcut_baselines.get("raw_plus_kmer_model", {}), "test_auroc"),
        "raw_plus_kmer_plus_metadata_baseline_performance": value(
            shortcut_baselines.get("raw_plus_kmer_plus_metadata_model", {}),
            "test_auroc",
        ),
        "family_label_capability_label_correlation": family_corr,
        "group_heldout_status": group_status,
        "group_heldout_generalization_if_feasible": group_result,
        "split_column": "split",
        "manifest_hash": "",
    }


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            clean = sanitize(row)
            writer.writerow({field: clean.get(field, "") for field in METRIC_FIELDS})


def write_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            clean = sanitize(row)
            writer.writerow({field: clean.get(field, "") for field in PREDICTION_FIELDS})


def separability(auroc: float) -> float:
    if math.isnan(auroc):
        return float("nan")
    return float(max(auroc, 1.0 - auroc))


def baseline_matrices(rows: list[dict[str, str]]):
    raw_matrix, _ = build_raw_features(rows)
    family_only, _ = family_matrix(rows)
    raw_plus_metadata, _ = build_raw_plus_metadata_features(rows)
    kmer_matrix, _ = build_kmer_features(rows, 3, 6)
    return {
        "raw_only_model": raw_matrix,
        "metadata_only_model": family_only,
        "raw_plus_metadata_model": raw_plus_metadata,
        "family_only_model": family_only,
        "kmer_only_model": kmer_matrix,
        "raw_plus_kmer_model": hstack([raw_matrix, kmer_matrix], format="csr"),
        "raw_plus_kmer_plus_metadata_model": hstack([raw_matrix, family_only, kmer_matrix], format="csr"),
    }


def family_corr_from_audit(audit: dict[str, Any], task: str) -> float | None:
    payload = (
        audit.get("tasks", {})
        .get(task, {})
        .get("family_label_capability_label_correlation", {})
    )
    raw = payload.get("test_separability")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def group_status_from_audit(audit: dict[str, Any], task: str) -> str:
    return (
        audit.get("tasks", {})
        .get(task, {})
        .get("group_feasibility", {})
        .get("group_heldout_status", "infeasible_for_this_dataset")
    )


def task_role(rows: list[dict[str, str]]) -> str:
    roles = sorted({row.get("capability_role", "") for row in rows if row.get("capability_role", "")})
    return "+".join(roles) if roles else "unknown"


def fit_model(matrix, labels: np.ndarray, splits: np.ndarray, c_grid: list[float], seed: int, task: str, model_name: str):
    return fit_eval_logistic(
        matrix,
        labels,
        splits,
        c_grid,
        seed=seed,
        n_bootstrap=0,
        target=task,
        baseline=model_name,
    )


def fit_model_with_scores(
    matrix,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    masks = {name: splits == name for name in ("train", "val", "test")}
    result: dict[str, Any] = {
        "status": "ok",
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
    }
    if masks["train"].sum() == 0 or len(np.unique(labels[masks["train"]])) < 2:
        result["status"] = "missing_or_single_class_train"
        return result, None
    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(matrix[masks["train"]])
    x_all = scaler.transform(matrix)
    selection_split = "val" if masks["val"].sum() and len(np.unique(labels[masks["val"]])) >= 2 else "train"
    selection_mask = masks[selection_split]
    result["selection_split"] = selection_split
    best = None
    best_score = -float("inf")
    for c_value in c_grid:
        clf = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(x_train, labels[masks["train"]])
        probs = clf.predict_proba(x_all[selection_mask])[:, 1]
        score = separability(safe_auroc(labels[selection_mask], probs))
        if not math.isnan(score) and score > best_score:
            best_score = score
            best = (c_value, clf)
    if best is None:
        result["status"] = "fit_failed"
        return result, None
    best_c, clf = best
    result["best_c"] = float(best_c)
    all_scores = clf.predict_proba(x_all)[:, 1]
    for split in ("train", "val", "test"):
        mask = masks[split]
        if mask.sum() == 0:
            result[f"{split}_status"] = "missing"
            continue
        if len(np.unique(labels[mask])) < 2:
            result[f"{split}_status"] = "single_class"
            continue
        probs = all_scores[mask]
        result[f"{split}_auroc"] = safe_auroc(labels[mask], probs)
        result[f"{split}_separability"] = separability(result[f"{split}_auroc"])
        result[f"{split}_log_loss"] = safe_log_loss(labels[mask], probs)
    return result, all_scores.astype(np.float32, copy=False)


def append_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    scores: np.ndarray,
    task: str,
    checkpoint: dict[str, Any],
    seed: int,
    model_name: str,
    layer: int,
    source_artifact: str,
    manifest_hash: str,
) -> None:
    for idx, score in enumerate(scores):
        rows.append(
            {
                "sample_id": str(sample_ids[idx]),
                "task": task,
                "split": str(splits[idx]),
                "split_seed": "",
                "checkpoint_name": checkpoint["source_checkpoint_name"],
                "method_family": checkpoint["method_family"],
                "checkpoint_path": checkpoint["checkpoint_path"],
                "probe_protocol": "fresh_probe_from_feature_cache",
                "probe_type": "capability_probe",
                "probe_seed": seed,
                "model_name": model_name,
                "layer": layer,
                "label": int(labels[idx]),
                "score": float(score),
                "score_type": "predict_proba_positive",
                "source_artifact": source_artifact,
                "manifest_hash": manifest_hash,
            }
        )


def load_checkpoint_model(args: argparse.Namespace, checkpoint: dict[str, Any]):
    from phase1.utils import load_local_checkpoint

    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if checkpoint["checkpoint_path"]:
        apply_checkpoint(model, checkpoint["checkpoint_path"], checkpoint_format=args.checkpoint_format)
    model.eval()
    return model


def cache_features(
    feature_root: Path,
    checkpoint: dict[str, Any],
    task: str,
    layer: int,
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    row_ids: np.ndarray,
    feature_keys: np.ndarray,
    manifest_hash: str,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "checkpoint_name": checkpoint["checkpoint_name"],
            "task": task,
            "layer": layer,
            "cached": False,
            "reason": "disabled",
        }
    out_dir = feature_root / checkpoint["checkpoint_name"] / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_{layer}.npz"
    np.savez_compressed(
        out_path,
        features=features.astype(np.float16, copy=False),
        labels=labels.astype(np.int8, copy=False),
        splits=splits.astype("U16", copy=False),
        row_ids=row_ids.astype("U256", copy=False),
        feature_cache_keys=feature_keys.astype("U64", copy=False),
        manifest_hash=np.array([manifest_hash]),
    )
    return {
        "checkpoint_name": checkpoint["checkpoint_name"],
        "task": task,
        "layer": layer,
        "cached": True,
        "path": str(out_path),
        "shape": list(features.shape),
        "dtype": "float16",
        "manifest_hash": manifest_hash,
        "row_id_rows": int(row_ids.shape[0]),
        "feature_key_rows": int(feature_keys.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", default="data/phase2/audits/task7_capability_probe_20260713/capability_dataset_manifest.csv")
    parser.add_argument("--dataset-audit", default="data/phase2/audits/task7_capability_probe_20260713/capability_dataset_audit.json")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--out-dir", default="data/phase2/audits/task7_capability_probe_20260713")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-format", default="auto")
    parser.add_argument("--layers", default="0-15")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--save-feature-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--prediction-output", default="")
    parser.add_argument(
        "--prediction-models",
        default="hidden_only_model",
        help="Comma-separated capability probe model names to export when --export-predictions is set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_manifest)
    dataset_audit_path = Path(args.dataset_audit)
    checkpoint_manifest_path = Path(args.checkpoint_manifest)
    manifest_hash = file_sha1(dataset_path)
    signature = probe_signature(
        dataset_path=dataset_path,
        dataset_audit_path=dataset_audit_path,
        checkpoint_manifest=checkpoint_manifest_path,
        args=args,
    )
    rows = apply_split_column(read_dataset(dataset_path), args.split_column)
    audit = read_json(dataset_audit_path)
    checkpoints = parse_checkpoint_manifest(checkpoint_manifest_path)
    layers = parse_layers(args.layers)
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    c_grid = [float(part.strip()) for part in args.c_grid.split(",") if part.strip()]
    prediction_models = {part.strip() for part in args.prediction_models.split(",") if part.strip()}
    write_checkpoint_manifest_copy(out_dir, checkpoints, checkpoint_manifest_path)
    dataset_lineage = {
        "manifest_hash": manifest_hash,
        "candidate_manifest_hash": manifest_hash,
        "dataset_audit_hash": file_sha256(dataset_audit_path) if dataset_audit_path.exists() else "missing",
        "row_order_hash": hash_sequence([row.get("id") or row.get("sample_id", "") for row in rows]),
        "label_hash": hash_sequence([row.get("label", "") for row in rows]),
        "sample_id_hash": hash_sequence([row.get("sample_id", "") for row in rows]),
        "feature_cache_key_hash": hash_sequence([row.get("feature_cache_key", "") for row in rows]),
        "split_hash": hash_sequence([row.get("split", "") for row in rows]),
    }

    if args.dry_run:
        write_json(
            out_dir / "capability_feature_cache_manifest.json",
            {
                "created_at": now(),
                "task": capability_probe_task_name(out_dir),
                "dry_run": True,
                "tasks": sorted(rows_by_task(rows)),
                "checkpoints": checkpoints,
                "layers": layers,
                "split_column": args.split_column,
                "manifest_hash": manifest_hash,
                "dataset_lineage": dataset_lineage,
                "run_signature": signature,
            },
        )
        write_json(out_dir / "capability_probe_signature.json", signature)
        write_metrics(out_dir / "capability_probe_metrics.csv", [])
        write_probe_run_metadata(
            args=args,
            out_dir=out_dir,
            signature=signature,
            dataset_lineage=dataset_lineage,
            checkpoints=checkpoints,
            layers=layers,
            seeds=seeds,
            c_grid=c_grid,
            feature_entries=[],
            metric_row_count=0,
            prediction_row_count=0,
            dry_run=True,
        )
        print(f"[cap-probe] dry-run wrote manifests to {out_dir}")
        return

    from evo.tokenizer import CharLevelTokenizer

    feature_root = out_dir / "feature_cache"
    tokenizer = CharLevelTokenizer(512)
    all_metric_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    feature_entries: list[dict[str, Any]] = []
    grouped = rows_by_task(rows)

    task_baselines: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    task_matrices: dict[str, dict[str, Any]] = {}
    task_labels_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task, task_rows in grouped.items():
        labels, splits = labels_and_splits(task_rows)
        task_labels_splits[task] = (labels, splits)
        task_matrices[task] = baseline_matrices(task_rows)
        for seed in seeds:
            task_baselines[(task, seed)] = {
                name: fit_model(matrix, labels, splits, c_grid, seed, task, name)
                for name, matrix in task_matrices[task].items()
            }

    for checkpoint in checkpoints:
        if checkpoint["checkpoint_path"] and not Path(checkpoint["checkpoint_path"]).exists():
            print(f"[cap-probe] skip missing checkpoint {checkpoint['checkpoint_name']}: {checkpoint['checkpoint_path']}")
            continue

        print(f"[cap-probe] loading checkpoint {checkpoint['checkpoint_name']}")
        model = load_checkpoint_model(args, checkpoint)
        for task, task_rows in grouped.items():
            labels, splits = task_labels_splits[task]
            row_ids = np.array([row.get("id") or row.get("sample_id", "") for row in task_rows])
            feature_keys = np.array([row.get("feature_cache_key", "") for row in task_rows])
            sequences = [row["sequence"] for row in task_rows]
            capability_role = task_role(task_rows)
            print(f"[cap-probe] checkpoint={checkpoint['checkpoint_name']} task={task} n={len(task_rows)}")
            features_by_layer = extract_features_for_layers(
                model,
                sequences,
                tokenizer,
                layers,
                args.batch_size,
                args.max_length,
                args.device,
            )
            family_corr = family_corr_from_audit(audit, task)
            group_status = group_status_from_audit(audit, task)
            group_result = "not_run_infeasible" if group_status != "feasible_basic" else "not_implemented_basic_feasibility_only"

            for layer in layers:
                hidden_dense = features_by_layer[layer].astype(np.float32, copy=False)
                hidden = csr_matrix(hidden_dense)
                raw = task_matrices[task]["raw_only_model"]
                metadata = task_matrices[task]["metadata_only_model"]
                family = task_matrices[task]["family_only_model"]
                raw_family = hstack([raw, family], format="csr")
                matrices = {
                    "hidden_only_model": hidden,
                    "raw_hidden_joint_model": hstack([raw, hidden], format="csr"),
                    "metadata_hidden_joint_model": hstack([metadata, hidden], format="csr"),
                    "raw_plus_kmer_plus_metadata_hidden_joint_model": hstack(
                        [task_matrices[task]["raw_plus_kmer_plus_metadata_model"], hidden],
                        format="csr",
                    ),
                    "family_hidden_joint_model": hstack([family, hidden], format="csr"),
                    "raw_family_hidden_joint_model": hstack([raw_family, hidden], format="csr"),
                }
                feature_entry = cache_features(
                    feature_root,
                    checkpoint,
                    task,
                    layer,
                    hidden_dense,
                    labels,
                    splits,
                    row_ids,
                    feature_keys,
                    manifest_hash,
                    args.save_feature_cache,
                )
                feature_entries.append(feature_entry)
                for seed in seeds:
                    shortcut = task_baselines[(task, seed)]
                    for model_name, baseline_name in (
                        ("raw_only_model", "raw_only_model"),
                        ("metadata_only_model", "metadata_only_model"),
                        ("raw_plus_metadata_model", "raw_plus_metadata_model"),
                        ("family_only_model", "family_only_model"),
                        ("kmer_only_model", "kmer_only_model"),
                        ("raw_plus_kmer_model", "raw_plus_kmer_model"),
                        ("raw_plus_kmer_plus_metadata_model", "raw_plus_kmer_plus_metadata_model"),
                    ):
                        all_metric_rows.append(
                            model_result_row(
                                shortcut[model_name],
                                task=task,
                                capability_role=capability_role,
                                checkpoint=checkpoint,
                                layer=layer,
                                seed=seed,
                                model_name=model_name,
                                comparator_model=baseline_name,
                                shortcut_baselines=shortcut,
                                family_corr=family_corr,
                                group_status=group_status,
                                group_result=group_result,
                            )
                        )
                    for model_name, matrix in matrices.items():
                        comparator = {
                            "hidden_only_model": "shortcut_best",
                            "raw_hidden_joint_model": "raw_only_model",
                            "metadata_hidden_joint_model": "metadata_only_model",
                            "raw_plus_kmer_plus_metadata_hidden_joint_model": "raw_plus_kmer_plus_metadata_model",
                            "family_hidden_joint_model": "family_only_model",
                            "raw_family_hidden_joint_model": "shortcut_best",
                        }[model_name]
                        if args.export_predictions and model_name in prediction_models:
                            result, scores = fit_model_with_scores(matrix, labels, splits, c_grid, seed)
                            if result.get("status") == "ok" and scores is not None:
                                append_prediction_rows(
                                    all_prediction_rows,
                                    sample_ids=row_ids,
                                    labels=labels,
                                    splits=splits,
                                    scores=scores,
                                    task=task,
                                    checkpoint=checkpoint,
                                    seed=seed,
                                    model_name=model_name,
                                    layer=layer,
                                    source_artifact=str(feature_entry.get("path") or feature_entry.get("reason") or "in_memory_features"),
                                    manifest_hash=manifest_hash,
                                )
                        else:
                            result = fit_model(matrix, labels, splits, c_grid, seed, task, model_name)
                        all_metric_rows.append(
                            model_result_row(
                                result,
                                task=task,
                                capability_role=capability_role,
                                checkpoint=checkpoint,
                                layer=layer,
                                seed=seed,
                                model_name=model_name,
                                comparator_model=comparator,
                                shortcut_baselines=shortcut,
                                family_corr=family_corr,
                                group_status=group_status,
                                group_result=group_result,
                            )
                        )
            del features_by_layer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_metrics(out_dir / "capability_probe_metrics.csv", all_metric_rows)
    if args.export_predictions:
        prediction_path = Path(args.prediction_output) if args.prediction_output else out_dir / "capability_probe_predictions.csv"
        write_prediction_rows(prediction_path, all_prediction_rows)
        print(f"[cap-probe] wrote {len(all_prediction_rows)} prediction rows to {prediction_path}")
    write_json(out_dir / "capability_probe_signature.json", signature)
    write_json(
        out_dir / "capability_feature_cache_manifest.json",
        {
            "created_at": now(),
            "task": capability_probe_task_name(out_dir),
            "dataset_manifest": args.dataset_manifest,
            "dataset_audit": args.dataset_audit,
            "checkpoint_manifest": args.checkpoint_manifest,
            "task3_context": TASK3_CONTEXT,
            "layers": layers,
            "seeds": seeds,
            "c_grid": c_grid,
            "save_feature_cache": args.save_feature_cache,
            "dataset_lineage": dataset_lineage,
            "run_signature": signature,
            "entries": feature_entries,
            "metric_rows": len(all_metric_rows),
        },
    )
    write_probe_run_metadata(
        args=args,
        out_dir=out_dir,
        signature=signature,
        dataset_lineage=dataset_lineage,
        checkpoints=checkpoints,
        layers=layers,
        seeds=seeds,
        c_grid=c_grid,
        feature_entries=feature_entries,
        metric_row_count=len(all_metric_rows),
        prediction_row_count=len(all_prediction_rows),
        dry_run=False,
    )
    print(f"[cap-probe] wrote {len(all_metric_rows)} metric rows to {out_dir / 'capability_probe_metrics.csv'}")


if __name__ == "__main__":
    main()
