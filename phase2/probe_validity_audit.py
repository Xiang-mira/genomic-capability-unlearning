"""Minimum probe-validity audit for Phase 2 internal targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.eval_unlearn import parse_layers, separability
from phase2.run_metadata import file_sha256, git_info, stable_hash


NUCLEOTIDES = "ACGT"
GROUP_CANDIDATES = [
    "source",
    "category",
    "higher_group",
    "group_id",
    "similarity_cluster_id",
    "sample_id",
    "accession",
    "tax_id",
    "genus",
    "species",
    "virus_tax_id",
    "virus_name",
    "host_tax_id",
    "host_name",
    "family",
]

EXCLUDED_METADATA_FIELDS = {
    "id",
    "sample_id",
    "feature_cache_key",
    "label",
    "split",
    "split_random",
    "source_split",
    "category_split",
    "similarity_split",
    "sequence",
    "length",
}
EXCLUDED_METADATA_PREFIXES = ("matched_",)
MANDATORY_SPLIT_LEAKAGE_FIELDS = (
    "matched_pair_id",
    "sample_id",
    "feature_cache_key",
    "similarity_cluster_id",
)
HARD_STOP_ACTIONS = {
    "pause_fix_probe_pipeline",
    "pause_fix_split_or_cache",
    "pause_fix_split_leakage",
    "pause_fix_feature_matrix",
    "pause_fix_label_balance",
}


def parse_list(spec: str, cast=str) -> list:
    return [cast(part.strip()) for part in spec.split(",") if part.strip()]


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def safe_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(log_loss(labels, probs, labels=[0, 1]))
    except ValueError:
        return float("nan")


def bootstrap_auc_ci(labels: np.ndarray, probs: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    if n_bootstrap <= 0 or labels.size < 2 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, labels.size, size=labels.size)
        if len(np.unique(labels[idx])) < 2:
            continue
        values.append(safe_auroc(labels[idx], probs[idx]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def sequence_stats(seq: str) -> dict[str, float]:
    seq = (seq or "").upper()
    length = len(seq)
    denom = max(length, 1)
    counts = Counter(seq)
    result = {
        "length": float(length),
        "gc": float((counts["G"] + counts["C"]) / denom),
        "other_freq": float(sum(count for base, count in counts.items() if base not in "ACGTN") / denom),
    }
    for base in "ACGTN":
        result[f"{base}_freq"] = float(counts[base] / denom)
    for a in NUCLEOTIDES:
        result[f"1mer_{a}"] = float(counts[a] / denom)
    two_total = max(length - 1, 1)
    two_counts = Counter(seq[i : i + 2] for i in range(max(length - 1, 0)))
    for a in NUCLEOTIDES:
        for b in NUCLEOTIDES:
            result[f"2mer_{a}{b}"] = float(two_counts[a + b] / two_total)
    return result


def build_numeric_features(rows: list[dict[str, str]]):
    numeric_dicts = [sequence_stats(row.get("sequence", "")) for row in rows]
    numeric_keys = sorted({key for item in numeric_dicts for key in item})
    numeric = np.array([[item.get(key, 0.0) for key in numeric_keys] for item in numeric_dicts], dtype=np.float32)
    return csr_matrix(numeric), {
        "numeric_feature_count": len(numeric_keys),
        "numeric_features": numeric_keys,
    }


def build_metadata_features(rows: list[dict[str, str]]):
    metadata_dicts = []
    for row in rows:
        features = {}
        for key, value in row.items():
            if key in EXCLUDED_METADATA_FIELDS:
                continue
            if any(key.startswith(prefix) for prefix in EXCLUDED_METADATA_PREFIXES):
                continue
            if value:
                features[f"{key}={value}"] = 1
        metadata_dicts.append(features)
    metadata_vectorizer = DictVectorizer(sparse=True)
    metadata = metadata_vectorizer.fit_transform(metadata_dicts)
    return metadata, {
        "metadata_feature_count": len(metadata_vectorizer.feature_names_),
        "metadata_feature_names": list(metadata_vectorizer.feature_names_),
        "excluded_metadata_fields": sorted(EXCLUDED_METADATA_FIELDS),
        "excluded_metadata_prefixes": list(EXCLUDED_METADATA_PREFIXES),
        "feature_exclusion_hash": stable_hash(
            {
                "fields": sorted(EXCLUDED_METADATA_FIELDS),
                "prefixes": list(EXCLUDED_METADATA_PREFIXES),
            }
        ),
    }


def build_raw_features(rows: list[dict[str, str]]):
    return build_numeric_features(rows)


def build_raw_plus_metadata_features(rows: list[dict[str, str]]):
    numeric, numeric_info = build_numeric_features(rows)
    metadata, metadata_info = build_metadata_features(rows)
    return hstack([numeric, metadata], format="csr"), {
        **numeric_info,
        **metadata_info,
        "combined_feature_count": int(numeric.shape[1] + metadata.shape[1]),
    }


def build_kmer_features(rows: list[dict[str, str]], kmer_min: int, kmer_max: int):
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(kmer_min, kmer_max),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    sequences = [(row.get("sequence") or "").upper() for row in rows]
    return vectorizer.transform(sequences), {"n_features": 2**18, "kmer_min": kmer_min, "kmer_max": kmer_max}


def masks_for_splits(splits: np.ndarray) -> dict[str, np.ndarray]:
    return {name: splits == name for name in ("train", "val", "test")}


def fit_eval_logistic(
    matrix,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: list[float],
    *,
    seed: int,
    n_bootstrap: int,
    target: str,
    baseline: str,
) -> dict[str, object]:
    masks = masks_for_splits(splits)
    result = {
        "target": target,
        "baseline": baseline,
        "seed": seed,
        "status": "ok",
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
    }
    if masks["train"].sum() == 0 or len(np.unique(labels[masks["train"]])) < 2:
        result["status"] = "missing_or_single_class_train"
        return result

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
        auc = safe_auroc(labels[selection_mask], probs)
        score = separability(auc)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best = (c_value, clf)

    if best is None:
        result["status"] = "fit_failed"
        return result

    best_c, clf = best
    result["best_c"] = float(best_c)
    for split in ("train", "val", "test"):
        mask = masks[split]
        if mask.sum() == 0:
            result[f"{split}_status"] = "missing"
            continue
        if len(np.unique(labels[mask])) < 2:
            result[f"{split}_status"] = "single_class"
            continue
        probs = clf.predict_proba(x_all[mask])[:, 1]
        auc = safe_auroc(labels[mask], probs)
        split_offset = {"train": 101, "val": 202, "test": 303}[split]
        ci_low, ci_high = bootstrap_auc_ci(labels[mask], probs, n_bootstrap, seed + split_offset)
        result[f"{split}_auroc"] = auc
        result[f"{split}_separability"] = separability(auc)
        result[f"{split}_log_loss"] = safe_log_loss(labels[mask], probs)
        result[f"{split}_auroc_ci_low"] = ci_low
        result[f"{split}_auroc_ci_high"] = ci_high
    return result


def permutation_eval(
    matrix,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: list[float],
    seeds: Iterable[int],
    *,
    n_bootstrap: int,
    target: str,
    baseline: str,
) -> list[dict[str, object]]:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        permuted = labels.copy()
        rng.shuffle(permuted)
        row = fit_eval_logistic(
            matrix,
            permuted,
            splits,
            c_grid,
            seed=seed,
            n_bootstrap=n_bootstrap,
            target=target,
            baseline=f"permutation_{baseline}",
        )
        rows.append(row)
    return rows


def duplicate_audit(rows: list[dict[str, str]], target: str) -> list[dict[str, object]]:
    output = []
    for key_name, getter in (
        ("id", lambda row: row.get("id", "")),
        ("sequence", lambda row: row.get("sequence", "")),
    ):
        buckets = defaultdict(list)
        for idx, row in enumerate(rows):
            buckets[getter(row)].append((idx, row))
        duplicate_values = [items for items in buckets.values() if len(items) > 1]
        cross_split = []
        conflicting_label = []
        for items in duplicate_values:
            splits = {item[1].get("split", "") for item in items}
            labels = {item[1].get("label", "") for item in items}
            if len(splits) > 1:
                cross_split.append(items)
            if len(labels) > 1:
                conflicting_label.append(items)
        output.append(
            {
                "target": target,
                "duplicate_type": key_name,
                "duplicate_values": len(duplicate_values),
                "duplicate_rows": sum(len(items) for items in duplicate_values),
                "cross_split_duplicate_values": len(cross_split),
                "conflicting_label_values": len(conflicting_label),
                "failure": bool(cross_split or conflicting_label),
            }
        )
    return output


def split_leakage_audit(rows: list[dict[str, str]], target: str) -> list[dict[str, object]]:
    output = []
    for field in MANDATORY_SPLIT_LEAKAGE_FIELDS:
        values = defaultdict(set)
        row_count = defaultdict(int)
        for row in rows:
            value = row.get(field, "")
            if not value:
                continue
            values[value].add(row.get("split", ""))
            row_count[value] += 1
        cross_values = [value for value, splits in values.items() if len(splits) > 1]
        cross_rows = sum(row_count[value] for value in cross_values)
        output.append(
            {
                "target": target,
                "field": field,
                "available": bool(values),
                "n_values": len(values),
                "cross_split_value_count": len(cross_values),
                "cross_split_row_count": cross_rows,
                "failure": len(cross_values) > 0,
                "status": "ok" if values and not cross_values else ("unavailable" if not values else "cross_split_leakage"),
            }
        )
    return output


def cache_mapping_audit(rows: list[dict[str, str]], manifest_path: str, target: str) -> list[dict[str, object]]:
    feature_dir = Path(manifest_path).parent / "features"
    output = []
    if not feature_dir.exists():
        return [
            {
                "target": target,
                "feature_dir": str(feature_dir),
                "status": "cache_missing",
                "failure": False,
            }
        ]
    manifest_ids = np.array([row.get("id", "") for row in rows])
    manifest_labels = np.array([int(row.get("label", 0)) for row in rows])
    status = {
        "target": target,
        "feature_dir": str(feature_dir),
        "status": "ok",
        "failure": False,
        "manifest_rows": len(rows),
    }
    ids_path = feature_dir / "ids.npy"
    labels_path = feature_dir / "labels.npy"
    if ids_path.exists():
        ids = np.load(ids_path, allow_pickle=True)
        status["ids_rows"] = int(ids.shape[0])
        status["ids_match_manifest_order"] = bool(ids.shape[0] == manifest_ids.shape[0] and np.all(ids == manifest_ids))
        status["failure"] = status["failure"] or not status["ids_match_manifest_order"]
    else:
        status["ids_rows"] = "missing"
        status["failure"] = True
    if labels_path.exists():
        labels = np.load(labels_path, allow_pickle=True)
        status["labels_rows"] = int(labels.shape[0])
        status["labels_match_manifest_order"] = bool(
            labels.shape[0] == manifest_labels.shape[0] and np.all(labels.astype(int) == manifest_labels)
        )
        status["failure"] = status["failure"] or not status["labels_match_manifest_order"]
    else:
        status["labels_rows"] = "missing"
        status["failure"] = True

    for layer_dir in sorted(feature_dir.glob("layer_*")):
        if not layer_dir.is_dir():
            continue
        total_rows = 0
        for chunk in sorted(layer_dir.glob("chunk_*.npy")):
            total_rows += int(np.load(chunk, mmap_mode="r").shape[0])
        output.append(
            {
                "target": target,
                "feature_dir": str(feature_dir),
                "layer": layer_dir.name,
                "chunk_rows": total_rows,
                "manifest_rows": len(rows),
                "status": "ok" if total_rows == len(rows) else "row_mismatch",
                "failure": total_rows != len(rows),
            }
        )
        status["failure"] = status["failure"] or total_rows != len(rows)
    output.insert(0, status)
    return output


def feature_matrix_audit(matrix, labels: np.ndarray, splits: np.ndarray, target: str, baseline: str) -> dict[str, object]:
    train_mask = splits == "train"
    train = matrix[train_mask]
    total_features = int(matrix.shape[1])
    train_rows = int(train.shape[0])
    varying_nonzero_cols = 0
    if total_features == 0:
        return {
            "target": target,
            "baseline": baseline,
            "total_features": total_features,
            "train_rows": train_rows,
            "status": "no_features",
            "failure": True,
        }
    train_nonzero_cols = int(np.asarray(train.getnnz(axis=0) > 0).sum()) if train_rows else 0
    labels_train = labels[train_mask]
    status = "ok"
    failure = False
    if train_nonzero_cols == 0:
        status = "all_zero_train_columns"
        failure = True
    elif train_rows and len(np.unique(labels_train)) >= 2 and train_rows >= 2:
        train_dense_mean = np.asarray(train.mean(axis=0)).ravel()
        nonzero_mask = np.asarray(train.getnnz(axis=0)).ravel() > 0
        if np.any(nonzero_mask):
            try:
                col_sq_mean = np.asarray(train.power(2).mean(axis=0)).ravel()
                variance = np.maximum(col_sq_mean - np.square(train_dense_mean), 0.0)
                varying_nonzero_cols = int(np.sum((variance > 1e-12) & nonzero_mask))
            except Exception:
                varying_nonzero_cols = train_nonzero_cols
        else:
            varying_nonzero_cols = 0
        if varying_nonzero_cols == 0:
            status = "constant_train_columns"
            failure = True
    else:
        varying_nonzero_cols = train_nonzero_cols
    return {
        "target": target,
        "baseline": baseline,
        "total_features": total_features,
        "train_rows": train_rows,
        "train_nonzero_cols": train_nonzero_cols,
        "train_varying_nonzero_cols": varying_nonzero_cols,
        "status": status,
        "failure": failure,
    }


def label_balance_audit(rows: list[dict[str, str]], target: str, min_fraction: float = 0.10) -> list[dict[str, object]]:
    output = []
    for split in ("all", "train", "val", "test"):
        current = rows if split == "all" else [row for row in rows if row.get("split", "") == split]
        counts = Counter(row.get("label", "") for row in current)
        total = len(current)
        minority_fraction = None
        failure = False
        status = "ok"
        if total == 0:
            status = "missing_split"
            failure = True
        elif len([label for label, count in counts.items() if count > 0]) < 2:
            status = "single_class"
            failure = True
        else:
            minority_fraction = min(counts.values()) / total
            if minority_fraction < min_fraction:
                status = "severe_imbalance"
                failure = True
        output.append(
            {
                "target": target,
                "split": split,
                "n_rows": total,
                "label_counts": json.dumps(dict(counts), sort_keys=True),
                "minority_fraction": minority_fraction,
                "threshold": min_fraction,
                "status": status,
                "failure": failure,
            }
        )
    return output


def group_split_feasibility(rows: list[dict[str, str]], target: str) -> list[dict[str, object]]:
    output = []
    total = max(len(rows), 1)
    for field in GROUP_CANDIDATES:
        values = [row.get(field, "") for row in rows if row.get(field, "")]
        if not values:
            continue
        counts = Counter(values)
        group_labels = defaultdict(set)
        group_splits = defaultdict(set)
        for row in rows:
            value = row.get(field, "")
            if not value:
                continue
            group_labels[value].add(row.get("label", ""))
            group_splits[value].add(row.get("split", ""))
        output.append(
            {
                "target": target,
                "group_field": field,
                "n_groups": len(counts),
                "largest_group_fraction": max(counts.values()) / total,
                "groups_with_both_labels": sum(1 for labels in group_labels.values() if len(labels) > 1),
                "groups_crossing_splits": sum(1 for splits in group_splits.values() if len(splits) > 1),
                "feasible_basic_group_heldout": len(counts) >= 3 and max(counts.values()) / total < 0.8,
            }
        )
    return output


def target_specs(config_path: str) -> list[dict[str, object]]:
    with open(config_path) as f:
        payload = json.load(f)
    specs = []
    for entry in payload.get("targets", []):
        specs.append(
            {
                "name": entry["name"],
                "manifest": entry["manifest"],
                "probe_dir": entry.get("probe_dir", ""),
                "layers": parse_layers(entry.get("layers", "5-9")),
            }
        )
    if not specs:
        raise ValueError(f"No targets found in {config_path}")
    return specs


def probe_validity_signature(
    *,
    config_path: Path,
    target_hashes: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, object]:
    git = git_info()
    audit_config = {
        "seeds": args.seeds,
        "c_grid": args.c_grid,
        "kmer_min": args.kmer_min,
        "kmer_max": args.kmer_max,
        "n_bootstrap": args.n_bootstrap,
    }
    script_paths = [
        "phase2/probe_validity_audit.py",
        "phase2/build_capability_probe_dataset.py",
        "phase2/eval_capability_probe.py",
        "phase2/summarize_identity_capability_calibration.py",
    ]
    script_hashes = {
        path: file_sha256(path)
        for path in script_paths
        if Path(path).exists()
    }
    return {
        "task": "probe_validity_audit",
        "git_commit_hash": git.get("commit_hash", ""),
        "audit_config": audit_config,
        "audit_config_hash": stable_hash(audit_config),
        "feature_exclusion_hash": stable_hash(
            {
                "fields": sorted(EXCLUDED_METADATA_FIELDS),
                "prefixes": list(EXCLUDED_METADATA_PREFIXES),
            }
        ),
        "internal_target_config": str(config_path),
        "internal_target_config_hash": file_sha256(config_path),
        "target_manifest_hashes": target_hashes,
        "script_hashes": script_hashes,
        "script_version": stable_hash(script_hashes),
    }


def decision_from_rows(
    permutation_rows,
    duplicate_rows,
    cache_rows,
    split_leak_rows,
    feature_rows,
    label_balance_rows,
    baseline_rows,
    group_rows,
) -> dict[str, object]:
    permutation_failures = [
        row for row in permutation_rows if row.get("test_separability") is not None and row.get("test_separability", 0) >= 0.60
    ]
    permutation_warnings = [
        row for row in permutation_rows if row.get("test_separability") is not None and 0.55 <= row.get("test_separability", 0) < 0.60
    ]
    duplicate_failures = [row for row in duplicate_rows if row.get("failure")]
    cache_failures = [row for row in cache_rows if row.get("failure")]
    split_leak_failures = [row for row in split_leak_rows if row.get("failure")]
    feature_failures = [row for row in feature_rows if row.get("failure")]
    label_balance_failures = [row for row in label_balance_rows if row.get("failure")]
    strong_identity = [
        row for row in baseline_rows if row.get("test_separability") is not None and row.get("test_separability", 0) >= 0.90
    ]
    identity = [
        row for row in baseline_rows if row.get("test_separability") is not None and row.get("test_separability", 0) >= 0.80
    ]
    feasible_group_rows = [row for row in group_rows if row.get("feasible_basic_group_heldout")]
    group_isolation_failures = [row for row in feasible_group_rows if int(row.get("groups_crossing_splits", 0) or 0) > 0]
    missing_group_isolation = not feasible_group_rows
    if permutation_failures:
        action = "pause_fix_probe_pipeline"
    elif duplicate_failures or cache_failures:
        action = "pause_fix_split_or_cache"
    elif split_leak_failures:
        action = "pause_fix_split_leakage"
    elif feature_failures:
        action = "pause_fix_feature_matrix"
    elif label_balance_failures:
        action = "pause_fix_label_balance"
    elif strong_identity:
        action = "continue_with_strong_identity_confound_risk"
    elif identity:
        action = "continue_with_identity_confound_risk"
    elif group_isolation_failures or missing_group_isolation:
        action = "continue_group_isolation_unavailable"
    else:
        action = "continue"
    hard_stop = action in HARD_STOP_ACTIONS
    hard_stop_reasons = []
    if permutation_failures:
        hard_stop_reasons.append("permutation_abnormal")
    if duplicate_failures:
        hard_stop_reasons.append("duplicate_leakage")
    if cache_failures:
        hard_stop_reasons.append("cache_mapping_mismatch")
    if split_leak_failures:
        hard_stop_reasons.append("split_leakage")
    if feature_failures:
        hard_stop_reasons.append("empty_or_constant_feature_columns")
    if label_balance_failures:
        hard_stop_reasons.append("label_imbalance_too_severe")
    return {
        "action": action,
        "hard_stop": hard_stop,
        "hard_stop_reasons": hard_stop_reasons,
        "permutation_failures": len(permutation_failures),
        "permutation_warnings": len(permutation_warnings),
        "duplicate_failures": len(duplicate_failures),
        "cache_failures": len(cache_failures),
        "split_leak_failures": len(split_leak_failures),
        "feature_failures": len(feature_failures),
        "label_balance_failures": len(label_balance_failures),
        "identity_confound_rows": len(identity),
        "strong_identity_confound_rows": len(strong_identity),
        "group_isolation_failures": len(group_isolation_failures),
        "missing_group_isolation": missing_group_isolation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-target-config", default="phase2/internal_eval_targets_coro0_10.json")
    parser.add_argument("--out-dir", default="data/phase2/audits/task0_3_20260713/probe_validity")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1.0,10.0")
    parser.add_argument("--kmer-min", type=int, default=3)
    parser.add_argument("--kmer-max", type=int, default=6)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args()

    config_path = Path(args.internal_target_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Required corrected target config missing: {config_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_list(args.seeds, int)
    c_grid = parse_list(args.c_grid, float)

    raw_rows = []
    metadata_rows = []
    raw_plus_metadata_rows = []
    kmer_rows = []
    raw_plus_kmer_rows = []
    raw_plus_kmer_plus_metadata_rows = []
    permutation_rows = []
    duplicate_rows = []
    cache_rows = []
    split_leak_rows = []
    feature_rows = []
    label_balance_rows = []
    group_rows = []
    audit_targets = {}
    target_hashes = {}

    for spec in target_specs(str(config_path)):
        target = str(spec["name"])
        manifest = str(spec["manifest"])
        rows = read_rows(manifest)
        target_hashes[manifest] = file_sha256(manifest)
        labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
        splits = np.array([row["split"] for row in rows])

        raw_matrix, raw_info = build_raw_features(rows)
        metadata_matrix, metadata_info = build_metadata_features(rows)
        raw_plus_metadata_matrix, raw_plus_metadata_info = build_raw_plus_metadata_features(rows)
        kmer_matrix, kmer_info = build_kmer_features(rows, args.kmer_min, args.kmer_max)
        raw_plus_kmer_matrix = hstack([raw_matrix, kmer_matrix], format="csr")
        raw_plus_kmer_plus_metadata_matrix = hstack([raw_matrix, metadata_matrix, kmer_matrix], format="csr")
        raw_eval = fit_eval_logistic(
            raw_matrix,
            labels,
            splits,
            c_grid,
            seed=seeds[0],
            n_bootstrap=args.n_bootstrap,
            target=target,
            baseline="raw_only_model",
        )
        metadata_eval = fit_eval_logistic(
            metadata_matrix,
            labels,
            splits,
            c_grid,
            seed=seeds[0],
            n_bootstrap=args.n_bootstrap,
            target=target,
            baseline="metadata_only_model",
        )
        raw_plus_metadata_eval = fit_eval_logistic(
            raw_plus_metadata_matrix,
            labels,
            splits,
            c_grid,
            seed=seeds[0],
            n_bootstrap=args.n_bootstrap,
            target=target,
            baseline="raw_plus_metadata_model",
        )
        kmer_eval = fit_eval_logistic(
            kmer_matrix,
            labels,
            splits,
            c_grid,
            seed=seeds[0],
            n_bootstrap=args.n_bootstrap,
            target=target,
            baseline=f"kmer_{args.kmer_min}_{args.kmer_max}",
        )
        raw_plus_kmer_eval = fit_eval_logistic(
            raw_plus_kmer_matrix,
            labels,
            splits,
            c_grid,
            seed=seeds[0],
            n_bootstrap=args.n_bootstrap,
            target=target,
            baseline="raw_plus_kmer_model",
        )
        raw_plus_kmer_plus_metadata_eval = fit_eval_logistic(
            raw_plus_kmer_plus_metadata_matrix,
            labels,
            splits,
            c_grid,
            seed=seeds[0],
            n_bootstrap=args.n_bootstrap,
            target=target,
            baseline="raw_plus_kmer_plus_metadata_model",
        )
        raw_eval.update(raw_info)
        metadata_eval.update(metadata_info)
        raw_plus_metadata_eval.update(raw_plus_metadata_info)
        kmer_eval.update(kmer_info)
        raw_rows.append(raw_eval)
        metadata_rows.append(metadata_eval)
        raw_plus_metadata_rows.append(raw_plus_metadata_eval)
        kmer_rows.append(kmer_eval)
        raw_plus_kmer_rows.append(raw_plus_kmer_eval)
        raw_plus_kmer_plus_metadata_rows.append(raw_plus_kmer_plus_metadata_eval)

        permutation_rows.extend(
            permutation_eval(
                raw_matrix,
                labels,
                splits,
                c_grid,
                seeds,
                n_bootstrap=args.n_bootstrap,
                target=target,
                baseline="raw_only_model",
            )
        )
        permutation_rows.extend(
            permutation_eval(
                metadata_matrix,
                labels,
                splits,
                c_grid,
                seeds,
                n_bootstrap=args.n_bootstrap,
                target=target,
                baseline="metadata_only_model",
            )
        )
        permutation_rows.extend(
            permutation_eval(
                kmer_matrix,
                labels,
                splits,
                c_grid,
                seeds,
                n_bootstrap=args.n_bootstrap,
                target=target,
                baseline=f"kmer_{args.kmer_min}_{args.kmer_max}",
            )
        )
        permutation_rows.extend(
            permutation_eval(
                raw_plus_kmer_plus_metadata_matrix,
                labels,
                splits,
                c_grid,
                seeds,
                n_bootstrap=args.n_bootstrap,
                target=target,
                baseline="raw_plus_kmer_plus_metadata_model",
            )
        )
        duplicate_rows.extend(duplicate_audit(rows, target))
        cache_rows.extend(cache_mapping_audit(rows, manifest, target))
        split_leak_rows.extend(split_leakage_audit(rows, target))
        label_balance_rows.extend(label_balance_audit(rows, target))
        group_rows.extend(group_split_feasibility(rows, target))
        feature_rows.extend(
            [
                feature_matrix_audit(raw_matrix, labels, splits, target, "raw_only_model"),
                feature_matrix_audit(metadata_matrix, labels, splits, target, "metadata_only_model"),
                feature_matrix_audit(raw_plus_metadata_matrix, labels, splits, target, "raw_plus_metadata_model"),
                feature_matrix_audit(kmer_matrix, labels, splits, target, "kmer_only_model"),
                feature_matrix_audit(raw_plus_kmer_matrix, labels, splits, target, "raw_plus_kmer_model"),
                feature_matrix_audit(
                    raw_plus_kmer_plus_metadata_matrix,
                    labels,
                    splits,
                    target,
                    "raw_plus_kmer_plus_metadata_model",
                ),
            ]
        )
        audit_targets[target] = {
            "manifest": manifest,
            "n_rows": len(rows),
            "split_counts": dict(Counter(splits)),
            "label_counts": {str(key): int(value) for key, value in Counter(labels).items()},
        }

    write_csv(out_dir / "raw_covariate_baseline.csv", raw_rows)
    write_csv(out_dir / "raw_numeric_baseline.csv", raw_rows)
    write_csv(out_dir / "metadata_baseline.csv", metadata_rows)
    write_csv(out_dir / "raw_plus_metadata_baseline.csv", raw_plus_metadata_rows)
    write_csv(out_dir / "kmer_baseline.csv", kmer_rows)
    write_csv(out_dir / "raw_plus_kmer_baseline.csv", raw_plus_kmer_rows)
    write_csv(out_dir / "raw_plus_kmer_plus_metadata_baseline.csv", raw_plus_kmer_plus_metadata_rows)
    write_csv(out_dir / "label_permutation.csv", permutation_rows)
    write_csv(out_dir / "duplicate_audit.csv", duplicate_rows)
    write_csv(out_dir / "cache_mapping_audit.csv", cache_rows)
    write_csv(out_dir / "split_leakage_audit.csv", split_leak_rows)
    write_csv(out_dir / "feature_matrix_audit.csv", feature_rows)
    write_csv(out_dir / "label_balance_audit.csv", label_balance_rows)
    write_csv(out_dir / "group_split_feasibility.csv", group_rows)

    baseline_rows = (
        raw_rows
        + metadata_rows
        + raw_plus_metadata_rows
        + kmer_rows
        + raw_plus_kmer_rows
        + raw_plus_kmer_plus_metadata_rows
    )
    decision = decision_from_rows(
        permutation_rows,
        duplicate_rows,
        cache_rows,
        split_leak_rows,
        feature_rows,
        label_balance_rows,
        baseline_rows,
        group_rows,
    )
    signature = probe_validity_signature(config_path=config_path, target_hashes=target_hashes, args=args)
    payload = {
        "internal_target_config": str(config_path),
        "targets": audit_targets,
        "raw_covariate_baseline": raw_rows,
        "raw_numeric_baseline": raw_rows,
        "metadata_baseline": metadata_rows,
        "raw_plus_metadata_baseline": raw_plus_metadata_rows,
        "kmer_baseline": kmer_rows,
        "raw_plus_kmer_baseline": raw_plus_kmer_rows,
        "raw_plus_kmer_plus_metadata_baseline": raw_plus_kmer_plus_metadata_rows,
        "label_permutation": permutation_rows,
        "exact_duplicate_audit": [row for row in duplicate_rows if row["duplicate_type"] == "sequence"],
        "id_duplicate_audit": [row for row in duplicate_rows if row["duplicate_type"] == "id"],
        "cache_mapping_audit": cache_rows,
        "split_leakage_audit": split_leak_rows,
        "feature_matrix_audit": feature_rows,
        "label_balance_audit": label_balance_rows,
        "group_split_feasibility": group_rows,
        "matched_result": {
            "status": "deferred_to_task4_or_task8",
            "reason": "minimum_audit_only",
        },
        "conditional_incremental_result": {
            "status": "deferred_to_task4_or_task8",
            "reason": "minimum_audit_only",
        },
        "decision": decision,
        "run_signature": signature,
    }
    (out_dir / "probe_validity_audit.json").write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n")
    (out_dir / "probe_validity_signature.json").write_text(json.dumps(sanitize(signature), indent=2, sort_keys=True) + "\n")

    summary = f"""# Probe Validity Minimum Audit

## Files Produced

- probe_validity_audit.json
- probe_validity_signature.json
- raw_covariate_baseline.csv
- raw_numeric_baseline.csv
- metadata_baseline.csv
- raw_plus_metadata_baseline.csv
- kmer_baseline.csv
- raw_plus_kmer_baseline.csv
- raw_plus_kmer_plus_metadata_baseline.csv
- label_permutation.csv
- duplicate_audit.csv
- cache_mapping_audit.csv
- split_leakage_audit.csv
- feature_matrix_audit.csv
- label_balance_audit.csv
- group_split_feasibility.csv
- probe_validity_summary.md

## Decision

- Action: {decision['action']}
- Hard stop: {decision['hard_stop']}
- Hard stop reasons: {', '.join(decision['hard_stop_reasons']) if decision['hard_stop_reasons'] else 'none'}
- Permutation failures: {decision['permutation_failures']}
- Duplicate failures: {decision['duplicate_failures']}
- Cache failures: {decision['cache_failures']}
- Split leak failures: {decision['split_leak_failures']}
- Feature failures: {decision['feature_failures']}
- Label balance failures: {decision['label_balance_failures']}
- Group isolation failures: {decision['group_isolation_failures']}
- Missing group isolation: {decision['missing_group_isolation']}
- Identity confound rows: {decision['identity_confound_rows']}
- Strong identity confound rows: {decision['strong_identity_confound_rows']}

matched_result and conditional_incremental_result are deferred to Task 4/8 as planned.
"""
    (out_dir / "probe_validity_summary.md").write_text(summary)
    print(f"[probe-validity] wrote reports to {out_dir}")
    print(f"[probe-validity] decision={decision['action']}")


if __name__ == "__main__":
    main()
