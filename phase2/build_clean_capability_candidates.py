"""Build deconfounded capability-gate candidates from the current Task 7-R manifest.

The current source/category fields are constant for the known HVUE tasks, so the
practical clean-gate route is matched hard negatives while preserving the
existing similarity-held-out split. This script constructs a small number of
candidate manifests, writes candidate-specific formal-task configs, and records
shortcut audits for later smoke evaluation.
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
from scipy.optimize import linear_sum_assignment


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.build_capability_probe_dataset import (
    STANDARD_MANIFEST_FIELDS,
    group_feasibility,
    numeric_stats,
    shortcut_audit,
    split_counts,
    write_formal_task_manifests,
    write_manifest,
)
from phase2.probe_validity_audit import (
    EXCLUDED_METADATA_FIELDS,
    EXCLUDED_METADATA_PREFIXES,
    build_kmer_features,
    sequence_stats,
)
from phase2.run_metadata import file_sha256, git_info, stable_hash
from phase2.run_task5a_identity_reaudit import TASK3_CONTEXT


MATCH_KEYS = [
    "length",
    "gc",
    "A_freq",
    "C_freq",
    "G_freq",
    "T_freq",
    "1mer_A",
    "1mer_C",
    "1mer_G",
    "1mer_T",
    "2mer_AA",
    "2mer_AC",
    "2mer_AG",
    "2mer_AT",
    "2mer_CA",
    "2mer_CC",
    "2mer_CG",
    "2mer_CT",
    "2mer_GA",
    "2mer_GC",
    "2mer_GG",
    "2mer_GT",
    "2mer_TA",
    "2mer_TC",
    "2mer_TG",
    "2mer_TT",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def distribution_summary(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    counts = Counter(row.get(field, "") or "missing" for row in rows)
    total = max(len(rows), 1)
    return {
        "field": field,
        "n_values": len(counts),
        "counts": dict(counts),
        "fractions": {key: float(value / total) for key, value in counts.items()},
    }


def numeric_distribution(rows: list[dict[str, str]], field: str) -> dict[str, float | None]:
    values = []
    for row in rows:
        raw = row.get(field, "")
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    if not values:
        return {"mean": None, "std": None, "median": None, "p90": None, "min": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def class_ratio(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row.get("label", "") for row in rows)
    pos = int(counts.get("1", 0))
    neg = int(counts.get("0", 0))
    return {
        "n_total": len(rows),
        "n_positive": pos,
        "n_negative": neg,
        "positive_negative_ratio": float(pos / neg) if neg else None,
        "label_counts": dict(counts),
    }


def cross_split_count(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    values = defaultdict(set)
    row_counts = Counter()
    for row in rows:
        value = row.get(field, "")
        if not value:
            continue
        values[value].add(row.get("split", ""))
        row_counts[value] += 1
    cross = [value for value, splits in values.items() if len(splits) > 1]
    return {
        "field": field,
        "available": bool(values),
        "n_values": len(values),
        "cross_split_count": len(cross),
        "cross_split_rows": int(sum(row_counts[value] for value in cross)),
    }


def composition_centroid_distance(rows: list[dict[str, str]]) -> float | None:
    pos_rows = [row for row in rows if row.get("label") == "1"]
    neg_rows = [row for row in rows if row.get("label") == "0"]
    if not pos_rows or not neg_rows:
        return None
    pos = np.stack([feature_vector(row) for row in pos_rows], axis=0)
    neg = np.stack([feature_vector(row) for row in neg_rows], axis=0)
    both = np.concatenate([pos, neg], axis=0)
    std = both.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    mean = both.mean(axis=0, keepdims=True)
    pos_center = ((pos - mean) / std).mean(axis=0)
    neg_center = ((neg - mean) / std).mean(axis=0)
    return float(np.linalg.norm(pos_center - neg_center))


def ngram_centroid_distance(rows: list[dict[str, str]]) -> float | None:
    pos_rows = [row for row in rows if row.get("label") == "1"]
    neg_rows = [row for row in rows if row.get("label") == "0"]
    if not pos_rows or not neg_rows:
        return None
    pos_matrix, _ = build_kmer_features(pos_rows, 3, 6)
    neg_matrix, _ = build_kmer_features(neg_rows, 3, 6)
    pos_center = np.asarray(pos_matrix.mean(axis=0)).ravel()
    neg_center = np.asarray(neg_matrix.mean(axis=0)).ravel()
    denom = np.linalg.norm(pos_center) * np.linalg.norm(neg_center)
    if denom < 1e-12:
        return None
    cosine = float(np.dot(pos_center, neg_center) / denom)
    return float(1.0 - cosine)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_signature(source_manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    git = git_info()
    build_config = {
        "source_manifest": str(source_manifest),
        "quantiles": args.quantiles,
        "seeds": args.seeds,
        "c_grid": args.c_grid,
        "n_bootstrap": args.n_bootstrap,
    }
    script_paths = [
        "phase2/build_clean_capability_candidates.py",
        "phase2/build_capability_probe_dataset.py",
        "phase2/probe_validity_audit.py",
        "phase2/eval_capability_probe.py",
    ]
    script_hashes = {
        path: file_sha256(path)
        for path in script_paths
        if Path(path).exists()
    }
    return {
        "task": "clean_capability_candidate_build",
        "git_commit_hash": git.get("commit_hash", ""),
        "source_manifest_hash": file_sha256(source_manifest),
        "build_config": build_config,
        "build_config_hash": stable_hash(build_config),
        "feature_exclusion_hash": stable_hash(
            {
                "fields": sorted(EXCLUDED_METADATA_FIELDS),
                "prefixes": list(EXCLUDED_METADATA_PREFIXES),
            }
        ),
        "script_hashes": script_hashes,
        "script_version": stable_hash(script_hashes),
    }


def feature_vector(row: dict[str, str]) -> np.ndarray:
    stats = sequence_stats(row.get("sequence", ""))
    return np.array([float(stats.get(key, 0.0)) for key in MATCH_KEYS], dtype=np.float32)


def standardized_distance_matrix(pos_rows: list[dict[str, str]], neg_rows: list[dict[str, str]]) -> np.ndarray:
    pos = np.stack([feature_vector(row) for row in pos_rows], axis=0)
    neg = np.stack([feature_vector(row) for row in neg_rows], axis=0)
    both = np.concatenate([pos, neg], axis=0)
    mean = both.mean(axis=0, keepdims=True)
    std = both.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    pos = (pos - mean) / std
    neg = (neg - mean) / std
    distances = np.sqrt(((pos[:, None, :] - neg[None, :, :]) ** 2).sum(axis=2))

    # Prefer rows that stay within the same similarity cluster when possible.
    for i, pos_row in enumerate(pos_rows):
        pos_cluster = pos_row.get("similarity_cluster_id", "")
        if not pos_cluster:
            continue
        for j, neg_row in enumerate(neg_rows):
            if neg_row.get("similarity_cluster_id", "") == pos_cluster:
                distances[i, j] *= 0.85
    return distances


def match_split_rows(
    rows: list[dict[str, str]],
    *,
    task: str,
    split: str,
    quantile: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    pos_rows = [row for row in rows if row.get("label") == "1"]
    neg_rows = [row for row in rows if row.get("label") == "0"]
    if not pos_rows or not neg_rows:
        return [], {
            "task": task,
            "split": split,
            "status": "missing_class",
            "selected_pairs": 0,
        }

    if len(pos_rows) > len(neg_rows):
        pos_rows, neg_rows = neg_rows, pos_rows
        anchor_label = "0"
    else:
        anchor_label = "1"

    distance = standardized_distance_matrix(pos_rows, neg_rows)
    row_idx, col_idx = linear_sum_assignment(distance)
    pair_distances = distance[row_idx, col_idx]
    keep_pairs = len(pair_distances)
    if 0.0 < quantile < 1.0:
        keep_pairs = max(1, int(math.floor(len(pair_distances) * quantile)))
    order = np.argsort(pair_distances)[:keep_pairs]

    matched_rows: list[dict[str, str]] = []
    kept_distances = []
    same_cluster_pairs = 0
    for rank, order_idx in enumerate(order):
        pos_idx = int(row_idx[order_idx])
        neg_idx = int(col_idx[order_idx])
        left = dict(pos_rows[pos_idx])
        right = dict(neg_rows[neg_idx])
        pair_id = f"{task}|{split}|pair_{rank:04d}"
        pair_distance = float(pair_distances[order_idx])
        kept_distances.append(pair_distance)
        if left.get("similarity_cluster_id", "") and left.get("similarity_cluster_id") == right.get("similarity_cluster_id"):
            same_cluster_pairs += 1

        for side, row in enumerate((left, right)):
            row["split"] = split
            row["matched_pair_id"] = pair_id
            row["matched_pair_rank"] = str(rank)
            row["matched_pair_distance"] = f"{pair_distance:.8f}"
            row["matched_anchor_label"] = anchor_label
            row["matched_pair_side"] = "anchor" if side == 0 else "match"
            matched_rows.append(row)

    summary = {
        "task": task,
        "split": split,
        "status": "ok",
        "candidate_quantile": quantile,
        "original_positive_rows": len([row for row in rows if row.get("label") == "1"]),
        "original_negative_rows": len([row for row in rows if row.get("label") == "0"]),
        "selected_pairs": keep_pairs,
        "selected_rows": len(matched_rows),
        "all_pair_distance": percentile_summary(pair_distances.tolist()),
        "matched_pair_distance": percentile_summary(kept_distances),
        "same_similarity_cluster_fraction": float(same_cluster_pairs / keep_pairs) if keep_pairs else None,
    }
    return matched_rows, summary


def build_candidate_rows(rows: list[dict[str, str]], quantile: float) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("task", ""), row.get("similarity_split", row.get("split", "train")))].append(row)
    for (task, split), items in sorted(grouped.items()):
        matched, summary = match_split_rows(items, task=task, split=split, quantile=quantile)
        summaries.append(summary)
        candidate_rows.extend(matched)
    candidate_rows.sort(
        key=lambda row: (
            row.get("task", ""),
            row.get("split", ""),
            row.get("label", ""),
            row.get("matched_pair_rank", ""),
            row.get("sample_id", ""),
        )
    )
    return candidate_rows, summaries


def candidate_name_for_quantile(quantile: float) -> str:
    if quantile >= 0.999:
        return "matched_all_pairs"
    return f"matched_top{int(round(quantile * 100)):02d}"


def candidate_audit(
    candidate_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    pair_summary: list[dict[str, Any]],
    *,
    task7r_source_manifest: str,
    formal_target_config: str,
    quantile: float,
    seeds: list[int],
    c_grid: list[float],
    n_bootstrap: int,
) -> dict[str, Any]:
    task_payloads = {}
    for task in sorted({row["task"] for row in candidate_rows}):
        rows = [row for row in candidate_rows if row["task"] == task]
        original_rows = [row for row in source_rows if row["task"] == task]
        shortcut = shortcut_audit(rows, task, seeds, c_grid, n_bootstrap)
        original_class = class_ratio(original_rows)
        matched_class = class_ratio(rows)
        task_pairs = [item for item in pair_summary if item.get("task") == task]
        pair_distances = {}
        for row in rows:
            pair_id = row.get("matched_pair_id", "")
            raw_distance = row.get("matched_pair_distance", "")
            if not pair_id or raw_distance in ("", None):
                continue
            pair_distances[pair_id] = float(raw_distance)
        matched_distance_values = sorted(pair_distances.values())
        task_payloads[task] = {
            "n_rows": len(rows),
            "n_total": matched_class["n_total"],
            "n_positive": matched_class["n_positive"],
            "n_negative": matched_class["n_negative"],
            "positive_negative_ratio": matched_class["positive_negative_ratio"],
            "label_counts": matched_class["label_counts"],
            "split_counts": split_counts(rows),
            "family_counts": dict(Counter(row.get("family", "") or "missing" for row in rows)),
            "group_counts": dict(Counter(row.get("group", "") or "missing" for row in rows)),
            "length_gc_composition_stats": numeric_stats(rows),
            "length_distribution": numeric_distribution(rows, "length"),
            "source_distribution": distribution_summary(rows, "source"),
            "category_distribution": distribution_summary(rows, "category"),
            "selection_bias_audit": {
                "before_matching": {
                    **original_class,
                    "source_distribution": distribution_summary(original_rows, "source"),
                    "category_distribution": distribution_summary(original_rows, "category"),
                    "length_distribution": numeric_distribution(original_rows, "length"),
                    "composition_distance": composition_centroid_distance(original_rows),
                    "ngram_distance": ngram_centroid_distance(original_rows),
                },
                "after_matching": {
                    **matched_class,
                    "source_distribution": distribution_summary(rows, "source"),
                    "category_distribution": distribution_summary(rows, "category"),
                    "length_distribution": numeric_distribution(rows, "length"),
                    "composition_distance": composition_centroid_distance(rows),
                    "ngram_distance": ngram_centroid_distance(rows),
                },
            },
            "matching_quality": {
                "per_split": task_pairs,
                "matched_distance_mean_median_p90": percentile_summary(matched_distance_values),
            },
            "split_leakage_audit": {
                "pair_id_cross_split_count": cross_split_count(rows, "matched_pair_id")["cross_split_count"],
                "sample_id_cross_split_count": cross_split_count(rows, "sample_id")["cross_split_count"],
                "feature_cache_key_cross_split_count": cross_split_count(rows, "feature_cache_key")["cross_split_count"],
                "similarity_cluster_cross_split_count": cross_split_count(rows, "similarity_cluster_id")["cross_split_count"],
                "details": [
                    cross_split_count(rows, "matched_pair_id"),
                    cross_split_count(rows, "sample_id"),
                    cross_split_count(rows, "feature_cache_key"),
                    cross_split_count(rows, "similarity_cluster_id"),
                ],
            },
            "group_feasibility": group_feasibility(rows),
            **shortcut,
        }
    return {
        "created_at": now(),
        "task": "task7_clean_capability_candidate",
        "candidate_type": "matched_hard_negative",
        "candidate_quantile": quantile,
        "task3_context": TASK3_CONTEXT,
        "task7r_source_manifest": task7r_source_manifest,
        "task7r_source_manifest_hash": file_sha256(Path(task7r_source_manifest)),
        "formal_split_column": "split",
        "formal_target_config": formal_target_config,
        "formal_target_config_hash": file_sha256(Path(formal_target_config)),
        "matching_summary": pair_summary,
        "tasks": task_payloads,
    }


def parse_list(spec: str, cast=float) -> list[Any]:
    return [cast(part.strip()) for part in spec.split(",") if part.strip()]


def build_skip_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    report = {"source_heldout": {}, "category_heldout": {}, "similarity_split_existing": {}}
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        for key, field in (("source_heldout", "source"), ("category_heldout", "category")):
            values = [row.get(field, "") for row in task_rows if row.get(field, "")]
            report[key][task] = {
                "field": field,
                "n_values": len(set(values)),
                "status": "skip_constant_or_missing" if len(set(values)) < 3 else "available",
            }
        report["similarity_split_existing"][task] = {
            "field": "similarity_cluster_id",
            "n_values": len({row.get("similarity_cluster_id", "") for row in task_rows if row.get("similarity_cluster_id", "")}),
            "status": "already_available",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="data/phase2/audits/task7r_capability_probe_20260714/capability_dataset_manifest.csv",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--quantiles", default="1.0,0.75,0.50")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source_manifest)
    quantiles = parse_list(args.quantiles, float)
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    c_grid = parse_list(args.c_grid, float)
    signature = build_signature(source_manifest, args)

    index = {
        "created_at": now(),
        "task": "task7_clean_candidate_builder",
        "source_manifest": str(source_manifest),
        "source_manifest_hash": file_sha256(source_manifest),
        "task3_context": TASK3_CONTEXT,
        "run_signature": signature,
        "skipped_candidate_reasons": build_skip_report(rows),
        "candidates": [],
    }

    for quantile in quantiles:
        name = candidate_name_for_quantile(quantile)
        candidate_dir = out_root / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows, pair_summary = build_candidate_rows(rows, quantile)
        write_manifest(candidate_dir / "capability_dataset_manifest.csv", candidate_rows, list(STANDARD_MANIFEST_FIELDS) + [
            "matched_pair_id",
            "matched_pair_rank",
            "matched_pair_distance",
            "matched_anchor_label",
            "matched_pair_side",
        ])
        config_path = write_formal_task_manifests(candidate_dir, candidate_rows, "split")
        audit = candidate_audit(
            candidate_rows,
            rows,
            pair_summary,
            task7r_source_manifest=str(source_manifest),
            formal_target_config=str(config_path),
            quantile=quantile,
            seeds=seeds,
            c_grid=c_grid,
            n_bootstrap=args.n_bootstrap,
        )
        write_json(candidate_dir / "capability_dataset_audit.json", audit)
        write_json(candidate_dir / "candidate_metadata.json", {
            "created_at": now(),
            "candidate_name": name,
            "candidate_quantile": quantile,
            "candidate_type": "matched_hard_negative",
            "rows": len(candidate_rows),
            "pair_summary": pair_summary,
            "run_signature": signature,
            "manifest_hash": file_sha256(candidate_dir / "capability_dataset_manifest.csv"),
            "audit_hash": file_sha256(candidate_dir / "capability_dataset_audit.json"),
        })
        index["candidates"].append(
            {
                "candidate_name": name,
                "candidate_quantile": quantile,
                "candidate_dir": str(candidate_dir),
                "manifest": str(candidate_dir / "capability_dataset_manifest.csv"),
                "dataset_audit": str(candidate_dir / "capability_dataset_audit.json"),
                "formal_target_config": str(config_path),
                "rows": len(candidate_rows),
            }
        )

    write_json(out_root / "candidate_index.json", index)
    write_json(out_root / "candidate_build_signature.json", signature)
    print(f"[clean-candidates] wrote {len(index['candidates'])} candidates to {out_root}")


if __name__ == "__main__":
    main()
