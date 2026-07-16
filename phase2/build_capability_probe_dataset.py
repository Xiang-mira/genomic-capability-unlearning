"""Build the Task 7 capability-probe dataset and shortcut audit."""
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
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.probe_validity_audit import (
    build_kmer_features,
    build_raw_features,
    build_raw_plus_metadata_features,
    fit_eval_logistic,
    safe_auroc,
    separability,
    sequence_stats,
)
from phase2.run_task5a_identity_reaudit import TASK3_CONTEXT


csv.field_size_limit(sys.maxsize)


PRIMARY_TASK = "hvue_human_transmissibility_coronaviridae"
AUX_TASK = "hvue_human_host_tropism"
STANDARD_MANIFEST_FIELDS = [
    "benchmark",
    "task",
    "capability_role",
    "split",
    "split_random",
    "source_split",
    "category_split",
    "similarity_split",
    "sequence",
    "label",
    "category",
    "source",
    "higher_group",
    "group_id",
    "similarity_cluster_id",
    "sample_id",
    "feature_cache_key",
    "family",
    "group",
    "id",
    "length",
    "gc",
    "A_freq",
    "C_freq",
    "G_freq",
    "T_freq",
    "N_freq",
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


def read_task_rows(path: Path, task: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task") == task:
                rows.append(row)
    return rows


def cap_split_label(rows: list[dict[str, str]], max_per_split_label: int, seed: int) -> list[dict[str, str]]:
    if max_per_split_label <= 0:
        return rows
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("split", ""), row.get("label", ""))].append(row)
    rng = np.random.default_rng(seed)
    capped: list[dict[str, str]] = []
    for key in sorted(buckets):
        recs = buckets[key]
        if len(recs) > max_per_split_label:
            idx = rng.choice(len(recs), size=max_per_split_label, replace=False)
            recs = [recs[int(i)] for i in idx]
        capped.extend(recs)
    capped.sort(key=lambda row: (row.get("task", ""), row.get("split", ""), row.get("label", ""), row.get("id", "")))
    return capped



def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def infer_category(row: dict[str, str], task: str) -> str:
    family = row.get("family", "")
    if family:
        return family
    parts = task.split("_")
    if "host" in parts and "tropism" in parts:
        return "host_tropism"
    if "transmissibility" in parts:
        return "transmissibility"
    if "pathogenicity" in parts:
        return "pathogenicity"
    return row.get("benchmark", "") or "unknown_category"


def infer_source(row: dict[str, str]) -> str:
    row_id = row.get("id", "")
    if "|" in row_id:
        return row_id.split("|", 1)[0]
    return row.get("benchmark", "") or "unknown_source"


def infer_higher_group(row: dict[str, str], category: str) -> str:
    return row.get("family", "") or row.get("group", "") or category


def similarity_cluster_id(sequence: str, buckets: int = 2048) -> str:
    seq = (sequence or "").upper()
    if len(seq) < 8:
        return f"sim_{stable_hash(seq) % buckets:04d}"
    kmers = {seq[idx : idx + 8] for idx in range(0, max(len(seq) - 7, 1), 4)}
    min_hash = min(stable_hash(kmer) for kmer in kmers) if kmers else stable_hash(seq)
    return f"sim_{min_hash % buckets:04d}"


def assign_group_split(rows: list[dict[str, str]], group_field: str) -> dict[str, str]:
    groups = sorted({row.get(group_field, "") or "missing" for row in rows})
    if len(groups) < 3:
        return {group: "train" for group in groups}
    ordered = sorted(groups, key=lambda value: stable_hash(f"{group_field}|{value}"))
    n_groups = len(ordered)
    n_train = max(1, int(round(n_groups * 0.60)))
    n_val = max(1, int(round(n_groups * 0.20)))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1
    split_map = {}
    for idx, group in enumerate(ordered):
        if idx < n_train:
            split_map[group] = "train"
        elif idx < n_train + n_val:
            split_map[group] = "val"
        else:
            split_map[group] = "test"
    return split_map


def add_standard_fields(rows: list[dict[str, str]], task: str, role: str) -> list[dict[str, str]]:
    enriched = []
    for idx, row in enumerate(rows):
        sequence = row.get("sequence", "").upper()
        category = infer_category(row, task)
        source = infer_source(row)
        higher_group = infer_higher_group(row, category)
        sample_id = row.get("sample_id") or row.get("id") or f"{task}|{idx}"
        sim_id = row.get("similarity_cluster_id") or similarity_cluster_id(sequence)
        group_id = row.get("group_id") or "|".join(
            [source, category, higher_group, row.get("group", "") or "missing_group"]
        )
        stats = sequence_stats(sequence)
        enriched.append(
            {
                **row,
                "benchmark": row.get("benchmark", source),
                "task": task,
                "capability_role": role,
                "split": row.get("split", "").lower(),
                "split_random": row.get("split", "").lower(),
                "sequence": sequence,
                "label": row.get("label", ""),
                "category": category,
                "source": source,
                "higher_group": higher_group,
                "group_id": group_id,
                "similarity_cluster_id": sim_id,
                "sample_id": sample_id,
                "feature_cache_key": hashlib.sha1(f"{task}|{sample_id}|{sequence}".encode("utf-8")).hexdigest(),
                "family": row.get("family", ""),
                "group": row.get("group", ""),
                "id": row.get("id", sample_id),
                "length": str(int(stats["length"])),
                "gc": f"{stats['gc']:.8f}",
                "A_freq": f"{stats['A_freq']:.8f}",
                "C_freq": f"{stats['C_freq']:.8f}",
                "G_freq": f"{stats['G_freq']:.8f}",
                "T_freq": f"{stats['T_freq']:.8f}",
                "N_freq": f"{stats['N_freq']:.8f}",
            }
        )
    split_specs = {
        "source_split": ("source", assign_group_split(enriched, "source")),
        "category_split": ("category", assign_group_split(enriched, "category")),
        "similarity_split": ("similarity_cluster_id", assign_group_split(enriched, "similarity_cluster_id")),
    }
    for row in enriched:
        for field, (group_field, split_map) in split_specs.items():
            row[field] = split_map.get(row.get(group_field, "") or "missing", "train")
    return enriched

def resolve_task_rows(
    task: str,
    role: str,
    primary_manifest: Path,
    fallback_manifest: Path,
    max_per_split_label: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    primary_rows = read_task_rows(primary_manifest, task)
    source = str(primary_manifest)
    source_status = "primary_manifest"
    rows = primary_rows
    if not rows and fallback_manifest.exists():
        rows = read_task_rows(fallback_manifest, task)
        source = str(fallback_manifest)
        source_status = "fallback_manifest_used_because_primary_task_missing"
    capped = cap_split_label(rows, max_per_split_label, seed)
    return capped, {
        "task": task,
        "role": role,
        "source_manifest": source,
        "source_status": source_status if rows else "missing_task",
        "source_rows": len(rows),
        "selected_rows": len(capped),
        "max_per_split_label": max_per_split_label,
    }


def family_matrix(rows: list[dict[str, str]]):
    dicts = []
    metadata_keys = ("family", "group", "benchmark", "category", "source", "higher_group", "group_id")
    for row in rows:
        features = {}
        for key in metadata_keys:
            value = row.get(key, "")
            if value:
                features[f"{key}={value}"] = 1
        if not features:
            features["metadata_missing_or_constant"] = 1
        dicts.append(features)
    vectorizer = DictVectorizer(sparse=True)
    return vectorizer.fit_transform(dicts), {
        "family_feature_count": len(vectorizer.feature_names_),
        "metadata_keys": list(metadata_keys),
    }


def family_label_correlation(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = np.array([int(row.get("label", "0")) for row in rows], dtype=np.int64)
    family_values = [row.get("family", "") or "missing" for row in rows]
    counts = Counter(family_values)
    if len(counts) < 2:
        return {
            "status": "infeasible_constant_family",
            "n_family_values": len(counts),
            "largest_family_fraction": max(counts.values()) / max(len(rows), 1) if counts else None,
            "test_separability": None,
        }
    positive_family = counts.most_common(1)[0][0]
    family_binary = np.array([1 if value == positive_family else 0 for value in family_values], dtype=np.int64)
    auc = safe_auroc(labels, family_binary)
    return {
        "status": "ok",
        "n_family_values": len(counts),
        "largest_family": positive_family,
        "largest_family_fraction": max(counts.values()) / max(len(rows), 1),
        "test_separability": separability(auc),
    }


def group_feasibility(rows: list[dict[str, str]]) -> dict[str, Any]:
    result = {}
    total = max(len(rows), 1)
    for field in ("family", "group", "source", "id"):
        values = [row.get(field, "") for row in rows if row.get(field, "")]
        counts = Counter(values)
        if not counts:
            continue
        result[field] = {
            "n_groups": len(counts),
            "largest_group_fraction": max(counts.values()) / total,
            "feasible_basic_group_heldout": len(counts) >= 3 and max(counts.values()) / total < 0.8,
        }
    feasible = any(payload["feasible_basic_group_heldout"] for payload in result.values())
    return {
        "group_heldout_status": "feasible_basic" if feasible else "infeasible_for_this_dataset",
        "fields": result,
    }


def split_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row.get("split", "")][row.get("label", "")] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def numeric_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    stats = [sequence_stats(row.get("sequence", "")) for row in rows]
    result = {}
    for key in ("length", "gc", "A_freq", "C_freq", "G_freq", "T_freq", "N_freq"):
        values = np.array([item.get(key, 0.0) for item in stats], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return result


def shortcut_audit(rows: list[dict[str, str]], task: str, seeds: list[int], c_grid: list[float], n_bootstrap: int):
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    splits = np.array([row["split"] for row in rows])
    raw_matrix, raw_info = build_raw_features(rows)
    kmer_matrix, kmer_info = build_kmer_features(rows, 3, 6)
    fam_matrix, fam_info = family_matrix(rows)
    raw_plus_metadata_matrix, raw_plus_metadata_info = build_raw_plus_metadata_features(rows)
    raw_plus_kmer_matrix = hstack([raw_matrix, kmer_matrix], format="csr")
    raw_plus_kmer_plus_metadata_matrix = hstack([raw_matrix, fam_matrix, kmer_matrix], format="csr")
    raw = fit_eval_logistic(
        raw_matrix, labels, splits, c_grid, seed=seeds[0], n_bootstrap=n_bootstrap, target=task, baseline="raw_only_model"
    )
    metadata = fit_eval_logistic(
        fam_matrix, labels, splits, c_grid, seed=seeds[0], n_bootstrap=n_bootstrap, target=task, baseline="metadata_only_model"
    )
    kmer = fit_eval_logistic(
        kmer_matrix, labels, splits, c_grid, seed=seeds[0], n_bootstrap=n_bootstrap, target=task, baseline="kmer_only_model"
    )
    raw_plus_metadata = fit_eval_logistic(
        raw_plus_metadata_matrix,
        labels,
        splits,
        c_grid,
        seed=seeds[0],
        n_bootstrap=n_bootstrap,
        target=task,
        baseline="raw_plus_metadata_model",
    )
    raw_plus_kmer = fit_eval_logistic(
        raw_plus_kmer_matrix,
        labels,
        splits,
        c_grid,
        seed=seeds[0],
        n_bootstrap=n_bootstrap,
        target=task,
        baseline="raw_plus_kmer_model",
    )
    raw_plus_kmer_plus_metadata = fit_eval_logistic(
        raw_plus_kmer_plus_metadata_matrix,
        labels,
        splits,
        c_grid,
        seed=seeds[0],
        n_bootstrap=n_bootstrap,
        target=task,
        baseline="raw_plus_kmer_plus_metadata_model",
    )
    fam = dict(metadata)
    raw.update(raw_info)
    kmer.update(kmer_info)
    metadata.update(fam_info)
    fam.update(fam_info)
    raw_plus_metadata.update(raw_plus_metadata_info)
    return {
        "raw_only_model": raw,
        "metadata_only_model": metadata,
        "kmer_only_model": kmer,
        "raw_plus_metadata_model": raw_plus_metadata,
        "raw_plus_kmer_model": raw_plus_kmer,
        "raw_plus_kmer_plus_metadata_model": raw_plus_kmer_plus_metadata,
        "family_only_model": fam,
        "family_label_capability_label_correlation": family_label_correlation(rows),
    }


def write_manifest(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(STANDARD_MANIFEST_FIELDS)
        fields.extend(sorted({key for row in rows for key in row if key not in fields}))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_formal_task_manifests(out_dir: Path, rows: list[dict[str, str]], split_column: str) -> Path:
    manifest_dir = out_dir / "formal_task_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = []
        for row in rows:
            if row["task"] != task:
                continue
            copy = dict(row)
            copy["split"] = row.get(split_column, row.get("split", ""))
            task_rows.append(copy)
        manifest_path = manifest_dir / f"{task}.csv"
        write_manifest(manifest_path, task_rows)
        targets.append(
            {
                "name": task,
                "manifest": str(manifest_path),
                "layers": "0-15",
                "split_column": "split",
                "group_field": "similarity_cluster_id",
            }
        )
    config_path = out_dir / "task7r_internal_target_config.json"
    write_json(
        config_path,
        {
            "created_at": now(),
            "task": "task7r_formal_target_config",
            "formal_split_column": split_column,
            "targets": targets,
        },
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-manifest", default="data/benchmarks/final_fast_eval_manifest.csv")
    parser.add_argument("--fallback-benchmark-manifest", default="data/benchmarks/hvue_gue_manifest.csv")
    parser.add_argument("--primary-task", default=PRIMARY_TASK)
    parser.add_argument("--aux-task", default=AUX_TASK)
    parser.add_argument("--out-dir", default="data/phase2/audits/task7_capability_probe_20260713")
    parser.add_argument("--max-per-split-label", type=int, default=400)
    parser.add_argument("--formal-split-column", default="similarity_split")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    c_grid = [float(part.strip()) for part in args.c_grid.split(",") if part.strip()]

    selected_rows: list[dict[str, str]] = []
    sources = []
    for task, role, offset in (
        (args.primary_task, "primary_pt_capability", 0),
        (args.aux_task, "auxiliary_am_monitor", 1000),
    ):
        rows, source = resolve_task_rows(
            task,
            role,
            Path(args.benchmark_manifest),
            Path(args.fallback_benchmark_manifest),
            args.max_per_split_label,
            seeds[0] + offset,
        )
        sources.append(source)
        selected_rows.extend(add_standard_fields(rows, task, role))

    write_manifest(out_dir / "capability_dataset_manifest.csv", selected_rows)
    config_path = write_formal_task_manifests(out_dir, selected_rows, args.formal_split_column)

    task_payloads = {}
    for task in sorted({row["task"] for row in selected_rows}):
        rows = [row for row in selected_rows if row["task"] == task]
        if args.dry_run:
            shortcut = {"status": "dry_run"}
        else:
            shortcut = shortcut_audit(rows, task, seeds, c_grid, args.n_bootstrap)
        task_payloads[task] = {
            "n_rows": len(rows),
            "label_counts": dict(Counter(row["label"] for row in rows)),
            "split_counts": split_counts(rows),
            "family_counts": dict(Counter(row.get("family", "") or "missing" for row in rows)),
            "group_counts": dict(Counter(row.get("group", "") or "missing" for row in rows)),
            "length_gc_composition_stats": numeric_stats(rows),
            "group_feasibility": group_feasibility(rows),
            **shortcut,
        }

    audit = {
        "created_at": now(),
        "task": "task7_capability_probe_dataset",
        "primary_task": args.primary_task,
        "aux_task": args.aux_task,
        "task3_context": TASK3_CONTEXT,
        "source_resolution": sources,
        "max_per_split_label": args.max_per_split_label,
        "formal_split_column": args.formal_split_column,
        "formal_target_config": str(config_path),
        "tasks": task_payloads,
    }
    write_json(out_dir / "capability_dataset_audit.json", audit)
    print(f"[task7-dataset] wrote {len(selected_rows)} rows to {out_dir / 'capability_dataset_manifest.csv'}")


if __name__ == "__main__":
    main()
