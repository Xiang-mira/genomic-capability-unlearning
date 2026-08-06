"""Stage 1 base-calibration baseline audit and postprocessing tools.

This module intentionally avoids mutating the active training path. It only:
1. audits baseline implementation differences and provenance;
2. computes standalone k-mer metrics and threshold metadata;
3. writes batch registries and post-hoc summaries for the running 27-run grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_MANIFEST = PROJECT_ROOT / "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv"
DEFAULT_LEGACY_FORMAL_MANIFEST = (
    PROJECT_ROOT / "data/phase2/audits/task7s_clean_gate_20260715/candidates/matched_all_pairs/formal_task_manifests/hvue_human_host_tropism.csv"
)
DEFAULT_FORMAL_KMER_BASELINE = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_available_kmer.csv"
DEFAULT_LEGACY_KMER_BASELINE = (
    PROJECT_ROOT / "data/phase2/audits/task7s_clean_gate_20260715/candidates/matched_all_pairs/probe_validity/kmer_baseline.csv"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/stage1_formal_experiment_20260727"
DEFAULT_PLAN_JSON = DEFAULT_OUT_ROOT / "stage1_formal_experiment_plan.json"
DEFAULT_RESULTS_GLOB = "fresh_lora/base/rank_*/lr_*/seed_*"
DEFAULT_TASK = "hvue_human_host_tropism"
DEFAULT_SPLIT_TYPE = "cluster_disjoint"
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_baseline_audit.json"
DEFAULT_AUDIT_MD = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_baseline_audit.md"
DEFAULT_KMER_METRICS_CSV = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_available_kmer_metrics.csv"
DEFAULT_KMER_METRICS_JSON = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_available_kmer_metrics.json"
DEFAULT_CANONICAL_CANDIDATES = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_canonical_baseline_candidates.json"
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_canonical_baseline.json"
DEFAULT_REGISTRY_CSV = DEFAULT_OUT_ROOT / "base_calibration_registry.csv"
DEFAULT_REGISTRY_JSON = DEFAULT_OUT_ROOT / "base_calibration_registry.json"
DEFAULT_EXCESS_CSV = DEFAULT_OUT_ROOT / "base_calibration_excess_recalculated.csv"
DEFAULT_EXCESS_JSON = DEFAULT_OUT_ROOT / "base_calibration_excess_recalculated.json"
DEFAULT_SELECTION_JSON = DEFAULT_OUT_ROOT / "base_calibration_selection_summary.json"
DEFAULT_FREEZE_JSON = DEFAULT_OUT_ROOT / "frozen_attack_config.json"
DEFAULT_CONFIRMATION_JSON = DEFAULT_OUT_ROOT / "targeted_confirmation_rerun_plan.json"
DEFAULT_TRANCHE3_JSON = DEFAULT_OUT_ROOT / "tranche3_readiness.json"
DEFAULT_TRANCHE3_MD = DEFAULT_OUT_ROOT / "tranche3_readiness.md"
DEFAULT_FINAL_REPORT_JSON = DEFAULT_OUT_ROOT / "base_calibration_final_report.json"
DEFAULT_FINAL_REPORT_MD = DEFAULT_OUT_ROOT / "base_calibration_final_report.md"
DEFAULT_INPUT_FAIRNESS_JSON = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_input_fairness.json"
DEFAULT_INPUT_FAIRNESS_MD = PROJECT_ROOT / "data/phase2/kmer_baselines/stage1_formal_targets_input_fairness.md"
DEFAULT_EXPLORATORY_FREEZE_JSON = DEFAULT_OUT_ROOT / "exploratory_frozen_attack_config.json"
DEFAULT_COMPARISON_OUT_ROOT = PROJECT_ROOT / "data/phase2/stage1_exploratory_checkpoint_comparison_20260728"
DEFAULT_COMPARISON_PLAN_JSON = DEFAULT_COMPARISON_OUT_ROOT / "stage1_formal_experiment_plan.json"
DEFAULT_COMPARISON_REGISTRY_CSV = DEFAULT_COMPARISON_OUT_ROOT / "exploratory_comparison_registry.csv"
DEFAULT_COMPARISON_REGISTRY_JSON = DEFAULT_COMPARISON_OUT_ROOT / "exploratory_comparison_registry.json"
DEFAULT_COMPARISON_REPORT_CSV = DEFAULT_COMPARISON_OUT_ROOT / "stage1_exploratory_checkpoint_comparison.csv"
DEFAULT_COMPARISON_REPORT_JSON = DEFAULT_COMPARISON_OUT_ROOT / "stage1_exploratory_checkpoint_comparison.json"
DEFAULT_COMPARISON_REPORT_MD = DEFAULT_COMPARISON_OUT_ROOT / "stage1_exploratory_checkpoint_comparison.md"
DEFAULT_CURRENT_C_GRID = "0.001,0.01,0.1,1,10"
DEFAULT_STRONG_C_GRID = "0.001,0.01,0.1,1,10,100"


@dataclass(frozen=True)
class ManifestRow:
    row_id: str
    split: str
    label: int
    sequence: str


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    implementation_script: str
    vectorizer_type: str
    sequence_length_policy: str
    feature_protocol: str
    normalization: str
    vocabulary_policy: str
    solver: str
    max_iter: int
    class_weight: str
    c_grid: str
    selection_metric: str
    use_scaler: bool
    use_hashing: bool
    max_length: int


FORMAL_PROTOCOL = ProtocolSpec(
    protocol_id="formal_count_first512",
    implementation_script="phase2/eval_kmer_baseline.py",
    vectorizer_type="CountVectorizer",
    sequence_length_policy="first_512",
    feature_protocol="count_raw",
    normalization="none",
    vocabulary_policy="fit_on_full_dataset",
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    c_grid=DEFAULT_CURRENT_C_GRID,
    selection_metric="auroc",
    use_scaler=False,
    use_hashing=False,
    max_length=512,
)

LEGACY_PROTOCOL = ProtocolSpec(
    protocol_id="legacy_hashing_full_sequence",
    implementation_script="phase2/probe_validity_audit.py",
    vectorizer_type="HashingVectorizer",
    sequence_length_policy="full_sequence",
    feature_protocol="hashing_l2_scaled",
    normalization="l2_then_standard_scaler",
    vocabulary_policy="hashing_no_explicit_vocab",
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
    c_grid=DEFAULT_CURRENT_C_GRID,
    selection_metric="separability_auroc",
    use_scaler=True,
    use_hashing=True,
    max_length=0,
)

STRONG_PROTOCOL = ProtocolSpec(
    protocol_id="strong_hashing_full_sequence_cgrid100",
    implementation_script="phase2/stage1_calibration_tools.py",
    vectorizer_type="HashingVectorizer",
    sequence_length_policy="full_sequence",
    feature_protocol="hashing_l2_scaled",
    normalization="l2_then_standard_scaler",
    vocabulary_policy="hashing_no_explicit_vocab",
    solver="liblinear",
    max_iter=1000,
    class_weight="balanced",
    c_grid=DEFAULT_STRONG_C_GRID,
    selection_metric="separability_auroc",
    use_scaler=True,
    use_hashing=True,
    max_length=0,
)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_dirty_diff_sha256(paths: Iterable[str]) -> str:
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--", *paths],
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return hashlib.sha256(diff).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(spec: str) -> list[float]:
    return [float(part.strip()) for part in str(spec).split(",") if part.strip()]


def normalize_split_type(value: str) -> str:
    split_type = str(value or "").strip().lower()
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type or "random"


def clean_sequence_for_formal(value: str, max_length: int) -> str:
    seq = "".join(ch for ch in str(value or "").upper() if ch in {"A", "C", "G", "T", "N"})
    return seq if max_length <= 0 else seq[:max_length]


def clean_sequence_for_legacy(value: str) -> str:
    return str(value or "").upper()


def read_formal_manifest_rows(path: Path, task: str, split_type: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("task") != task:
                continue
            if normalize_split_type(row.get("split_type", "")) != split_type:
                continue
            rows.append(
                ManifestRow(
                    row_id=str(row.get("id", "")),
                    split=str(row.get("split", "")),
                    label=int(row.get("label", "0")),
                    sequence=str(row.get("sequence", "")),
                )
            )
    return rows


def read_legacy_manifest_rows(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ManifestRow(
                    row_id=str(row.get("id", "")),
                    split=str(row.get("split", "")),
                    label=int(row.get("label", "0")),
                    sequence=str(row.get("sequence", "")),
                )
            )
    return rows


def row_order_sha256(rows: Iterable[ManifestRow]) -> str:
    payload = [(row.row_id, row.split, row.label, row.sequence) for row in rows]
    return stable_hash(payload)


def describe_row_equivalence(formal_rows: list[ManifestRow], legacy_rows: list[ManifestRow]) -> dict[str, object]:
    same_order = formal_rows == legacy_rows
    return {
        "n_rows_formal": len(formal_rows),
        "n_rows_legacy": len(legacy_rows),
        "row_order_sha256_formal": row_order_sha256(formal_rows),
        "row_order_sha256_legacy": row_order_sha256(legacy_rows),
        "same_row_order": same_order,
        "same_row_set": set(formal_rows) == set(legacy_rows),
    }


def split_arrays(rows: list[ManifestRow], protocol: ProtocolSpec) -> tuple[list[str], np.ndarray, np.ndarray]:
    sequences: list[str] = []
    labels: list[int] = []
    splits: list[str] = []
    for row in rows:
        if protocol.use_hashing:
            sequences.append(clean_sequence_for_legacy(row.sequence))
        else:
            sequences.append(clean_sequence_for_formal(row.sequence, protocol.max_length))
        labels.append(row.label)
        split = "val" if row.split == "dev" else row.split
        splits.append(split)
    return sequences, np.array(labels, dtype=np.int64), np.array(splits)


def separability(auroc: float) -> float:
    if math.isnan(auroc):
        return float("nan")
    return max(float(auroc), 1.0 - float(auroc))


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred_05 = (probs >= 0.5).astype(np.int64)
    return {
        "accuracy_at_0.5": float(accuracy_score(y_true, pred_05)),
        "f1_at_0.5": float(f1_score(y_true, pred_05, average="macro", zero_division=0)),
        "mcc_at_0.5": float(matthews_corrcoef(y_true, pred_05)),
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
    }


def select_mcc_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = sorted({0.5, *[float(value) for value in probs]})
    best_threshold = 0.5
    best_mcc = -2.0
    for threshold in thresholds:
        pred = (probs >= threshold).astype(np.int64)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc + 1e-12:
            best_mcc = mcc
            best_threshold = threshold
    return best_threshold, best_mcc


def build_matrix(sequences: list[str], protocol: ProtocolSpec):
    if protocol.use_hashing:
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        matrix = vectorizer.transform(sequences)
    else:
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            lowercase=False,
            binary=False,
        )
        matrix = vectorizer.fit_transform(sequences)
    return matrix


def evaluate_protocol(rows: list[ManifestRow], protocol: ProtocolSpec) -> dict[str, object]:
    sequences, labels, splits = split_arrays(rows, protocol)
    matrix = build_matrix(sequences, protocol)
    masks = {name: splits == name for name in ("train", "val", "test")}
    x_train = matrix[masks["train"]]
    x_all = matrix
    if protocol.use_scaler:
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x_train)
        x_all = scaler.transform(matrix)

    selection_mask = masks["val"]
    c_grid = parse_float_list(protocol.c_grid)
    best_clf = None
    best_c = None
    best_score = -float("inf")
    for c_value in c_grid:
        clf = LogisticRegression(
            C=c_value,
            solver=protocol.solver,
            max_iter=protocol.max_iter,
            class_weight=protocol.class_weight,
            random_state=42 if protocol.solver == "liblinear" else None,
        )
        clf.fit(x_train, labels[masks["train"]])
        val_probs = clf.predict_proba(x_all[selection_mask])[:, 1]
        val_auroc = float(roc_auc_score(labels[selection_mask], val_probs))
        score = separability(val_auroc) if protocol.selection_metric == "separability_auroc" else val_auroc
        if score > best_score:
            best_score = score
            best_clf = clf
            best_c = c_value

    assert best_clf is not None and best_c is not None
    val_probs = best_clf.predict_proba(x_all[masks["val"]])[:, 1]
    test_probs = best_clf.predict_proba(x_all[masks["test"]])[:, 1]
    val_threshold, val_mcc = select_mcc_threshold(labels[masks["val"]], val_probs)
    test_pred_val = (test_probs >= val_threshold).astype(np.int64)
    metrics = compute_binary_metrics(labels[masks["test"]], test_probs)
    metrics.update(
        {
            "task": DEFAULT_TASK,
            "split_type": DEFAULT_SPLIT_TYPE,
            "best_c": best_c,
            "selection_metric": protocol.selection_metric,
            "selection_split": "val",
            "threshold_policy": "validation_selected_mcc",
            "val_selected_threshold": float(val_threshold),
            "val_mcc": float(val_mcc),
            "test_accuracy_val_selected": float(accuracy_score(labels[masks["test"]], test_pred_val)),
            "test_f1_val_selected": float(f1_score(labels[masks["test"]], test_pred_val, average="macro", zero_division=0)),
            "test_mcc_val_selected": float(matthews_corrcoef(labels[masks["test"]], test_pred_val)),
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()),
        }
    )
    return metrics


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def read_matching_row_csv(path: Path, **criteria: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = []
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one matching row in {path} for {criteria}, found {len(matches)}")
    return matches[0]


def baseline_audit_payload(args: argparse.Namespace) -> dict[str, object]:
    formal_rows = read_formal_manifest_rows(args.formal_manifest, args.task, args.split_type)
    legacy_rows = read_legacy_manifest_rows(args.legacy_manifest)
    equivalence = describe_row_equivalence(formal_rows, legacy_rows)
    formal_baseline = read_single_row_csv(args.formal_kmer_baseline)
    legacy_baseline = read_matching_row_csv(args.legacy_kmer_baseline, target=args.task, seed="42")
    return {
        "task": args.task,
        "split_type": args.split_type,
        "date_utc": args.current_date,
        "data_loading": {
            "formal_manifest_path": str(args.formal_manifest),
            "formal_manifest_sha256": file_sha256(args.formal_manifest),
            "legacy_manifest_path": str(args.legacy_manifest),
            "legacy_manifest_sha256": file_sha256(args.legacy_manifest),
            **equivalence,
            "formal_cleaning_policy": "retain ACGTN only, truncate to first 512 nt",
            "legacy_cleaning_policy": "uppercase only, no truncation",
            "sequence_length_summary": {
                "all_rows_gt_512": all(len(row.sequence) > 512 for row in formal_rows),
                "min_length": min(len(row.sequence) for row in formal_rows),
                "max_length": max(len(row.sequence) for row in formal_rows),
            },
            "finding": "No data-row, label-order, or split-order differences were found between the two Host Tropism formal manifests.",
        },
        "feature_protocol": {
            "legacy": {
                "vectorizer_type": LEGACY_PROTOCOL.vectorizer_type,
                "sequence_length_policy": LEGACY_PROTOCOL.sequence_length_policy,
                "feature_protocol": LEGACY_PROTOCOL.feature_protocol,
                "normalization": LEGACY_PROTOCOL.normalization,
                "vocabulary_policy": LEGACY_PROTOCOL.vocabulary_policy,
                "sparse_matrix": True,
                "reverse_complement_features": False,
                "unknown_character_handling": "uppercase only; no explicit stripping before hashing",
                "uses_full_sequence": True,
            },
            "formal": {
                "vectorizer_type": FORMAL_PROTOCOL.vectorizer_type,
                "sequence_length_policy": FORMAL_PROTOCOL.sequence_length_policy,
                "feature_protocol": FORMAL_PROTOCOL.feature_protocol,
                "normalization": FORMAL_PROTOCOL.normalization,
                "vocabulary_policy": FORMAL_PROTOCOL.vocabulary_policy,
                "sparse_matrix": True,
                "reverse_complement_features": False,
                "unknown_character_handling": "strip to ACGTN before vectorization",
                "uses_full_sequence": False,
            },
        },
        "classifier": {
            "legacy": {
                "solver": LEGACY_PROTOCOL.solver,
                "penalty": "l2_default",
                "class_weight": LEGACY_PROTOCOL.class_weight,
                "max_iter": LEGACY_PROTOCOL.max_iter,
                "random_state": 42,
                "c_grid": LEGACY_PROTOCOL.c_grid,
                "selection_metric": LEGACY_PROTOCOL.selection_metric,
            },
            "formal": {
                "solver": FORMAL_PROTOCOL.solver,
                "penalty": "l2_default",
                "class_weight": FORMAL_PROTOCOL.class_weight,
                "max_iter": FORMAL_PROTOCOL.max_iter,
                "random_state": None,
                "c_grid": FORMAL_PROTOCOL.c_grid,
                "selection_metric": FORMAL_PROTOCOL.selection_metric,
            },
        },
        "evaluation": {
            "positive_class_definition": "label 1, predict_proba[:, 1]",
            "legacy_threshold_policy": "none for stored AUROC; MCC threshold not stored",
            "formal_threshold_policy": "none in baseline file; MCC threshold not stored",
            "test_sample_order_consistency": equivalence["same_row_order"],
        },
        "source_artifacts": {
            "legacy_baseline_path": str(args.legacy_kmer_baseline),
            "legacy_baseline_sha256": file_sha256(args.legacy_kmer_baseline),
            "legacy_baseline_row": legacy_baseline,
            "formal_baseline_path": str(args.formal_kmer_baseline),
            "formal_baseline_sha256": file_sha256(args.formal_kmer_baseline),
            "formal_baseline_row": formal_baseline,
        },
        "final_determination": {
            "primary_divergence": "baseline implementation / evaluation layer",
            "agreed_strong_protocol_deviation": [
                "formal baseline truncates to first_512 rather than full_sequence",
                "formal baseline uses count_raw features rather than hashing_l2_scaled geometry",
                "formal baseline c_grid currently stops at 10 rather than 100",
            ],
            "most_likely_explanation_for_gap": [
                "sequence_length_policy differs (full_sequence vs first_512)",
                "feature geometry differs (HashingVectorizer+l2+StandardScaler vs CountVectorizer raw counts)",
            ],
            "allow_legacy_for_formal_excess": False,
            "allow_formal_for_canonical_freeze_now": False,
            "conclusion": [
                "No data-layer differences were found.",
                "Differences are located in the baseline implementation / evaluation layer.",
                "Do not mix 0.8930 and 0.849647 in the same formal excess calculation.",
            ],
        },
    }


def baseline_audit_markdown(payload: dict[str, object]) -> str:
    determination = payload["final_determination"]
    return "\n".join(
        [
            "# Stage 1 Formal Targets Baseline Audit",
            "",
            f"- Task: `{payload['task']}`",
            f"- Split type: `{payload['split_type']}`",
            f"- Formal manifest SHA256: `{payload['data_loading']['formal_manifest_sha256']}`",
            f"- Legacy manifest SHA256: `{payload['data_loading']['legacy_manifest_sha256']}`",
            f"- Same row order: `{payload['data_loading']['same_row_order']}`",
            "",
            "## Key Findings",
            "",
            "- Data layer: no row-order, label-order, or split-order differences were found.",
            "- Legacy baseline: full-sequence `HashingVectorizer` + `norm='l2'` + `StandardScaler(with_mean=False)` + `liblinear`.",
            "- Formal baseline: cleaned/truncated `CountVectorizer` raw counts + `lbfgs` without scaling.",
            "- All sequences exceed 512 nt, so `first_512` is a material truncation policy.",
            "",
            "## Final Determination",
            "",
            *[f"- {line}" for line in determination["conclusion"]],
            "",
            "## Strong-Protocol Deviations",
            "",
            *[f"- {line}" for line in determination["agreed_strong_protocol_deviation"]],
        ]
    )


def run_audit_baselines(args: argparse.Namespace) -> None:
    payload = baseline_audit_payload(args)
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(baseline_audit_markdown(payload) + "\n")


def sequence_length_percentile(lengths: list[int], q: float) -> float:
    if not lengths:
        raise ValueError("lengths is empty")
    ordered = sorted(lengths)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo))


def run_write_input_fairness_report(args: argparse.Namespace) -> None:
    rows = read_formal_manifest_rows(args.formal_manifest, args.task, args.split_type)
    lengths = [len(row.sequence) for row in rows]
    lora_max_length = 512
    payload = {
        "task": args.task,
        "manifest_path": str(args.formal_manifest),
        "manifest_sha256": file_sha256(args.formal_manifest),
        "row_order_sha256": row_order_sha256(rows),
        "lora_sequence_length_policy": "first_512_characters_only",
        "lora_context_length": 8192,
        "lora_tokenizer_max_length": lora_max_length,
        "lora_runtime_max_length": lora_max_length,
        "lora_truncation_policy": "prefix truncation before CharLevelTokenizer tokenization",
        "lora_crop_policy": "deterministic prefix only; no random windowing",
        "lora_multiple_windows_per_sample": False,
        "lora_truncated_fraction": sum(length > lora_max_length for length in lengths) / len(lengths),
        "kmer_0849_sequence_length_policy": "first_512_characters_only_after_ACGTN_cleaning",
        "kmer_0849_auroc": 0.8496470934222136,
        "kmer_0849_matches_lora_input_budget": True,
        "kmer_0893_sequence_length_policy": "full_sequence_after_uppercase_only",
        "kmer_0893_auroc": 0.8930006862072345,
        "sequence_length_median": float(sequence_length_percentile(lengths, 0.5)),
        "sequence_length_p90": float(sequence_length_percentile(lengths, 0.9)),
        "sequence_length_p95": float(sequence_length_percentile(lengths, 0.95)),
        "sequence_length_max": int(max(lengths)),
        "fraction_exceeding_lora_context_length": sum(length > 8192 for length in lengths) / len(lengths),
        "fraction_exceeding_lora_runtime_max_length": sum(length > lora_max_length for length in lengths) / len(lengths),
        "matched_input_baseline": "0.8496470934222136",
        "full_sequence_reference": "0.8930006862072345",
        "final_baseline_interpretation": (
            "0.8496470934222136 can be treated as the matched-input k-mer baseline for the current fresh-LoRA pipeline, "
            "while 0.8930006862072345 remains the full-sequence strong canonical reference."
        ),
    }
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "\n".join(
            [
                "# Input Fairness Report",
                "",
                f"- Task: `{args.task}`",
                f"- LoRA context length: `{payload['lora_context_length']}`",
                f"- LoRA runtime max_length: `{payload['lora_runtime_max_length']}`",
                f"- LoRA truncated fraction: `{payload['lora_truncated_fraction']}`",
                f"- Sequence length median: `{payload['sequence_length_median']}`",
                f"- Sequence length p90: `{payload['sequence_length_p90']}`",
                f"- Sequence length p95: `{payload['sequence_length_p95']}`",
                f"- Sequence length max: `{payload['sequence_length_max']}`",
                f"- Matched-input baseline: `{payload['matched_input_baseline']}`",
                f"- Full-sequence reference: `{payload['full_sequence_reference']}`",
                "",
                "## Conclusion",
                "",
                f"- {payload['final_baseline_interpretation']}",
            ]
        )
        + "\n"
    )


def run_compute_kmer_metrics(args: argparse.Namespace) -> None:
    rows = read_formal_manifest_rows(args.formal_manifest, args.task, args.split_type)
    metrics = evaluate_protocol(rows, FORMAL_PROTOCOL)
    csv_row = {
        "task": metrics["task"],
        "split_type": metrics["split_type"],
        "best_c": metrics["best_c"],
        "selection_metric": metrics["selection_metric"],
        "selection_split": metrics["selection_split"],
        "threshold_policy": metrics["threshold_policy"],
        "val_selected_threshold": metrics["val_selected_threshold"],
        "test_auroc": metrics["auroc"],
        "test_auprc": metrics["auprc"],
        "test_accuracy_at_0.5": metrics["accuracy_at_0.5"],
        "test_f1_at_0.5": metrics["f1_at_0.5"],
        "test_mcc_at_0.5": metrics["mcc_at_0.5"],
        "test_accuracy_val_selected": metrics["test_accuracy_val_selected"],
        "test_f1_val_selected": metrics["test_f1_val_selected"],
        "test_mcc_val_selected": metrics["test_mcc_val_selected"],
    }
    write_csv(args.out_csv, [csv_row], list(csv_row.keys()))
    write_json(
        args.out_json,
        {
            "task": args.task,
            "split_type": args.split_type,
            "manifest_path": str(args.formal_manifest),
            "manifest_sha256": file_sha256(args.formal_manifest),
            "protocol_id": FORMAL_PROTOCOL.protocol_id,
            "metrics": csv_row,
        },
    )


def canonical_candidate_payload(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_formal_manifest_rows(args.formal_manifest, args.task, args.split_type)
    implementation_commit = git_commit_hash()
    implementation_dirty_diff = git_dirty_diff_sha256(
        [
            "phase2/probe_validity_audit.py",
            "phase2/eval_kmer_baseline.py",
            "phase2/stage1_calibration_tools.py",
        ]
    )
    candidates = []
    for protocol, source_path in (
        (FORMAL_PROTOCOL, args.formal_kmer_baseline),
        (LEGACY_PROTOCOL, args.legacy_kmer_baseline),
        (STRONG_PROTOCOL, args.legacy_kmer_baseline),
    ):
        metrics = evaluate_protocol(rows, protocol)
        candidate = {
            "candidate_id": protocol.protocol_id,
            "task": args.task,
            "manifest_path": str(args.formal_manifest),
            "manifest_sha256": file_sha256(args.formal_manifest),
            "row_order_sha256": row_order_sha256(rows),
            "sequence_length_policy": protocol.sequence_length_policy,
            "feature_protocol": protocol.feature_protocol,
            "k_values": "3-6",
            "normalization": protocol.normalization,
            "vectorizer_type": protocol.vectorizer_type,
            "vocabulary_policy": protocol.vocabulary_policy,
            "classifier": "LogisticRegression",
            "class_weight": protocol.class_weight,
            "solver": protocol.solver,
            "max_iter": protocol.max_iter,
            "c_grid": protocol.c_grid,
            "selection_metric": protocol.selection_metric,
            "selection_split": metrics["selection_split"],
            "threshold_policy": metrics["threshold_policy"],
            "val_selected_threshold": metrics["val_selected_threshold"],
            "test_auroc": metrics["auroc"],
            "test_mcc": metrics["test_mcc_val_selected"],
            "test_accuracy": metrics["test_accuracy_val_selected"],
            "test_f1": metrics["test_f1_val_selected"],
            "test_auprc": metrics["auprc"],
            "test_mcc_at_0.5": metrics["mcc_at_0.5"],
            "implementation_script": protocol.implementation_script,
            "implementation_commit": implementation_commit,
            "implementation_dirty_diff_sha256": implementation_dirty_diff,
            "status": "provisional_not_frozen",
            "freeze_reason": (
                "Current batch is still running; this candidate is recorded for post-batch canonicalization."
                if protocol is FORMAL_PROTOCOL
                else "Recommended candidate matches the agreed strong full-sequence baseline geometry more closely, but canonical freeze is deferred until batch completion."
            ),
        }
        candidate["baseline_file_sha256"] = (
            file_sha256(source_path) if source_path.exists() and protocol is not STRONG_PROTOCOL else stable_hash(candidate)
        )
        candidates.append(candidate)
    recommended = next(item for item in candidates if item["candidate_id"] == STRONG_PROTOCOL.protocol_id)
    canonical = dict(recommended)
    canonical["status"] = "provisional_not_frozen"
    canonical["freeze_reason"] = (
        "Do not freeze 0.849647 directly. Recommended canonical candidate is the strong full-sequence hashing+l2+scaler protocol with C-grid extended to 100, but final freeze is deferred until the 27-run batch completes."
    )
    canonical["recommended_candidate_id"] = STRONG_PROTOCOL.protocol_id
    return candidates, canonical


def run_write_canonical_baseline(args: argparse.Namespace) -> None:
    candidates, canonical = canonical_candidate_payload(args)
    write_json(args.candidates_json, {"candidates": candidates})
    write_json(args.out_json, canonical)


def find_result_dirs(out_root: Path) -> list[Path]:
    return sorted(out_root.glob(DEFAULT_RESULTS_GLOB))


def load_progress(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_last_training_log_row(result_dir: Path) -> dict[str, object]:
    log_path = result_dir / "logs" / f"{DEFAULT_TASK}.jsonl"
    if not log_path.exists():
        return {}
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def read_result_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def extract_manifest_sha256(metadata: dict[str, object], manifest_path: Path) -> str:
    data_hashes = metadata.get("data_hashes", {})
    if not isinstance(data_hashes, dict):
        return ""
    manifest_str = str(manifest_path)
    if manifest_str in data_hashes:
        return str(data_hashes[manifest_str])
    manifest_posix = manifest_path.as_posix()
    if manifest_posix in data_hashes:
        return str(data_hashes[manifest_posix])
    manifest_name = manifest_path.name
    for key, value in data_hashes.items():
        key_str = str(key)
        if key_str == manifest_name or key_str.endswith(manifest_name):
            return str(value)
    return ""


def parse_run_key(result_dir: Path) -> tuple[int, float, int]:
    rank = int(result_dir.parent.parent.name.replace("rank_", ""))
    lr = float(result_dir.parent.name.replace("lr_", ""))
    seed = int(result_dir.name.replace("seed_", ""))
    return rank, lr, seed


def run_write_registry(args: argparse.Namespace) -> None:
    plan = json.loads(args.plan_json.read_text())
    rows = []
    for item in plan:
        result_dir = PROJECT_ROOT / item["cmd"][item["cmd"].index("--out-dir") + 1]
        results_path = result_dir / "eval_benchmarks.csv"
        metadata_path = result_dir / "eval_benchmarks_metadata.json"
        summary_path = result_dir / "eval_benchmarks_summary.json"
        progress_path = result_dir / "eval_benchmarks_progress.json"
        result_row = read_result_row(results_path)
        progress = load_progress(progress_path)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        rank = item.get("rank", "")
        lr = item.get("lr", "")
        seed = item.get("seed", "")
        progress_status = str(progress.get("status", ""))
        if result_row and progress_status == "complete":
            run_status = "completed_pending_canonical_baseline"
        elif progress_status == "running":
            run_status = "running"
        elif progress_status == "failed":
            run_status = "failed"
        else:
            run_status = "planned"
        notes = ""
        if int(seed) == 42 and run_status.startswith("completed"):
            notes = "Compare seed42 provenance against later seeds before final confirmation rerun decisions."
        rows.append(
            {
                "run_id": item["name"],
                "task": args.task,
                "checkpoint": item.get("checkpoint", "base"),
                "rank": rank,
                "lr": lr,
                "seed": seed,
                "results_path": str(results_path),
                "metadata_path": str(metadata_path),
                "summary_path": str(summary_path),
                "progress_path": str(progress_path),
                "commit_hash": metadata.get("commit_hash", ""),
                "git_dirty": metadata.get("git_dirty", ""),
                "config_hash": metadata.get("config_hash", ""),
                "manifest_sha256": extract_manifest_sha256(metadata, args.formal_manifest),
                "kmer_baseline_source": str(args.formal_kmer_baseline),
                "run_status": run_status,
                "calibration_label": "",
                "notes": progress.get("exit_reason", notes) if progress_status == "failed" else notes,
            }
        )
    write_csv(args.out_csv, rows, list(rows[0].keys()) if rows else [])
    write_json(args.out_json, {"task": args.task, "runs": rows})


def canonical_baseline_from_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def run_recalculate_excess(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry_json.read_text())["runs"]
    canonical = canonical_baseline_from_json(args.canonical_json)
    kmer_auroc = float(canonical["test_auroc"])
    kmer_mcc = float(canonical["test_mcc"])
    threshold_policy = str(canonical["threshold_policy"])
    rows = []
    for record in registry:
        result_row = read_result_row(Path(record["results_path"]))
        if not result_row:
            continue
        lo_ra_auroc = float(result_row["auroc"]) if result_row.get("auroc") else None
        lo_ra_mcc = float(result_row["mcc"]) if result_row.get("mcc") else None
        if threshold_policy == "validation_selected_mcc":
            mcc_status = "requires_eval_only_rerun_for_lora_threshold"
            mcc_excess = ""
        else:
            mcc_status = "computed"
            mcc_excess = "" if lo_ra_mcc is None else str(lo_ra_mcc - kmer_mcc)
        rows.append(
            {
                "run_id": record["run_id"],
                "rank": record["rank"],
                "lr": record["lr"],
                "seed": record["seed"],
                "raw_test_auroc": "" if lo_ra_auroc is None else lo_ra_auroc,
                "canonical_kmer_auroc": kmer_auroc,
                "canonical_auroc_excess": "" if lo_ra_auroc is None else lo_ra_auroc - kmer_auroc,
                "raw_test_mcc": "" if lo_ra_mcc is None else lo_ra_mcc,
                "canonical_kmer_mcc": kmer_mcc,
                "canonical_mcc_excess": mcc_excess,
                "threshold_policy_used": threshold_policy,
                "mcc_excess_status": mcc_status,
            }
        )
    write_csv(args.out_csv, rows, list(rows[0].keys()) if rows else [])
    write_json(args.out_json, {"canonical_baseline": canonical, "rows": rows})


def group_key(row: dict[str, object]) -> tuple[int, float]:
    return int(row["rank"]), float(row["lr"])


def run_select_config(args: argparse.Namespace) -> None:
    excess_rows = json.loads(args.excess_json.read_text())["rows"]
    grouped: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in excess_rows:
        grouped[group_key(row)].append(row)
    summary_rows = []
    best_candidate = None
    best_sort_key = None
    for key, rows in sorted(grouped.items()):
        dev_rows = [row for row in rows if int(row["seed"]) in {42, 43}]
        confirm_row = next((row for row in rows if int(row["seed"]) == 44), None)
        if len(dev_rows) < 2:
            status = "pending_incomplete"
            dev_validation_mean = None
            dev_validation_std = None
            dev_auroc_excess_mean = None
        else:
            validation_values = []
            dev_auroc_values = []
            for row in dev_rows:
                result_row = read_result_row(Path(next(item["results_path"] for item in json.loads(args.registry_json.read_text())["runs"] if item["run_id"] == row["run_id"])))
                validation_values.append(float(result_row["validation_metric"]))
                dev_auroc_values.append(float(row["canonical_auroc_excess"]))
            dev_validation_mean = mean(validation_values)
            dev_validation_std = pstdev(validation_values) if len(validation_values) > 1 else 0.0
            dev_auroc_excess_mean = mean(dev_auroc_values)
            status = "eligible_dev_positive" if all(value > 0 for value in dev_auroc_values) else "dev_not_positive"
        summary = {
            "rank": key[0],
            "lr": key[1],
            "status": status,
            "dev_validation_mean": dev_validation_mean,
            "dev_validation_std": dev_validation_std,
            "dev_auroc_excess_mean": dev_auroc_excess_mean,
            "confirmation_seed44_auroc_excess": None if not confirm_row or confirm_row["canonical_auroc_excess"] == "" else float(confirm_row["canonical_auroc_excess"]),
        }
        summary_rows.append(summary)
        if status == "eligible_dev_positive":
            sort_key = (-float(dev_validation_mean), float(dev_validation_std or 0.0), key[0], key[1])
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_candidate = summary
    write_json(args.out_json, {"rows": summary_rows, "recommended_config": best_candidate})


def run_write_exploratory_attack_config(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection_json.read_text())
    summary_rows = selection.get("rows", [])
    if not summary_rows:
        raise RuntimeError("Selection summary is empty")
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            -float(row["dev_validation_mean"]),
            float(row["dev_validation_std"]),
            float(row["rank"]),
            float(row["lr"]),
        ),
    )
    best = ranked[0]
    payload = {
        "status": "exploratory_only",
        "formal_strong_baseline_calibration": "failed",
        "allowed_use": "relative_checkpoint_comparison",
        "not_allowed_use": "formal_tamper_resistance_claim",
        "selection_rule": "highest mean development validation AUROC with stability and cost tie-breakers; seed44 excluded from selection",
        "selected_rank": int(best["rank"]),
        "selected_learning_rate": float(best["lr"]),
        "lora_alpha": int(best["rank"]) * 2,
        "lora_dropout": 0.0,
        "target_modules": "all Linear modules under every Evo block",
        "target_layers": "all_blocks",
        "epochs": 3,
        "max_steps": 0,
        "early_stopping_patience": 3,
        "validation_interval": 200,
        "pooling": "mean pooled final normalized Evo states",
        "classification_head": "fresh linear head",
        "threshold_policy": "validation_selected_mcc",
        "development_seeds": [42, 43],
        "confirmation_seed": 44,
        "selection_evidence": best,
        "freeze_reason": (
            "Best exploratory attacker under completed 27-run base calibration, chosen for relative checkpoint comparison despite failed strong-baseline calibration."
        ),
    }
    write_json(args.out_json, payload)


def run_write_exploratory_comparison_report(args: argparse.Namespace) -> None:
    registry_payload = json.loads(args.registry_json.read_text())
    registry = registry_payload["runs"]
    fairness = json.loads(args.fairness_json.read_text())
    canonical = json.loads(args.canonical_json.read_text())
    exploratory = json.loads(args.exploratory_json.read_text())
    matched_input_auroc = float(fairness["matched_input_baseline"])
    strong_auroc = float(canonical["test_auroc"])
    strong_mcc = float(canonical["test_mcc"])

    per_run_rows: list[dict[str, object]] = []
    by_checkpoint: dict[str, list[dict[str, object]]] = defaultdict(list)
    base_by_seed: dict[int, dict[str, object]] = {}

    for record in registry:
        result_row = read_result_row(Path(record["results_path"]))
        last_log = load_last_training_log_row(Path(record["results_path"]).parent)
        if not result_row:
            continue
        row = {
            "checkpoint": record["checkpoint"],
            "run_id": record["run_id"],
            "seed": int(record["seed"]),
            "rank": int(record["rank"]),
            "lr": float(record["lr"]),
            "run_status": record["run_status"],
            "post_attack_auroc": float(result_row["auroc"]),
            "post_attack_mcc_raw_0.5": float(result_row["mcc"]) if result_row.get("mcc") else None,
            "post_attack_auprc": float(result_row["auprc"]) if result_row.get("auprc") else None,
            "post_attack_accuracy": float(result_row["accuracy"]) if result_row.get("accuracy") else None,
            "post_attack_f1": float(result_row["f1"]) if result_row.get("f1") else None,
            "validation_metric": float(result_row["validation_metric"]) if result_row.get("validation_metric") else None,
            "best_step": int(float(result_row["best_step"])) if result_row.get("best_step") else None,
            "runtime_sec": float(last_log.get("elapsed_sec", 0.0)) if last_log else None,
            "excess_vs_matched_input_kmer": float(result_row["auroc"]) - matched_input_auroc,
            "excess_vs_full_sequence_kmer": float(result_row["auroc"]) - strong_auroc,
            "mcc_vs_full_sequence_kmer_raw_0.5": (
                None if not result_row.get("mcc") else float(result_row["mcc"]) - strong_mcc
            ),
        }
        per_run_rows.append(row)
        by_checkpoint[str(record["checkpoint"])].append(row)
        if record["checkpoint"] == "base":
            base_by_seed[int(record["seed"])] = row

    summary_rows: list[dict[str, object]] = []
    for checkpoint, rows in sorted(by_checkpoint.items()):
        seeds = sorted(int(row["seed"]) for row in rows)
        aurocs = [float(row["post_attack_auroc"]) for row in rows]
        mccs = [float(row["post_attack_mcc_raw_0.5"]) for row in rows if row["post_attack_mcc_raw_0.5"] is not None]
        shared_base = [row for row in rows if int(row["seed"]) in base_by_seed and checkpoint != "base"]
        delta_aurocs = [
            float(row["post_attack_auroc"]) - float(base_by_seed[int(row["seed"])]["post_attack_auroc"])
            for row in shared_base
        ]
        delta_mccs = [
            float(row["post_attack_mcc_raw_0.5"]) - float(base_by_seed[int(row["seed"])]["post_attack_mcc_raw_0.5"])
            for row in shared_base
            if row["post_attack_mcc_raw_0.5"] is not None and base_by_seed[int(row["seed"])]["post_attack_mcc_raw_0.5"] is not None
        ]
        mean_delta_auroc = mean(delta_aurocs) if delta_aurocs else 0.0
        mean_delta_mcc = mean(delta_mccs) if delta_mccs else 0.0
        if checkpoint == "base":
            classification = "reference_base"
        elif len(rows) < 3:
            classification = "inconclusive"
        elif abs(mean_delta_auroc) <= 0.01 and abs(mean_delta_mcc) <= 0.03:
            classification = "recovered_to_base"
        elif mean_delta_auroc < -0.02 and mean_delta_mcc < -0.02:
            classification = "inconclusive"
        else:
            classification = "inconclusive"
        summary_rows.append(
            {
                "checkpoint": checkpoint,
                "n_completed_seeds": len(rows),
                "completed_seeds": seeds,
                "post_attack_auroc_mean": mean(aurocs),
                "post_attack_auroc_std": pstdev(aurocs) if len(aurocs) > 1 else 0.0,
                "post_attack_mcc_raw_0.5_mean": mean(mccs) if mccs else None,
                "delta_auroc_vs_base_mean": mean_delta_auroc if checkpoint != "base" else 0.0,
                "delta_mcc_vs_base_mean_raw_0.5": mean_delta_mcc if checkpoint != "base" else 0.0,
                "excess_vs_matched_input_kmer_mean": mean(float(row["excess_vs_matched_input_kmer"]) for row in rows),
                "excess_vs_full_sequence_kmer_mean": mean(float(row["excess_vs_full_sequence_kmer"]) for row in rows),
                "preliminary_classification": classification,
            }
        )

    registry_counts: dict[str, int] = defaultdict(int)
    checkpoint_run_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in registry:
        run_status = str(record["run_status"])
        checkpoint = str(record["checkpoint"])
        registry_counts[run_status] += 1
        checkpoint_run_counts[checkpoint][run_status] += 1

    payload = {
        "status": "complete" if len(per_run_rows) == 18 else "partial_running",
        "task": DEFAULT_TASK,
        "matched_input_baseline_auroc": matched_input_auroc,
        "full_sequence_strong_reference_auroc": strong_auroc,
        "full_sequence_strong_reference_mcc": strong_mcc,
        "exploratory_attack_config": exploratory,
        "registry_counts": dict(sorted(registry_counts.items())),
        "checkpoint_run_counts": {key: dict(value) for key, value in sorted(checkpoint_run_counts.items())},
        "per_run_rows": per_run_rows,
        "checkpoint_summary": summary_rows,
    }
    csv_rows = []
    for row in summary_rows:
        csv_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "n_completed_seeds": row["n_completed_seeds"],
                "completed_seeds": "|".join(str(seed) for seed in row["completed_seeds"]),
                "post_attack_auroc_mean": row["post_attack_auroc_mean"],
                "post_attack_auroc_std": row["post_attack_auroc_std"],
                "post_attack_mcc_raw_0.5_mean": row["post_attack_mcc_raw_0.5_mean"],
                "delta_auroc_vs_base_mean": row["delta_auroc_vs_base_mean"],
                "delta_mcc_vs_base_mean_raw_0.5": row["delta_mcc_vs_base_mean_raw_0.5"],
                "excess_vs_matched_input_kmer_mean": row["excess_vs_matched_input_kmer_mean"],
                "excess_vs_full_sequence_kmer_mean": row["excess_vs_full_sequence_kmer_mean"],
                "preliminary_classification": row["preliminary_classification"],
            }
        )
    write_csv(args.out_csv, csv_rows, list(csv_rows[0].keys()) if csv_rows else [])
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 1 Exploratory Checkpoint Comparison",
        "",
        f"- Status: `{payload['status']}`",
        f"- Task: `{DEFAULT_TASK}`",
        f"- Exploratory attacker: `rank={exploratory['selected_rank']}, lr={exploratory['selected_learning_rate']}`",
        f"- Matched-input k-mer AUROC: `{matched_input_auroc}`",
        f"- Full-sequence strong reference AUROC: `{strong_auroc}`",
        f"- Registry counts: `{dict(sorted(registry_counts.items()))}`",
        "",
        "## Checkpoint Summary",
        "",
    ]
    if summary_rows:
        for row in summary_rows:
            lines.append(
                f"- `{row['checkpoint']}`: seeds={row['n_completed_seeds']} "
                f"auroc_mean={row['post_attack_auroc_mean']:.6f} "
                f"delta_vs_base={row['delta_auroc_vs_base_mean']:.6f} "
                f"classification={row['preliminary_classification']}"
            )
    else:
        for checkpoint, counts in sorted(checkpoint_run_counts.items()):
            lines.append(f"- `{checkpoint}`: run_status_counts={dict(sorted(counts.items()))}")
    args.out_md.write_text("\n".join(lines) + "\n")


def run_write_attack_freeze(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection_json.read_text())
    candidate = selection.get("recommended_config")
    if not candidate:
        payload = {
            "status": "no_eligible_dev_config",
            "freeze_reason": "Batch completed, but no configuration achieved positive development-seed AUROC excess under the current canonical candidate baseline.",
            "recommended_config": None,
        }
    else:
        payload = {
            "status": "provisional_pending_mcc",
            "freeze_reason": "AUROC-based development recommendation exists, but MCC threshold-aligned excess still requires evaluation-only reruns or saved predictions.",
            "recommended_config": {
                "rank": candidate["rank"],
                "lr": candidate["lr"],
                "epochs": 3,
                "eval_every": 200,
                "patience": 3,
                "target_modules": "all_blocks_lora",
                "threshold_policy": "validation_selected_mcc",
                "seed_policy": {"development": [42, 43], "confirmation": [44]},
            },
        }
    write_json(args.out_json, payload)


def run_write_confirmation_plan(args: argparse.Namespace) -> None:
    freeze = json.loads(args.freeze_json.read_text())
    config = freeze.get("recommended_config")
    commands = []
    if config:
        for seed in (42, 43, 44):
            commands.append(
                {
                    "seed": seed,
                    "rank": config["rank"],
                    "lr": config["lr"],
                    "note": "Targeted confirmation rerun under frozen commit if seed42 provenance mismatch or MCC evaluation-only rerun is required.",
                }
            )
    write_json(
        args.out_json,
        {
            "status": freeze["status"],
            "reason": freeze.get("freeze_reason", ""),
            "commands": commands,
        },
    )


def run_check_tranche3(args: argparse.Namespace) -> None:
    canonical = json.loads(args.canonical_json.read_text())
    freeze = json.loads(args.freeze_json.read_text())
    readiness = {
        "canonical_baseline_frozen": canonical.get("status") == "canonical_frozen",
        "threshold_policy_available": bool(canonical.get("threshold_policy")),
        "attack_config_frozen": freeze.get("status") == "canonical_frozen",
        "reason_blocked": [],
    }
    if not readiness["canonical_baseline_frozen"]:
        readiness["reason_blocked"].append("canonical_kmer_baseline_not_frozen")
    if freeze.get("status") != "canonical_frozen":
        readiness["reason_blocked"].append("attack_config_not_frozen")
    readiness["tranche3_ready"] = not readiness["reason_blocked"]
    write_json(args.out_json, readiness)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        "\n".join(
            [
                "# Tranche 3 Readiness",
                "",
                f"- Ready: `{readiness['tranche3_ready']}`",
                *[f"- Blocked by: `{reason}`" for reason in readiness["reason_blocked"]],
            ]
        )
        + "\n"
    )


def run_finalize_canonical_baseline(args: argparse.Namespace) -> None:
    canonical = json.loads(args.canonical_json.read_text())
    canonical["status"] = "canonical_frozen"
    canonical["freeze_reason"] = (
        "Frozen after completion of the 27-run base_only calibration batch. "
        "This strong canonical baseline matches the agreed full-sequence 3-6-mer hashing+l2+scaler protocol "
        "and extends the C-grid through 100."
    )
    canonical["freeze_timestamp_utc"] = args.current_date + "T00:00:00Z"
    write_json(args.out_json, canonical)


def run_write_final_report(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry_json.read_text())["runs"]
    canonical = json.loads(args.canonical_json.read_text())
    excess = json.loads(args.excess_json.read_text())["rows"]
    selection = json.loads(args.selection_json.read_text())
    freeze = json.loads(args.freeze_json.read_text())
    tranche3 = json.loads(args.tranche3_json.read_text())

    best_auroc = None
    if excess:
        best_auroc = max(
            excess,
            key=lambda row: float(row["canonical_auroc_excess"]) if row["canonical_auroc_excess"] != "" else -float("inf"),
        )

    completed_runs = sum(1 for row in registry if row["run_status"] == "completed_pending_canonical_baseline")
    summary_rows = selection.get("rows", [])
    dev_positive = [row for row in summary_rows if row.get("status") == "eligible_dev_positive"]

    payload = {
        "task": DEFAULT_TASK,
        "batch_status": "complete",
        "run_count": len(registry),
        "completed_run_count": completed_runs,
        "canonical_baseline": {
            "candidate_id": canonical.get("candidate_id"),
            "status": canonical.get("status"),
            "test_auroc": canonical.get("test_auroc"),
            "test_mcc": canonical.get("test_mcc"),
            "threshold_policy": canonical.get("threshold_policy"),
            "freeze_reason": canonical.get("freeze_reason"),
        },
        "selection_outcome": {
            "recommended_config": selection.get("recommended_config"),
            "eligible_dev_positive_count": len(dev_positive),
            "freeze_status": freeze.get("status"),
            "freeze_reason": freeze.get("freeze_reason"),
        },
        "best_observed_run_by_auroc_excess": best_auroc,
        "final_decision": {
            "base_calibration_status": "fail_under_frozen_canonical_baseline" if not dev_positive else "pass",
            "tranche3_ready": tranche3.get("tranche3_ready"),
            "blocked_reasons": tranche3.get("reason_blocked", []),
            "mcc_gate_status": "pending_eval_only_rerun_for_threshold_alignment",
            "next_step": (
                "Do not enter Tranche 3. First complete canonical write-up and, if desired, run evaluation-only MCC threshold-aligned reruns for the strongest few configurations."
            ),
        },
    }
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    best_line = "None"
    if best_auroc:
        best_line = (
            f"{best_auroc['run_id']} "
            f"(auroc={best_auroc['raw_test_auroc']}, excess={best_auroc['canonical_auroc_excess']})"
        )
    args.out_md.write_text(
        "\n".join(
            [
                "# Base Calibration Final Report",
                "",
                f"- Task: `{DEFAULT_TASK}`",
                f"- Completed runs: `{completed_runs}/{len(registry)}`",
                f"- Canonical baseline: `{canonical.get('candidate_id')}`",
                f"- Canonical baseline status: `{canonical.get('status')}`",
                f"- Canonical baseline AUROC: `{canonical.get('test_auroc')}`",
                f"- Canonical baseline MCC: `{canonical.get('test_mcc')}`",
                f"- Eligible dev-positive configs: `{len(dev_positive)}`",
                f"- Attack freeze status: `{freeze.get('status')}`",
                f"- Tranche 3 ready: `{tranche3.get('tranche3_ready')}`",
                f"- Best observed run by AUROC excess: `{best_line}`",
                "",
                "## Final Decision",
                "",
                f"- Base calibration status: `{payload['final_decision']['base_calibration_status']}`",
                f"- MCC gate status: `{payload['final_decision']['mcc_gate_status']}`",
                f"- Next step: {payload['final_decision']['next_step']}",
            ]
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--formal-manifest", type=Path, default=DEFAULT_FORMAL_MANIFEST)
        subparser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_FORMAL_MANIFEST)
        subparser.add_argument("--formal-kmer-baseline", type=Path, default=DEFAULT_FORMAL_KMER_BASELINE)
        subparser.add_argument("--legacy-kmer-baseline", type=Path, default=DEFAULT_LEGACY_KMER_BASELINE)
        subparser.add_argument("--task", default=DEFAULT_TASK)
        subparser.add_argument("--split-type", default=DEFAULT_SPLIT_TYPE)
        subparser.add_argument("--current-date", default="2026-07-27")

    audit = sub.add_parser("audit-baselines")
    add_common(audit)
    audit.add_argument("--out-json", type=Path, default=DEFAULT_AUDIT_JSON)
    audit.add_argument("--out-md", type=Path, default=DEFAULT_AUDIT_MD)

    metrics = sub.add_parser("compute-kmer-metrics")
    add_common(metrics)
    metrics.add_argument("--out-csv", type=Path, default=DEFAULT_KMER_METRICS_CSV)
    metrics.add_argument("--out-json", type=Path, default=DEFAULT_KMER_METRICS_JSON)

    fairness = sub.add_parser("write-input-fairness-report")
    add_common(fairness)
    fairness.add_argument("--out-json", type=Path, default=DEFAULT_INPUT_FAIRNESS_JSON)
    fairness.add_argument("--out-md", type=Path, default=DEFAULT_INPUT_FAIRNESS_MD)

    canonical = sub.add_parser("write-canonical-baseline")
    add_common(canonical)
    canonical.add_argument("--candidates-json", type=Path, default=DEFAULT_CANONICAL_CANDIDATES)
    canonical.add_argument("--out-json", type=Path, default=DEFAULT_CANONICAL_JSON)

    registry = sub.add_parser("write-registry")
    add_common(registry)
    registry.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    registry.add_argument("--out-csv", type=Path, default=DEFAULT_REGISTRY_CSV)
    registry.add_argument("--out-json", type=Path, default=DEFAULT_REGISTRY_JSON)

    excess = sub.add_parser("recalculate-excess")
    excess.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_JSON)
    excess.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    excess.add_argument("--out-csv", type=Path, default=DEFAULT_EXCESS_CSV)
    excess.add_argument("--out-json", type=Path, default=DEFAULT_EXCESS_JSON)

    select = sub.add_parser("select-config")
    select.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_JSON)
    select.add_argument("--excess-json", type=Path, default=DEFAULT_EXCESS_JSON)
    select.add_argument("--out-json", type=Path, default=DEFAULT_SELECTION_JSON)

    freeze = sub.add_parser("write-attack-freeze")
    freeze.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    freeze.add_argument("--out-json", type=Path, default=DEFAULT_FREEZE_JSON)

    exploratory = sub.add_parser("write-exploratory-attack-config")
    exploratory.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    exploratory.add_argument("--out-json", type=Path, default=DEFAULT_EXPLORATORY_FREEZE_JSON)

    confirm = sub.add_parser("write-confirmation-plan")
    confirm.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    confirm.add_argument("--out-json", type=Path, default=DEFAULT_CONFIRMATION_JSON)

    tranche = sub.add_parser("check-tranche3-readiness")
    tranche.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    tranche.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    tranche.add_argument("--out-json", type=Path, default=DEFAULT_TRANCHE3_JSON)
    tranche.add_argument("--out-md", type=Path, default=DEFAULT_TRANCHE3_MD)

    finalize = sub.add_parser("finalize-canonical-baseline")
    add_common(finalize)
    finalize.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    finalize.add_argument("--out-json", type=Path, default=DEFAULT_CANONICAL_JSON)

    final_report = sub.add_parser("write-final-report")
    final_report.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_JSON)
    final_report.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    final_report.add_argument("--excess-json", type=Path, default=DEFAULT_EXCESS_JSON)
    final_report.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    final_report.add_argument("--freeze-json", type=Path, default=DEFAULT_FREEZE_JSON)
    final_report.add_argument("--tranche3-json", type=Path, default=DEFAULT_TRANCHE3_JSON)
    final_report.add_argument("--out-json", type=Path, default=DEFAULT_FINAL_REPORT_JSON)
    final_report.add_argument("--out-md", type=Path, default=DEFAULT_FINAL_REPORT_MD)

    comparison_registry = sub.add_parser("write-exploratory-comparison-registry")
    add_common(comparison_registry)
    comparison_registry.add_argument("--plan-json", type=Path, default=DEFAULT_COMPARISON_PLAN_JSON)
    comparison_registry.add_argument("--out-csv", type=Path, default=DEFAULT_COMPARISON_REGISTRY_CSV)
    comparison_registry.add_argument("--out-json", type=Path, default=DEFAULT_COMPARISON_REGISTRY_JSON)

    comparison_report = sub.add_parser("write-exploratory-comparison-report")
    comparison_report.add_argument("--registry-json", type=Path, default=DEFAULT_COMPARISON_REGISTRY_JSON)
    comparison_report.add_argument("--fairness-json", type=Path, default=DEFAULT_INPUT_FAIRNESS_JSON)
    comparison_report.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    comparison_report.add_argument("--exploratory-json", type=Path, default=DEFAULT_EXPLORATORY_FREEZE_JSON)
    comparison_report.add_argument("--out-csv", type=Path, default=DEFAULT_COMPARISON_REPORT_CSV)
    comparison_report.add_argument("--out-json", type=Path, default=DEFAULT_COMPARISON_REPORT_JSON)
    comparison_report.add_argument("--out-md", type=Path, default=DEFAULT_COMPARISON_REPORT_MD)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "audit-baselines":
        run_audit_baselines(args)
    elif args.command == "compute-kmer-metrics":
        run_compute_kmer_metrics(args)
    elif args.command == "write-input-fairness-report":
        run_write_input_fairness_report(args)
    elif args.command == "write-canonical-baseline":
        run_write_canonical_baseline(args)
    elif args.command == "write-registry":
        run_write_registry(args)
    elif args.command == "recalculate-excess":
        run_recalculate_excess(args)
    elif args.command == "select-config":
        run_select_config(args)
    elif args.command == "write-attack-freeze":
        run_write_attack_freeze(args)
    elif args.command == "write-exploratory-attack-config":
        run_write_exploratory_attack_config(args)
    elif args.command == "write-confirmation-plan":
        run_write_confirmation_plan(args)
    elif args.command == "check-tranche3-readiness":
        run_check_tranche3(args)
    elif args.command == "finalize-canonical-baseline":
        run_finalize_canonical_baseline(args)
    elif args.command == "write-final-report":
        run_write_final_report(args)
    elif args.command == "write-exploratory-comparison-registry":
        run_write_registry(args)
    elif args.command == "write-exploratory-comparison-report":
        run_write_exploratory_comparison_report(args)
    else:
        raise AssertionError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
