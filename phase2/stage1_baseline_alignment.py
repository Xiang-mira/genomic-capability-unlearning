"""Host Tropism Stage 1 baseline-alignment package generator.

This module is intentionally evaluation-only. It reuses the Stage 1 manifests
and k-mer protocol geometry, exports matched-input k-mer predictions, audits the
LoRA input budget, and aggregates existing LoRA result CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "hvue_human_host_tropism"
DEFAULT_SPLIT_TYPE = "cluster_disjoint"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "data/phase2/stage1_formal_experiment_20260727"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data/phase2/stage1_baseline_alignment_20260729"
EARLIER_MATCHED_AUROC = 0.8496470934222136
FULL_SEQUENCE_STRONG_AUROC = 0.8930006862072345
FULL_SEQUENCE_STRONG_MCC = 0.6452985196139533
BEST_LORA_RUN_ID = "fresh_lora_base_r32_lr5e-5_seed44"
LORA_MAX_LENGTH = 512


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    label: int
    sequence: str


@dataclass(frozen=True)
class KmerProtocol:
    config_id: str
    vectorizer_type: str
    sequence_length_policy: str
    c_grid: tuple[float, ...]
    solver: str
    max_iter: int
    use_hashing: bool
    use_scaler: bool
    cleaning_policy: str
    historical_auroc: float | None = None


EARLIER_MATCHED_PROTOCOL = KmerProtocol(
    config_id="earlier_matched_input_count_first512_cgrid10",
    vectorizer_type="CountVectorizer",
    sequence_length_policy="first_512_after_ACGTN_cleaning",
    c_grid=(0.001, 0.01, 0.1, 1.0, 10.0),
    solver="lbfgs",
    max_iter=2000,
    use_hashing=False,
    use_scaler=False,
    cleaning_policy="retain ACGTN only, then prefix truncate to 512",
    historical_auroc=EARLIER_MATCHED_AUROC,
)

STRONG_MATCHED_PROTOCOL = KmerProtocol(
    config_id="strong_matched_input_hashing_first512_cgrid100",
    vectorizer_type="HashingVectorizer",
    sequence_length_policy="first_512_raw_characters_uppercase",
    c_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    solver="liblinear",
    max_iter=1000,
    use_hashing=True,
    use_scaler=True,
    cleaning_policy="same raw prefix region as LoRA, uppercase before hashing",
)

FULL_SEQUENCE_STRONG_PROTOCOL = KmerProtocol(
    config_id="full_sequence_strong_hashing_cgrid100",
    vectorizer_type="HashingVectorizer",
    sequence_length_policy="full_sequence_uppercase",
    c_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    solver="liblinear",
    max_iter=1000,
    use_hashing=True,
    use_scaler=True,
    cleaning_policy="uppercase full sequence before hashing",
    historical_auroc=FULL_SEQUENCE_STRONG_AUROC,
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_split(value: str) -> str:
    split = str(value or "").strip().lower()
    return "val" if split in {"dev", "valid", "validation"} else split


def normalize_split_type(value: str) -> str:
    split_type = str(value or "").strip().lower()
    if split_type in {"cluster-disjoint", "cluster_disjoint", "disjoint"}:
        return "cluster_disjoint"
    return split_type or "random"


def read_manifest_samples(path: Path, task: str, split_type: str) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_csv_rows(path):
        if row.get("task") != task:
            continue
        if normalize_split_type(row.get("split_type", "")) != split_type:
            continue
        samples.append(
            Sample(
                sample_id=str(row.get("id") or row.get("record_id") or ""),
                split=normalize_split(row.get("split", "")),
                label=int(float(row.get("label", "0"))),
                sequence=str(row.get("sequence", "")),
            )
        )
    if not samples:
        raise RuntimeError(f"No samples found for task={task} split_type={split_type} in {path}")
    return samples


def lora_retained_region(sequence: str, max_length: int = LORA_MAX_LENGTH) -> tuple[int, int, str]:
    end = min(len(sequence), max_length)
    return 0, end, sequence[:end]


def cleaned_acgtn_prefix(sequence: str, max_length: int) -> str:
    cleaned = "".join(ch for ch in sequence.upper() if ch in {"A", "C", "G", "T", "N"})
    return cleaned[:max_length]


def sequence_for_protocol(sample: Sample, protocol: KmerProtocol) -> str:
    if protocol is EARLIER_MATCHED_PROTOCOL:
        return cleaned_acgtn_prefix(sample.sequence, LORA_MAX_LENGTH)
    if protocol is STRONG_MATCHED_PROTOCOL:
        return lora_retained_region(sample.sequence, LORA_MAX_LENGTH)[2].upper()
    if protocol is FULL_SEQUENCE_STRONG_PROTOCOL:
        return sample.sequence.upper()
    raise AssertionError(protocol.config_id)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo))


def length_summary(samples: list[Sample], protocol: KmerProtocol | None = None) -> dict[str, object]:
    rows = samples
    lengths = [len(sequence_for_protocol(row, protocol)) if protocol else len(row.sequence) for row in rows]
    by_split = {}
    for split in ("train", "val", "test"):
        split_lengths = [lengths[idx] for idx, row in enumerate(rows) if row.split == split]
        if not split_lengths:
            continue
        by_split[split] = {
            "n": len(split_lengths),
            "min": min(split_lengths),
            "median": percentile(split_lengths, 0.5),
            "mean": float(mean(split_lengths)),
            "p90": percentile(split_lengths, 0.9),
            "p95": percentile(split_lengths, 0.95),
            "max": max(split_lengths),
            "truncated_count": sum(len(row.sequence) > LORA_MAX_LENGTH for row in rows if row.split == split),
        }
        by_split[split]["truncated_fraction"] = by_split[split]["truncated_count"] / by_split[split]["n"]
    return {
        "overall": {
            "n": len(lengths),
            "min": min(lengths),
            "median": percentile(lengths, 0.5),
            "mean": float(mean(lengths)),
            "p90": percentile(lengths, 0.9),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "by_split": by_split,
    }


def export_lora_input_audit(samples: list[Sample], out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        start, end, retained = lora_retained_region(sample.sequence)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "raw_sequence_length": len(sample.sequence),
                "tokenized_length_before_truncation": len(sample.sequence),
                "tokenized_length_after_truncation": len(retained),
                "retained_raw_start": start,
                "retained_raw_end": end,
                "truncated": len(sample.sequence) > LORA_MAX_LENGTH,
                "crop_policy": "deterministic_prefix_raw_before_tokenization",
                "seed": "not_applicable",
            }
        )
    csv_path = out_dir / "lora_input_policy_per_sample.csv"
    write_csv(csv_path, rows)
    summary = {
        "configured_max_sequence_length": LORA_MAX_LENGTH,
        "actual_model_context_length": 8192,
        "length_measure": "raw characters before CharLevelTokenizer.tokenize",
        "truncation_side": "right",
        "retained_region": "prefix, raw offsets [0, min(raw_length, 512))",
        "random_cropping": False,
        "train_val_test_policy_identical": True,
        "padding_policy": "pad_batch pads token IDs to batch max with tokenizer.pad_id",
        "special_token_handling": "no explicit special-token insertion in phase2/utils.py::tokenize_batch",
        "token_length_note": "The installed environment lacks evo, but the traced runtime code passes seq[:512] directly to CharLevelTokenizer; retained token count is therefore recorded as the retained raw character count for this char-level policy.",
        "length_summary_raw": length_summary(samples),
        "length_summary_retained": length_summary(samples, STRONG_MATCHED_PROTOCOL),
        "per_sample_csv": str(csv_path),
    }
    md_path = out_dir / "lora_input_policy_audit.md"
    md_path.write_text(
        "\n".join(
            [
                "# LoRA Input Policy Audit",
                "",
                f"- Configured max sequence length: `{LORA_MAX_LENGTH}`",
                "- Truncation policy: raw-prefix truncation via `seq[:max_length]` before `CharLevelTokenizer.tokenize`.",
                "- Truncation side: right; retained raw offsets are `[0, min(raw_length, 512))`.",
                "- Cropping: deterministic prefix only; no random cropping or multi-window sampling found.",
                "- Train/validation/test policy: identical tokenization and truncation.",
                "- Padding: batch-local padding with `tokenizer.pad_id` in `pad_batch`.",
                f"- Truncated fraction overall: `{sum(len(s.sequence) > LORA_MAX_LENGTH for s in samples) / len(samples)}`",
                f"- Per-sample audit: `{csv_path}`",
            ]
        )
        + "\n"
    )
    return md_path, csv_path, summary


def build_matrix(sequences: list[str], protocol: KmerProtocol):
    if protocol.use_hashing:
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        return vectorizer.transform(sequences)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 6), lowercase=False, binary=False)
    return vectorizer.fit_transform(sequences)


def select_mcc_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = sorted({0.5, *[float(value) for value in probs]})
    best_threshold = 0.5
    best_mcc = -2.0
    for threshold in thresholds:
        pred = (probs >= threshold).astype(np.int64)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc + 1e-12:
            best_threshold = threshold
            best_mcc = mcc
    return best_threshold, best_mcc


def binary_metric_block(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y_true, pred, labels=[0, 1]).ravel()]
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "predicted_positive_rate": float(pred.mean()),
        "label_positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_kmer_protocol(samples: list[Sample], protocol: KmerProtocol, pred_dir: Path, seed: int) -> dict[str, object]:
    sequences = [sequence_for_protocol(sample, protocol) for sample in samples]
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    splits = np.array([sample.split for sample in samples])
    masks = {split: splits == split for split in ("train", "val", "test")}
    for split, mask in masks.items():
        if int(mask.sum()) == 0 or len(set(labels[mask])) < 2:
            raise RuntimeError(f"{protocol.config_id} split {split} is missing rows or classes")

    matrix = build_matrix(sequences, protocol)
    x_all = matrix
    x_train = matrix[masks["train"]]
    if protocol.use_scaler:
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x_train)
        x_all = scaler.transform(matrix)

    best: tuple[float, float, LogisticRegression] | None = None
    c_candidates = []
    for c_value in protocol.c_grid:
        clf = LogisticRegression(
            C=c_value,
            solver=protocol.solver,
            max_iter=protocol.max_iter,
            class_weight="balanced",
            random_state=seed if protocol.solver == "liblinear" else None,
        )
        clf.fit(x_train, labels[masks["train"]])
        val_probs = clf.predict_proba(x_all[masks["val"]])[:, 1]
        val_auroc = float(roc_auc_score(labels[masks["val"]], val_probs))
        score = max(val_auroc, 1.0 - val_auroc)
        c_candidates.append({"C": c_value, "validation_auroc": val_auroc, "selection_score": score})
        if best is None or score > best[0]:
            best = (score, c_value, clf)
    assert best is not None
    _, selected_c, clf = best
    val_probs = clf.predict_proba(x_all[masks["val"]])[:, 1]
    test_probs = clf.predict_proba(x_all[masks["test"]])[:, 1]
    selected_threshold, val_mcc = select_mcc_threshold(labels[masks["val"]], val_probs)

    prediction_paths = {}
    prediction_hashes = {}
    for split, probs in (("val", val_probs), ("test", test_probs)):
        split_samples = [sample for sample in samples if sample.split == split]
        path = pred_dir / f"{protocol.config_id}_{split}_predictions.csv"
        pred_rows = []
        for sample, prob in zip(split_samples, probs):
            start, end, retained = lora_retained_region(sample.sequence)
            pred_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "split": split,
                    "label": sample.label,
                    "probability_positive": float(prob),
                    "selected_threshold": selected_threshold,
                    "predicted_label": int(float(prob) >= selected_threshold),
                    "retained_raw_start": 0 if protocol is not FULL_SEQUENCE_STRONG_PROTOCOL else 0,
                    "retained_raw_end": end if protocol is not FULL_SEQUENCE_STRONG_PROTOCOL else len(sample.sequence),
                    "input_sequence_length": len(sequence_for_protocol(sample, protocol)),
                }
            )
        write_csv(path, pred_rows)
        prediction_paths[split] = str(path)
        prediction_hashes[split] = file_sha256(path)

    metrics = {
        "config_id": protocol.config_id,
        "vectorizer_type": protocol.vectorizer_type,
        "feature_implementation": "hashing_l2_scaled" if protocol.use_hashing else "count_raw",
        "sequence_length_policy": protocol.sequence_length_policy,
        "cleaning_policy": protocol.cleaning_policy,
        "kmer_range": "3-6",
        "c_grid": ",".join(f"{value:g}" for value in protocol.c_grid),
        "selected_c": selected_c,
        "selection_split": "val",
        "selection_metric": "separability_auroc",
        "threshold_policy": "validation_selected_mcc",
        "validation_selected_threshold": selected_threshold,
        "validation_mcc_at_selected_threshold": val_mcc,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "length_summary": length_summary(samples, protocol),
        "test_metrics_validation_threshold": binary_metric_block(labels[masks["test"]], test_probs, selected_threshold),
        "test_metrics_threshold_0_5": binary_metric_block(labels[masks["test"]], test_probs, 0.5),
        "prediction_paths": prediction_paths,
        "prediction_hashes": prediction_hashes,
        "c_candidates": c_candidates,
        "historical_auroc": protocol.historical_auroc,
    }
    return metrics


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def result_row_for_run(results_root: Path, rank: int, lr_dir: str, seed: int) -> dict[str, str] | None:
    path = results_root / f"fresh_lora/base/rank_{rank}/lr_{lr_dir}/seed_{seed}/eval_benchmarks.csv"
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    return rows[0] if rows else None


def discover_lora_runs(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(results_root.glob("fresh_lora/base/rank_*/lr_*/seed_*/eval_benchmarks.csv")):
        parts = path.parts
        rank = int(parts[-4].split("_", 1)[1])
        lr_dir = parts[-3].split("_", 1)[1]
        seed = int(parts[-2].split("_", 1)[1])
        result_rows = read_csv_rows(path)
        if not result_rows:
            continue
        row = result_rows[0]
        run_id = f"fresh_lora_base_r{rank}_lr{lr_dir}_seed{seed}"
        rows.append(
            {
                "run_id": run_id,
                "rank": rank,
                "lr": float(lr_dir),
                "lr_label": lr_dir,
                "seed": seed,
                "results_path": str(path),
                "raw_lora_auroc": float(row["auroc"]),
                "raw_lora_mcc": float(row["mcc"]) if row.get("mcc") else None,
                "validation_metric": float(row["validation_metric"]) if row.get("validation_metric") else None,
                "best_step": int(float(row["best_step"])) if row.get("best_step") else None,
            }
        )
    return rows


def aggregate_lora_runs(
    runs: list[dict[str, object]],
    earlier_kmer: dict[str, object],
    strong_matched: dict[str, object],
    full_strong: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    earlier_auroc = float(earlier_kmer["test_metrics_validation_threshold"]["auroc"])
    strong_auroc = float(strong_matched["test_metrics_validation_threshold"]["auroc"])
    full_auroc = float(full_strong["test_metrics_validation_threshold"]["auroc"])
    earlier_mcc = float(earlier_kmer["test_metrics_validation_threshold"]["mcc"])
    strong_mcc = float(strong_matched["test_metrics_validation_threshold"]["mcc"])
    full_mcc = float(full_strong["test_metrics_validation_threshold"]["mcc"])

    enriched = []
    for row in runs:
        item = dict(row)
        item.update(
            {
                "excess_vs_earlier_matched_input_kmer": item["raw_lora_auroc"] - earlier_auroc,
                "excess_vs_strong_matched_input_kmer": item["raw_lora_auroc"] - strong_auroc,
                "gap_vs_full_sequence_strong_kmer": item["raw_lora_auroc"] - full_auroc,
                "validation_selected_threshold": None,
                "mcc_excess_vs_earlier_matched_input_kmer": None,
                "mcc_excess_vs_strong_matched_input_kmer": None,
                "mcc_gap_vs_full_sequence_strong_kmer": None,
                "mcc_status": "missing_lora_validation_and_test_prediction_exports",
                "diagnostic_raw_mcc_minus_earlier_kmer_mcc": None if item["raw_lora_mcc"] is None else item["raw_lora_mcc"] - earlier_mcc,
                "diagnostic_raw_mcc_minus_strong_matched_kmer_mcc": None if item["raw_lora_mcc"] is None else item["raw_lora_mcc"] - strong_mcc,
                "diagnostic_raw_mcc_minus_full_strong_kmer_mcc": None if item["raw_lora_mcc"] is None else item["raw_lora_mcc"] - full_mcc,
            }
        )
        enriched.append(item)

    grouped: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        grouped[(int(row["rank"]), float(row["lr"]))].append(row)

    summary = []
    for (rank, lr), group_rows in sorted(grouped.items()):
        aurocs = [float(row["raw_lora_auroc"]) for row in group_rows]
        mccs = [float(row["raw_lora_mcc"]) for row in group_rows if row["raw_lora_mcc"] is not None]
        dev_rows = [row for row in group_rows if int(row["seed"]) in {42, 43}]
        confirm = next((row for row in group_rows if int(row["seed"]) == 44), None)
        summary.append(
            {
                "rank": rank,
                "lr": lr,
                "n_completed": len(group_rows),
                "seeds": "|".join(str(row["seed"]) for row in sorted(group_rows, key=lambda r: int(r["seed"]))),
                "auroc_mean": mean(aurocs),
                "auroc_std": pstdev(aurocs) if len(aurocs) > 1 else 0.0,
                "auroc_min": min(aurocs),
                "auroc_max": max(aurocs),
                "individual_aurocs": "|".join(f"{row['seed']}:{float(row['raw_lora_auroc']):.12g}" for row in sorted(group_rows, key=lambda r: int(r["seed"]))),
                "mcc_mean_raw": mean(mccs) if mccs else None,
                "mcc_std_raw": pstdev(mccs) if len(mccs) > 1 else 0.0,
                "mcc_min_raw": min(mccs) if mccs else None,
                "mcc_max_raw": max(mccs) if mccs else None,
                "individual_raw_mccs": "|".join(f"{row['seed']}:{float(row['raw_lora_mcc']):.12g}" for row in sorted(group_rows, key=lambda r: int(r["seed"])) if row["raw_lora_mcc"] is not None),
                "auroc_excess_vs_earlier_matched_mean": mean(float(row["excess_vs_earlier_matched_input_kmer"]) for row in group_rows),
                "auroc_excess_vs_strong_matched_mean": mean(float(row["excess_vs_strong_matched_input_kmer"]) for row in group_rows),
                "auroc_gap_vs_full_sequence_strong_mean": mean(float(row["gap_vs_full_sequence_strong_kmer"]) for row in group_rows),
                "dev_seed_auroc_excess_vs_strong_matched_all_positive": (
                    len(dev_rows) == 2 and all(float(row["excess_vs_strong_matched_input_kmer"]) > 0 for row in dev_rows)
                ),
                "confirmation_seed44_auroc_excess_vs_strong_matched": None
                if confirm is None
                else float(confirm["excess_vs_strong_matched_input_kmer"]),
                "formal_mcc_status": "missing_lora_prediction_exports",
            }
        )

    best_run = max(enriched, key=lambda row: float(row["raw_lora_auroc"])) if enriched else None
    dev_eligible = [row for row in summary if row["dev_seed_auroc_excess_vs_strong_matched_all_positive"]]
    best_dev = None
    if dev_eligible:
        best_dev = max(dev_eligible, key=lambda row: (row["auroc_excess_vs_strong_matched_mean"], -row["auroc_std"]))
    most_stable = min(summary, key=lambda row: float(row["auroc_std"])) if summary else None
    judgement = {
        "best_individual_run": best_run,
        "best_development_configuration": best_dev,
        "confirmation_seed_result": None
        if best_dev is None
        else next(
            (
                row
                for row in enriched
                if int(row["rank"]) == int(best_dev["rank"]) and float(row["lr"]) == float(best_dev["lr"]) and int(row["seed"]) == 44
            ),
            None,
        ),
        "most_stable_configuration": most_stable,
        "selected_formal_attacker": None,
        "selected_formal_attacker_status": "not_selected_missing_formal_mcc_and_unstable_strong_matched_excess",
    }
    return enriched, summary, judgement


def copy_prediction_for_mcc(config: dict[str, object], out_dir: Path) -> dict[str, str]:
    copied = {}
    for split, src in dict(config["prediction_paths"]).items():
        dst = out_dir / Path(src).name
        shutil.copy2(src, dst)
        copied[split] = str(dst)
    return copied


def write_reports(
    out_dir: Path,
    samples: list[Sample],
    lora_audit: dict[str, object],
    kmer_metrics: list[dict[str, object]],
    lora_runs: list[dict[str, object]],
    lora_summary: list[dict[str, object]],
    lora_judgement: dict[str, object],
    commands: list[str],
) -> None:
    by_id = {row["config_id"]: row for row in kmer_metrics}
    earlier = by_id[EARLIER_MATCHED_PROTOCOL.config_id]
    strong = by_id[STRONG_MATCHED_PROTOCOL.config_id]
    full = by_id[FULL_SEQUENCE_STRONG_PROTOCOL.config_id]
    best = lora_judgement["best_individual_run"]
    strong_auroc = float(strong["test_metrics_validation_threshold"]["auroc"])
    strong_mcc = float(strong["test_metrics_validation_threshold"]["mcc"])
    best_excess = None if best is None else float(best["raw_lora_auroc"]) - strong_auroc
    stable_auroc_configs = [
        row
        for row in lora_summary
        if bool(row["dev_seed_auroc_excess_vs_strong_matched_all_positive"])
        and row["confirmation_seed44_auroc_excess_vs_strong_matched"] is not None
        and float(row["confirmation_seed44_auroc_excess_vs_strong_matched"]) > 0
    ]
    conclusion = {
        "lora_actual_input": "first 512 raw sequence characters, right-truncated before CharLevelTokenizer tokenization",
        "input_matched_kmer_baseline": STRONG_MATCHED_PROTOCOL.config_id,
        "strong_matched_input_kmer_auroc": strong_auroc,
        "strong_matched_input_kmer_mcc": strong_mcc,
        "best_lora_positive_auroc_excess_vs_strong_matched": best_excess,
        "stable_positive_auroc_advantage_across_development_and_confirmation_seeds": bool(stable_auroc_configs),
        "stable_positive_auroc_configurations": stable_auroc_configs,
        "formal_lora_mcc_excess": None,
        "formal_lora_mcc_excess_status": "unavailable: Stage 1 LoRA validation/test prediction exports and retained checkpoints are missing",
        "supports_starting_formal_TAR": False,
        "tar_reason": "Some configurations retain stable positive AUROC excess under the strong matched-input baseline, but formal validation-threshold LoRA MCC cannot be reconstructed from aggregate-only outputs; the current evidence therefore does not establish a reproducible AUROC/MCC margin for TAR.",
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": DEFAULT_TASK,
        "split_type": DEFAULT_SPLIT_TYPE,
        "conclusion": conclusion,
        "lora_input_policy": lora_audit,
        "kmer_results": kmer_metrics,
        "lora_runs": lora_runs,
        "lora_configuration_summary": lora_summary,
        "lora_judgement": lora_judgement,
        "sample_counts": {split: sum(1 for sample in samples if sample.split == split) for split in ("train", "val", "test")},
    }
    write_json(out_dir / "stage1_baseline_alignment_report.json", report)

    comparison_rows = []
    for row in kmer_metrics:
        comparison_rows.append(
            {
                "config_id": row["config_id"],
                "n_train": row["n_train"],
                "n_val": row["n_val"],
                "n_test": row["n_test"],
                "sequence_length_policy": row["sequence_length_policy"],
                "length_mean": row["length_summary"]["overall"]["mean"],
                "length_median": row["length_summary"]["overall"]["median"],
                "kmer_range": row["kmer_range"],
                "feature_implementation": row["feature_implementation"],
                "c_grid": row["c_grid"],
                "selected_c": row["selected_c"],
                "auroc": row["test_metrics_validation_threshold"]["auroc"],
                "mcc": row["test_metrics_validation_threshold"]["mcc"],
                "mcc_at_0_5": row["test_metrics_threshold_0_5"]["mcc"],
                "threshold_policy": row["threshold_policy"],
                "prediction_val_path": row["prediction_paths"]["val"],
                "prediction_test_path": row["prediction_paths"]["test"],
            }
        )
    write_csv(out_dir / "kmer_input_fairness_comparison.csv", comparison_rows)
    write_csv(out_dir / "base_calibration_27run_unified.csv", lora_runs)

    summary_lines = [
        "# Base Calibration 27-Run Unified Summary",
        "",
        "| rank | lr | n | AUROC mean | AUROC sd | raw MCC mean | excess vs strong matched mean | dev seeds positive? |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in lora_summary:
        summary_lines.append(
            f"| {row['rank']} | {row['lr']} | {row['n_completed']} | {row['auroc_mean']:.6f} | "
            f"{row['auroc_std']:.6f} | {row['mcc_mean_raw']:.6f} | "
            f"{row['auroc_excess_vs_strong_matched_mean']:.6f} | {row['dev_seed_auroc_excess_vs_strong_matched_all_positive']} |"
        )
    (out_dir / "base_calibration_27run_unified.md").write_text("\n".join(summary_lines) + "\n")

    mcc_payload = {
        "threshold_policy": "select threshold on validation predictions only; freeze once on test",
        "kmer_results": kmer_metrics,
        "lora_results": {
            "status": "missing_prediction_exports",
            "reason": "Only aggregate eval_benchmarks.csv files are present; runs used --discard-task-checkpoint and no validation/test probability tables were saved.",
            "formal_mcc_excess": None,
        },
    }
    write_json(out_dir / "mcc_threshold_alignment_report.json", mcc_payload)
    (out_dir / "mcc_threshold_alignment_report.md").write_text(
        "\n".join(
            [
                "# MCC Threshold Alignment",
                "",
                "- Rule: select MCC-maximizing threshold on validation predictions only, then apply it once to test predictions.",
                f"- Earlier matched-input k-mer MCC: `{earlier['test_metrics_validation_threshold']['mcc']}`",
                f"- Strong matched-input k-mer MCC: `{strong_mcc}`",
                f"- Full-sequence strong k-mer MCC: `{full['test_metrics_validation_threshold']['mcc']}`",
                "- LoRA formal MCC excess: unavailable because validation/test probability exports are missing and checkpoints were discarded.",
            ]
        )
        + "\n"
    )

    md = [
        "# Stage 1 Baseline Alignment Report",
        "",
        "## Executive Conclusion",
        "",
        f"LoRA received the first `{LORA_MAX_LENGTH}` raw sequence characters. The fair strong input-matched k-mer baseline is `{STRONG_MATCHED_PROTOCOL.config_id}`, not the full-sequence reference. Its AUROC is `{strong_auroc:.12f}` and its validation-threshold MCC is `{strong_mcc:.12f}`.",
        "",
        f"The best individual LoRA run is `{best['run_id'] if best else ''}` with AUROC `{best['raw_lora_auroc'] if best else ''}`; its AUROC excess against the strong matched-input k-mer baseline is `{best_excess}`. Positive AUROC excess is stable for at least one rank/LR configuration, but formal LoRA MCC excess cannot be recomputed without per-sample validation/test probabilities, so the combined AUROC/MCC evidence is not sufficient for formal TAR.",
        "",
        "## Why 0.849647 and 0.893001 Differ",
        "",
        "The `0.849647` result used a first-512, raw-count `CountVectorizer` path with C only through 10. The `0.893001` result used full sequences with the stronger hashing+l2+StandardScaler geometry and C through 100. The discrepancy is therefore caused by both input length and baseline implementation strength; the new strong matched-input run isolates the LoRA-length budget while retaining the stronger implementation.",
        "",
        "## Key Results",
        "",
        f"- Earlier matched-input k-mer AUROC: `{earlier['test_metrics_validation_threshold']['auroc']}`",
        f"- New strong matched-input k-mer AUROC: `{strong_auroc}`",
        f"- Full-sequence strong k-mer AUROC: `{full['test_metrics_validation_threshold']['auroc']}`",
        f"- Strong matched-input selected C: `{strong['selected_c']}`",
        f"- Strong matched-input selected MCC threshold: `{strong['validation_selected_threshold']}`",
        "",
        "## TAR Judgment",
        "",
        "The current evidence does not support starting formal TAR. It supports preserving the exploratory artifacts, but the formal gate needs either saved LoRA validation/test prediction exports or an evaluation-only rerun that exports them, followed by a stable positive excess check across development seeds.",
    ]
    (out_dir / "stage1_baseline_alignment_report.md").write_text("\n".join(md) + "\n")

    registry = {
        "repository_commit": git_text(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_text(["status", "--porcelain"])),
        "git_status_short": git_text(["status", "--short"]).splitlines(),
        "dirty_diff_sha256": hashlib.sha256(git_text(["diff"]).encode("utf-8")).hexdigest(),
        "manifest_path": str(DEFAULT_MANIFEST),
        "manifest_sha256": file_sha256(DEFAULT_MANIFEST),
        "row_order_sha256": stable_hash([(s.sample_id, s.split, s.label, s.sequence) for s in samples]),
        "commands": commands,
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "prediction_hashes": {row["config_id"]: row["prediction_hashes"] for row in kmer_metrics},
        "selected_c": {row["config_id"]: row["selected_c"] for row in kmer_metrics},
        "selected_mcc_thresholds": {row["config_id"]: row["validation_selected_threshold"] for row in kmer_metrics},
        "model_and_adapter_hashes": {
            "lora": "unavailable: checkpoints discarded by Stage 1 run arguments",
        },
    }
    write_json(out_dir / "baseline_alignment_registry.json", registry)


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "kmer_input_fairness_predictions"
    mcc_pred_dir = out_dir / "mcc_threshold_alignment_predictions"
    pred_dir.mkdir(exist_ok=True)
    mcc_pred_dir.mkdir(exist_ok=True)

    samples = read_manifest_samples(args.manifest, args.task, args.split_type)
    _md_path, _csv_path, lora_audit = export_lora_input_audit(samples, out_dir)
    kmer_metrics = [
        evaluate_kmer_protocol(samples, EARLIER_MATCHED_PROTOCOL, pred_dir, args.seed),
        evaluate_kmer_protocol(samples, STRONG_MATCHED_PROTOCOL, pred_dir, args.seed),
        evaluate_kmer_protocol(samples, FULL_SEQUENCE_STRONG_PROTOCOL, pred_dir, args.seed),
    ]
    for metric in kmer_metrics:
        copied = copy_prediction_for_mcc(metric, mcc_pred_dir)
        metric["mcc_prediction_paths"] = copied
    lora_runs = discover_lora_runs(args.results_root)
    enriched, summary, judgement = aggregate_lora_runs(lora_runs, kmer_metrics[0], kmer_metrics[1], kmer_metrics[2])
    write_reports(out_dir, samples, lora_audit, kmer_metrics, enriched, summary, judgement, [" ".join(sys.argv)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--split-type", default=DEFAULT_SPLIT_TYPE)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
