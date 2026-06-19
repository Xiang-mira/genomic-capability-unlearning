"""Shortcut audit for BacBench antibiotic resistance.

The audit asks whether AMR prediction survives stronger taxonomy controls, or
whether simple shortcut baselines can explain most of the signal.

Inputs are intentionally plain local files so the script can be run against the
Hugging Face BacBench exports, a downloaded subset, or an already joined table.
At minimum you need:

  * a sequence table with genome_name and DNA sequence columns
  * a labels CSV with genome_name plus one binary column per antibiotic

Optional taxonomy metadata can either live in the sequence table or be supplied
as a separate table with genome_name plus species/genus/family columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


GENOME_COLUMNS = ("genome_name", "Genome Name", "genome", "strain_name", "assembly_accession", "id")
SEQUENCE_COLUMNS = ("dna_seq", "sequence", "Sequence", "genome_sequence", "contig_sequence")
TAXONOMY_COLUMNS = ("species", "genus", "family")
SPLITS = ("random", "species_heldout", "genus_heldout", "family_heldout")
BASELINES = ("taxonomy", "gc", "kmer", "mash")
POSITIVE_LABELS = {"1", "true", "yes", "y", "r", "resistant", "resistance", "positive", "+"}
NEGATIVE_LABELS = {"0", "false", "no", "n", "s", "susceptible", "susceptibility", "negative", "-"}


@dataclass(frozen=True)
class SplitData:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    group_column: str
    status: str = "ok"
    reason: str = ""


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"Unsupported table format: {path}")


def first_existing(columns: Iterable[str], candidates: Iterable[str], required_name: str) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(f"Could not find {required_name} column. Tried: {', '.join(candidates)}")


def clean_sequence(value: object) -> str:
    text = str(value or "").upper()
    return "".join(ch for ch in text if ch in {"A", "C", "G", "T", "N"})


def normalize_taxonomy(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "na", "unclassified", "unknown"}:
        return ""
    return text


def normalize_binary_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not np.isfinite(numeric):
        return None
    if numeric == 1:
        return 1
    if numeric == 0:
        return 0
    return None


def load_sequence_metadata(args: argparse.Namespace) -> pd.DataFrame:
    seq_df = read_table(args.sequence_table)
    genome_col = first_existing(seq_df.columns, GENOME_COLUMNS, "genome id")
    seq_col = first_existing(seq_df.columns, SEQUENCE_COLUMNS, "DNA sequence")

    keep_cols = [genome_col, seq_col] + [col for col in TAXONOMY_COLUMNS if col in seq_df.columns]
    out = seq_df[keep_cols].copy()
    out = out.rename(columns={genome_col: "genome_name", seq_col: "sequence"})
    out["genome_name"] = out["genome_name"].astype(str)
    out["sequence"] = out["sequence"].map(clean_sequence)

    if args.taxonomy_table:
        tax_df = read_table(args.taxonomy_table)
        tax_genome_col = first_existing(tax_df.columns, GENOME_COLUMNS, "taxonomy genome id")
        tax_keep = [tax_genome_col] + [col for col in TAXONOMY_COLUMNS if col in tax_df.columns]
        tax_df = tax_df[tax_keep].copy().rename(columns={tax_genome_col: "genome_name"})
        tax_df["genome_name"] = tax_df["genome_name"].astype(str)
        out = out.drop(columns=[col for col in TAXONOMY_COLUMNS if col in out.columns], errors="ignore")
        out = out.merge(tax_df, on="genome_name", how="left")

    for col in TAXONOMY_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(normalize_taxonomy)

    out = out[out["sequence"].str.len() > 0].drop_duplicates("genome_name")
    return out.reset_index(drop=True)


def load_labels(path: str | Path) -> pd.DataFrame:
    labels = read_table(path)
    genome_col = first_existing(labels.columns, GENOME_COLUMNS, "label genome id")
    labels = labels.rename(columns={genome_col: "genome_name"})
    labels["genome_name"] = labels["genome_name"].astype(str)
    return labels


def label_columns(labels: pd.DataFrame, requested: str) -> list[str]:
    if requested:
        cols = [col.strip() for col in requested.split(",") if col.strip()]
        missing = [col for col in cols if col not in labels.columns]
        if missing:
            raise ValueError(f"Requested antibiotic columns missing from labels file: {missing}")
        return cols
    return [col for col in labels.columns if col != "genome_name"]


def build_task_frame(metadata: pd.DataFrame, labels: pd.DataFrame, antibiotic: str) -> pd.DataFrame:
    task = metadata.merge(labels[["genome_name", antibiotic]], on="genome_name", how="inner")
    task["label"] = task[antibiotic].map(normalize_binary_label)
    task = task.dropna(subset=["label"]).copy()
    task["label"] = task["label"].astype(int)
    task = task.drop(columns=[antibiotic])
    return task.reset_index(drop=True)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def split_random(y: np.ndarray, seed: int, val_size: float, test_size: float) -> SplitData:
    idx = np.arange(len(y))
    stratify = y if len(np.unique(y)) == 2 and min(Counter(y).values()) >= 3 else None
    train_idx, tmp_idx = train_test_split(
        idx,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=stratify,
    )
    tmp_y = y[tmp_idx]
    tmp_stratify = tmp_y if len(np.unique(tmp_y)) == 2 and min(Counter(tmp_y).values()) >= 2 else None
    rel_val_size = val_size / (val_size + test_size)
    val_rel, test_rel = train_test_split(
        np.arange(len(tmp_idx)),
        train_size=rel_val_size,
        random_state=seed + 1,
        stratify=tmp_stratify,
    )
    return SplitData("random", train_idx, tmp_idx[val_rel], tmp_idx[test_rel], "")


def split_grouped(df: pd.DataFrame, y: np.ndarray, group_col: str, seed: int, val_size: float, test_size: float) -> SplitData:
    groups = df[group_col].astype(str).to_numpy()
    valid = np.array([bool(g) for g in groups])
    if valid.sum() != len(df):
        return SplitData(
            f"{group_col}_heldout",
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            group_col,
            status="skipped",
            reason=f"missing {group_col} values for {len(df) - int(valid.sum())} rows",
        )
    if len(np.unique(groups)) < 3:
        return SplitData(
            f"{group_col}_heldout",
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            group_col,
            status="skipped",
            reason=f"need at least 3 unique {group_col} groups",
        )

    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(np.unique(groups)), dtype=object)
    rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n_test = min(max(1, int(round(test_size * n_groups))), n_groups - 2)
    n_val = min(max(1, int(round(val_size * n_groups))), n_groups - n_test - 1)
    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val :])

    idx = np.arange(len(df))
    train_idx = idx[np.isin(groups, list(train_groups))]
    val_idx = idx[np.isin(groups, list(val_groups))]
    test_idx = idx[np.isin(groups, list(test_groups))]
    for split_name, split_idx in {"train": train_idx, "val": val_idx, "test": test_idx}.items():
        if len(np.unique(y[split_idx])) < 2:
            return SplitData(
                f"{group_col}_heldout",
                train_idx,
                val_idx,
                test_idx,
                group_col,
                status="skipped",
                reason=f"{split_name} split has only one class",
            )
    return SplitData(f"{group_col}_heldout", train_idx, val_idx, test_idx, group_col)


def make_splits(df: pd.DataFrame, seed: int, val_size: float, test_size: float) -> list[SplitData]:
    y = df["label"].to_numpy()
    return [
        split_random(y, seed, val_size, test_size),
        split_grouped(df, y, "species", seed, val_size, test_size),
        split_grouped(df, y, "genus", seed, val_size, test_size),
        split_grouped(df, y, "family", seed, val_size, test_size),
    ]


def fit_predict_logistic(
    x,
    y: np.ndarray,
    split: SplitData,
    c_grid: list[float],
    max_iter: int,
    scale_dense: bool = False,
) -> tuple[np.ndarray, dict]:
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
    train_x = x[train_idx]
    val_x = x[val_idx]
    test_x = x[test_idx]

    if scale_dense and not sparse.issparse(train_x):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)

    best_c = c_grid[0]
    best_val = -np.inf
    best_model: LogisticRegression | None = None
    for c_value in c_grid:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=max_iter,
            solver="liblinear" if x.shape[1] < 1000 else "saga",
        )
        model.fit(train_x, y[train_idx])
        val_score = model.predict_proba(val_x)[:, 1]
        val_auc = safe_auc(y[val_idx], val_score)
        if np.nan_to_num(val_auc, nan=-np.inf) > best_val:
            best_val = np.nan_to_num(val_auc, nan=-np.inf)
            best_c = c_value
            best_model = model

    if best_model is None:
        raise RuntimeError("No logistic model was fitted.")
    return best_model.predict_proba(test_x)[:, 1], {"C": best_c, "val_auroc": float(best_val)}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def taxonomy_features(df: pd.DataFrame, split: SplitData):
    cols = [col for col in TAXONOMY_COLUMNS if df[col].astype(bool).any()]
    if not cols:
        raise ValueError("No taxonomy columns available.")
    enc = make_one_hot_encoder()
    enc.fit(df.loc[split.train_idx, cols].astype(str))
    return enc.transform(df[cols].astype(str)), {"taxonomy_columns": ",".join(cols)}


def gc_features(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(df), 1), dtype=np.float32)
    for i, seq in enumerate(df["sequence"]):
        length = max(len(seq), 1)
        out[i, 0] = (seq.count("G") + seq.count("C")) / length
    return out


def kmer_features(df: pd.DataFrame, k_min: int, k_max: int, n_features: int):
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(k_min, k_max),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    return vectorizer.transform(df["sequence"]), {
        "k_min": k_min,
        "k_max": k_max,
        "n_features": n_features,
    }


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("ascii"), digest_size=8).digest(), "little")


def sketch_sequence(seq: str, k: int, sketch_size: int, max_kmers: int) -> np.ndarray:
    n = len(seq) - k + 1
    if n <= 0:
        return np.empty((0,), dtype=np.uint64)
    step = max(1, math.ceil(n / max_kmers)) if max_kmers > 0 else 1
    hashes: list[int] = []
    for start in range(0, n, step):
        kmer = seq[start : start + k]
        if "N" in kmer:
            continue
        hashes.append(stable_u64(kmer))
    if not hashes:
        return np.empty((0,), dtype=np.uint64)
    unique = np.unique(np.asarray(hashes, dtype=np.uint64))
    unique.sort()
    return unique[:sketch_size]


def minhash_sketches(df: pd.DataFrame, k: int, sketch_size: int, max_kmers: int) -> list[np.ndarray]:
    return [sketch_sequence(seq, k, sketch_size, max_kmers) for seq in df["sequence"]]


def mash_minhash_predict(
    sketches: list[np.ndarray],
    y: np.ndarray,
    split: SplitData,
    top_k: int,
) -> tuple[np.ndarray, dict]:
    train_idx = split.train_idx
    train_prior = float(y[train_idx].mean())
    inverted: dict[int, list[int]] = defaultdict(list)
    train_pos = {idx: pos for pos, idx in enumerate(train_idx)}
    train_labels = y[train_idx]
    for idx in train_idx:
        for value in sketches[idx]:
            inverted[int(value)].append(train_pos[idx])

    scores = []
    for idx in split.test_idx:
        counts: Counter[int] = Counter()
        for value in sketches[idx]:
            for train_position in inverted.get(int(value), []):
                counts[train_position] += 1
        if not counts:
            scores.append(train_prior)
            continue
        neighbors = counts.most_common(top_k)
        weights = np.asarray([count for _, count in neighbors], dtype=np.float32)
        labels = np.asarray([train_labels[pos] for pos, _ in neighbors], dtype=np.float32)
        if weights.sum() == 0:
            scores.append(train_prior)
        else:
            scores.append(float(np.average(labels, weights=weights)))
    return np.asarray(scores, dtype=np.float32), {
        "mash_mode": "python_minhash",
        "top_k": top_k,
    }


def write_fasta(path: Path, names: Iterable[str], sequences: Iterable[str]) -> None:
    with path.open("w") as f:
        for name, seq in zip(names, sequences, strict=True):
            f.write(f">{name}\n")
            for start in range(0, len(seq), 80):
                f.write(seq[start : start + 80] + "\n")


def mash_binary_predict(
    df: pd.DataFrame,
    y: np.ndarray,
    split: SplitData,
    mash_bin: str,
    k: int,
    sketch_size: int,
    top_k: int,
) -> tuple[np.ndarray, dict]:
    if not mash_bin or shutil.which(mash_bin) is None:
        raise FileNotFoundError("mash binary not found")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "genomes.fna"
        write_fasta(fasta, df["genome_name"], df["sequence"])
        sketch_prefix = tmp_path / "genomes"
        sketch_path = tmp_path / "genomes.msh"
        subprocess.run(
            [mash_bin, "sketch", "-k", str(k), "-s", str(sketch_size), "-i", "-o", str(sketch_prefix), str(fasta)],
            check=True,
            capture_output=True,
            text=True,
        )
        dist = subprocess.run(
            [mash_bin, "dist", str(sketch_path), str(sketch_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    name_to_idx = {name: i for i, name in enumerate(df["genome_name"])}
    train_set = set(split.train_idx.tolist())
    neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for line in dist.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        left = Path(parts[0]).name
        right = Path(parts[1]).name
        if left not in name_to_idx or right not in name_to_idx:
            continue
        left_idx = name_to_idx[left]
        right_idx = name_to_idx[right]
        if left_idx == right_idx or right_idx not in train_set:
            continue
        try:
            distance = float(parts[2])
        except ValueError:
            continue
        neighbors[left_idx].append((distance, right_idx))

    train_prior = float(y[split.train_idx].mean())
    scores = []
    for idx in split.test_idx:
        row_neighbors = sorted(neighbors.get(idx, []))[:top_k]
        if not row_neighbors:
            scores.append(train_prior)
            continue
        weights = np.asarray([1.0 / max(distance, 1e-6) for distance, _ in row_neighbors], dtype=np.float32)
        labels = np.asarray([y[neighbor_idx] for _, neighbor_idx in row_neighbors], dtype=np.float32)
        scores.append(float(np.average(labels, weights=weights)))
    return np.asarray(scores, dtype=np.float32), {"mash_mode": "binary", "top_k": top_k}


def classification_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    return {
        "auroc": safe_auc(y_true, score),
        "auprc": safe_auprc(y_true, score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def split_summary(df: pd.DataFrame, split: SplitData, antibiotic: str) -> dict:
    row = {
        "antibiotic": antibiotic,
        "split": split.name,
        "status": split.status,
        "reason": split.reason,
        "group_column": split.group_column,
    }
    if split.status != "ok":
        return row

    y = df["label"].to_numpy()
    split_indices = {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}
    for split_name, idx in split_indices.items():
        row[f"n_{split_name}"] = int(len(idx))
        counts = Counter(y[idx].tolist())
        row[f"{split_name}_pos"] = int(counts.get(1, 0))
        row[f"{split_name}_neg"] = int(counts.get(0, 0))
    for col in TAXONOMY_COLUMNS:
        for left, right in (("train", "val"), ("train", "test")):
            left_groups = set(df.loc[split_indices[left], col].astype(str))
            right_groups = set(df.loc[split_indices[right], col].astype(str))
            left_groups.discard("")
            right_groups.discard("")
            row[f"{col}_{left}_{right}_overlap"] = int(len(left_groups & right_groups))
    return row


def parse_c_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def audit_antibiotic(
    df: pd.DataFrame,
    antibiotic: str,
    args: argparse.Namespace,
    c_grid: list[float],
) -> tuple[list[dict], list[dict]]:
    y = df["label"].to_numpy(dtype=int)
    if len(df) < args.min_samples or len(np.unique(y)) < 2:
        return [], [
            {
                "antibiotic": antibiotic,
                "split": "",
                "status": "skipped",
                "reason": "not enough binary labeled samples",
            }
        ]
    if min(Counter(y).values()) < args.min_class_samples:
        return [], [
            {
                "antibiotic": antibiotic,
                "split": "",
                "status": "skipped",
                "reason": "minority class below --min-class-samples",
            }
        ]

    splits = make_splits(df, args.seed, args.val_size, args.test_size)
    summary_rows = [split_summary(df, split, antibiotic) for split in splits]
    metric_rows: list[dict] = []
    gc_x = gc_features(df)
    kmer_x = None
    sketches = None

    for split in splits:
        if split.status != "ok":
            continue
        for baseline in BASELINES:
            meta: dict = {}
            try:
                if baseline == "taxonomy":
                    tax_x, tax_meta = taxonomy_features(df, split)
                    score, meta = fit_predict_logistic(tax_x, y, split, c_grid, args.max_iter)
                    meta.update(tax_meta)
                elif baseline == "gc":
                    score, meta = fit_predict_logistic(gc_x, y, split, c_grid, args.max_iter, scale_dense=True)
                elif baseline == "kmer":
                    if kmer_x is None:
                        kmer_x, meta = kmer_features(df, args.kmer_min, args.kmer_max, args.kmer_features)
                    score, fit_meta = fit_predict_logistic(kmer_x, y, split, c_grid, args.max_iter)
                    meta.update(fit_meta)
                elif baseline == "mash":
                    try:
                        score, meta = mash_binary_predict(
                            df,
                            y,
                            split,
                            args.mash_bin,
                            args.mash_k,
                            args.mash_sketch_size,
                            args.mash_top_k,
                        )
                    except Exception as exc:
                        if not args.allow_python_minhash:
                            raise
                        if sketches is None:
                            sketches = minhash_sketches(df, args.mash_k, args.mash_sketch_size, args.mash_max_kmers)
                        score, meta = mash_minhash_predict(sketches, y, split, args.mash_top_k)
                        meta["mash_fallback_reason"] = str(exc)
                else:
                    raise ValueError(f"Unknown baseline: {baseline}")
            except Exception as exc:
                metric_rows.append(
                    {
                        "antibiotic": antibiotic,
                        "split": split.name,
                        "baseline": baseline,
                        "status": "error",
                        "reason": str(exc),
                    }
                )
                continue

            metrics = classification_metrics(y[split.test_idx], score)
            metric_rows.append(
                {
                    "antibiotic": antibiotic,
                    "split": split.name,
                    "baseline": baseline,
                    "status": "ok",
                    "n_train": int(len(split.train_idx)),
                    "n_val": int(len(split.val_idx)),
                    "n_test": int(len(split.test_idx)),
                    **metrics,
                    **meta,
                }
            )
    return metric_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BacBench AMR shortcut baselines under taxonomy-held-out splits.")
    parser.add_argument("--sequence-table", required=True, help="CSV/TSV/JSONL/Parquet with genome_name and DNA sequence.")
    parser.add_argument("--labels-table", required=True, help="CSV/TSV/JSONL/Parquet with genome_name and antibiotic labels.")
    parser.add_argument("--taxonomy-table", default="", help="Optional metadata table with genome_name/species/genus/family.")
    parser.add_argument("--antibiotics", default="", help="Comma-separated label columns. Default: all label columns.")
    parser.add_argument("--out-dir", default="data/phase2/bacbench_amr_shortcut_audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-class-samples", type=int, default=20)
    parser.add_argument("--c-grid", default="0.01,0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--kmer-min", type=int, default=3)
    parser.add_argument("--kmer-max", type=int, default=6)
    parser.add_argument("--kmer-features", type=int, default=262144)
    parser.add_argument("--mash-bin", default="mash")
    parser.add_argument("--mash-k", type=int, default=21)
    parser.add_argument("--mash-sketch-size", type=int, default=1000)
    parser.add_argument("--mash-top-k", type=int, default=5)
    parser.add_argument("--mash-max-kmers", type=int, default=250000)
    parser.add_argument(
        "--allow-python-minhash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to internal Mash-style MinHash nearest-neighbor when mash binary is unavailable.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metadata = load_sequence_metadata(args)
    labels = load_labels(args.labels_table)
    antibiotics = label_columns(labels, args.antibiotics)
    c_grid = parse_c_grid(args.c_grid)

    all_metrics: list[dict] = []
    all_summaries: list[dict] = []
    for antibiotic in antibiotics:
        task_df = build_task_frame(metadata, labels, antibiotic)
        metric_rows, summary_rows = audit_antibiotic(task_df, antibiotic, args, c_grid)
        all_metrics.extend(metric_rows)
        all_summaries.extend(summary_rows)
        print(f"[amr-shortcut] {antibiotic}: rows={len(task_df)} metrics={len(metric_rows)}")

    metrics_path = Path(args.out_dir) / "shortcut_metrics.csv"
    splits_path = Path(args.out_dir) / "split_summary.csv"
    config_path = Path(args.out_dir) / "config.json"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(all_summaries).to_csv(splits_path, index=False)
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[amr-shortcut] wrote {metrics_path}")
    print(f"[amr-shortcut] wrote {splits_path}")
    print(f"[amr-shortcut] wrote {config_path}")


if __name__ == "__main__":
    csv.field_size_limit(2**31 - 1)
    main()
