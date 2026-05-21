import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
<<<<<<< HEAD
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
=======
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import read_manifest


def gc_1gram_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    length = max(len(seq), 1)
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
        else:
            counts["N"] += 1
    gc = (counts["G"] + counts["C"]) / length
    freqs = np.array([
        counts["A"] / length,
        counts["C"] / length,
        counts["G"] / length,
        counts["T"] / length,
        counts["N"] / length,
    ], dtype=np.float32)
    return np.concatenate([[gc], freqs], axis=0)


<<<<<<< HEAD
def length_feature(seq: str) -> np.ndarray:
    return np.array([len(seq)], dtype=np.float32)


=======
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
def gc_only_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    length = max(len(seq), 1)
    gc = (seq.count("G") + seq.count("C")) / length
    return np.array([gc], dtype=np.float32)


def train_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: List[float],
    max_iter: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    best_val = -1.0
    best_metrics: Dict[str, float] = {}
    best_meta: Dict[str, float] = {}

    for c in c_grid:
        clf = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=max_iter,
<<<<<<< HEAD
            class_weight="balanced",
=======
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
            n_jobs=None,
        )
        clf.fit(features[train_mask], labels[train_mask])

        metrics = {}
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            probs = clf.predict_proba(features[mask])[:, 1]
            preds = (probs >= 0.5).astype(np.int64)
            metrics[f"{name}_acc"] = accuracy_score(labels[mask], preds)
            metrics[f"{name}_auroc"] = roc_auc_score(labels[mask], probs)

        if metrics["val_auroc"] > best_val:
            best_val = metrics["val_auroc"]
            best_metrics = metrics
            best_meta = {"C": c}

    return best_metrics, best_meta


def main() -> None:
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Simple sequence baselines for viral vs non-viral.")
=======
    parser = argparse.ArgumentParser(description="GC + 1-gram baseline for viral vs non-viral.")
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
    parser.add_argument("--manifest", default="data/phase1/manifest.csv")
    parser.add_argument("--out-dir", default="data/phase1/baselines")
    parser.add_argument("--c-grid", default="0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
<<<<<<< HEAD
    parser.add_argument(
        "--feature",
        choices=["gc", "gc_1gram", "length", "gc_1gram_length", "kmer"],
        default="gc_1gram",
    )
    parser.add_argument("--kmer-max", type=int, default=4)
    parser.add_argument(
        "--kmer-binary",
        action="store_true",
        help="Use k-mer presence/absence instead of counts.",
    )
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    sequences = [r.sequence for r in records]
    if args.feature == "gc":
        features = np.stack([gc_only_features(seq) for seq in sequences], axis=0)
    else:
        if args.feature == "gc_1gram":
            features = np.stack([gc_1gram_features(seq) for seq in sequences], axis=0)
        elif args.feature == "length":
            features = np.stack([length_feature(seq) for seq in sequences], axis=0)
        elif args.feature == "gc_1gram_length":
            features = np.stack(
                [
                    np.concatenate([gc_1gram_features(seq), length_feature(seq)], axis=0)
                    for seq in sequences
                ],
                axis=0,
            )
        else:
            vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=(1, args.kmer_max),
                lowercase=False,
                binary=args.kmer_binary,
            )
            features = vectorizer.fit_transform(sequences)
    labels = np.array([r.label for r in records], dtype=np.int64)
    splits = np.array([r.split for r in records])

    if not sparse.issparse(features):
        train_mask = splits == "train"
        scaler = StandardScaler()
        scaled = np.empty(features.shape, dtype=np.float32)
        scaled[train_mask] = scaler.fit_transform(features[train_mask])
        for split in ["val", "test"]:
            mask = splits == split
            scaled[mask] = scaler.transform(features[mask])
        features = scaled

=======
    parser.add_argument("--feature", choices=["gc", "gc_1gram"], default="gc_1gram")
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    if args.feature == "gc":
        features = np.stack([gc_only_features(r.sequence) for r in records], axis=0)
    else:
        features = np.stack([gc_1gram_features(r.sequence) for r in records], axis=0)
    labels = np.array([r.label for r in records], dtype=np.int64)
    splits = np.array([r.split for r in records])

>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
    c_grid = [float(x) for x in args.c_grid.split(",")]
    metrics, meta = train_baseline(features, labels, splits, c_grid, args.max_iter)

    os.makedirs(args.out_dir, exist_ok=True)
<<<<<<< HEAD
    if args.feature == "kmer":
        suffix = f"kmer_1-{args.kmer_max}"
        if args.kmer_binary:
            suffix += "_binary"
        out_name = f"{suffix}_metrics.csv"
    else:
        out_name = f"{args.feature}_metrics.csv"
=======
    out_name = "gc_metrics.csv" if args.feature == "gc" else "gc_1gram_metrics.csv"
>>>>>>> a41d6a7edeb16aead36fb9da8b2cd4b77a380a87
    out_path = os.path.join(args.out_dir, out_name)
    row = {"C": meta["C"], **metrics}
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Wrote baseline metrics to {out_path}")


if __name__ == "__main__":
    main()
