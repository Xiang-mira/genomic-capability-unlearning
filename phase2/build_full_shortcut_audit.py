"""
Full shortcut baseline audit across all HVUE and local candidate tasks.

Runs: length_only, gc_only, mono_di, kmer3, kmer4, kmer5, kmer6,
      kmer3_6, raw_plus_kmer (gc+mono+di+kmer3_6), nearest_neighbor.
Outputs: reports/shortcut_audit.csv
"""
import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)


# ---------- feature extraction ----------

def gc_content(seq):
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return np.array([gc / max(len(seq), 1)])


def mono_di_features(seq):
    seq = seq.upper()
    alpha = "ACGT"
    n = len(seq)
    mono = np.array([seq.count(c) / max(n, 1) for c in alpha])
    di_counts = defaultdict(int)
    for i in range(n - 1):
        di_counts[seq[i:i+2]] += 1
    di = np.array([di_counts.get(a+b, 0) / max(n - 1, 1) for a in alpha for b in alpha])
    return np.concatenate([mono, di])


def kmer_features(seq, k):
    seq = seq.upper()
    alpha = "ACGT"
    vocab = {}
    idx = 0
    def _enum(prefix, depth):
        nonlocal idx
        if depth == k:
            vocab[prefix] = idx
            idx += 1
            return
        for c in alpha:
            _enum(prefix + c, depth + 1)
    _enum("", 0)
    counts = defaultdict(int)
    n = len(seq)
    for i in range(n - k + 1):
        mer = seq[i:i+k]
        if mer in vocab:
            counts[mer] += 1
    total = max(n - k + 1, 1)
    return np.array([counts.get(m, 0) / total for m in sorted(vocab.keys())])


def length_feature(seq):
    return np.array([len(seq)])


def build_features(seqs, feature_name):
    """Build feature matrix for a list of sequences."""
    if feature_name == "length_only":
        return np.array([length_feature(s) for s in seqs])
    if feature_name == "gc_only":
        return np.array([gc_content(s) for s in seqs])
    if feature_name == "mono_di":
        return np.array([mono_di_features(s) for s in seqs])
    if feature_name == "kmer3":
        return np.array([kmer_features(s, 3) for s in seqs])
    if feature_name == "kmer4":
        return np.array([kmer_features(s, 4) for s in seqs])
    if feature_name == "kmer5":
        return np.array([kmer_features(s, 5) for s in seqs])
    if feature_name == "kmer6":
        return np.array([kmer_features(s, 6) for s in seqs])
    if feature_name == "kmer3_6":
        return np.hstack([np.array([kmer_features(s, k) for s in seqs]) for k in [3, 4, 5, 6]])
    if feature_name == "raw_plus_kmer":
        gc = np.array([gc_content(s) for s in seqs])
        md = np.array([mono_di_features(s) for s in seqs])
        km = np.hstack([np.array([kmer_features(s, k) for s in seqs]) for k in [3, 4, 5, 6]])
        return np.hstack([gc, md, km])
    raise ValueError(f"Unknown feature: {feature_name}")


# ---------- evaluation ----------

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def train_eval_logreg(X_tr, y_tr, X_te, y_te):
    best_auroc = -1
    best_mcc = 0
    best_C = C_GRID[0]
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        if proba.shape[1] == 2:
            score = proba[:, 1]
        else:
            score = proba[:, 0]
        try:
            auroc = roc_auc_score(y_te, score)
        except Exception:
            auroc = 0.5
        if auroc > best_auroc:
            best_auroc = auroc
            pred = clf.predict(X_te)
            best_mcc = matthews_corrcoef(y_te, pred)
            best_C = C
    return best_auroc, best_mcc, best_C


def nearest_neighbor_eval(X_tr, y_tr, X_te, y_te):
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    score = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
    try:
        auroc = roc_auc_score(y_te, score)
    except Exception:
        auroc = 0.5
    pred = clf.predict(X_te)
    mcc = matthews_corrcoef(y_te, pred)
    return auroc, mcc


# ---------- data loaders ----------

def load_csv_manifest(path, train_split="train", test_split="test"):
    rows = list(csv.DictReader(open(path)))
    tr_seqs, tr_labels = [], []
    te_seqs, te_labels = [], []
    for r in rows:
        s = r["sequence"]
        l = int(r["label"])
        sp = r.get("split", train_split)
        if sp == train_split:
            tr_seqs.append(s); tr_labels.append(l)
        elif sp == test_split:
            te_seqs.append(s); te_labels.append(l)
    return tr_seqs, np.array(tr_labels), te_seqs, np.array(te_labels)


def load_parquet_hvue(train_path, test_path, max_train=10000, max_test=5000, seed=42):
    import pandas as pd
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(train_path)
    te = pd.read_parquet(test_path)
    if len(tr) > max_train:
        tr = tr.sample(n=max_train, random_state=seed)
    if len(te) > max_test:
        te = te.sample(n=max_test, random_state=seed)
    return (list(tr["sequence"]), np.array(tr["label"].astype(int)),
            list(te["sequence"]), np.array(te["label"].astype(int)))


# ---------- main ----------

FEATURES = [
    "length_only", "gc_only", "mono_di",
    "kmer3", "kmer4", "kmer5", "kmer6",
    "kmer3_6", "raw_plus_kmer",
]

TASKS = {
    # Local manifest tasks (already have kmer_3_6 combined, need per-k breakdown)
    "bvbrc_cov": {
        "loader": "csv",
        "path": "data/shortcut_audit/bvbrc_cov_manifest.csv",
        "benchmark": "hvue",
        "category": "forget_candidate",
    },
    "cini": {
        "loader": "csv",
        "path": "data/shortcut_audit/cini_manifest.csv",
        "benchmark": "hvue",
        "category": "forget_candidate",
    },
    "host_tropism": {
        "loader": "csv",
        "path": "data/shortcut_audit/host_tropism_manifest.csv",
        "benchmark": "hvue",
        "category": "forget_candidate",
    },
    # HVUE parquet tasks (from glm-locking)
    "hvue_host_tropism": {
        "loader": "parquet",
        "train": "/home/nvidia/glm-locking/data/hvue/Host_Tropism_train.parquet",
        "test": "/home/nvidia/glm-locking/data/hvue/Host_Tropism_test.parquet",
        "benchmark": "hvue",
        "category": "forget_candidate",
        "max_train": 10000, "max_test": 5000,
    },
    "hvue_pathogenicity": {
        "loader": "parquet",
        "train": "/home/nvidia/glm-locking/data/hvue/Pathogenecity_train.parquet",
        "test": "/home/nvidia/glm-locking/data/hvue/Pathogenecity_test.parquet",
        "benchmark": "hvue",
        "category": "forget_candidate",
        "max_train": 10000, "max_test": 5000,
    },
    "hvue_transmissibility": {
        "loader": "parquet",
        "train": "/home/nvidia/glm-locking/data/hvue/Transmissibility_train.parquet",
        "test": "/home/nvidia/glm-locking/data/hvue/Transmissibility_test.parquet",
        "benchmark": "hvue",
        "category": "forget_candidate",
        "max_train": 10000, "max_test": 5000,
    },
}


def run_task(task_name, task_cfg):
    print(f"\n[{task_name}] Loading data...")
    if task_cfg["loader"] == "csv":
        tr_seqs, tr_y, te_seqs, te_y = load_csv_manifest(task_cfg["path"])
    else:
        tr_seqs, tr_y, te_seqs, te_y = load_parquet_hvue(
            task_cfg["train"], task_cfg["test"],
            max_train=task_cfg.get("max_train", 10000),
            max_test=task_cfg.get("max_test", 5000),
        )
    print(f"  train: {len(tr_seqs)} ({int(tr_y.sum())} pos), test: {len(te_seqs)} ({int(te_y.sum())} pos)")

    results = {}
    for feat in FEATURES:
        t0 = time.time()
        print(f"  [{feat}]...", end=" ", flush=True)
        try:
            # Limit to 5000 train for k5/k6 to keep fast
            max_tr = 5000 if feat in ("kmer5", "kmer6", "kmer3_6", "raw_plus_kmer") else len(tr_seqs)
            X_tr = build_features(tr_seqs[:max_tr], feat)
            y_tr = tr_y[:max_tr]
            X_te = build_features(te_seqs, feat)
            y_te = te_y

            if feat == "length_only":
                # Scale
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

            auroc, mcc, best_C = train_eval_logreg(X_tr, y_tr, X_te, y_te)
            elapsed = time.time() - t0
            results[feat] = {"auroc": auroc, "mcc": mcc, "best_C": best_C}
            print(f"AUROC={auroc:.4f}  MCC={mcc:.4f}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            results[feat] = {"auroc": float("nan"), "mcc": float("nan"), "best_C": None}

    # Nearest neighbor (on kmer3 features, small subsample)
    print(f"  [nearest_neighbor]...", end=" ", flush=True)
    try:
        t0 = time.time()
        max_nn = min(2000, len(tr_seqs))
        X_tr_nn = build_features(tr_seqs[:max_nn], "kmer3")
        X_te_nn = build_features(te_seqs[:2000], "kmer3")
        auroc_nn, mcc_nn = nearest_neighbor_eval(X_tr_nn, tr_y[:max_nn], X_te_nn, te_y[:2000])
        results["nearest_neighbor"] = {"auroc": auroc_nn, "mcc": mcc_nn}
        print(f"AUROC={auroc_nn:.4f}  MCC={mcc_nn:.4f}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"ERROR: {e}")
        results["nearest_neighbor"] = {"auroc": float("nan"), "mcc": float("nan")}

    # Compute best shortcut
    all_aurocs = {k: v["auroc"] for k, v in results.items() if not np.isnan(v["auroc"])}
    best_feat = max(all_aurocs, key=all_aurocs.get)
    best_auroc = all_aurocs[best_feat]
    best_mcc = results[best_feat]["mcc"]

    return {
        "task": task_name,
        "benchmark": task_cfg.get("benchmark", "?"),
        "category": task_cfg.get("category", "?"),
        "n_train": len(tr_seqs),
        "n_test": len(te_seqs),
        "pos_frac_train": f"{tr_y.mean():.3f}",
        "pos_frac_test": f"{te_y.mean():.3f}",
        **{f"{feat}_auroc": f"{results[feat]['auroc']:.4f}" for feat in FEATURES + ['nearest_neighbor']},
        **{f"{feat}_mcc": f"{results[feat]['mcc']:.4f}" for feat in FEATURES + ['nearest_neighbor']},
        "best_shortcut_feat": best_feat,
        "best_shortcut_auroc": f"{best_auroc:.4f}",
        "best_shortcut_mcc": f"{best_mcc:.4f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        help="Tasks to audit (default: all)")
    parser.add_argument("--out", default="reports/shortcut_audit.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for task_name in args.tasks:
        if task_name not in TASKS:
            print(f"Warning: unknown task {task_name}, skipping")
            continue
        result = run_task(task_name, TASKS[task_name])
        rows.append(result)
        print(f"  ✓ {task_name}: best={result['best_shortcut_feat']} AUROC={result['best_shortcut_auroc']}")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
