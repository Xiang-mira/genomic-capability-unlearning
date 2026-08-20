"""
Composition-cluster holdout diagnostic.

Tests whether the k-mer shortcut (and, if features available, the model probe)
survives when whole composition-defined clusters are held out of training.

Procedure per task:
  1. Compute k-mer(k_cluster) spectra for all sequences.
  2. PCA -> KMeans into N_CLUSTERS composition clusters.
  3. Build a CLUSTER-DISJOINT split: whole clusters assigned to test (~1/3 of data),
     rest to train. No test cluster's composition neighbourhood appears in train.
  4. Train kmer3_6 logistic regression on train, evaluate on:
        (a) a RANDOM split of the same data (composition overlap present)
        (b) the CLUSTER-DISJOINT split (composition overlap removed)
  5. Report the AUROC drop = composition-overlap contribution to the shortcut.

A model with genuine generalizable capability should degrade LESS than the
k-mer baseline under the cluster-disjoint split. If k-mer and model degrade
together, the "capability" is composition retrieval.
"""
import argparse
import csv
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)

ALPHA = "ACGT"


def kmer_vocab(k):
    vocab = {}
    def _enum(prefix, depth):
        if depth == k:
            vocab[prefix] = len(vocab); return
        for c in ALPHA:
            _enum(prefix + c, depth + 1)
    _enum("", 0)
    return {m: i for i, m in enumerate(sorted(vocab))}


def kmer_matrix(seqs, k, vocab=None):
    if vocab is None:
        vocab = kmer_vocab(k)
    D = len(vocab)
    X = np.zeros((len(seqs), D), dtype=np.float32)
    for r, seq in enumerate(seqs):
        seq = seq.upper()
        n = len(seq)
        total = max(n - k + 1, 1)
        for i in range(n - k + 1):
            j = vocab.get(seq[i:i + k])
            if j is not None:
                X[r, j] += 1.0
        X[r] /= total
    return X


def kmer3_6_matrix(seqs):
    return np.hstack([kmer_matrix(seqs, k) for k in (3, 4, 5, 6)])


C_GRID = [0.01, 0.1, 1.0, 10.0]


def fit_eval(X_tr, y_tr, X_te, y_te):
    best = (-1, 0.0, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=1500, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)
        score = p[:, 1] if p.shape[1] == 2 else p[:, 0]
        try:
            auroc = roc_auc_score(y_te, score)
        except Exception:
            auroc = 0.5
        if auroc > best[0]:
            best = (auroc, matthews_corrcoef(y_te, clf.predict(X_te)), C)
    return best


def nn_eval(X_tr, y_tr, X_te, y_te):
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)
    score = p[:, 1] if p.shape[1] == 2 else p[:, 0]
    try:
        auroc = roc_auc_score(y_te, score)
    except Exception:
        auroc = 0.5
    return auroc, matthews_corrcoef(y_te, clf.predict(X_te))


def load_manifest(path, max_n):
    rows = list(csv.DictReader(open(path)))
    if max_n and len(rows) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rows), size=max_n, replace=False)
        rows = [rows[i] for i in idx]
    seqs = [r["sequence"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    return seqs, labels


def make_cluster_disjoint_split(cluster_ids, labels, test_frac=0.33, seed=42):
    """Assign whole clusters to test until ~test_frac of rows are in test,
    keeping both labels present on each side."""
    rng = np.random.default_rng(seed)
    clusters = np.unique(cluster_ids)
    rng.shuffle(clusters)
    n = len(labels)
    test_mask = np.zeros(n, dtype=bool)
    for c in clusters:
        if test_mask.sum() >= test_frac * n:
            break
        test_mask |= (cluster_ids == c)
    # guard: both labels present on each side
    tr_lab = np.unique(labels[~test_mask])
    te_lab = np.unique(labels[test_mask])
    if len(tr_lab) < 2 or len(te_lab) < 2:
        return None
    return test_mask


def make_random_split(labels, test_frac=0.33, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = rng.permutation(n)
    n_te = int(test_frac * n)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[idx[:n_te]] = True
    return test_mask


def run_task(name, path, max_n, n_clusters, k_cluster):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    seqs, labels = load_manifest(path, max_n)
    print(f"  n={len(seqs)} pos_frac={labels.mean():.3f}", flush=True)

    # cluster features
    print(f"  computing kmer{k_cluster} spectra for clustering...", flush=True)
    Xc = kmer_matrix(seqs, k_cluster)
    ncomp = min(50, Xc.shape[1], Xc.shape[0] - 1)
    Xc_red = PCA(n_components=ncomp, random_state=42).fit_transform(Xc)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10, batch_size=1024)
    cluster_ids = km.fit_predict(Xc_red)
    csizes = np.bincount(cluster_ids)
    print(f"  {n_clusters} clusters, sizes min/med/max = {csizes.min()}/{int(np.median(csizes))}/{csizes.max()}", flush=True)

    # classifier features (shared)
    print("  computing kmer3_6 classifier features...", flush=True)
    Xk = kmer3_6_matrix(seqs)

    results = {}
    for split_name, mask in [
        ("random", make_random_split(labels)),
        ("cluster_disjoint", make_cluster_disjoint_split(cluster_ids, labels)),
    ]:
        if mask is None:
            print(f"  [{split_name}] SKIP (label imbalance)", flush=True)
            continue
        Xtr, ytr = Xk[~mask], labels[~mask]
        Xte, yte = Xk[mask], labels[mask]
        auroc, mcc, C = fit_eval(Xtr, ytr, Xte, yte)
        nn_auroc, nn_mcc = nn_eval(kmer_matrix(seqs, 4)[~mask], ytr,
                                    kmer_matrix(seqs, 4)[mask], yte)
        results[split_name] = dict(kmer_auroc=auroc, kmer_mcc=mcc, C=C,
                                   nn_auroc=nn_auroc, nn_mcc=nn_mcc,
                                   n_train=int((~mask).sum()), n_test=int(mask.sum()),
                                   test_pos=float(yte.mean()))
        print(f"  [{split_name}] kmer3_6 AUROC={auroc:.4f} MCC={mcc:.4f} | NN AUROC={nn_auroc:.4f} "
              f"(n_tr={int((~mask).sum())} n_te={int(mask.sum())})", flush=True)

    if "random" in results and "cluster_disjoint" in results:
        drop_k = results["random"]["kmer_auroc"] - results["cluster_disjoint"]["kmer_auroc"]
        drop_nn = results["random"]["nn_auroc"] - results["cluster_disjoint"]["nn_auroc"]
        print(f"  >> k-mer AUROC drop (random -> cluster-disjoint): {drop_k:+.4f}", flush=True)
        print(f"  >> NN AUROC drop:    {drop_nn:+.4f}", flush=True)
        results["kmer_drop"] = drop_k
        results["nn_drop"] = drop_nn
    results["elapsed"] = time.time() - t0
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["bvbrc_cov", "cini", "host_tropism"])
    ap.add_argument("--max-n", type=int, default=8000)
    ap.add_argument("--n-clusters", type=int, default=25)
    ap.add_argument("--k-cluster", type=int, default=5)
    ap.add_argument("--out", default="reports/composition_cluster_holdout.csv")
    args = ap.parse_args()

    manifests = {
        "bvbrc_cov": "data/shortcut_audit/bvbrc_cov_manifest.csv",
        "cini": "data/shortcut_audit/cini_manifest.csv",
        "host_tropism": "data/shortcut_audit/host_tropism_manifest.csv",
    }

    rows = []
    for t in args.tasks:
        r = run_task(t, manifests[t], args.max_n, args.n_clusters, args.k_cluster)
        for split in ("random", "cluster_disjoint"):
            if split in r:
                rows.append(dict(task=t, split=split,
                                 kmer_auroc=round(r[split]["kmer_auroc"], 4),
                                 kmer_mcc=round(r[split]["kmer_mcc"], 4),
                                 nn_auroc=round(r[split]["nn_auroc"], 4),
                                 n_train=r[split]["n_train"], n_test=r[split]["n_test"],
                                 test_pos_frac=round(r[split]["test_pos"], 3),
                                 kmer_drop_vs_random=round(r.get("kmer_drop", 0), 4) if split == "cluster_disjoint" else "",
                                 nn_drop_vs_random=round(r.get("nn_drop", 0), 4) if split == "cluster_disjoint" else ""))

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
