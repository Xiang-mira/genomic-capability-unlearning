"""
Decisive GPU-free test for bvbrc_cov:
does the MODEL frozen-probe keep its edge over the k-mer shortcut when whole
composition-clusters are held out of training?

Uses cached Evo hidden-state features (data/bvbrc_probes/features, layers 0-9,
first 5632 sequences with intact chunks). For the SAME sequences and the SAME
random vs cluster-disjoint split, compares:
    - kmer3_6 logistic regression
    - model frozen-probe logistic regression (best of layers 0,3,6,9)
on both splits, and reports the excess (model - kmer) on each.

Interpretation:
    excess stays >0 under cluster-disjoint  -> generalizable model capability
    excess collapses to ~0 or negative      -> capability was composition retrieval
"""
import csv
import glob
import sys
import warnings

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)

BASE = "data/bvbrc_probes/features"
MANIFEST = "data/shortcut_audit/bvbrc_cov_manifest.csv"
ALPHA = "ACGT"
PROBE_LAYERS = [0, 3, 6, 9]
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_layer(layer, n_rows):
    chunks = sorted(glob.glob(f"{BASE}/layer_{layer}/chunk_*.npy"))
    mats = []
    for f in chunks:
        try:
            mats.append(np.load(f))
        except Exception:
            break  # stop at first corrupted chunk
    X = np.vstack(mats)
    return X[:n_rows]


def kmer_vocab(k):
    vocab = {}
    def _enum(p, d):
        if d == k:
            vocab[p] = len(vocab); return
        for c in ALPHA:
            _enum(p + c, d + 1)
    _enum("", 0)
    return {m: i for i, m in enumerate(sorted(vocab))}


def kmer_matrix(seqs, k):
    vocab = kmer_vocab(k)
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for r, seq in enumerate(seqs):
        seq = seq.upper(); n = len(seq); tot = max(n - k + 1, 1)
        for i in range(n - k + 1):
            j = vocab.get(seq[i:i + k])
            if j is not None:
                X[r, j] += 1.0
        X[r] /= tot
    return X


def kmer3_6(seqs):
    return np.hstack([kmer_matrix(seqs, k) for k in (3, 4, 5, 6)])


def fit_eval(Xtr, ytr, Xte, yte, scale=False):
    if scale:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    best = (-1, 0.0)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)
        s = p[:, 1] if p.shape[1] == 2 else p[:, 0]
        try:
            a = roc_auc_score(yte, s)
        except Exception:
            a = 0.5
        if a > best[0]:
            best = (a, matthews_corrcoef(yte, clf.predict(Xte)))
    return best


def main():
    ids = np.load(f"{BASE}/ids.npy", allow_pickle=True)
    labels = np.load(f"{BASE}/labels.npy", allow_pickle=True).astype(int)

    # number of intact rows in layer 0
    X0 = load_layer(0, 10 ** 9)
    n = X0.shape[0]
    print(f"intact cached rows: {n}", flush=True)
    ids_n = ids[:n]
    y = labels[:n]

    # map id -> sequence
    id2seq = {}
    for row in csv.DictReader(open(MANIFEST)):
        id2seq[row["id"]] = row["sequence"]
    seqs = [id2seq[str(i)] for i in ids_n]
    print(f"matched sequences: {len(seqs)}, pos_frac={y.mean():.3f}", flush=True)

    # composition clusters (kmer5 -> PCA -> KMeans)
    print("clustering by kmer5 composition...", flush=True)
    Xc = kmer_matrix(seqs, 5)
    Xc = PCA(n_components=min(50, Xc.shape[1]), random_state=42).fit_transform(Xc)
    cl = MiniBatchKMeans(n_clusters=25, random_state=42, n_init=10, batch_size=1024).fit_predict(Xc)

    # splits
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    rand_test = np.zeros(n, bool); rand_test[idx[: int(0.33 * n)]] = True

    clusters = np.unique(cl); rng.shuffle(clusters)
    clus_test = np.zeros(n, bool)
    for c in clusters:
        if clus_test.sum() >= 0.33 * n:
            break
        clus_test |= (cl == c)
    print(f"random test n={rand_test.sum()}, cluster-disjoint test n={clus_test.sum()}", flush=True)

    # feature matrices
    print("building kmer3_6...", flush=True)
    Xk = kmer3_6(seqs)
    probe_layers = {L: load_layer(L, n) for L in PROBE_LAYERS}

    rows = []
    for split_name, mask in [("random", rand_test), ("cluster_disjoint", clus_test)]:
        tr, te = ~mask, mask
        ka, kmcc = fit_eval(Xk[tr], y[tr], Xk[te], y[te])
        # best probe layer
        best_layer, best_pa, best_pmcc = None, -1, 0
        for L, XL in probe_layers.items():
            pa, pmcc = fit_eval(XL[tr], y[tr], XL[te], y[te], scale=True)
            if pa > best_pa:
                best_layer, best_pa, best_pmcc = L, pa, pmcc
        excess = best_pa - ka
        print(f"[{split_name}] kmer={ka:.4f} model(L{best_layer})={best_pa:.4f} "
              f"excess={excess:+.4f}", flush=True)
        rows.append(dict(task="bvbrc_cov", split=split_name,
                         kmer_auroc=round(ka, 4), kmer_mcc=round(kmcc, 4),
                         model_best_layer=best_layer, model_auroc=round(best_pa, 4),
                         model_mcc=round(best_pmcc, 4),
                         excess_model_minus_kmer=round(excess, 4),
                         n_train=int(tr.sum()), n_test=int(te.sum())))

    # drops
    kd = rows[0]["kmer_auroc"] - rows[1]["kmer_auroc"]
    md = rows[0]["model_auroc"] - rows[1]["model_auroc"]
    print(f"\nk-mer drop (random->cluster): {kd:+.4f}", flush=True)
    print(f"model drop (random->cluster): {md:+.4f}", flush=True)
    print(f"excess random={rows[0]['excess_model_minus_kmer']:+.4f}  "
          f"cluster-disjoint={rows[1]['excess_model_minus_kmer']:+.4f}", flush=True)

    out = "reports/model_vs_kmer_cluster_disjoint.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
