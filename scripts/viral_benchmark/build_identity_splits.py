"""
Rebuild HVUE disjoint splits WITHOUT the baseline-performance gate.

Why: build_splits_v2.py accepts a candidate split only if
    passed = kmer_auroc(split) <= kmer_auroc(random) - 0.03      # GATE = 0.03
i.e. the evaluation split is selected on the condition that the k-mer baseline
loses at least 0.03 AUROC. Composition clusters are additionally built in
kmer5-PCA space -- the baseline's own feature space. Both bias the split against
the baseline.

Consequence, from logs_v2 / phaseA.log (Host_Tropism):
    random split                       kmer AUROC = 0.9213
    composition-cluster-disjoint       kmer AUROC = 0.8034   -> PASS (kept)
    MMseqs2 90%-identity-disjoint      kmer AUROC = 0.9131   -> FAIL (discarded)
The genuine homology-disjoint split barely moved the baseline and was thrown away
for exactly that reason, on all three tasks ("VALIDITY ... INVALID").

This script writes the identity-disjoint splits with NO gate, over several holdout
seeds, so models can be evaluated on a split that was not chosen to flatter them.
"""
import json, os, shutil, subprocess, sys, tempfile, time, warnings
import numpy as np, pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
warnings.filterwarnings("ignore")
ROOT = P.LOCK_ROOT
HVUE = P.HVUE_DIR
MMSEQS = P.MMSEQS
SCRATCH = P.SCRATCH
OUT = P.sub("splits_identity")
TASKS = ["Host_Tropism", "Pathogenecity", "Transmissibility"]
POOL_CAP_PER_CLASS, SEED, TEST_FRAC = 15000, 42, 0.30
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CODE = np.full(256, 255, np.uint8)
for i, c in enumerate("ACGT"):
    CODE[ord(c)] = i; CODE[ord(c.lower())] = i
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P


def kmer(seqs, ks=(3, 4, 5, 6)):
    out = []
    for k in ks:
        V = 4 ** k; X = np.zeros((len(seqs), V), np.float32)
        pw = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
        for r, s in enumerate(seqs):
            c = CODE[np.frombuffer(s.encode(), np.uint8)]; ok = c != 255
            if ok.sum() < k: continue
            c = c.astype(np.int64)
            if len(c) - k + 1 <= 0: continue
            w = np.lib.stride_tricks.sliding_window_view(c, k)
            v = np.lib.stride_tricks.sliding_window_view(ok, k).all(1)
            idx = (w @ pw)[v]
            if idx.size: np.add.at(X[r], idx, 1.0); X[r] /= idx.size
        out.append(X)
    return np.hstack(out)


def mmseqs_cluster(seqs, min_id, cov=0.9, tag="cl"):
    os.makedirs(SCRATCH, exist_ok=True)
    d = tempfile.mkdtemp(prefix=f"{tag}_", dir=SCRATCH)
    fa = os.path.join(d, "in.fasta")
    with open(fa, "w") as f:
        for i, s in enumerate(seqs): f.write(f">{i}\n{s}\n")
    pref, tmp = os.path.join(d, "res"), os.path.join(d, "tmp")
    subprocess.run([MMSEQS, "easy-cluster", fa, pref, tmp, "--min-seq-id", str(min_id),
                    "-c", str(cov), "--threads", "16", "-v", "1"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rep = {}
    with open(pref + "_cluster.tsv") as f:
        for line in f:
            r, m = line.split(); rep[int(m)] = int(r)
    uniq = {r: i for i, r in enumerate(sorted(set(rep.values())))}
    out = np.array([uniq[rep[i]] for i in range(len(seqs))], dtype=int)
    shutil.rmtree(d, ignore_errors=True)
    return out


def group_holdout(groups, labels, seed, target=TEST_FRAC):
    """Balanced whole-group holdout. NO baseline-performance criterion."""
    uniq = np.unique(groups); rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    val = np.zeros(len(groups), bool); vp = vn = 0.0; n = len(groups)
    for g in order:
        if val.sum() >= target * n: break
        m = groups == g
        gp, gn = labels[m].sum(), (~labels[m].astype(bool)).sum()
        if vp + vn > 0 and abs((vp + gp) / (vp + gp + vn + gn) - 0.5) > 0.12: continue
        val |= m; vp += gp; vn += gn
    return val


def kmer_eval(Xk, y, val):
    sc = StandardScaler().fit(Xk[~val])
    A, B = sc.transform(Xk[~val]), sc.transform(Xk[val])
    clf = LogisticRegressionCV(Cs=C_GRID, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                               scoring="roc_auc", solver="lbfgs", max_iter=5000, n_jobs=-1,
                               refit=True).fit(A, y[~val])
    s = clf.predict_proba(B)[:, 1]
    fpr, tpr, th = roc_curve(y[~val], clf.predict_proba(A)[:, 1]); thr = th[np.argmax(tpr - fpr)]
    return dict(auroc=float(roc_auc_score(y[val], s)),
                mcc=float(matthews_corrcoef(y[val], (s >= thr).astype(int))),
                bestC=float(clf.C_[0]))


def main():
    os.makedirs(OUT, exist_ok=True); rep = {}
    for task in TASKS:
        t0 = time.time()
        tr = pd.read_parquet(f"{HVUE}/{task}_train.parquet")
        va = pd.read_parquet(f"{HVUE}/{task}_validation.parquet")
        df = pd.concat([tr, va], ignore_index=True)
        # NOTE: not df.groupby("label", group_keys=False).apply(lambda g: g.sample(...)) --
        # pandas >=2.2 dropped the grouping column from `g` by default (include_groups),
        # silently breaking the subsequent df.label access. Explicit loop is version-safe.
        df = pd.concat([g.sample(min(POOL_CAP_PER_CLASS, len(g)), random_state=SEED)
                        for _, g in df.groupby("label")], ignore_index=True)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        seqs = df.sequence.tolist(); y = df.label.values.astype(int)
        print(f"=== {task}: pool n={len(df)} pos={y.mean():.3f} ===", flush=True)
        g99 = mmseqs_cluster(seqs, 0.99, 0.9, "d99")
        keep = pd.Series(range(len(seqs))).groupby(g99).first().values
        df = df.iloc[keep].reset_index(drop=True); seqs = df.sequence.tolist()
        y = df.label.values.astype(int); df["id"] = np.arange(len(df))
        print(f"  dedup@99%: {len(keep)} kept", flush=True)
        Xk = kmer(seqs)
        g90 = mmseqs_cluster(seqs, 0.90, 0.9, "id90")
        print(f"  mmseqs@90%: {len(np.unique(g90))} clusters over {len(seqs)} seqs "
              f"({time.time()-t0:.0f}s)", flush=True)
        rep[task] = {"n": len(seqs), "n_clusters_90": int(len(np.unique(g90))), "splits": {}}
        for hsd in [0, 1, 2]:
            val = group_holdout(g90, y, hsd)
            vf, vp = val.mean(), y[val].mean()
            if not (0.18 <= vf <= 0.45 and 0.35 <= vp <= 0.65):
                print(f"  [hsd={hsd}] skipped (valfrac={vf:.2f} pos={vp:.3f})", flush=True); continue
            ke = kmer_eval(Xk, y, val)
            name = f"{task}__identity_disjoint_hsd{hsd}"
            o = df[["id", "sequence", "label"]].copy(); o["group"] = g90
            o["partition"] = np.where(val, "val", "train")
            o.to_parquet(f"{OUT}/{name}.parquet")
            rep[task]["splits"][f"hsd{hsd}"] = dict(kmer_auroc=round(ke["auroc"], 4),
                kmer_mcc=round(ke["mcc"], 4), bestC=ke["bestC"], val_frac=round(float(vf), 3),
                val_pos=round(float(vp), 3), n_train=int((~val).sum()), n_val=int(val.sum()))
            print(f"  [hsd={hsd}] NO GATE -> kmer AUROC={ke['auroc']:.4f} MCC={ke['mcc']:.4f} "
                  f"C={ke['bestC']:g} valfrac={vf:.2f} pos={vp:.3f}  wrote {name}", flush=True)
        json.dump(rep, open(f"{OUT}/ungated_split_report.json", "w"), indent=2)
    print("\nDONE. Reference (gated, from their logs):")
    print("  Host_Tropism     random 0.9213 | composition-cluster 0.8034 (kept) | identity 0.9131 (DISCARDED)")
    print("  Pathogenecity    random 0.9685 | composition-cluster 0.8044 (kept) | identity FAILED gate")
    print("  Transmissibility random 0.9238 | composition-cluster 0.7395 (kept) | identity FAILED gate")


if __name__ == "__main__":
    main()
