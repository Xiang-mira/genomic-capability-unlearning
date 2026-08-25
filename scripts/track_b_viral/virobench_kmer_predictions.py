"""
k-mer3-6 baseline on ViroBench ALL/times/family, saving PER-EXAMPLE predictions.

Needed because the alignment baselines (Kraken2 especially) must be compared on
matched subsets: Kraken2's RefSeq-viral reference DB contains 15.4% of ViroBench's
test taxids verbatim, so its headline score is inflated by reference leakage that
our train-split-only k-mer never gets. Stratified comparison requires per-example
predictions, which virobench_baselines.py does not save (it only writes aggregate
metrics). Also satisfies the cross-cluster reproducibility requirement that every
run emit raw per-example predictions.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
from virobench_baselines import load, kmer_feats

OUT = P.sub("virobench")

def main():
    tr, dv, te = (load("ALL", "times", s) for s in ["train", "val", "test"])
    lv = "family"
    keep = tr[lv].value_counts()
    keep = set(keep[keep >= 1].index) & set(te[lv].dropna())
    tr, dv, te = (d[d[lv].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
    cls = sorted(keep); ix = {c: i for i, c in enumerate(cls)}
    for d in (tr, dv, te): d["y"] = d[lv].map(ix)
    print(f"classes={len(cls)} train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)

    ks = (3, 4, 5, 6)
    Xtr, Xdv, Xte = (kmer_feats(d.sequence.tolist(), ks, None) for d in (tr, dv, te))
    sc = StandardScaler().fit(Xtr)
    Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)

    best = (-1, None, None)
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=1500, n_jobs=-1).fit(Xtr, tr.y)
        s = f1_score(dv.y, clf.predict(Xdv), average="macro", zero_division=0)
        print(f"  C={C} dev_macro_f1={s:.4f}", flush=True)
        if s > best[0]: best = (s, C, clf)
    pred = best[2].predict(Xte)

    inv = {v: k for k, v in ix.items()}
    np.savez(f"{OUT}/ALL_times_family__kmer36_predictions.npz",
             taxid=te.taxid.astype(str).values,
             y_true=te.y.values, y_pred=pred,
             classes=np.array(cls, dtype=object))
    res = dict(macro_f1=round(float(f1_score(te.y, pred, average="macro", zero_division=0)), 4),
               accuracy=round(float(accuracy_score(te.y, pred)), 4),
               mcc=round(float(matthews_corrcoef(te.y, pred)), 4),
               C=best[1], dev_macro_f1=round(best[0], 4), n_classes=len(cls),
               context="whole_genome")
    json.dump(res, open(f"{OUT}/ALL_times_family__kmer36_withpreds.json", "w"), indent=2)
    print("DONE", res, flush=True)

if __name__ == "__main__":
    main()
