"""Phase 3: paired bootstrap, frozen-probe gLMs vs k-mer LR on ViroBench ALL/times.

Regenerates k-mer per-example test predictions under the SAME protocol as
virobench_baselines.py (whole-genome counts, StandardScaler, C on dev macro-F1),
then pairs them by taxid against each saved frozen-probe prediction vector.

Reports the paired delta with a bootstrap CI over test examples, and an equivalence
verdict against pre-declared practical margins delta in {0.01, 0.02, 0.03, 0.05}.
Margins are declared HERE, before any CI is computed, per the project guardrail.
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "track_b_viral"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import virobench_baselines as VBB
import warnings; warnings.filterwarnings("ignore")

DELTAS = [0.01, 0.02, 0.03, 0.05]        # pre-declared, not chosen after seeing CIs
OUT = "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel/virobench_frozen"
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "family"
NBOOT = 2000

tr, dv, te = (VBB.load("ALL", "times", x) for x in ("train", "val", "test"))
keep = tr[LEVEL].value_counts(); keep = set(keep[keep >= 1].index) & set(te[LEVEL].dropna())
tr, dv, te = (d[d[LEVEL].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
ix = {c: i for i, c in enumerate(sorted(keep))}
for d in (tr, dv, te): d["y"] = d[LEVEL].map(ix)
print(f"{LEVEL}: classes={len(keep)} train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)

t0 = time.time()
Xtr, Xdv, Xte = (VBB.kmer_feats(d.sequence.tolist(), (3,4,5,6), None) for d in (tr, dv, te))
sc = StandardScaler().fit(Xtr); Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
print(f"  kmer3-6 feats {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)
best = (-1, None, None)
for C in [0.01, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=1500, n_jobs=-1).fit(Xtr, tr.y)
    s = f1_score(dv.y, clf.predict(Xdv), average="macro", zero_division=0)
    print(f"  C={C:<6} dev macroF1={s:.4f}", flush=True)
    if s > best[0]: best = (s, C, clf.predict(Xte))
dev_kmer, C_kmer, kpred = best
kmer_f1 = f1_score(te.y, kpred, average="macro", zero_division=0)
print(f"  kmer3-6 selected C={C_kmer} (dev {dev_kmer:.4f}) -> TEST macroF1 {kmer_f1:.4f}", flush=True)
np.savez(f"{OUT}/kmer3-6__ALL_times_{LEVEL}__preds.npz",
         taxid=te.taxid.values, y_true=te.y.values, y_pred=kpred, C=C_kmer, dev=dev_kmer)

kmap = dict(zip(te.taxid.values, zip(te.y.values, kpred)))
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob(f"{OUT}/*ALL_times_{LEVEL}__W*__preds.npz")):
    d = np.load(f, allow_pickle=True)
    tid, yt, yp = d["taxid"], d["y_true"], d["y_pred"]
    ok = np.array([t in kmap for t in tid])
    tid, yt, yp = tid[ok], yt[ok], yp[ok]
    ky = np.array([kmap[t][0] for t in tid]); kp = np.array([kmap[t][1] for t in tid])
    assert (ky == yt).all(), f"label mismatch in {f}"
    n = len(yt)
    fm_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    km_f1 = f1_score(yt, kp, average="macro", zero_division=0)
    bs = np.empty(NBOOT)
    for b in range(NBOOT):
        i = rng.integers(0, n, n)
        bs[b] = (f1_score(yt[i], yp[i], average="macro", zero_division=0)
                 - f1_score(yt[i], kp[i], average="macro", zero_division=0))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    name = os.path.basename(f).replace("__preds.npz", "")
    eq = {f"delta_{dd}": bool(hi < dd and lo > -dd) for dd in DELTAS}
    rows.append(dict(run=name, n=n, fm=round(fm_f1,4), kmer=round(km_f1,4),
                     delta=round(fm_f1-km_f1,4), ci_lo=round(lo,4), ci_hi=round(hi,4),
                     favors_fm_p=round(float((bs>0).mean()),4), **eq))
    print(f"  {name:<52} n={n} FM={fm_f1:.4f} kmer={km_f1:.4f} d={fm_f1-km_f1:+.4f} "
          f"CI[{lo:+.4f},{hi:+.4f}] P(d>0)={float((bs>0).mean()):.3f}", flush=True)

res = dict(level=LEVEL, n_classes=len(keep), metric="macro_f1", nboot=NBOOT,
           kmer_C=C_kmer, kmer_dev=dev_kmer, kmer_test=round(kmer_f1,4),
           pre_declared_deltas=DELTAS, comparisons=rows)
json.dump(res, open(f"{OUT}/../paired_bootstrap_viro_{LEVEL}.json","w"), indent=2)
print(f"\nwrote paired_bootstrap_viro_{LEVEL}.json", flush=True)
