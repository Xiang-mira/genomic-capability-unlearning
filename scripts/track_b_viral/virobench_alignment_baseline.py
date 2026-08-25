"""Alignment nearest-hit taxonomy baseline on ViroBench, MATCHED to our frozen-probe test set.

Cluster 2's stratified analysis shows that on ViroBench family macro-F1 the k-mer is NOT the
strongest train-only baseline: BLASTn nearest-hit beats it by ~+0.05 on matched clean examples,
while k-mer wins on accuracy. macro-F1 weights every family equally, so one close reference match
resolves a rare family the k-mer has almost no training signal for.

That matters because our new NT-v2 result (+0.0337 macro-F1 at the dev-selected layer L-2) is
measured against the k-mer alone. If an alignment baseline also clears the k-mer by ~+0.05, then
NT-v2's advantage is over the WRONG comparator and may not survive against max(k-mer, alignment).

BLAST is not installed here, so this uses MMseqs2 easy-search nucleotide mode as the aligner --
same nearest-hit-transfers-its-label protocol. Scored on EXACTLY the frozen-probe test taxids so
macro-F1 is comparable (Cluster 2's own warning: macro-F1 is not comparable across subsets).
"""
import os, subprocess, tempfile, shutil, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "track_b_viral"))
import virobench_baselines as VBB
from sklearn.metrics import f1_score, accuracy_score
MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
M = "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel"
SCRATCH = "/data/nvidia/tmp_mmseqs"; os.makedirs(SCRATCH, exist_ok=True)
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "family"

tr, dv, te = (VBB.load("ALL", "times", x) for x in ("train", "val", "test"))
keep = tr[LEVEL].value_counts(); keep = set(keep[keep >= 1].index) & set(te[LEVEL].dropna())
tr, dv, te = (d[d[LEVEL].isin(keep)].reset_index(drop=True) for d in (tr, dv, te))
ix = {c: i for i, c in enumerate(sorted(keep))}
for d in (tr, dv, te): d["y"] = d[LEVEL].map(ix)
print(f"{LEVEL}: classes={len(keep)} train={len(tr)} test={len(te)}", flush=True)

d = tempfile.mkdtemp(dir=SCRATCH)
try:
    for nm, df in (("tr", tr), ("te", te)):
        with open(f"{d}/{nm}.fa", "w") as f:
            for i, s in enumerate(df.sequence.values):
                if isinstance(s, str) and len(s) > 20: f.write(f">{i}\n{s}\n")
    print("  running mmseqs easy-search (nucleotide)...", flush=True)
    r = subprocess.run([MMSEQS, "easy-search", f"{d}/te.fa", f"{d}/tr.fa", f"{d}/h.tsv",
                        f"{d}/tmp", "--search-type", "3", "--min-seq-id", "0.0", "-c", "0.05",
                        "--cov-mode", "1", "-e", "10", "--max-seqs", "50", "--threads", "16",
                        "-v", "1", "--format-output", "query,target,bits,pident,alnlen"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("mmseqs failed:", r.stderr[-500:]); sys.exit(1)
    h = pd.read_csv(f"{d}/h.tsv", sep="\t", header=None, names=["q","t","bits","pident","alnlen"])
finally:
    shutil.rmtree(d, ignore_errors=True)

# nearest hit by bitscore transfers its label
best = h.sort_values("bits", ascending=False).drop_duplicates("q")
pred = np.full(len(te), -1, np.int64)
tr_y = tr.y.values
for q, t in zip(best.q.values, best.t.values):
    pred[int(q)] = tr_y[int(t)]
hit = pred >= 0
print(f"  test genomes with >=1 alignment hit: {hit.sum()}/{len(te)} ({100*hit.mean():.1f}%)", flush=True)
# unhit examples fall back to the majority training class (a real method must predict something)
maj = int(pd.Series(tr_y).mode()[0]); pred_full = np.where(hit, pred, maj)
yte = te.y.values
res = dict(level=LEVEL, n_classes=len(keep), n_test=len(te), hit_rate=float(hit.mean()),
           macro_f1=round(float(f1_score(yte, pred_full, average="macro", zero_division=0)), 4),
           accuracy=round(float(accuracy_score(yte, pred_full)), 4),
           macro_f1_hit_only=round(float(f1_score(yte[hit], pred[hit], average="macro", zero_division=0)), 4),
           accuracy_hit_only=round(float(accuracy_score(yte[hit], pred[hit])), 4))
print(f"  ALIGNMENT nearest-hit: macro-F1={res['macro_f1']:.4f}  acc={res['accuracy']:.4f}", flush=True)
np.savez(f"{M}/virobench_frozen/alignment__ALL_times_{LEVEL}__preds.npz",
         taxid=te.taxid.values, y_true=yte, y_pred=pred_full)
json.dump(res, open(f"{M}/virobench_alignment_{LEVEL}.json","w"), indent=2)
print(json.dumps(res, indent=2))
