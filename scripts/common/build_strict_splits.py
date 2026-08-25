"""Build genuinely homology-clean HVUE splits via easy-SEARCH filtering.

The existing `identity_disjoint` splits used easy-CLUSTER with -c 0.9, which requires 90%
BIDIRECTIONAL coverage and is therefore blind to partial overlap. Measured leakage on those
splits: 42-83% of test rows have a train hit at >=90% id over >=50% of the test sequence.

Here we keep the same train set, then DROP any test row with a train hit above threshold.
This shrinks the test set but makes it defensible. Both the k-mer baseline and every FM are
then evaluated on the identical surviving rows, so the comparison stays paired.
"""
import os, subprocess, tempfile, shutil, json, sys
import pandas as pd, numpy as np
MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
M = "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel"
OUT = f"{M}/splits_strict"; os.makedirs(OUT, exist_ok=True)
SCRATCH = "/data/nvidia/tmp_mmseqs"; os.makedirs(SCRATCH, exist_ok=True)
ID_T, COV_T = 0.70, 0.30      # strict: drop test rows sharing >=30% of their length at >=70% id

rep = {}
for task in ("Host_Tropism", "Pathogenecity", "Transmissibility"):
    src = f"{M}/splits_ungated/{task}__identity_disjoint_hsd0.parquet"
    if not os.path.exists(src): continue
    d = pd.read_parquet(src)
    tr = d[d.partition == "train"].reset_index(drop=True)
    te = d[d.partition == "val"].reset_index(drop=True)
    dtmp = tempfile.mkdtemp(dir=SCRATCH)
    try:
        for nm, df in (("tr", tr), ("te", te)):
            with open(f"{dtmp}/{nm}.fa", "w") as f:
                for i, s in enumerate(df.sequence.values):
                    if isinstance(s, str) and len(s) > 20: f.write(f">{i}\n{s}\n")
        r = subprocess.run([MMSEQS, "easy-search", f"{dtmp}/te.fa", f"{dtmp}/tr.fa",
                            f"{dtmp}/h.tsv", f"{dtmp}/tmp", "--search-type", "3",
                            "--min-seq-id", "0.3", "-c", "0.1", "--cov-mode", "1", "-e", "1e-3",
                            "--threads", "16", "-v", "1",
                            "--format-output", "query,target,pident,alnlen,qlen,qcov"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{task}: mmseqs failed {r.stderr[-300:]}", flush=True); continue
        drop = set()
        if os.path.getsize(f"{dtmp}/h.tsv"):
            h = pd.read_csv(f"{dtmp}/h.tsv", sep="\t", header=None,
                            names=["q","t","pident","alnlen","qlen","qcov"])
            h["pident"] = h["pident"]/100.0 if h["pident"].max() > 1.5 else h["pident"]
            h["qcov"] = h["qcov"]/100.0 if h["qcov"].max() > 1.5 else h["qcov"]
            h["covf"] = np.maximum(h["qcov"], h["alnlen"] / h["qlen"].clip(lower=1))
            drop = set(h[(h["pident"] >= ID_T) & (h["covf"] >= COV_T)]["q"].unique())
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)
    keep = [i for i in range(len(te)) if i not in drop]
    te_clean = te.iloc[keep].reset_index(drop=True)
    if te_clean.label.nunique() < 2 or len(te_clean) < 100:
        print(f"{task}: STRICT TEST TOO SMALL/DEGENERATE n={len(te_clean)} "
              f"classes={te_clean.label.nunique()} -- cannot evaluate", flush=True)
        rep[task] = dict(n_test_before=len(te), n_test_after=len(te_clean),
                         pct_dropped=round(100*len(drop)/len(te),2),
                         usable=False, class_balance=te_clean.label.value_counts().to_dict())
        continue
    out = pd.concat([tr.assign(partition="train"), te_clean.assign(partition="val")], ignore_index=True)
    out.to_parquet(f"{OUT}/{task}__strict_id{int(ID_T*100)}_cov{int(COV_T*100)}.parquet", index=False)
    rep[task] = dict(n_train=len(tr), n_test_before=len(te), n_test_after=len(te_clean),
                     pct_dropped=round(100*len(drop)/len(te),2), usable=True,
                     class_balance=te_clean.label.value_counts().to_dict())
    print(f"{task}: test {len(te)} -> {len(te_clean)} ({rep[task]['pct_dropped']}% dropped) "
          f"balance={rep[task]['class_balance']}", flush=True)
json.dump(dict(id_threshold=ID_T, cov_threshold=COV_T, tasks=rep),
          open(f"{OUT}/strict_split_report.json","w"), indent=2)
print("\nwrote splits_strict/")
