"""Items 2 & 7: PARTIAL-overlap leakage between train and test.

MMseqs2 easy-cluster with `-c 0.9` requires 90% bidirectional coverage, so it is blind to a
test genome that shares only (say) 50% of its length with a training genome at high identity.
This measures what that blindness costs us, using easy-SEARCH (local alignment) instead of
clustering, which is the right tool for partial overlap.

Reports, for each split, the % of TEST rows having >=1 training hit at >=IDENT identity over
>=COVFRAC of the test sequence. This cuts AGAINST our own negative results, so it is measured
rather than assumed.
"""
import os, subprocess, sys, json, tempfile, shutil
import pandas as pd, numpy as np
MMSEQS = os.environ.get("VB_MMSEQS", "/home/nvidia/tools/mmseqs/bin/mmseqs")
M = "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel"
SCRATCH = "/data/nvidia/tmp_mmseqs"; os.makedirs(SCRATCH, exist_ok=True)
IDENTS = [0.90, 0.70, 0.50]
COVFRACS = [0.50, 0.30]

def fasta(path, ids, seqs):
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            if isinstance(s, str) and len(s) > 20: f.write(f">{i}\n{s}\n")

def search(train_df, test_df, tag, nt=True):
    d = tempfile.mkdtemp(dir=SCRATCH)
    try:
        fasta(f"{d}/tr.fa", train_df.index, train_df.sequence.values)
        fasta(f"{d}/te.fa", test_df.index, test_df.sequence.values)
        cmd = [MMSEQS, "easy-search", f"{d}/te.fa", f"{d}/tr.fa", f"{d}/hits.tsv", f"{d}/tmp",
               "--min-seq-id", "0.3", "-c", "0.1", "--cov-mode", "1", "-e", "1e-3",
               "--threads", "16", "-v", "1",
               "--format-output", "query,target,pident,alnlen,qlen,qcov"]
        if nt: cmd += ["--search-type", "3"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {tag}: mmseqs FAILED: {r.stderr[-400:]}", flush=True); return None
        if not os.path.getsize(f"{d}/hits.tsv"):
            print(f"  {tag}: no hits at all", flush=True)
            return {f"id{int(i*100)}_cov{int(c*100)}": 0.0 for i in IDENTS for c in COVFRACS}
        h = pd.read_csv(f"{d}/hits.tsv", sep="\t", header=None,
                        names=["q","t","pident","alnlen","qlen","qcov"])
        h["pident"] = h["pident"]/100.0 if h["pident"].max() > 1.5 else h["pident"]
        h["qcov"] = h["qcov"]/100.0 if h["qcov"].max() > 1.5 else h["qcov"]
        h["covfrac"] = np.maximum(h["qcov"], h["alnlen"] / h["qlen"].clip(lower=1))
        out = {}
        n = len(test_df)
        for i in IDENTS:
            for c in COVFRACS:
                leak = h[(h["pident"] >= i) & (h["covfrac"] >= c)]["q"].nunique()
                out[f"id{int(i*100)}_cov{int(c*100)}"] = round(100.0 * leak / n, 2)
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)

res = {}
# ---- item 2: HVUE identity-disjoint splits ----
for task in ("Host_Tropism", "Pathogenecity", "Transmissibility"):
    f = f"{M}/splits_ungated/{task}__identity_disjoint_hsd0.parquet"
    if not os.path.exists(f): continue
    d = pd.read_parquet(f)
    tr, te = d[d.partition == "train"], d[d.partition == "val"]
    r = search(tr, te, f"HVUE/{task}")
    if r: res[f"HVUE__{task}__identity_disjoint_hsd0"] = dict(n_train=len(tr), n_test=len(te), **r)
    print(f"  HVUE/{task}: {r}", flush=True)

# ---- item 7: ViroBench times split (previously assumed clean, never measured) ----
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "track_b_viral"))
    import virobench_baselines as VBB
    tr = VBB.load("ALL", "times", "train"); te = VBB.load("ALL", "times", "test")
    r = search(tr, te, "ViroBench/times")
    if r: res["VIROBENCH__ALL_times"] = dict(n_train=len(tr), n_test=len(te), **r)
    print(f"  ViroBench/times: {r}", flush=True)
except Exception as e:
    print(f"  ViroBench: {type(e).__name__}: {e}", flush=True)

json.dump(res, open(f"{M}/partial_overlap_audit.json", "w"), indent=2)
print(f"\nwrote partial_overlap_audit.json")
print("\n%% of TEST rows with a train hit at >=identity over >=coverage-of-test-seq:")
if res:
    print(pd.DataFrame(res).T.to_string())
