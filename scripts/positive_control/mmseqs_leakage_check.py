"""
True sequence-identity leakage check (not just exact-string-match) for the
GUE-nonviral and EPI splits, using MMseqs2 the same way build_identity_splits.py
does for HVUE (--min-seq-id 0.90 -c 0.9). Pools each task's train+test, clusters,
and reports what fraction of TEST sequences land in a cluster that also contains
a TRAIN sequence -- i.e. genuine >=90%-identity leakage, not just exact dupes.

Usage: python mmseqs_leakage_check.py
"""
import os, subprocess, sys, tempfile
import pandas as pd

MMSEQS = os.environ.get("VB_MMSEQS", "/scratch/10906/arisk/biojepa-env/.pixi/envs/default/bin/mmseqs")
GUE_DIR = os.environ.get("VB_GUE_DIR", "/scratch/10906/arisk/genomic_unlearning_pc/results/gue_dir")
SCRATCH = os.environ.get("VB_SCRATCH", "/scratch/10906/arisk/genomic_unlearning_pc/results/mmseqs_scratch")

GUE_TASKS = [
    "gue_prom_core_all_official", "gue_prom_core_notata_official", "gue_prom_core_tata_official",
    "gue_prom_300_all_official", "gue_prom_300_notata_official", "gue_prom_300_tata_official",
    "gue_splice_reconstructed_official", "gue_tf_0_official", "gue_tf_1_official",
    "gue_tf_2_official", "gue_tf_3_official", "gue_tf_4_official",
]
EPI_CELLS = ["gm12878", "helas3", "huvec", "imr90", "k562", "nhek"]


def cluster_overlap(seqs_train, seqs_test, tag, min_id=0.90, cov=0.9):
    os.makedirs(SCRATCH, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        fa = f"{td}/pool.fasta"
        ids, part = [], []
        with open(fa, "w") as f:
            for i, s in enumerate(seqs_train):
                rid = f"tr{i}"; f.write(f">{rid}\n{s}\n"); ids.append(rid); part.append("train")
            for i, s in enumerate(seqs_test):
                rid = f"te{i}"; f.write(f">{rid}\n{s}\n"); ids.append(rid); part.append("test")
        pref = f"{td}/clu"
        subprocess.run([MMSEQS, "easy-cluster", fa, pref, f"{td}/tmp",
                        "--min-seq-id", str(min_id), "-c", str(cov), "--threads", "16", "-v", "1"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tsv = f"{pref}_cluster.tsv"
        rep_of = {}
        with open(tsv) as f:
            for line in f:
                rep, mem = line.strip().split("\t")
                rep_of[mem] = rep
        part_of = dict(zip(ids, part))
        train_reps = {rep_of[i] for i in ids if part_of[i] == "train"}
        n_test = sum(1 for i in ids if part_of[i] == "test")
        n_test_leaked = sum(1 for i in ids if part_of[i] == "test" and rep_of[i] in train_reps)
        frac = n_test_leaked / n_test if n_test else float("nan")
        print(f"{tag:<40} n_train={len(seqs_train):<6} n_test={n_test:<6} "
              f"test_in_train_cluster={n_test_leaked} ({frac:.4f})", flush=True)
        return dict(tag=tag, n_train=len(seqs_train), n_test=n_test,
                    n_test_leaked=n_test_leaked, frac_leaked=round(frac, 4))


def main():
    results = []
    print("== GUE non-viral (official split, >=90% identity clustering) ==")
    for task in GUE_TASKS:
        tr = pd.read_csv(f"{GUE_DIR}/{task}__train.csv")
        te = pd.read_csv(f"{GUE_DIR}/{task}__test.csv")
        results.append(cluster_overlap(tr.sequence.tolist(), te.sequence.tolist(), task))

    print("\n== EPI (official split, >=90% identity clustering, enhancer and promoter separately) ==")
    for cell in EPI_CELLS:
        tr = pd.read_csv(f"{GUE_DIR}/epi_{cell}_official__train.csv")
        te = pd.read_csv(f"{GUE_DIR}/epi_{cell}_official__test.csv")
        results.append(cluster_overlap(tr.enhancer.tolist(), te.enhancer.tolist(), f"epi_{cell}_enhancer"))
        results.append(cluster_overlap(tr.promoter.tolist(), te.promoter.tolist(), f"epi_{cell}_promoter"))

    df = pd.DataFrame(results)
    out = "/work/10906/arisk/vista/genomic-capability-unlearning/reports/mmseqs_leakage_check.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
