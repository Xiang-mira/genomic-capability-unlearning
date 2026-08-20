"""
Positive-control task prep: materializes NTv3 + GUE non-viral tasks into the
{task}__{train,dev,test}.csv schema that scripts/viral_benchmark/gue_baselines.py
already consumes unmodified, under two split variants:

  {task}_identity_disjoint   NTv3 only. The official split IS chromosome-disjoint
                             (verified: every task's test.parquet is chr20/chr21
                             only, train.parquet never touches them) -- kept as-is,
                             just carving a dev set out of train by stratified
                             random sampling (dev never touches test).
  {task}_random              NTv3 + GUE. Pool official train+test, stratified
                             shuffle 70/15/15 (seed 42). No MMseqs2, no clustering --
                             these are short fixed-window regulatory snippets, not
                             whole genomes with homology to control for.
  {task}_official            GUE only. Official train/test as released (disjointness
                             unverified -- no coordinate metadata locally), dev
                             carved from train. Kept so the baseline number is
                             directly comparable to competitor_comparison_TEST.md.

Source data (already local, not re-downloaded):
  NTv3:  /scratch/10906/arisk/biojepa_data/ntv3/{task}/{train,test}.parquet
         columns: sequence, name (encodes chrom:start-end), label, task
  GUE:   /scratch/10906/arisk/biojepa_data/gue/GUE/{subdir}/{train,test}.csv
         columns: sequence, label

Output: $VB_GUE_DIR/{task}_{split}__{train,dev,test}.csv, columns [sequence, label].
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/viral_benchmark")
import paths as P

NTV3_SRC = os.environ.get("PC_NTV3_DIR", "/scratch/10906/arisk/biojepa_data/ntv3")
GUE_SRC = os.environ.get("PC_GUE_SRC_DIR", "/scratch/10906/arisk/biojepa_data/gue/GUE")
OUT_DIR = P.GUE_DIR
SEED = 42

NTV3_TASKS = [
    "enhancers", "enhancers_types", "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3",
    "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
    "promoter_all", "promoter_no_tata", "promoter_tata",
    "splice_sites_acceptors", "splice_sites_donors", "splice_sites_all",
]

GUE_TASKS = {
    "gue_prom_core_all": "prom/prom_core_all",
    "gue_prom_core_notata": "prom/prom_core_notata",
    "gue_prom_core_tata": "prom/prom_core_tata",
    "gue_prom_300_all": "prom/prom_300_all",
    "gue_prom_300_notata": "prom/prom_300_notata",
    "gue_prom_300_tata": "prom/prom_300_tata",
    "gue_splice_reconstructed": "splice/reconstructed",
    "gue_tf_0": "tf/0",
    "gue_tf_1": "tf/1",
    "gue_tf_2": "tf/2",
    "gue_tf_3": "tf/3",
    "gue_tf_4": "tf/4",
}


def chrom(name_series):
    return name_series.str.extract(r"^(chr[0-9XYM]+)")[0]


def carve_dev(train_df, frac=0.15, seed=SEED):
    rest, dev = train_test_split(train_df, test_size=frac, stratify=train_df.label, random_state=seed)
    return rest.reset_index(drop=True), dev.reset_index(drop=True)


def random_split(pool_df, seed=SEED):
    tr, rest = train_test_split(pool_df, test_size=0.30, stratify=pool_df.label, random_state=seed)
    dv, te = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=seed)
    return tr.reset_index(drop=True), dv.reset_index(drop=True), te.reset_index(drop=True)


def write(name, tr, dv, te):
    for part, df in [("train", tr), ("dev", dv), ("test", te)]:
        df[["sequence", "label"]].to_csv(f"{OUT_DIR}/{name}__{part}.csv", index=False)
    print(f"  wrote {name}: train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)


def prep_ntv3(task):
    base = f"{NTV3_SRC}/{task}"
    tr_full = pd.read_parquet(f"{base}/train.parquet")
    te_full = pd.read_parquet(f"{base}/test.parquet")

    tr_chr, te_chr = chrom(tr_full.name), chrom(te_full.name)
    overlap = set(tr_chr.dropna().unique()) & set(te_chr.dropna().unique())
    assert not overlap, f"{task}: train/test chromosome overlap {overlap} -- not identity-disjoint!"

    tr, dv = carve_dev(tr_full)
    write(f"{task}_identity_disjoint", tr, dv, te_full)

    pool = pd.concat([tr_full, te_full], ignore_index=True)
    tr_r, dv_r, te_r = random_split(pool)
    write(f"{task}_random", tr_r, dv_r, te_r)


def prep_gue(name, subdir):
    base = f"{GUE_SRC}/{subdir}"
    tr_full = pd.read_csv(f"{base}/train.csv")
    te_full = pd.read_csv(f"{base}/test.csv")

    tr, dv = carve_dev(tr_full)
    write(f"{name}_official", tr, dv, te_full)

    pool = pd.concat([tr_full, te_full], ignore_index=True)
    tr_r, dv_r, te_r = random_split(pool)
    write(f"{name}_random", tr_r, dv_r, te_r)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("\n== NTv3 (identity_disjoint + random) ==")
    for task in NTV3_TASKS:
        prep_ntv3(task)

    print("\n== GUE (official + random) ==")
    for name, subdir in GUE_TASKS.items():
        prep_gue(name, subdir)


if __name__ == "__main__":
    main()
