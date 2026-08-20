"""
Prepares GUE's EPI (enhancer-promoter interaction) task for the positive-control
baseline: 6 cell lines, columns [enhancer (3000bp), promoter (2000bp), label],
10,000 train / 2,000 test rows each. No coordinate metadata -- same treatment as
the other GUE tasks: an "official" split (as released, dev carved from train)
and a "random" split (pool + stratified reshuffle 70/15/15).

Source: /scratch/10906/arisk/biojepa_data/gue/GUE/EPI/{cell}/{train,test}.csv
Output: $VB_GUE_DIR/epi_{cell}_{official,random}__{train,dev,test}.csv
        columns [enhancer, promoter, label] (paired -- epi_baselines.py reads both).
"""
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/viral_benchmark")
import paths as P

EPI_SRC = os.environ.get("PC_EPI_SRC_DIR", "/scratch/10906/arisk/biojepa_data/gue/GUE/EPI")
OUT_DIR = P.GUE_DIR
SEED = 42
CELLS = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
COLS = ["enhancer", "promoter", "label"]


def carve_dev(train_df, frac=0.15, seed=SEED):
    rest, dev = train_test_split(train_df, test_size=frac, stratify=train_df.label, random_state=seed)
    return rest.reset_index(drop=True), dev.reset_index(drop=True)


def random_split(pool_df, seed=SEED):
    tr, rest = train_test_split(pool_df, test_size=0.30, stratify=pool_df.label, random_state=seed)
    dv, te = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=seed)
    return tr.reset_index(drop=True), dv.reset_index(drop=True), te.reset_index(drop=True)


def write(name, tr, dv, te):
    for part, df in [("train", tr), ("dev", dv), ("test", te)]:
        df[COLS].to_csv(f"{OUT_DIR}/{name}__{part}.csv", index=False)
    print(f"  wrote {name}: train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)


def prep_cell(cell):
    slug = cell.lower().replace("-", "")
    base = f"{EPI_SRC}/{cell}"
    tr_full = pd.read_csv(f"{base}/train.csv")
    te_full = pd.read_csv(f"{base}/test.csv")

    tr, dv = carve_dev(tr_full)
    write(f"epi_{slug}_official", tr, dv, te_full)

    pool = pd.concat([tr_full, te_full], ignore_index=True)
    tr_r, dv_r, te_r = random_split(pool)
    write(f"epi_{slug}_random", tr_r, dv_r, te_r)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    for cell in CELLS:
        prep_cell(cell)


if __name__ == "__main__":
    main()
