"""Final splice table: supervised baseline ladder vs frozen probe vs full fine-tune.

Keeps the three adaptation regimes in SEPARATE columns (never mixed), and marks published
comparators as EXTERNAL/PUBLISHED rather than OUR RUN.
"""
import json, glob, os, sys
import pandas as pd
B = "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel"
TASKS = ["splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]

# --- our supervised baseline ladder (dev-selected) ---
base = {}
for f in glob.glob(f"{B}/capacity_sweep/splice__*.json"):
    d = json.load(open(f)); t = d["task"]
    ds = d["dev_selected"]
    base[t] = dict(dev_sel=f"{ds['arch']} {ds['params_M']}M", test=ds["test"],
                   incumbent=(d.get("incumbent") or {}).get("test"),
                   oracle=d["oracle_best_test"])

# --- frozen probes (Phase 2 positive control) ---
froz = {}
for f in glob.glob(f"{B}/splice_control/*.json") + glob.glob(f"{B}/splice_positive_control/*.json"):
    d = json.load(open(f))
    t = d.get("task")
    for k, v in d.items():
        if isinstance(v, dict) and "mcc" in v:
            froz.setdefault(t, {})[k] = v["mcc"]

# --- full fine-tune ---
ft = {}
for f in glob.glob(f"{B}/splice_finetune/*.json"):
    d = json.load(open(f))
    ft.setdefault(d["task"], {})[d["model"]] = dict(
        test=d["test_mean"], dev=max(r["dev"] for r in d["runs"]), lr=d["lr"])

rows = []
for t in TASKS:
    b = base.get(t, {})
    for mdl in ("nt_v2_500m", "gena_lm", "hyenadna"):
        fz = (froz.get(t) or {}).get(mdl)
        f_ = (ft.get(t) or {}).get(mdl)
        rows.append(dict(
            task=t.replace("splice_sites_", ""), model=mdl,
            baseline_incumbent=b.get("incumbent"),
            baseline_dev_selected=b.get("test"), baseline_arch=b.get("dev_sel"),
            frozen_probe=fz, full_ft=(f_ or {}).get("test"), ft_lr=(f_ or {}).get("lr"),
            ft_minus_baseline=(round(f_["test"] - b["test"], 4)
                               if f_ and b.get("test") is not None else None),
            frozen_minus_baseline=(round(fz - b["test"], 4)
                                   if fz is not None and b.get("test") is not None else None)))
d = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("METRIC = MCC. All cells OUR RUN. Baseline = dev-selected cell from the 13-point")
print("architecture/capacity ladder. Published comparators (EXTERNAL/PUBLISHED, fine-tuned):")
print("  NT-v2 / DNABERT-2 / GENA-LM splice MCC 0.971-0.984 -- NOT our runs, not like-for-like.\n")
print(d.to_string(index=False))
d.to_csv(f"{B}/splice_final_table.csv", index=False)
print(f"\nwrote splice_final_table.csv")
miss = d[d.full_ft.isna()]
if len(miss): print(f"PENDING full-FT cells: {len(miss)}")
