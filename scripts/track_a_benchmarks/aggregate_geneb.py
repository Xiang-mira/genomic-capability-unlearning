"""Four-way GENEB comparison on identical splits and identical dev rows.

  fair k-mer (C2)  |  CNN ladder (dev-selected cell)  |  frozen probe (C2)  |  full FT (LR-swept)

All four select on the SAME stratified-15%-seed-42 dev carve, because GENEB ships no dev split.
Regimes are kept in separate columns and never mixed: on splice the frozen->FT gap alone is
0.59 MCC, larger than any gap between models.

The question this answers: do the GENEB probe wins over a k-mer survive a real CNN baseline?
Until now every non-viral positive in the programme was k-mer-anchored.
"""
import json, glob, os, sys
import numpy as np, pandas as pd
M = os.environ.get("VB_OUT", "/data/nvidia/genomic-capability-unlearning/scratchpad/multimodel")
R = "/data/nvidia/genomic-capability-unlearning/reports"
MODELS = ("nt_v2_500m", "gena_lm", "hyenadna")
TIE = 0.005

# --- Cluster 2 reference numbers (frozen probe + fair k-mer), from the handoff table ---
C2 = {  # task-label: (naive kmer, fair kmer, nt_v2, hyenadna, gena_lm, best_published)
 "InstaDeepAI_nucleotide_transformer_downstream_tasks_H3":            (0.602,0.590,0.662,0.671,0.705,0.781),
 "InstaDeepAI_nucleotide_transformer_downstream_tasks_promoter_all":  (0.754,0.813,0.882,0.835,0.917,0.930),
 "InstaDeepAI_nucleotide_transformer_downstream_tasks_enhancers":     (0.456,0.425,0.396,0.485,0.463,0.526),
 "deep4mc_A.thaliana_4mC":                                            (0.202,0.204,0.331,0.071,0.185,0.402),
 "InstaDeepAI_nucleotide_transformer_downstream_tasks_revised_splice_sites_acceptors":
                                                                      (0.269,0.387,0.479,0.402,0.543,0.685),
 "InstaDeepAI_plant-genomic-benchmark_lncrna.g_max":                  (0.111,0.155,0.233,0.156,0.225,0.475),
 "leannmlindsey_GUE_mouse_0":                                         (0.437,0.437,0.378,0.156,0.464,0.667),
 "leannmlindsey_GUE_human_tf_0":                                      (0.611,0.537,0.576,0.563,0.672,0.690),
 "katarinagresova_Genomic_Benchmarks_demo_human_or_worm":             (0.815,0.812,0.893,0.782,0.931,0.948),
 "katarinagresova_Genomic_Benchmarks_human_ensembl_regulatory":       (0.348,0.289,0.526,0.555,0.526,0.597),
 "leannmlindsey_GUE_phage_fragments":                                 (0.512,0.604,0.854,0.479,0.659,0.950),
 "katarinagresova_Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs":(0.706,0.734,0.780,0.677,0.853,0.904),
 "iDHS-EL_DNase_I":                                                   (0.000,0.589,0.593,0.413,0.509,0.728),
}
SHORT = {k: (k.replace("InstaDeepAI_nucleotide_transformer_downstream_tasks_","NT_")
              .replace("katarinagresova_Genomic_Benchmarks_","GB_")
              .replace("leannmlindsey_","").replace("InstaDeepAI_plant-genomic-benchmark_","plant_")
              .replace("demo_","")[:34]) for k in C2}

# --- our runs ---
ladder = {}
for f in glob.glob(f"{M}/capacity_sweep/geneb__*.json"):
    d = json.load(open(f)); ladder[d["task"]] = d
ft = {}
for f in glob.glob(f"{M}/geneb_finetune/*__fullft.json"):
    d = json.load(open(f))
    if d.get("collapsed_to_majority_class"): continue
    k = (d["task"], d["model"])
    cur = ft.get(k)
    dev = max(r["dev_mcc"] for r in d["runs"])
    if cur is None or dev > cur[0]:
        ft[k] = (dev, d["test_mcc_mean"], d.get("test_mcc_sd"), d["lr"], len(d["runs"]))

rows = []
for t, (naive, fair, p_nt, p_hy, p_ge, pub) in C2.items():
    lad = ladder.get(t)
    cnn = lad["dev_selected"]["test"] if lad else None
    probes = {"nt_v2_500m": p_nt, "hyenadna": p_hy, "gena_lm": p_ge}
    best_probe = max(probes.values())
    ftv = {m: ft.get((t, m)) for m in MODELS}
    best_ft = max([v[1] for v in ftv.values() if v], default=None)
    base = max([x for x in (fair, cnn) if x is not None], default=fair)
    n_ft = sum(1 for v in ftv.values() if v)
    # PER-MODEL rows are the defensible unit. best_* is a max-over-models statistic and is only
    # meaningful once all 3 models are done -- otherwise it compares best-of-3 probe against
    # whichever single model happens to have finished fine-tuning.
    for m in MODELS:
        v = ftv[m]
        rows.append(dict(task=SHORT[t], model=m,
            fair_kmer=fair, CNN_ladder=cnn, baseline=round(base, 4),
            probe=probes[m], FT=v[1] if v else None, FT_lr=v[3] if v else None,
            FT_sd=v[2] if v else None, published=pub,
            probe_vs_kmer=round(probes[m]-fair, 4),
            probe_vs_baseline=round(probes[m]-base, 4) if cnn is not None else None,
            FT_vs_probe=round(v[1]-probes[m], 4) if v else None,
            FT_vs_baseline=round(v[1]-base, 4) if (v and cnn is not None) else None,
            complete=(n_ft == 3 and cnn is not None)))
d = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("GENEB — four-way comparison, identical splits, identical dev carve (stratified 15%, seed 42)")
print("metric = MCC.  fair_kmer/probe/published are Cluster 2 runs; CNN_ladder/FT are ours.")
print("Rows are PER MODEL. 'complete' = all 3 models fine-tuned AND the CNN ladder is in.\n")
print(d[d.FT.notna()].to_string(index=False) if d.FT.notna().any() else "(no FT cells yet)")

done_l = d[d.CNN_ladder.notna()].task.nunique()
done_f = d.FT.notna().sum()
print(f"\ncoverage: CNN ladder {done_l}/13 tasks | full FT {done_f}/39 model-task cells "
      f"| fully complete tasks {d[d.complete].task.nunique()}/13")
comp = d[d.complete]
if len(comp):
    print(f"\n--- {comp.task.nunique()} FULLY COMPLETE tasks (all 3 models FT'd + CNN ladder) ---")
    for lab, col in (("probe", "probe_vs_baseline"), ("FT", "FT_vs_baseline")):
        v = comp[col].dropna()
        print(f"  {lab:<6} beats max(kmer,CNN): {(v>TIE).sum()}/{len(v)} model-task cells  (mean {v.mean():+.4f})")
    for m in MODELS:
        sm = comp[comp.model == m]
        if len(sm):
            print(f"    {m:<12} probe {(sm.probe_vs_baseline>TIE).sum()}/{len(sm)} | "
                  f"FT {(sm.FT_vs_baseline>TIE).sum()}/{len(sm)} | FT-probe mean {sm.FT_vs_probe.mean():+.4f}")
if done_l:
    sub = d[d.CNN_ladder.notna()].drop_duplicates("task")
    print(f"\n--- on the {len(sub)} tasks WITH a CNN baseline ---")
    print(f"\n--- {len(sub)} tasks with a CNN ladder number (per-task, k-mer view) ---")
    print(f"  CNN is the binding baseline on {(sub.CNN_ladder>sub.fair_kmer).sum()}/{len(sub)} tasks "
          f"(i.e. CNN > fair k-mer)")
d.to_csv(f"{M}/geneb_four_way.csv", index=False)
print(f"\nwrote {M}/geneb_four_way.csv")
