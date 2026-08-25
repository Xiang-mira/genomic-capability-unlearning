
"""PHASE 5: proof-quality ViroBench split-integrity audit + PHASE 4: virus_covid dedup."""
import os, sys, json
sys.path.insert(0,'/data/nvidia/genomic-capability-unlearning/scripts/viral_benchmark')
import numpy as np, pandas as pd
import paths as P
OUT="/data/nvidia/genomic-capability-unlearning/reports"
rows=[]
print("="*100); print("PHASE 5  ViroBench split integrity -- every level, both splits, both modalities"); print("="*100)
for mod in ["DNA","ALL"]:
    for sp in ["genus","times"]:
        try:
            tr=pd.read_csv(f"{P.VIRO_DIR}/{mod}_taxon_{sp}__train.csv")
            te=pd.read_csv(f"{P.VIRO_DIR}/{mod}_taxon_{sp}__test.csv")
        except Exception as e:
            print(f"  MISSING {mod}/{sp}: {e}"); continue
        r=dict(mod=mod, split=sp, n_train=len(tr), n_test=len(te))
        for lvl in ["taxid","species","genus","family","order","class","phylum","kingdom"]:
            if lvl not in tr.columns: continue
            trs=set(tr[lvl].dropna()); tes=set(te[lvl].dropna())
            shared=trs&tes
            in_tr=te[lvl].isin(trs).mean()      # fraction of TEST EXAMPLES whose label seen in train
            r[f"{lvl}_train_uniq"]=len(trs); r[f"{lvl}_test_uniq"]=len(tes)
            r[f"{lvl}_shared"]=len(shared)
            r[f"{lvl}_frac_test_classes_in_train"]=round(len(shared)/max(len(tes),1),4)
            r[f"{lvl}_frac_test_examples_in_train"]=round(float(in_tr),4)
        if "first_release_date" in tr.columns:
            r["train_date_max"]=str(tr.first_release_date.max())[:10]
            r["test_date_min"]=str(te.first_release_date.min())[:10]
            r["date_overlap"]= r["test_date_min"] <= r["train_date_max"]
        rows.append(r)
        print(f"\n--- {mod}/taxon/{sp}  train={len(tr)} test={len(te)}")
        print(f"    {'level':<9}{'train':>7}{'test':>7}{'shared':>8}{'%test CLASSES seen':>20}{'%test EXAMPLES seen':>21}")
        for lvl in ["taxid","species","genus","family","order","class","phylum","kingdom"]:
            k=f"{lvl}_shared"
            if k not in r: continue
            print(f"    {lvl:<9}{r[lvl+'_train_uniq']:>7}{r[lvl+'_test_uniq']:>7}{r[k]:>8}"
                  f"{100*r[lvl+'_frac_test_classes_in_train']:>19.1f}%{100*r[lvl+'_frac_test_examples_in_train']:>20.1f}%")
        if "train_date_max" in r:
            print(f"    dates: train<={r['train_date_max']}  test>={r['test_date_min']}  overlap={r['date_overlap']}")
pd.DataFrame(rows).to_csv(f"{OUT}/virobench_split_audit.csv", index=False)
print(f"\nwrote {OUT}/virobench_split_audit.csv")

print("\n"+"="*100); print("PHASE 4  GUE virus_covid duplicate audit"); print("="*100)
tr=pd.read_csv(f"{P.GUE_DIR}/virus_covid__train.csv"); te=pd.read_csv(f"{P.GUE_DIR}/virus_covid__test.csv")
dv=pd.read_csv(f"{P.GUE_DIR}/virus_covid__dev.csv")
trmap={}
for s,l in zip(tr.sequence, tr.label): trmap.setdefault(s,set()).add(l)
dup=te.sequence.isin(trmap)
print(f"  train={len(tr)} dev={len(dv)} test={len(te)}")
print(f"  test rows whose sequence appears VERBATIM in train: {int(dup.sum())} ({100*dup.mean():.2f}%)")
conf=sum(1 for s,l in zip(te.sequence[dup], te.label[dup]) if l not in trmap[s])
print(f"  of those, LABEL CONFLICTS with the train copy: {conf} ({100*conf/max(int(dup.sum()),1):.1f}%)")
print(f"  internal train duplicates: {int(tr.sequence.duplicated().sum())} | internal test: {int(te.sequence.duplicated().sum())}")
clean=te[~dup].reset_index(drop=True)
print(f"  -> DEDUPED TEST: {len(clean)} rows ({100*len(clean)/len(te):.1f}% retained)")
print(f"     class balance before: {dict(sorted(te.label.value_counts().items()))}")
print(f"     class balance after : {dict(sorted(clean.label.value_counts().items()))}")
os.makedirs(P.GUE_DIR, exist_ok=True)
clean.to_csv(f"{P.GUE_DIR}/virus_covid__test_dedup.csv", index=False)
json.dump(dict(policy="remove leaked TEST rows, preserve all training data",
               duplicate_definition="exact full-sequence string match, case-sensitive",
               n_test_orig=len(te), n_test_dedup=len(clean), n_removed=int(dup.sum()),
               frac_removed=round(float(dup.mean()),4), label_conflicts=int(conf),
               note="near-duplicate (MMseqs2) leakage NOT yet removed -- exact match only"),
          open(f"{P.GUE_DIR}/virus_covid_dedup_manifest.json","w"), indent=2)
print(f"  wrote {P.GUE_DIR}/virus_covid__test_dedup.csv + manifest")
