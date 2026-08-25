#!/usr/bin/env bash
# Reference invocations. Each job needs one GPU; use setsid so it survives the parent shell.
# Bare `nohup ... &` gets SIGTERM'd when the launching wrapper exits -- use setsid.
set -euo pipefail
: "${VB_OUT:?set VB_OUT}" ; : "${HF_HOME:?set HF_HOME}"
LOG="$VB_OUT/logs"; mkdir -p "$LOG"
PY=${PY:-python}
launch () { g=$1; shift; setsid env CUDA_VISIBLE_DEVICES="$g" nohup $PY -u "$@" \
            > "$LOG/$(echo "$*" | tr ' /' '__' | cut -c1-90).log" 2>&1 < /dev/null & disown; }

# --- 0. one-off: build the identity-disjoint HVUE splits (CPU, needs mmseqs) ---
# $PY build_identity_splits.py

# --- P1  Evo: finish the LR sweep, then run it on a defensible split ----------
launch 0 hvue_evo_lora.py --task Host_Tropism --split cluster_disjoint --seeds 44 --lrs 3e-4 --max_steps 8000
launch 1 hvue_evo_lora.py --task Host_Tropism --split identity_disjoint_hsd0 --seeds 42 43 44 --lrs 3e-4 --max_steps 8000

# --- P3  ViroBench at the full protocol (no class filter, all levels) --------
for lvl in family order class phylum kingdom; do
  launch 2 virobench_baselines.py --mod ALL --split times --level $lvl --min_count 1
done

# --- P5  gLMs on the identity-disjoint splits, remaining split seeds ---------
SD=$VB_OUT/splits_identity
for m in hyenadna gena_lm nt_v2_500m; do
  launch 3 hvue_glm.py --model $m --regime full --split_dir $SD \
      --kmer_json $SD/kmer_baselines.json --tasks Host_Tropism Pathogenecity Transmissibility \
      --splits identity_disjoint_hsd1 identity_disjoint_hsd2 --seeds 42 43 44 --lrs 1e-5 --bs 32
done

# --- GUE viral, all four gLMs + baselines -----------------------------------
launch 4 gue_baselines.py --task virus_covid --maxlen 1000
launch 5 gue_glm.py --model nt_v2_500m --task virus_species_40 --regime full --seeds 42 43 44 --lrs 1e-5 --bs 8
