# Benchmark Runtime Estimate

## Current Runnable Suite

The final runnable suite contains 38 tasks and 1,670,176 total sequence rows:

- Primary Forget: 2 tasks, 72,930 rows
- Secondary Forget: 3 tasks, 732,085 rows
- GUE Retain: 33 tasks, 865,161 rows
- Viral Retain: 0 runnable tasks

The two excluded Caliciviridae tasks account for 111,071 rows and should not be part of the final external evaluation manifest.

## Assumptions

These estimates are based on the existing `phase2/eval_benchmarks.py` settings:

- Layers: `3-9`
- Max length: `512`
- Device: `cuda:0`
- Auto batch size: `96`
- Probe jobs: `7`
- CPU threads: `16`

Observed local pilot/full progress implies roughly 15-18 sequences/sec for feature extraction plus linear-probe fitting overhead. The completed lean-pilot runs for Base, `gd_full_ar5`, and `rmu_full_sc200` each took about 2.1-2.3 hours on a 134,254-row manifest. The estimates below assume a cold full-suite run with no checkpoint-specific feature cache hits.

## Per-Checkpoint Estimate

Expected runtime per checkpoint is approximately 32-36 hours.

Largest contributors:

- `hvue_human_transmissibility_orthomyxoviridae`: approximately 7.5-8.5 hours per checkpoint
- `hvue_human_transmissibility_coronaviridae`: approximately 3.0-3.5 hours per checkpoint
- `hvue_human_virus_pathogenicity_bvbrc_cov`: approximately 2.3-2.7 hours per checkpoint
- all GUE tasks together: approximately 16-18 hours per checkpoint
- primary forget tasks together: approximately 1.3-1.5 hours per checkpoint

## Three-Checkpoint Estimate

For Base, `gd_full_ar5`, and `rmu_full_sc200`, expect approximately 96-108 GPU-hours if run serially from scratch.

The existing `data/phase2/base_benchmarks/eval_benchmarks.csv` already contains complete base results for 39 tasks, but the cleanest final comparison is to run or regenerate Base against `data/benchmarks/final_external_eval_manifest.csv` so the output groups are exactly `primary_forget`, `secondary_forget`, and `gue_retain`.

If you choose to reuse existing full Base rows and only post-filter/relabel them, the remaining new work is approximately 64-72 GPU-hours for the two candidate checkpoints.

## Viral Retain Impact

The viral retain tasks are not currently runnable. Once task-ready vGUE/Vir2vec split tables are added under `data/benchmarks/raw/viral_retain/<task>/`, runtime will scale with the added row count. The preferred viral retain tasks should be integrated in this order:

1. `host_range_prediction`
2. `dna_vs_rna_virus`
3. `hiv1_vs_hiv2`
4. optional `sars_cov_2_lineage_typing`

Do not add `hiv1_tropism` to the final retain score because it overlaps with the forget objective.
