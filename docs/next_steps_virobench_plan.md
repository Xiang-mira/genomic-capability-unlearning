# ViroBench Unlearning Diagnostics Plan

> Superseded for scientific prioritization by
> `docs/causal_chain_validation_plan.md`. This document remains as the
> implementation checklist for ViroBench/trajectory diagnostics, but the active
> priority is host-tropism target validity and probe validity before full
> benchmark trajectory completion.

## Summary

This plan implements the next diagnostic pass for the Evo unlearning project. Viral-retain evaluation will use the repository-supported ViroBench CLS-Lite tasks instead of vGUE/Vir2vec. The goal is to make the current results interpretable by measuring checkpoint trajectories, method convergence, probe reliability, and host-tropism shortcut risk.

## Key Changes

- Save intermediate checkpoints for representative GD/RMU runs at steps `100,200,500,1000`.
- Log GD objective components: raw forget/retain losses, weighted forget/retain terms, and total objective.
- Log RMU representation diagnostics: forget-to-target distance, forget-to-original distance, retain MSE, original-vs-modified forget cosine similarity, steering target norm, and target variance.
- Evaluate every intermediate checkpoint on internal fixed probes, forget/retain PPL, HVUE, GUE, and ViroBench.
- Use ViroBench CLS-Lite tasks as viral-retain tasks:
  - `virobench_all_taxon_genus`
  - `virobench_all_taxon_times`
  - `virobench_dna_taxon_genus`
  - `virobench_dna_taxon_times`
  - `virobench_rna_taxon_genus`
  - `virobench_rna_taxon_times`
- Compare frozen linear probing with supervised fine-tuning on:
  - `hvue_human_host_tropism`
  - `hvue_human_virus_pathogenicity_cini`
  - `gue_prom_300_all`
  - `virobench_all_taxon_genus`

## Representative Runs

- GD localized: `gd_localized_ar5_s1000`
- GD random control: `gd_random_ar5`
- RMU localized: `rmu_localized_sc50_l4`
- RMU random control: `rmu_random_sc50`
- Reference: Base

If old checkpoint directories are incomplete, rerun the same hyperparameters with `--save-steps 100,200,500,1000`.

## Execution Order

1. Run a toy checkpoint-save smoke test with `--steps 3 --save-steps 1,2,3`.
2. Rerun the representative GD/RMU runs with intermediate checkpoint saving.
3. For every saved checkpoint, run `phase2/eval_unlearn.py` for internal fixed-probe AUROC and forget/retain PPL.
4. For every saved checkpoint, run `phase2/eval_benchmarks.py` on the final HVUE/GUE/ViroBench manifest.
5. Aggregate checkpoint trajectory metrics and plot forget-vs-retain trajectories.
6. Plot GD/RMU convergence diagnostics from training logs.
7. Run probe-vs-SFT comparison on the four-task subset with seeds `42,43,44`.
8. Summarize host-tropism controlled-split results: random, taxonomy/family-held-out, homology-aware, and within-family when available.

## Expected Outputs

- `trajectory_metrics.csv`
- `trajectory_taskwise_hvue_gue_virobench.csv`
- `gd_convergence_diagnostics.csv`
- `rmu_convergence_diagnostics.csv`
- `probe_vs_sft_results.csv`
- `probe_sft_correlation.json`
- `host_tropism_controlled_split_results.csv`

## Assumptions

- ViroBench is the formal viral-retain benchmark for this phase; vGUE is not part of the main result.
- ViroBench rows are already represented in the benchmark manifest with `benchmark=virobench` and `group=viral_retain`.
- If the full ViroBench suite is too slow for probe-vs-SFT, use `virobench_all_taxon_genus` as the representative viral-retain SFT task while keeping all six ViroBench tasks in trajectory benchmark evaluation.
