# Full Benchmark Artifact Audit

This inventory covers the completed 44-task benchmark and its candidate
selection stage.

## Programs and configuration

The repository includes the complete executable path:

- `phase2/unlearn_gd.py` and `phase2/unlearn_rmu.py`
- `phase2/run_task2_sweeps.py`
- `phase2/sweep_configs/lora_full_grid.json`
- `phase2/eval_benchmarks.py`
- `phase2/run_benchmark_pilot.py`
- `phase2/rank_benchmark_pilot.py`
- `phase2/run_hvue_complete_selection_and_full.sh`
- `phase2/run_optimized_full_benchmark.sh`
- `phase2/run_optimized_full_watchdog.sh`

## Candidate selection artifacts

`data/phase2/checkpoints_lora_grid/` contains the `meta.json`, training
`log.json`, and sweep status for all 12 candidate checkpoints. The candidate
pilot directory contains the base and 12 candidate task results, summaries,
progress records, per-task logs, and pilot ranking.

## Final benchmark artifacts

`data/phase2/full_benchmarks_lora_optimized_s600/` contains:

- the selected batch profile and preflight log;
- base plus four checkpoint result CSV files;
- progress and summary JSON files for all five runs;
- 44 per-task training logs for each run;
- the final CSV and JSON rankings;
- hashes and counts for the sequence-bearing input manifests.

The result coverage is 33 GUE retain tasks, 5 HVUE forget tasks, and 6
ViroBench retain tasks for every evaluated model.

## Deliberate exclusions

The following are not stored in ordinary Git:

- the 2.4 GB full sequence manifest and 339 MB pilot sequence manifest;
- the Evo base-model files;
- unlearning model weights;
- discarded downstream task checkpoints;
- raw downloaded benchmark corpora.

The exact manifest sizes, row counts, schemas, and SHA-256 hashes are recorded
in
`data/phase2/full_benchmarks_lora_optimized_s600/manifest_audit.json`.
