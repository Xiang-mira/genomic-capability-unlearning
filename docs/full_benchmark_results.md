# Completed 44-Task Full Benchmark

This document records the completed benchmark used in the current project
summary. It compares two GD checkpoints and two RMU checkpoints against the
same base model.

## Benchmark coverage

Each model was evaluated on the same 44 downstream tasks:

- 33 GUE tasks for general genomic capability retention;
- 5 HVUE tasks for target viral capability forgetting;
- 6 ViroBench tasks for non-target viral capability retention.

The downstream evaluator uses a fixed training protocol for every model and
selects the best task checkpoint using validation performance. The final
comparison uses paired task-level changes relative to the base model.

## Results

| Checkpoint | Balanced forget | HVUE drop | GUE delta | Viral delta | Selection score |
|:--|--:|--:|--:|--:|--:|
| `lora_gd_full_ar3_s200` | 0.2066 | 0.1981 | -0.1443 | -0.0266 | 0.0357 |
| `lora_gd_full_ar5_s500` | 0.0863 | 0.0840 | -0.1047 | -0.0218 | -0.0402 |
| `lora_rmu_full_sc50_s200` | 0.0045 | 0.0057 | +0.0012 | -0.0066 | -0.0021 |
| `lora_rmu_full_sc200_s200` | 0.0310 | 0.0363 | -0.0442 | -0.0058 | -0.0189 |

GD produces the strongest target-task reduction, but its GUE degradation shows
that the effect is not sufficiently selective. Weak RMU preserves GUE
performance but produces almost no forgetting. Stronger RMU increases
forgetting, but the gain remains limited and GUE performance begins to fall.
None of the four checkpoints reaches the desired high-forgetting,
low-retain-loss region.

## Reproducibility files

- Exact ranking and confidence intervals:
  `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv`
- Per-task metrics:
  `data/phase2/full_benchmarks_lora_optimized_s600/<run>/eval_benchmarks.csv`
- Run summaries:
  `data/phase2/full_benchmarks_lora_optimized_s600/<run>/eval_benchmarks_summary.json`
- Unlearning configuration and training history:
  `data/phase2/checkpoints_lora_grid/<checkpoint>/meta.json` and `log.json`
- Figure generator:
  `tools/plot_full_benchmark_results.py`

Model weights, raw sequence corpora, and temporary task checkpoints are not
stored in Git.
