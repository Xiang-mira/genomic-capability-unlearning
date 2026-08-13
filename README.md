# Genomic Capability Unlearning

This repository now contains two tightly linked lines of work:

1. capability-unlearning experiments on genomic foundation models; and
2. a benchmark-qualification search asking whether any biological foundation
   model shows stable, reproducible, model-specific headroom over strong
   conventional baselines.

The four completed qualification studies in this repository are all currently
negative. That result matters: it means these candidates do not provide a clean
scientific target for later targeted unlearning claims.

## Benchmark Qualification Status

| Benchmark | Model | Key Conventional Baseline / Signal | Experimental Conclusion |
|:--|:--|:--|:--|
| HVUE | Evo | Full-sequence k-mer / composition | The apparent advantage on host tropism and pathogenicity largely disappears once stronger full-sequence composition baselines are used. |
| ProteinGym | ESM2 | Evolutionary baselines such as VESPA, VESPAl, and S2F_MSA | Under strict position-held-out evaluation, evolutionary methods match or beat the model; positive excess is not stable, and fresh adaptation remains seed-sensitive. |
| PHIStruct | SaProt | BLASTp sequence homology | SaProt clearly beats simple and structure-only baselines, but it does not establish a statistically reliable gain over BLASTp. |
| EvoMIL | ESM-1b | AA 3-mer / proteome composition | The strongest AA 3-mer baseline beats all five ESM-1b + MIL seeds, leaving no qualifying model-specific headroom. |

Formal summary document:

- `docs/foundation_model_benchmark_summary.md`

Code/result index:

- `docs/benchmark_artifact_index.md`

## Final Conclusions For The Four Completed Studies

### HVUE / Evo

Status: `UNQUALIFIED`

The strongest checked-in HVUE evidence comes from the completed 44-task
HVUE/GUE/ViroBench evaluation stack. The best GD run does induce forgetting on
HVUE, but the broader repository record and follow-up baseline analyses do not
support a clean Evo-specific viral capability claim once stronger sequence
composition controls are used. This benchmark should not be treated as a clean
foundation-model capability target.

### ProteinGym / ESM2

Status: `UNQUALIFIED`

The checked-in ProteinGym qualification run finished with no preliminarily
qualified assay-model pair. Strong public evolutionary baselines saturate the
pilot assays, random-split gains do not transfer to position-held-out
evaluation, and the one assay that advanced to LoRA remained unstable across
three formal seeds.

### PHIStruct / SaProt

Status: `PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED`

SaProt test macro-F1 is `0.454732`, while reconstructed BLASTp reaches
`0.475180`, for an observed delta of `-0.020448` under the sign convention
`model - baseline`. The paired bootstrap 95% CI is
`[-0.113066, 0.071794]`, so no statistically reliable positive headroom over
BLASTp was established.

### EvoMIL / ESM-1b

Status: `NO_QUALIFYING_HEADROOM`

The strongest non-foundation baseline is
`logistic_regression:aa_3mer_tfidf` with macro-F1 `0.841270`. The best
ESM-1b + MIL seed reaches macro-F1 `0.782246`, for an observed excess of
`-0.059024`. The bootstrap 95% CI is `[-0.204111, 0.056453]`, and zero of
five model seeds show positive headroom.

## Current Viral Search State

The four benchmark candidates above are complete. A separate final viral
qualification attempt for VPF-PLM is implemented in the repository and is
currently being executed in a detached `screen` workflow, but it is not yet a
finished result and should not be cited as concluded.

The current VPF controller covers:

- official asset audit;
- dataset freeze and reconstruction;
- strict remote-homology split and leakage audit;
- sequence/domain/homology baseline ladder;
- official PLM embedding/classifier evaluation;
- grouped bootstrap and final qualification decision.

## Repository Guide

### Benchmark qualification code

- `phase2/proteingym_esm2_qualification.py`
- `phase2/proteingym_esm2_top20_expansion.py`
- `phase2/phistruct_qualification.py`
- `phase2/phistruct_failure_audit_evomil_controller.py`
- `phase2/evomil_esm1b_qualification.py`
- `phase2/vpf_plm_qualification.py`
- `phase2/vpf_plm_compat.py`
- `phase2/signed_bootstrap.py`

### Canonical result roots

- `data/phase2/protein_48h_esm2_qualification/`
- `data/phase2/phistruct_qualification/`
- `data/phase2/evomil_qualification/`
- `data/phase2/full_benchmarks_lora_optimized_s600/`

### Main documentation

- `docs/foundation_model_benchmark_summary.md`
- `docs/benchmark_artifact_index.md`
- `docs/full_benchmark_results.md`
- `docs/full_benchmark_artifact_audit.md`

## Shared Statistical Fixes Added For Formal Qualification

The repository now includes a reusable signed bootstrap utility in
`phase2/signed_bootstrap.py`. It standardizes the sign convention

`delta = model_metric - baseline_metric`

across observed deltas, bootstrap means/medians, confidence intervals, and
tail probabilities. PHIStruct and EvoMIL now both use this shared
implementation, and tests were added to guard against future sign mismatches
and resume-schema regressions.

## Capability-Unlearning Work

The original unlearning project remains in the repository. Its completed
checked-in large benchmark is the 44-task downstream comparison across GD and
RMU checkpoints on HVUE, GUE, and ViroBench.

Best checked-in 44-task result summary:

| Checkpoint | Method | Balanced Forget | HVUE Drop | GUE Delta | Viral Delta | Selection Score |
|:--|:--|--:|--:|--:|--:|--:|
| `lora_gd_full_ar3_s200` | GD | 0.2066 | 0.1981 | -0.1443 | -0.0266 | 0.0357 |
| `lora_gd_full_ar5_s500` | GD | 0.0863 | 0.0840 | -0.1047 | -0.0218 | -0.0402 |
| `lora_rmu_full_sc50_s200` | RMU | 0.0045 | 0.0057 | +0.0012 | -0.0066 | -0.0021 |
| `lora_rmu_full_sc200_s200` | RMU | 0.0310 | 0.0363 | -0.0442 | -0.0058 | -0.0189 |

Interpretation: GD forgets more but is not selective enough; RMU preserves
more but does not forget enough. None of the four checkpoints reaches a clean
high-forget, low-retain-loss regime.

## Minimal Layout

```text
phase2/                    experiment controllers, evaluation code, and run scripts
data/phase2/               checked-in experiment artifacts and benchmark summaries
docs/                      benchmark summaries, audits, and reproducibility notes
results/                   compact result tables used in top-level summaries
tests/                     regression tests for controllers and shared utilities
tools/                     auxiliary plotting/build helpers
logs/                      selected historical run logs; active/incremental logs are not canonical results
```

## Reproducibility Note

Large raw datasets, model weights, and some local runtime toolchains are not
stored as normal Git artifacts. The checked-in qualification result directories
and summary reports are the canonical source for scientific conclusions in this
repository.
