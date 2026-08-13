# Viral Benchmark Qualification Repository Guide

## Project Background

The broader project studies whether a biological foundation-model capability
can first be identified clearly enough, and then later become a valid target
for capability unlearning. Before any unlearning claim is scientifically
meaningful, the benchmark itself must show stable model-specific headroom over
strong conventional baselines.

This README focuses on the viral benchmark qualification stage of that larger
project. The goal here is not to maximize absolute score, but to decide whether
any viral benchmark provides a clean, reproducible foundation-model advantage
worth carrying forward.

This repository contains four completed viral benchmark qualification tasks.

Core question:

> can a biological foundation model show stable, reproducible,
> model-specific predictive headroom over the strongest reasonable
> non-foundation baseline?

Current answer for all four completed tasks:

`No.`

## Four Tasks At A Glance

| Task | Model | Final Status | Strongest Conventional Baseline / Signal | Bottom Line |
|:--|:--|:--|:--|:--|
| HVUE | Evo | `UNQUALIFIED` | full-sequence k-mer / composition | The apparent advantage on host tropism and pathogenicity largely disappears after stronger sequence-composition controls. |
| ProteinGym | ESM2 | `UNQUALIFIED` | evolutionary baselines (`VESPA`, `VESPAl`, `S2F_MSA`) | Under strict position-held-out evaluation, evolutionary baselines match or beat the model; adaptation is not stable. |
| PHIStruct | SaProt | `PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED` | BLASTp | SaProt beats weak baselines, but does not establish a statistically reliable gain over BLASTp. |
| EvoMIL | ESM-1b | `NO_QUALIFYING_HEADROOM` | AA 3-mer / proteome composition | The strongest AA 3-mer baseline beats all five ESM-1b seeds. |

## How To Read This Repository

If you are collaborating on the four viral tasks, use this order:

1. read the task block below for the benchmark you care about;
2. open the listed code entrypoint to see how the experiment is implemented;
3. open the listed result directory and summary file to see the final evidence;
4. use the listed supporting artifacts only if you need deeper audit detail.

The two supporting overview documents are:

- `docs/foundation_model_benchmark_summary.md`
- `docs/benchmark_artifact_index.md`

## 1. HVUE / Evo

### What this task is

HVUE is used here as a viral host/pathogenicity benchmark family for testing
whether Evo shows meaningful viral capability beyond strong sequence-based
controls.

### Code directory

- main code directory: `phase2/`

### Main code files

- `phase2/prepare_benchmarks.py`
- `phase2/eval_benchmarks.py`
- `phase2/eval_kmer_baseline.py`
- `phase2/run_hvue_complete_selection_and_full.sh`
- `phase2/run_hvue_pipeline_watchdog.sh`
- `phase2/aggregate_hvue_lora.py`

### What each main file does

- `prepare_benchmarks.py`: builds the unified HVUE/GUE/ViroBench benchmark manifest
- `eval_benchmarks.py`: runs the common supervised downstream benchmark protocol
- `eval_kmer_baseline.py`: computes the sequence k-mer baseline used as a strong non-foundation comparator
- `run_hvue_complete_selection_and_full.sh`: launches the checked-in HVUE-centered full evaluation flow
- `run_hvue_pipeline_watchdog.sh`: resume/watchdog wrapper for long runs
- `aggregate_hvue_lora.py`: aggregates benchmark outputs into comparison tables

### Main result directory

- `data/phase2/full_benchmarks_lora_optimized_s600/`

### Main result files

- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv`
- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.json`
- `results/full_benchmark_summary.csv`
- `docs/full_benchmark_results.md`
- `docs/full_benchmark_artifact_audit.md`

### How to understand the result

HVUE is not stored as a standalone `*_qualification/` result package. Its
checked-in evidence lives inside the larger HVUE/GUE/ViroBench evaluation
stack. The conclusion to carry forward is:

`UNQUALIFIED`

The earlier apparent signal does not survive stronger full-sequence
composition controls well enough to support a clean Evo-specific capability
claim.

## 2. ProteinGym / ESM2

### What this task is

ProteinGym is used here to test whether ESM2 has stable mutation-effect signal
beyond strong evolutionary baselines under strict held-out evaluation.

### Code directory

- main code directory: `phase2/`

### Main code files

- `phase2/proteingym_esm2_qualification.py`
- `phase2/proteingym_esm2_top20_expansion.py`

### What each main file does

- `proteingym_esm2_qualification.py`: main formal qualification controller
- `proteingym_esm2_top20_expansion.py`: supporting expansion/screening workflow for candidate assays

### Main result directory

- `data/phase2/protein_48h_esm2_qualification/`

### Main result files

- `protein_48h_summary_report.json`
- `protein_48h_summary_report.md`
- `protein_48h_evolutionary_baseline_report.json`
- `protein_48h_evolutionary_baseline_report.md`
- `protein_48h_lora_qualification_evidence.json`
- `protein_48h_candidate_ranking.csv`
- `protein_48h_esm2_pilot_metrics.csv`
- `protein_48h_lora_metrics.csv`

### How to understand the result

This task finished with:

`UNQUALIFIED`

The checked-in summary shows:

- `20` static candidate assays screened
- `3` pilot assays selected
- `0` preliminarily qualified task-model pairs
- random-split positives do not transfer to position-held-out evaluation
- the one LoRA-advanced assay remains unstable across three seeds

## 3. PHIStruct / SaProt

### What this task is

PHIStruct is used here to test whether SaProt provides bacteriophage
receptor-binding host prediction headroom beyond sequence homology.

### Code directory

- main code directory: `phase2/`

### Main code files

- `phase2/phistruct_qualification.py`
- `phase2/phistruct_failure_audit_evomil_controller.py`

### What each main file does

- `phistruct_qualification.py`: main PHIStruct formal qualification controller
- `phistruct_failure_audit_evomil_controller.py`: post-failure audit controller that reconstructs the BLAST comparison and bootstrap evidence

### Main result directory

- `data/phase2/phistruct_qualification/`

### Main result files

- `summary_report.json`
- `summary_report.md`
- `baseline_results.csv`
- `plm_results.csv`
- `per_genus_metrics.csv`
- `phistruct_failure_audit/audit_summary.json`
- `phistruct_failure_audit/audit_summary.md`
- `phistruct_failure_audit/paired_bootstrap_summary.json`
- `phistruct_failure_audit/paired_bootstrap_samples.csv`

### How to understand the result

This task finished with:

`PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED`

Key checked-in numbers:

- SaProt macro-F1: `0.454732`
- BLASTp macro-F1: `0.475180`
- observed delta (`SaProt - BLASTp`): `-0.020448`
- 95% CI: `[-0.113066, 0.071794]`

Interpretation:

SaProt is better than weak/simple baselines, but it does not show statistically
reliable positive headroom over BLASTp.

## 4. EvoMIL / ESM-1b

### What this task is

EvoMIL is used here to test whether ESM-1b embeddings plus MIL outperform
strong proteome-composition baselines on viral host prediction.

### Code directory

- main code directory: `phase2/`

### Main code files

- `phase2/evomil_esm1b_qualification.py`
- `phase2/signed_bootstrap.py`

### What each main file does

- `evomil_esm1b_qualification.py`: full formal qualification controller with sequence recovery, preprocessing, baselines, embeddings, MIL, bootstrap, and summary
- `signed_bootstrap.py`: shared signed bootstrap helper used to keep `delta = model - baseline` consistent

### Main result directory

- `data/phase2/evomil_qualification/`

### Main result files

- `evomil_summary_report.json`
- `evomil_summary_report.md`
- `evomil_bootstrap_summary.json`
- `evomil_bootstrap_samples.csv`
- `evomil_kmer_baselines.csv`
- `evomil_model_results.csv`
- `evomil_split_audit.json`
- `evomil_preprocessing_audit.json`
- `evomil_reproduction_sanity_report.json`

### How to understand the result

This task finished with:

`NO_QUALIFYING_HEADROOM`

Key checked-in numbers:

- strongest baseline: `logistic_regression:aa_3mer_tfidf`
- baseline macro-F1: `0.841270`
- best ESM-1b + MIL macro-F1: `0.782246`
- observed delta (`model - baseline`): `-0.059024`
- bootstrap 95% CI: `[-0.204111, 0.056453]`
- positive seeds: `0 / 5`

Interpretation:

The strongest AA 3-mer baseline beats every checked-in ESM-1b seed, so this
task does not provide qualifying model-specific headroom.

## In-Progress Final Viral Candidate

There is also an implemented VPF-PLM controller for the final viral benchmark
search, but as of `August 13, 2026` it is still in progress and must not be
reported as a completed result.

Main code:

- `phase2/vpf_plm_qualification.py`
- `phase2/vpf_plm_compat.py`

Current execution mode:

- detached `screen` controller

## Shared Utilities And Tests

Shared qualification utilities:

- `phase2/signed_bootstrap.py`

Relevant tests:

- `tests/test_signed_bootstrap.py`
- `tests/test_evomil_esm1b_qualification.py`

## Minimal Layout

```text
phase2/        experiment controllers and benchmark code
data/phase2/   checked-in result artifacts
docs/          benchmark summaries and artifact indexes
results/       compact top-level result tables
tests/         regression tests
logs/          historical logs; active logs are not canonical result artifacts
```

## Practical Rule For Collaborators

If you want to inspect one viral task quickly, use this rule:

1. open the task section in this README
2. open the listed main code file
3. open the listed main result directory
4. read the listed summary `.md` or `.json` first

That is the fastest path to understand what the task does, how it was run, and
what final conclusion the repository supports.
