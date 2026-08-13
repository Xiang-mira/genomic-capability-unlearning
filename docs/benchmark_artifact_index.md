# Benchmark Artifact Index

This index maps each completed benchmark study to its main code entrypoints and
its canonical checked-in result artifacts.

## 1. HVUE / Evo

### Main code

- `phase2/prepare_benchmarks.py`
- `phase2/eval_benchmarks.py`
- `phase2/run_hvue_complete_selection_and_full.sh`
- `phase2/run_hvue_pipeline_watchdog.sh`
- `phase2/aggregate_hvue_lora.py`
- `phase2/eval_kmer_baseline.py`

### Canonical checked-in result roots

- `data/phase2/full_benchmarks_lora_optimized_s600/`
- `results/full_benchmark_summary.csv`
- `docs/full_benchmark_results.md`

### Main artifacts to inspect

- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv`
- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.json`
- `results/full_benchmark_summary.csv`
- `docs/full_benchmark_results.md`
- `docs/full_benchmark_artifact_audit.md`

### What these artifacts support

These files capture the completed checked-in HVUE/GUE/ViroBench downstream
benchmark stack. In this repository, HVUE is not packaged as a standalone
`*_qualification` result directory like ProteinGym, PHIStruct, or EvoMIL; its
evidence is embedded in the larger benchmark pipeline and related baseline
audits.

## 2. ProteinGym / ESM2

### Main code

- `phase2/proteingym_esm2_qualification.py`
- `phase2/proteingym_esm2_top20_expansion.py`

### Canonical checked-in result root

- `data/phase2/protein_48h_esm2_qualification/`

### Main artifacts to inspect

- `protein_48h_summary_report.json`
- `protein_48h_summary_report.md`
- `protein_48h_evolutionary_baseline_report.json`
- `protein_48h_evolutionary_baseline_report.md`
- `protein_48h_lora_qualification_evidence.json`
- `protein_48h_candidate_ranking.csv`
- `protein_48h_esm2_pilot_metrics.csv`
- `protein_48h_lora_metrics.csv`
- `protein_48h_artifact_audit.json`

### Key facts recorded there

- workflow status `complete`
- `20` static candidate assays screened
- `3` pilot assays selected
- no preliminarily qualified assay-model pair
- evolutionary baselines saturate the strongest candidates
- the one LoRA-advanced assay remains unstable across three formal seeds

## 3. PHIStruct / SaProt

### Main code

- `phase2/phistruct_qualification.py`
- `phase2/phistruct_failure_audit_evomil_controller.py`

### Canonical checked-in result root

- `data/phase2/phistruct_qualification/`

### Main artifacts to inspect

- `summary_report.json`
- `summary_report.md`
- `baseline_results.csv`
- `plm_results.csv`
- `per_genus_metrics.csv`
- `phistruct_failure_audit/audit_summary.json`
- `phistruct_failure_audit/audit_summary.md`
- `phistruct_failure_audit/paired_bootstrap_summary.json`
- `phistruct_failure_audit/paired_bootstrap_samples.csv`
- `phistruct_failure_audit/per_genus_comparison.csv`
- `phistruct_failure_audit/controller_status.json`

### Key facts recorded there

- final status `PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED`
- SaProt macro-F1 `0.454732`
- BLASTp macro-F1 `0.475180`
- observed delta `-0.020448`
- 95% CI crosses zero

## 4. EvoMIL / ESM-1b

### Main code

- `phase2/evomil_esm1b_qualification.py`
- `phase2/signed_bootstrap.py`

### Canonical checked-in result root

- `data/phase2/evomil_qualification/`

### Main artifacts to inspect

- `evomil_summary_report.json`
- `evomil_summary_report.md`
- `evomil_bootstrap_summary.json`
- `evomil_bootstrap_samples.csv`
- `evomil_kmer_baselines.csv`
- `evomil_kmer_baseline_predictions.csv`
- `evomil_model_results.csv`
- `evomil_model_predictions.csv`
- `evomil_split_audit.json`
- `evomil_preprocessing_audit.json`
- `evomil_reproduction_sanity_report.json`
- `evomil_experiment_registry.json`
- `evomil_controller_status.json`

### Key facts recorded there

- final status `NO_QUALIFYING_HEADROOM`
- strongest baseline `logistic_regression:aa_3mer_tfidf`
- baseline macro-F1 `0.841270`
- best model macro-F1 `0.782246`
- observed delta `-0.059024`
- zero positive seeds out of five

## 5. Shared Qualification Utilities

These files are now shared across multiple qualification controllers:

- `phase2/signed_bootstrap.py`
  standardizes `delta = model - baseline`
- `tests/test_signed_bootstrap.py`
  regression tests for sign consistency and summary invariants
- `tests/test_evomil_esm1b_qualification.py`
  regression test for EvoMIL resume/schema reconciliation

## 6. VPF-PLM In-Progress Work

Implemented code:

- `phase2/vpf_plm_qualification.py`
- `phase2/vpf_plm_compat.py`

Important note:

This workflow was added to continue the final viral qualification search, but
its result directory is still in progress and should not be treated as a
completed benchmark artifact set yet.
