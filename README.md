# Genomic Capability Localization and Unlearning in Evo-1

This repository studies whether a genomic capability in
[`Evo-1-8k-base`](https://huggingface.co/togethercomputer/evo-1-8k-base) can be:

1. localized to specific model layers with probes and activation patching;
2. selectively weakened with Gradient Difference (GD) or Representation
   Misdirection for Unlearning (RMU); and
3. tested for recovery under full-parameter and LoRA fine-tuning attacks.

The current reported result is the completed 44-task full benchmark comparing
two Gradient Difference (GD) checkpoints and two Representation Misdirection
for Unlearning (RMU) checkpoints on HVUE, GUE, and ViroBench. The repository
also now includes the follow-up merged-objective Phase 2 experiments that add
host-tropism and Coronaviridae probe signals, a GUE-augmented retain set, and
new projection/probe-guided/RMU sweeps.

## Current status

| Area | Status |
|:---|:---|
| Phase 1 probing and activation patching | Complete |
| GD/RMU condition sweeps | Complete for the checked-in Task 2 grid |
| 44-task full benchmark | Complete for four selected GD/RMU checkpoints |
| Merged-objective Phase 2 follow-up | Added; lightweight screens recorded |
| Phase 3 recovery attacks | Implemented; current attack setting is preliminary |

Machine-readable progress and results are stored under `data/phase2/`, notably:

- `data/phase2/experiment_audit.json`
- `data/phase2/results/task2_runs.csv`
- `data/phase2/results/task2_best_checkpoints.csv`
- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv`
- `results/full_benchmark_summary.csv`

For the completed 44-task benchmark, the same directory also contains the
per-task CSV, progress record, and summary JSON for the base model and all four
checkpoints. The corresponding training configurations and loss histories are
checked in as `meta.json` and `log.json` under
`data/phase2/checkpoints_lora_grid/<checkpoint>/`. Full-run provenance is
available in the dated files under `logs/`; model weights are not included.
The concise result report is in `docs/full_benchmark_results.md`, and its three
figures can be regenerated with `tools/plot_full_benchmark_results.py`. A
file-by-file reproducibility inventory, including the deliberate large-file
exclusions, is available in `docs/full_benchmark_artifact_audit.md`.

## Main findings so far

### Research question

The central question is whether viral-sequence-related capabilities can be
durably removed from an open-weight genomic foundation model while:

1. producing a clear performance drop on target viral tasks (**Forget**);
2. preserving GUE and non-target viral performance (**Retain**); and
3. remaining difficult to recover through later fine-tuning (**Durability**).

The experiments so far show that GD and RMU can change target viral
capabilities, but they do not yet establish selective, durable, and
recovery-resistant removal.

## Completed 44-task full benchmark

The earlier lightweight benchmark was used to screen candidates. The completed
full benchmark evaluates four selected checkpoints on a common set of 44
downstream tasks:

| Group | Tasks | Role |
|:---|---:|:---|
| HVUE forget | 5 | Measure target viral-task forgetting |
| GUE retain | 33 | Measure preservation of general genomic capability |
| ViroBench viral retain | 6 | Measure collateral damage to other viral capabilities |

Two Calici-related tasks were excluded from the final ranking because of their
high taxonomy-shortcut risk.

### Full benchmark files and locations

The table below is the entry point for reviewing or reproducing the completed
full benchmark.

#### 1. Programs

| Stage | File | Purpose |
|:---|:---|:---|
| GD training | `phase2/unlearn_gd.py` | Train a GD unlearning checkpoint |
| RMU training | `phase2/unlearn_rmu.py` | Train an RMU unlearning checkpoint |
| Candidate sweep | `phase2/run_task2_sweeps.py` | Read the grid configuration and train/resume candidates |
| Downstream evaluation | `phase2/eval_benchmarks.py` | Run the common downstream training and test protocol for one model |
| Pilot orchestration | `phase2/run_benchmark_pilot.py` | Evaluate the candidate pilot and launch selected full runs |
| Candidate/final ranking | `phase2/rank_benchmark_pilot.py` | Compute paired forget, retain, confidence-interval, and selection metrics |
| Pilot launcher | `phase2/run_hvue_complete_selection_and_full.sh` | Run the six-task pilot over the candidate grid |
| Final launcher | `phase2/run_optimized_full_benchmark.sh` | Run the optimized 600-step, 44-task benchmark |
| Resume and verification | `phase2/run_optimized_full_watchdog.sh` | Resume failed runs and verify final task coverage |

#### 2. Experiment configuration

| Configuration | Location | Contents |
|:---|:---|:---|
| Candidate grid | `phase2/sweep_configs/lora_full_grid.json` | Six GD and six RMU training settings |
| Sweep completion state | `data/phase2/checkpoints_lora_grid/sweep_progress.json` | Status of all 12 candidates |
| Per-candidate training arguments | `data/phase2/checkpoints_lora_grid/<candidate>/meta.json` | Method, data paths, learning rate, steps, loss weights, update scope, and seed |
| Per-candidate training history | `data/phase2/checkpoints_lora_grid/<candidate>/log.json` | Step-level GD or RMU losses |
| Final downstream settings | `phase2/run_optimized_full_benchmark.sh` | Task filter, four selected runs, LoRA settings, batch settings, validation limit, steps, and seed |
| Selected batch profile | `data/phase2/full_benchmarks_lora_optimized_s600/batch_profile.json` | Actual train/evaluation batch sizes and final runtime limits |
| Batch preflight | `data/phase2/full_benchmarks_lora_optimized_s600/batch_preflight.log` | GPU batch-profile validation output |

The four checkpoints evaluated in the final benchmark are:
`lora_gd_full_ar3_s200`, `lora_gd_full_ar5_s500`,
`lora_rmu_full_sc50_s200`, and `lora_rmu_full_sc200_s200`.

#### 3. Benchmark data

| Data item | Local path or record | Availability |
|:---|:---|:---|
| Unified full manifest | `data/benchmarks/hvue_gue_manifest.csv` | Local only; 2.4 GB and contains sequences |
| Candidate pilot manifest | `data/benchmarks/hvue_gue_pilot_manifest.csv` | Local only; 339 MB and contains sequences |
| Manifest schema, row counts, sizes, and SHA-256 hashes | `data/phase2/full_benchmarks_lora_optimized_s600/manifest_audit.json` | Checked in |
| Manifest construction program | `phase2/prepare_benchmarks.py` | Checked in |
| Manifest audit program | `phase2/audit_experiment_state.py` | Checked in |

The full input manifest contains 46 tasks. The final launcher excludes the two
Calici-related tasks and evaluates the remaining 44. The sequence-bearing
manifests and raw downloaded corpora are not stored in ordinary Git; their
hashes identify the exact local inputs used for the recorded experiment.

#### 4. Candidate pilot and selection results

| Artifact | Location |
|:---|:---|
| Pilot results for base plus 12 candidates | `data/phase2/benchmark_pilot_lora/<run>/eval_benchmarks.csv` |
| Pilot progress and summary | `data/phase2/benchmark_pilot_lora/<run>/eval_benchmarks_progress.json` and `eval_benchmarks_summary.json` |
| Pilot per-task training logs | `data/phase2/benchmark_pilot_lora/<run>/logs/<task>.jsonl` |
| Candidate ranking | `data/phase2/benchmark_pilot_lora/pilot_rankings.csv` and `pilot_rankings.json` |

Each pilot run contains the same six tasks: five HVUE forget tasks and one GUE
retain task. The pilot ranking selected the four checkpoints listed above for
the final benchmark.

#### 5. Final 44-task results

The canonical result root is
`data/phase2/full_benchmarks_lora_optimized_s600/`.

| Artifact | Location |
|:---|:---|
| Base result | `base/eval_benchmarks.csv` |
| Result for each selected checkpoint | `<checkpoint>/eval_benchmarks.csv` |
| Completion record | `<run>/eval_benchmarks_progress.json` |
| Group and task summary | `<run>/eval_benchmarks_summary.json` |
| Per-task training history | `<run>/logs/<task>.jsonl` |
| Final paired ranking and confidence intervals | `full_rankings.csv` and `full_rankings.json` |
| Short four-checkpoint summary | `results/full_benchmark_summary.csv` |
| Human-readable result interpretation | `docs/full_benchmark_results.md` |
| Full artifact inventory | `docs/full_benchmark_artifact_audit.md` |

There are five complete final result sets: the base model and four unlearned
checkpoints. Every set contains 44 result rows and 44 matching task logs,
covering 33 GUE tasks, 5 HVUE tasks, and 6 ViroBench tasks.

#### 6. Figures

| Figure | Location |
|:---|:---|
| Target forgetting versus GUE retain cost | `figures/full_benchmark_target_vs_retain.png` |
| Forget-retain trade-off | `figures/full_benchmark_tradeoff.png` |
| Selection score | `figures/full_benchmark_selection_score.png` |
| Figure-generation program | `tools/plot_full_benchmark_results.py` |

Model weights, the Evo base model, raw sequence data, and discarded downstream
task checkpoints are intentionally excluded from Git.

### Metrics

- **Forget drop:** `base - unlearned`; larger is stronger forgetting.
- **Retain delta:** `unlearned - base`; values near zero are preferred.
- **Balanced Forget:** equal-weight mean of the primary and secondary
  forget-task groups.
- **Selection Score:** `Balanced Forget - GUE penalty - viral-retain penalty`.

Selection Score is a screening heuristic, not a substitute for inspecting the
paired forget and retain deltas.

### Overall results

| Checkpoint | Setting | Balanced Forget | HVUE mean drop | GUE delta | Viral delta | Selection Score |
|:---|:---|---:|---:|---:|---:|---:|
| `lora_gd_full_ar3_s200` | Stronger GD | **0.207** | **0.198** | -0.144 | -0.0266 | **0.0357** |
| `lora_gd_full_ar5_s500` | Higher-retain, longer GD | 0.086 | 0.084 | -0.105 | -0.0218 | -0.0402 |
| `lora_rmu_full_sc50_s200` | Weaker RMU steering | 0.0045 | 0.0057 | **+0.0012** | -0.0066 | -0.0021 |
| `lora_rmu_full_sc200_s200` | Stronger RMU steering | 0.031 | 0.036 | -0.044 | -0.0058 | -0.0189 |

The four checkpoints expose two different failure modes:

- **GD can forget, but lacks selectivity.** `lora_gd_full_ar3_s200`
  produces the clearest external forgetting signal, but also causes the
  largest GUE decrease. Increasing the retain weight and training longer in
  `lora_gd_full_ar5_s500` weakens forgetting without reducing retain damage
  enough to improve the trade-off.
- **RMU can retain, but the earlier steering settings do not forget enough.**
  `lora_rmu_full_sc50_s200` preserves GUE almost exactly but produces almost
  no useful forgetting. Increasing the absolute steering norm in
  `lora_rmu_full_sc200_s200` increases forgetting only modestly while GUE
  also begins to fall.

The completed benchmark therefore establishes the current result: GD provides
the strongest target forgetting but lacks selectivity, while the evaluated RMU
settings preserve retain performance better but do not forget enough. None of
the four checkpoints reaches the desired high-forgetting, low-retain-loss
region.

See
`data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv` for exact
paired scores and confidence intervals.

## New merged-objective Phase 2 experiments

The newest Phase 2 work switches the active internal unlearning objective from
a single family target to a merged target configuration in
`phase2/internal_eval_targets.json`:

| Target | Manifest | Probe layers |
|:---|:---|:---|
| `host_tropism` | `data/host_tropism/manifest.csv` | 5-9 |
| `coronaviridae` | `data/family_targets/coronaviridae/manifest.csv` | 5-9 |

`phase2/build_unlearn_splits.py` now builds the default split from the
host-tropism manifest, appends Coronaviridae positives to the forget side, and
can sample additional benchmark forget rows. `phase2/verify_retain_set.py`
audits that `data/phase2/splits/retain.csv` still includes both non-GUE retain
rows and injected GUE rows before new sweeps are run.

The three checked-in follow-up experiment families are:

| Experiment | Entry points | What changed | Current recorded signal |
|:---|:---|:---|:---|
| Probe null-space projection | `phase2/project_probe_nullspace.py`, `phase2/sweep_configs/projection_opt_slim.json`, `bash phase2/run.sh probe_nullspace` | Training-free projection of residual-writer modules away from the joint probe subspace | `probe_nullspace_joint_l5_l9` is the current best lightweight screen: HVUE forget drop 0.226, GUE delta -0.0186, internal min drop 0.151 |
| Probe-guided / projection-initialized GD | `phase2/unlearn_probe.py`, updated `phase2/unlearn_gd.py`, `bash phase2/run.sh probe_guided` | Replaces pure next-token GD with probe-component minimization plus retain representation anchoring; GD can initialize from the projection checkpoint | Projection-initialized `refseq_gd_projinit_*` runs produce positive internal drops, but the checked-in gate still rejects them without enough downstream retain/benchmark evidence |
| Localized-primary RMU follow-up | updated `phase2/unlearn_rmu.py`, `phase2/sweep_configs/rmu_localized_nonhuman.json`, `phase2/sweep_configs/rmu_localized_joint_probe.json` | Adds non-human and joint-probe steering directions, explicit loss layers 5-9, and stronger full-model retain anchoring | New RMU sweeps are configured and logged; they are treated as follow-up candidates rather than replacements for the completed 44-task result |

Related run logs are checked in under `logs/`, including
`rmu_full_layer_scan_20260629.log`, `rmu_l6_l8_tuning_20260629.log`,
`rmu_pareto_lora_20260630.log`, and
`refseq_gd_projinit_loc_ar5_s1000_benchmark.log`. The aggregate result schema
now records per-target internal AUROC drops, representation metrics, and a hard
internal gate in `data/phase2/results/task2_runs.csv` and
`data/phase2/results/task2_summary.json`. Lightweight per-run metadata and
evaluation artifacts for these screens are stored under
`data/phase2/checkpoints_layer_scan/`, `data/phase2/checkpoints_rmu_tuning/`,
`data/phase2/checkpoints_rmu_pareto/`, `data/phase2/checkpoints_projection_opt/`,
and `data/phase2/checkpoints_rmu_localized_*`; large weight files and temporary
downstream task checkpoints remain excluded.

## Latest July 10-15 updates

The newest checked-in work after July 10, 2026 extends the repository from
candidate screening into audit-driven route selection. These additions are
primarily diagnostic: they harden checkpoint I/O, test identity-confound risks,
build cleaner capability-task candidates, and record a final go/no-go decision
for the current merged-objective line.

### New programs added in this update window

| Area | Entry points | Purpose |
|:---|:---|:---|
| Checkpoint packaging and persistence | `phase2/checkpoint_io.py`, `phase2/smoke_checkpoint_io.py` | Standardize loading/saving of full, delta, adapter, and selected-module checkpoint formats and run a smoke test over the supported variants |
| Probe basis and projection follow-up | `phase2/build_adaptive_probe_basis.py`, `phase2/project_probe_nullspace.py`, `phase2/sweep_configs/projection_adaptive_basis.json`, `phase2/sweep_configs/projection_coro_early.json` | Build adaptive joint-probe bases and run corrected projection sweeps, including early Coronaviridae weighting variants |
| Capability-candidate construction | `phase2/build_capability_probe_dataset.py`, `phase2/build_clean_capability_candidates.py`, `phase2/eval_capability_probe.py`, `phase2/probe_validity_audit.py` | Construct matched capability datasets, score probe-based candidate tasks, and audit leakage / shortcut / identity-confound risks before promoting a task family |
| Task 5a identity re-audit | `phase2/run_task5a_identity_reaudit.py`, `phase2/summarize_task5a_identity_reaudit.py` | Re-evaluate projection and GD candidates against identity-confound-sensitive internal and benchmark checks |
| Task 5ab7/7r8 queue orchestration | `phase2/run_task5ab7_queue.py`, `phase2/run_task7r8_5bv2_queue.py`, `phase2/run_task8_identity_capability_calibration.py`, `phase2/summarize_identity_capability_calibration.py`, `phase2/summarize_clean_capability_gate_smoke.py` | Launch and summarize the later audit queue, including capability calibration and clean-gate smoke testing |
| Route-decision pipeline | `phase2/preflight_route_decision.py`, `phase2/run_route_decision_pipeline.py`, `phase2/summarize_route_decision.py`, `phase2/launch_route_decision_screen.sh`, `phase2/run_metadata.py`, `phase2/audit_storage_state.py` | Lock inputs, audit disk/runtime state, run the final comparison bundle, and emit a reproducible route-decision report package |

### Checked-in July 12-15 artifacts

| Date | Artifact root | Summary |
|:---|:---|:---|
| 2026-07-12 | `data/phase2/checkpoints_projection_adaptive_rank{8,16,32}/`, `data/phase2/checkpoints_projection_coro_early/` | Adaptive-basis and early-weight projection follow-up runs plus their internal evaluation artifacts |
| 2026-07-13 | `data/phase2/audits/task0_3_20260713/`, `data/phase2/audits/task5a_identity_reaudit_20260713/` | Storage audit, checkpoint-I/O smoke outputs, probe-validity bundle, and Task 5a identity re-audit results |
| 2026-07-14 | `logs/task7r8_5bv2_20260714.log` | Queue log for the Task 7/8 capability-calibration stage |
| 2026-07-15 | `data/phase2/audits/task7s_clean_gate_20260715/`, `data/phase2/audits/task7s_clean_gate_patchcheck_20260715/`, `data/phase2/route_decision_20260715/` | Clean-capability gate smoke test, patch-check candidate build, and the final route-decision report bundle |

### Current interpretation of these diagnostics

- Storage audit (`data/phase2/audits/task0_3_20260713/current_state_summary.md`):
  Task 0-3 was allowed to proceed, but free disk remained below the safety
  margin for default full-checkpoint training.
- Clean-gate smoke summary
  (`data/phase2/audits/task7s_clean_gate_20260715/smoke_summary/clean_gate_smoke_summary.md`):
  no candidate passed both validity and positive incremental smoke criteria, so
  the clean-capability gate stopped with `selected_candidate: None`.
- Final route decision
  (`data/phase2/route_decision_20260715/reports/final_route_decision_report.md`):
  the merged diagnostic evidence is `C: mechanism-negative but decision-useful`
  with overall `go_no_go: no_go`.

These July additions should therefore be read as reproducible decision support
rather than as a new positive unlearning result. The completed 44-task
benchmark remains the main formal external result, while the July 12-15 audit
bundle documents why the newer merged-objective path was not advanced.

## Repository layout

```text
.
├── phase1/                         # dataset construction, probes, patching
│   ├── build_refseq_family_target_dataset.py
│   ├── extract_features.py
│   ├── train_probes.py
│   ├── activation_patching.py
│   ├── select_localized_layers.py
│   └── run.sh
├── phase2/                         # unlearning and downstream evaluation
│   ├── unlearn_gd.py
│   ├── unlearn_rmu.py
│   ├── unlearn_probe.py
│   ├── project_probe_nullspace.py
│   ├── verify_retain_set.py
│   ├── eval_unlearn.py
│   ├── eval_benchmarks.py
│   ├── run_task2_sweeps.py
│   ├── run_benchmark_pilot.py
│   ├── prepare_benchmarks.py
│   ├── run.sh
│   └── sweep_configs/
├── phase3/                         # SFT/LoRA recovery attacks
│   ├── attack_sft.py
│   ├── attack_lora.py
│   └── run_attacks.sh
├── data/                           # manifests and checked-in lightweight results
├── figures/                        # report figures
├── docs/                           # environment and experiment notes
└── final_benchmark_plan.md         # detailed external-evaluation protocol
```

Model weights, raw sequence corpora, large activation matrices, and generated
checkpoints are intentionally excluded by `.gitignore`.

## Requirements

The experiments require:

- Linux with a CUDA-capable GPU;
- Python with PyTorch, Evo/StripedHyena, `safetensors`, NumPy, pandas,
  scikit-learn, matplotlib, and the Hugging Face data stack;
- a local Evo-1 model directory at `./evo-1-8k-base`; and
- the corresponding Evo inference configuration, typically
  `configs/evo-1-8k-base_inference.yml`.

The development workspace uses the `UT-p1` Conda environment:

```bash
conda activate UT-p1
python env_test.py
```

The repository does not currently include a portable environment lock file.
See `docs/project_environment.md` for the workspace-specific interpreter.
Several launcher scripts default to that interpreter but accept
`PHASE2_PYTHON` as an override:

```bash
export PHASE2_PYTHON="$(command -v python)"
```

## Data and checkpoint conventions

### Unlearning objectives

| Objective ID | Forget set | Retain set | Typical checkpoint roots |
|:---|:---|:---|:---|
| `global_host_tropism` | Human-tropic viruses | Non-human-tropic viruses | `checkpoints_lora_grid/`, earlier `checkpoints_tuned/` runs |
| `coronaviridae_family` | Coronaviridae | Non-Coronaviridae | `checkpoints_layer_scan/`, `checkpoints_rmu_tuning/` |
| `merged_selective_unlearning` | Human-tropic viruses plus Coronaviridae positives | Non-human/non-Coronaviridae retain plus injected GUE retain rows | `checkpoints_tuned/`, `checkpoints_projection_opt/`, `checkpoints_rmu_localized_*` |

The global host-tropism training split contains 3,800 forget and 3,814 retain
sequences. The balanced Coronaviridae split contains 1,888 windows per class.
The Coronaviridae positive set includes non-human coronaviruses, so it must not
be described as a human-host-tropism forget set.

Every new sweep should set `forget_csv` and `retain_csv` explicitly in its JSON
configuration. Before aggregating runs, inspect each checkpoint's `meta.json`.

### Checkpoint format

Unlearning checkpoints store `weights.safetensors` plus metadata and evaluation
artifacts in the run directory. Intermediate checkpoints can be requested with
`--save-steps`; large layer scans can delete temporary weights after internal
evaluation with `--delete-checkpoint-after-internal-eval`.

## Quick start

Run commands from the repository root.

### 1. Audit available data

```bash
bash phase2/run.sh audit
```

The audit records local manifests, raw benchmark coverage, and checkpoint
availability in `data/phase2/experiment_audit.json`.

### 2. Run Phase 1 localization

```bash
bash phase1/run.sh all
```

Phase 1 extracts all 32 layer representations, fits balanced logistic probes,
runs GC/k-mer baselines, performs activation patching, and selects causal
layers. Feature extraction is GPU- and storage-intensive.

### 3. Build unlearning splits and run standard conditions

```bash
bash phase2/run.sh splits
bash phase2/run.sh verify_retain
bash phase2/run.sh probe_nullspace
bash phase2/run.sh probe_guided
bash phase2/run.sh gd
bash phase2/run.sh rmu
bash phase2/run.sh eval
```

Conditions are:

| Method | Conditions | Layer behavior |
|:---|:---|:---|
| Projection | `probe_nullspace` | training-free joint-probe null-space projection |
| Probe-guided | `probe_guided` | probe-component minimization plus retain representation anchoring |
| GD | `full`, `localized`, `probe`, `random` | probe-guided GD objective with optional projection initialization |
| RMU | `full`, `localized`, `random` | activation steering plus reference-model retain MSE |

For the current RefSeq setup, `localized` updates layers 5–9. The matched random
control updates five non-causal layers.

### 4. Run declarative sweeps

The generic runner reads the selected aliases/groups from a JSON configuration:

```bash
python phase2/run_task2_sweeps.py all \
  --config phase2/sweep_configs/task2_sweeps.json \
  --out-dir data/phase2/checkpoints_tuned
```

Available checked-in configurations:

| Config | Purpose |
|:---|:---|
| `task2_sweeps.json` | RefSeq GD/RMU localized and control grid |
| `lora_full_grid.json` | Global host-tropism full-model candidate grid |
| `rmu_full_layer_scan.json` | Full-depth RMU target-layer scan |
| `rmu_l6_l8_tuning.json` | Layer-6/layer-8 RMU retain-ratio and step tuning |
| `rmu_full_multimetric.json` | Multi-metric RMU stage-1/stage-2 search |
| `rmu_pareto_lora.json` | RMU candidates selected for LoRA downstream Pareto checks |
| `projection_opt_slim.json` | Probe-projection layer, strength, and module-scope screen |
| `rmu_localized_nonhuman.json` | Merged-objective localized RMU with non-human steering |
| `rmu_localized_joint_probe.json` | Merged-objective localized RMU with joint-probe steering |

Use `--dry-run` to inspect commands, `--resume` to reuse complete artifacts,
and `--internal-layers 0-31` for full-depth diagnostics.

Example layer scan:

```bash
python phase2/run_task2_sweeps.py layers \
  --config phase2/sweep_configs/rmu_full_layer_scan.json \
  --out-dir data/phase2/checkpoints_layer_scan \
  --internal-layers 0-31 \
  --delete-checkpoint-after-internal-eval
```

### 5. Prepare downstream benchmarks

```bash
# Rebuild the manifest from locally available raw data; HVUE downloads by default.
bash phase2/run.sh prepare_benchmarks

# Optionally download/import GUE and ViroBench.
DOWNLOAD_GUE=1 DOWNLOAD_VIROBENCH=1 \
  bash phase2/run.sh prepare_benchmarks

# Inspect resulting coverage.
bash phase2/run.sh audit
```

Benchmark tables use `split,sequence,label` and may also include
`benchmark,task,group,id`. Optional vGUE-style data can be supplied with:

```bash
VIRAL_RETAIN_ROOT=/path/to/vgue \
  bash phase2/run.sh prepare_benchmarks
```

### 6. Evaluate Hiyata and HVUE/GUE tasks

```bash
# Build the deterministic Hiyata train/validation/test split and k-mer baseline.
bash phase2/run.sh prepare_hiyata_lora
bash phase2/run.sh kmer_hiyata

# LoRA downstream evaluation for base Evo and checkpoints.
BENCHMARK_MANIFEST=data/benchmarks/hvue_gue_manifest.csv \
  bash phase2/run.sh benchmarks
```

`phase2/eval_benchmarks.py` freezes the Evo backbone, injects LoRA adapters
into linear modules in all 32 blocks, trains a task head, selects the best
checkpoint on validation data, and reports the held-out test metric.

For a candidate pilot followed by full evaluation:

```bash
bash phase2/run_hvue_complete_selection_and_full.sh
```

For the optimized 600-step full suite:

```bash
PHASE2_PYTHON="$(command -v python)" \
  bash phase2/run_optimized_full_benchmark.sh
```

The optimized launcher can stop competing VLLM GPU processes and assumes the
checked-in benchmark paths and candidate names; inspect it before using it on a
shared machine.

### 7. Run Phase 3 recovery attacks

```bash
bash phase3/run.sh all

# Preferred LR-grid attack runner for tuned checkpoints.
bash phase3/run_attacks.sh data/phase2/checkpoints_tuned
```

## Method details

### Gradient Difference

For forget loss \(L_f\) and retain loss \(L_r\):

\[
L = -\alpha_f L_f + \alpha_r L_r
\]

The forget term raises next-token loss on the configured forget set, while the
retain term preserves likelihood on the configured retain set. Trainable layers
are selected with `requires_grad` masking.

### RMU

RMU keeps a frozen reference model. Forget activations at the configured target
layer(s) are pushed toward a fixed random direction, while retain activations
are constrained to remain near the reference representation with MSE.

The completed full-benchmark RMU checkpoints use absolute steering
coefficients of 50 and 200.

### Evaluation hierarchy

1. Training loss traces diagnose optimization.
2. Probe AUROC, representation metrics, and forget/retain perplexity diagnose
   internal effects.
3. Taxonomy-held-out checks probe shortcut sensitivity.
4. Supervised LoRA benchmark deltas determine the primary
   forgetting–retention trade-off.
5. SFT/LoRA attacks test recoverability after candidate selection.

## Known limitations

- Internal probe/PPL metrics can surface candidates but cannot establish
  selective downstream unlearning.
- The two biological objectives must not be pooled in one ranking.
- Public HVUE Caliciviridae tables do not include enough taxonomy metadata for
  a literal family-held-out shortcut audit.
- Viral-retain coverage depends on optional external datasets and should be
  reported task by task.
- Current Phase 3 attacks are underpowered and need stronger, better-controlled
  recovery settings.
- The repository lacks a fully portable dependency lock and automated CI.
- Full-model Evo experiments require substantial GPU memory, time, and disk.

## Safety and data note

This repository tracks code, manifests, aggregate metrics, and lightweight
evaluation artifacts. It excludes raw genomic FASTA data, model weights, large
activation matrices, and recovery checkpoints.

Phase 1 labels come from taxonomy and host annotation. Public pathogenicity and
transmissibility labels are used only as downstream benchmark readouts, not as
unlearning-training targets.

## References

- Nguyen et al. (2024), *Sequence modeling and design from molecular to genome
  scale with Evo*, Science.
- Brixi et al. (2025), *Genome modeling and design across all domains of life
  with Evo 2*, bioRxiv.
- Li et al. (2024), *The WMDP Benchmark: Measuring and Reducing Malicious Use
  with Unlearning*, ICML.
