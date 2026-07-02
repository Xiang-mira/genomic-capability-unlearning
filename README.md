# Genomic Capability Localization and Unlearning in Evo-1

This repository studies whether a genomic capability in
[`Evo-1-8k-base`](https://huggingface.co/togethercomputer/evo-1-8k-base) can be:

1. localized to specific model layers with probes and activation patching;
2. selectively weakened with Gradient Difference (GD) or Representation
   Misdirection for Unlearning (RMU); and
3. tested for recovery under full-parameter and LoRA fine-tuning attacks.

The current reported result is the completed 44-task full benchmark comparing
two Gradient Difference (GD) checkpoints and two Representation Misdirection
for Unlearning (RMU) checkpoints on HVUE, GUE, and ViroBench.

## Current status

| Area | Status |
|:---|:---|
| Phase 1 probing and activation patching | Complete |
| GD/RMU condition sweeps | Complete for the checked-in Task 2 grid |
| 44-task full benchmark | Complete for four selected GD/RMU checkpoints |
| Phase 3 recovery attacks | Implemented; current attack setting is preliminary |

Machine-readable progress and results are stored under `data/phase2/`, notably:

- `data/phase2/experiment_audit.json`
- `data/phase2/results/task2_runs.csv`
- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv`
- `results/full_benchmark_summary.csv`

For the completed 44-task benchmark, the same directory also contains the
per-task CSV, progress record, and summary JSON for the base model and all four
checkpoints. The corresponding training configurations and loss histories are
checked in as `meta.json` and `log.json` under
`data/phase2/checkpoints_lora_grid/<checkpoint>/`. Full-run provenance is
available in the dated files under `logs/`; model weights are not included.
The concise result report is in `docs/full_benchmark_results.md`, and its three
figures can be regenerated with `tools/plot_full_benchmark_results.py`.

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
bash phase2/run.sh gd
bash phase2/run.sh rmu
bash phase2/run.sh eval
```

Conditions are:

| Method | Conditions | Layer behavior |
|:---|:---|:---|
| GD | `full`, `localized`, `probe`, `random` | next-token forget maximization plus retain minimization |
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
| `rmu_full_multimetric.json` | Multi-metric RMU stage-1/stage-2 search |

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
