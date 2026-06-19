# Evo-1 Host-Tropism Representation Localization and Targeted Unlearning

This repository contains a three-phase pipeline for **capability localization and targeted machine unlearning** in the Evo-1-8k-base genomic foundation model (StripedHyena, 32 layers, ~7B parameters).

The research question is whether human-virus-relevant genomic capabilities can be (1) localized to specific layers via probing and causal analysis, (2) selectively removed via targeted unlearning while preserving non-human viral biology and general genomics competence, and (3) tested for robustness against fine-tuning recovery attacks.

Phase 1 uses taxonomy and host annotation only for localization. Phase 2 unlearning is evaluated primarily on external HVUE + GUE benchmarks rather than only on the training-data forget/retain perplexity proxy. A local audit of the current workspace is saved to `data/phase2/experiment_audit.json`; at present the requested vGUE retain tasks are not yet present locally, and the public HVUE Calici CSVs do not expose the taxonomy metadata needed for a literal family-held-out shortcut audit.

---

## Current Review Status

This section summarizes what the current code and checked-in result files can answer, and what still needs an additional run.

### What is already implemented

- **Baseline and controls.** The Phase 2 scripts implement `full`, `localized`, `probe`, and `random` conditions. The first Phase 1 host-tropism localization selected layers 3-9; the second RefSeq Coronaviridae Phase 1 rerun selected `[5, 6, 7, 8, 9]` in `data/family_targets/coronaviridae/localized_layers.json`, with primary target layer 6. Current RefSeq `localized` runs update layers 5-9. Current RefSeq `random` controls update the same number of non-causal layers sampled from 11-30; with seed 42 these are `[11, 14, 18, 19, 27]`. The older 3-9 experiments used 7 random layers: `[11, 13, 14, 18, 19, 22, 27]`.
- **Internal diagnostics.** `phase2/eval_unlearn.py` reports the legacy internal host-tropism probe AUROC and forget/retain validation perplexity. These results are aggregated in `data/phase2/results/task2_runs.csv` and `data/phase2/results/task2_best_checkpoints.csv`.
- **External benchmark protocol.** `phase2/eval_benchmarks.py` now evaluates downstream tasks with supervised LoRA finetuning on a frozen Evo backbone. Hiyata Host Tropism is used for lightweight task adaptation and k-mer comparison, while HVUE Host Tropism remains a non-overlapping held-out benchmark. The old frozen-representation probe evaluator is preserved as `phase2/eval_benchmarks_probe_legacy.py` for reference only.
- **Legacy candidate cross-checks.** External evaluations under `data/phase2/benchmark_pilot_lean/` and `data/phase2/final_fast_eval/` were run on checkpoint candidates that were originally chosen from internal probe/PPL diagnostics. They are useful for reference, but they should not be treated as the post-LoRA primary selection.
- **Training losses.** Each unlearning run writes a `log.json` with training losses every `--log-every` steps. For example, tuned runs under `data/phase2/checkpoints_tuned/<run>/log.json` contain the optimization loss trace.

### Current limitations

- The internal probe AUROC and forget/retain PPL are **diagnostics**, not the final selective-unlearning claim. They are useful for debugging whether a method affects the host-tropism representation and whether the language-model objective is damaged, but final method selection should rely on external HVUE/GUE deltas.
- The current training scripts save the final checkpoint for each run. They do **not** yet automatically save and evaluate intermediate checkpoints at fixed milestones such as 2k, 5k, 10k, or 25k steps. Therefore, the repository currently has final-checkpoint PPL/AUROC for multiple step settings, plus training-loss traces, but not a complete validation PPL/AUROC time series over training.
- vGUE retain tasks are not yet present locally as task-ready `sequence,label,split` tables. The current retain-side external benchmark is GUE, with viral-retain integration left as a data-engineering follow-up.

---

## Results Summary

### Phase 1 — Layer-wise Probe AUROC and Activation Patching

![Activation patching analysis](figures/patching_analysis.png)

**Figure:** (a) Layer-wise probe AUROC. (b) Activation patching causal effect |Δprob| per layer. (c) PPL delta per layer (flat — single-layer patching is compensated downstream).

- Layers 0–10 achieve probe AUROC 0.975–0.997, far above the k-mer baseline (0.851)
- First Phase 1 activation patching identified layers 3-9 as the original causal target region (layer 6: |Δprob| = 0.355, layer 8: 0.276)
- The second RefSeq Coronaviridae Phase 1 rerun selected the current sparse localized set `[5, 6, 7, 8, 9]`; layer 6 remains the primary target layer
- Layers 0–2 are linearly decodable but have near-zero patching effect — probe salience ≠ intervention salience
- PPL delta is flat across all layers (std < 0.0001), confirming that unlearning must target multiple layers simultaneously

### Phase 2 — Targeted Unlearning

![Phase 2 unlearning results](figures/phase2_results.png)

**Figure:** (a) Probe AUROC by layer after unlearning. (b) Legacy forget/retain PPL diagnostic (log scale; no longer the main evaluation claim).

Primary selective-unlearning evaluation now uses external benchmarks:

| Evaluation axis | Benchmark tasks | Desired outcome |
|:---|:---|:---|
| Forget: human-virus-relevant capability | HVUE human host tropism; HVUE human-virus pathogenicity; HVUE transmissibility for Coronaviridae, Orthomyxoviridae, and Caliciviridae | Large post-unlearning drop relative to the base model |
| Retain: general genomics | GUE promoter, splice-site, TF-binding, and chromatin-accessibility tasks | Minimal drop relative to the base model |

The original host-tropism probe AUROC and forget/retain perplexity table is kept as a legacy internal diagnostic for comparing unlearning methods before running the external benchmarks. These rows come from the first Phase 1 localization, where `localized` meant layers 3-9:

| Method | Updated layers | Probe AUROC L3–9 | Δ AUROC | Forget PPL diagnostic | Retain PPL diagnostic |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | — | 0.844 | — | ~4.2 | ~4.2 |
| GD full | all 32 | 0.524 | −0.320 | 31.2 | 37.9 |
| GD localized | 3–9 (patching) | 0.555 | −0.289 | 20.4 | 15.7 |
| GD probe | 0–10 (probe curve) | 0.524 | −0.320 | 137.5 | 63.3 |
| GD random | 7 random (11–30) | 0.847 | +0.003 | 4.2 | 4.2 |
| RMU full | all 32 | 0.700 | −0.144 | 4.5 | 4.48 |
| RMU localized | 3–9 | 0.765 | −0.079 | 4.4 | 4.42 |
| RMU random | 7 random (11–30) | 0.847 | +0.003 | 4.2 | 4.3 |

The completed RefSeq sweep from `bash phase2/run_sweep.sh all` uses the second Phase 1 localized set `[5, 6, 7, 8, 9]`. The table below is a legacy internal diagnostic table: it reports AUROC recomputed over those current L5-9 layers, not the legacy aggregate column name `internal_auroc_3_9` used by older summary files. These rows can guide debugging, but the strong rows here were identified with probe/PPL criteria and must be re-ranked under the supervised LoRA benchmark protocol before being used for primary claims.

| Current RefSeq run | Method / condition | Updated layers | Probe AUROC L5-9 | Drop vs random | Forget PPL | Retain PPL | Interpretation |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `refseq_gd_full_ar5_s200` | GD full | all 32 | 0.547 | 0.447 | 4.21 | 4.13 | Strong legacy probe/PPL diagnostic; needs LoRA benchmark re-ranking |
| `refseq_rmu_full_sc50_s200` | RMU full | all 32 | 0.604 | 0.390 | 4.62 | 4.43 | Strong legacy probe/PPL diagnostic; needs LoRA benchmark re-ranking |
| `refseq_gd_loc_af1_ar3_s200` | GD localized | 5-9 | 0.897 | 0.097 | 4.83 | 4.12 | Best stable localized GD trade-off |
| `refseq_gd_loc_af1_ar5_s200` | GD localized | 5-9 | 0.951 | 0.043 | 4.50 | 3.97 | More retain-preserving, weaker forgetting |
| `refseq_gd_loc_ar5_lr2e-5_s200` | GD localized | 5-9 | 0.913 | 0.081 | 14.89 | 5.46 | Some forgetting, but retain/forget PPL damage |
| `refseq_gd_loc_ar5_s500` | GD localized | 5-9 | 0.859 | 0.135 | 5.1e6 | 30.54 | Longer GD lowers AUROC but becomes unstable |
| `refseq_gd_loc_ar5_s1000` | GD localized | 5-9 | 0.809 | 0.185 | 5.0e16 | 1232.17 | Catastrophic LM damage despite lower probe AUROC |
| `refseq_rmu_loc_sc50_s1000` | RMU localized | 5-9 | 0.988 | 0.006 | 4.14 | 4.25 | Localized RMU preserves LM behavior but barely reduces probe signal |
| `refseq_gd_random_ar5_s1000` | GD random | 11,14,18,19,27 | 0.994 | 0.000 | 3.89 | 3.95 | Negative control: no forgetting |
| `refseq_rmu_random_sc50_s1000` | RMU random | 11,14,18,19,27 | 0.994 | 0.000 | 4.24 | 4.37 | Negative control: no forgetting |

### Phase 3 — Recovery Attacks

![Phase 3 recovery attack results](figures/phase3_results.png)

**Figure:** (a) Full method × attack matrix heatmap. (b) Tuned comparison: GD localized (α_retain=5.0) vs RMU full under SFT and LoRA attacks.

Controlled comparison after hyperparameter tuning (both methods with retain diagnostic PPL ≈ baseline):

| Method | After unlearning diagnostic AUROC | Retain diagnostic PPL | After SFT | SFT Δ | After LoRA | LoRA Δ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GD localized (α_retain=5.0) | 0.624 | 3.88 | 0.448 | −0.175 | 0.656 | +0.033 |
| RMU full | 0.700 | 4.48 | 0.651 | −0.049 | 0.665 | −0.036 |

---

## Repository Structure

```text
project1/
├── phase1/
│   ├── utils.py                    # Shared utilities: model loading, manifest I/O, feature writing
│   ├── extract_features.py         # Layer-wise mean-pooled activation extraction
│   ├── train_probes.py             # Layer-wise logistic probe training with C-grid search
│   ├── baseline_gc_1gram.py        # GC + k-mer sequence-level baselines
│   ├── activation_patching.py      # Causal layer identification via hidden-state patching
│   ├── plot_metrics.py             # Probe AUROC visualization
│   ├── plot_patching.py            # Patching analysis visualization
│   └── run.sh                      # Phase 1 end-to-end script
├── phase2/
│   ├── utils.py                    # Shared constants (LOCALIZED_LAYERS, PROBE_LAYERS), loss functions
│   ├── build_unlearn_splits.py     # Construct forget / retain / eval splits
│   ├── unlearn_gd.py               # Gradient Difference unlearning (4 conditions)
│   ├── unlearn_rmu.py              # RMU representation misdirection (3 conditions)
│   ├── eval_unlearn.py             # Internal post-unlearning probe AUROC and PPL diagnostics
│   ├── eval_benchmarks.py          # Primary LoRA finetuning benchmark evaluation
│   ├── eval_benchmarks_probe_legacy.py # Legacy frozen-representation probe benchmark
│   ├── prepare_hiyata_lora_manifest.py # Hiyata Host Tropism canonical LoRA split
│   ├── eval_kmer_baseline.py       # k-mer baseline on benchmark-style manifests
│   ├── plot_results.py             # Phase 2 results visualization
│   └── run.sh                      # Phase 2 end-to-end script
├── phase3/
│   ├── utils.py                    # Shared utilities: checkpoint loading, feature extraction, probe scoring
│   ├── attack_sft.py               # Full-parameter SFT recovery attack
│   ├── attack_lora.py              # LoRA adapter recovery attack
│   ├── plot_results.py             # Phase 3 method × attack matrix visualization
│   └── run.sh                      # Phase 3 end-to-end script
└── data/
    ├── host_tropism/               # Phase 1 dataset, probes, patching results
    ├── phase2/                     # Unlearning splits and checkpoints
    └── phase3/                     # Attack results
```

---

## Dataset

| Property | Value |
|:---|:---|
| Source | NCBI Virus (host-tropism subset) |
| Total sequences | 9,521 |
| Label definition | human-tropic = 1, non-human-tropic = 0 |
| Label basis | Taxonomy and host annotation only; no virulence labels |
| Train / Val / Test | 7,614 / 923 / 984 |
| Positive rate | 49.9% / 45.1% / 46.0% |
| Max sequence length | 512 bp (unlearning) / 2,048 bp (feature extraction) |

---

## Phase 1 — Probing and Causal Analysis

### Feature Extraction

For each sequence and each of the 32 Evo-1 blocks, the hidden state is captured via a forward hook. The `next_norm` representation is used: each layer's output is passed through the subsequent block's `pre_norm` (or the final `model.norm` for the last layer) before mask-aware mean pooling:

$$h_l = \frac{\sum_{t=1}^{T} m_t \cdot H_{l,t}}{\sum_{t=1}^{T} m_t}$$

where $m_t \in \{0,1\}$ is the padding mask. This produces one 4096-dimensional vector per sequence per layer.

### Probe Training

For each layer, a balanced L2-regularized logistic regression is trained on the train split with C-grid search over {0.001, 0.01, 0.1, 1.0}, selecting the best C by validation AUROC. Probe weights (coef, intercept, scaler parameters) are saved as `.npz` files for reuse in Phase 2 and Phase 3 evaluation.

### Activation Patching

For each layer $l$, the hidden state of a target sequence is replaced with the corresponding activation from a paired source sequence. Two readouts are measured:

- **|Δprob|**: change in probe prediction probability (causal effect on representation)
- **Δloss**: change in final-output perplexity (causal effect on model output)

The PPL delta is flat across all layers (mean Δloss ≈ 0.048, std < 0.0001), indicating that single-layer patching is compensated by downstream layers. The probe-level |Δprob| is therefore used as the causal localization signal.

**Key result:** There are two Phase 1 localization snapshots in this repository. The first host-tropism run identified layers 3-9 as the broader causal target region. The second RefSeq Coronaviridae rerun selected `[5, 6, 7, 8, 9]` as the current sparse localized set, with layer 6 as the primary target layer. Layers 0-2 have high probe AUROC but near-zero patching effect, so probe salience alone is not treated as intervention salience. Layers 11+ are excluded from localized targets because they are outside the causal region and can show numerically unstable bfloat16 activations.

---

## Phase 2 — Targeted Unlearning

### Data Construction

- **Forget set**: human-tropic viral sequences (label=1), train split, 3,800 sequences
- **Retain set**: non-human-tropic viral sequences (label=0), train split, 3,814 sequences
- **Internal diagnostic**: val + test split, Phase 1 probe AUROC, and forget/retain perplexity
- **Primary evaluation**: downstream benchmark rows evaluated with supervised LoRA finetuning, fixed manifest splits, validation early stopping, and test metrics from the best validation checkpoint

### Gradient Difference (GD)

At each training step, one forget batch and one retain batch are sampled. The loss is:

$$\mathcal{L} = -\alpha_{\text{forget}} \cdot \mathcal{L}_{\text{forget}} + \alpha_{\text{retain}} \cdot \mathcal{L}_{\text{retain}}$$

where $\mathcal{L}_{\text{forget}}$ and $\mathcal{L}_{\text{retain}}$ are next-token cross-entropy losses. Maximizing $\mathcal{L}_{\text{forget}}$ degrades the model's ability to predict human-tropic sequences; minimizing $\mathcal{L}_{\text{retain}}$ preserves non-human-tropic sequence modeling. Gradient updates are restricted to the selected layer condition via `requires_grad` masking. Checkpoints store only the weight deltas of modified layers.

Four conditions are implemented:

| Condition | Updated layers | Basis |
|:---|:---|:---|
| `full` | all 32 layers | full-model baseline |
| `localized` | current RefSeq: layers 5-9; legacy first run: layers 3-9 | activation patching (causal) |
| `probe` | layers 0–10 | probe curve (alternative target) |
| `random` | current RefSeq: 5 layers from 11-30; legacy first run: 7 layers from 11-30 | matched negative control |

### RMU (Li et al., ICML 2024)

A frozen reference model is maintained. At each step:
- **Forget**: push hidden activations at layer 6 (strongest causal layer) toward a fixed random unit direction scaled by `steer_coef`
- **Retain**: constrain hidden activations at layer 6 to stay close to the reference model via MSE loss

Three conditions: `full`, `localized` (current RefSeq: layers 5-9; legacy first run: layers 3-9), `random`.

### Benchmark Evaluation Protocol

Selective unlearning is judged by benchmark deltas, not by perplexity alone. After applying each checkpoint, `phase2/eval_benchmarks.py` freezes the Evo backbone, injects LoRA adapters into every `nn.Linear` module inside all 32 Evo blocks, trains a supervised task head plus LoRA adapters on each downstream task, selects the best checkpoint by validation metric, and reports test metrics from that best checkpoint.

The benchmark manifest must contain `split,sequence,label` columns and may also include `benchmark,task,group,id`. The default LoRA evaluator filters to HVUE rows (`benchmark=hvue` or `group=hvue_forget`); pass `--benchmark-scope task --task-filter host_tropism_hiyata` for the Hiyata adaptation task. The retained legacy probe evaluator can still read the broader HVUE/GUE/viral-retain manifest for reference analyses. The intended task set is:

- **Hiyata Host Tropism adaptation**: `data/host_tropism_hiyata/manifest_no_gemini.csv` is converted into `data/host_tropism_hiyata/eval_manifest_lora.csv` with a deterministic label-stratified validation split; k-mer and LoRA use this same derived split.
- **HVUE forget tasks**: human host tropism; human-virus pathogenicity; transmissibility for human-relevant viral families (`Coronaviridae`, `Orthomyxoviridae`, `Caliciviridae`). The checked-in manifest has now been rebuilt from the local raw files and includes the `train/val/test` splits for `hvue_human_transmissibility_caliciviridae`; see `data/phase2/experiment_audit.json` for the current coverage audit.
- **GUE retain tasks**: promoter, splice-site, TF-binding, and chromatin-accessibility tasks, preserving general DNA modeling competence.
- **Viral retain tasks**: ViroBench CLS-Lite taxon tasks are the preferred viral-retain source because they ship metadata CSVs plus sequence JSONL splits. The default imported tasks are `virobench_all_taxon_genus`, `virobench_all_taxon_times`, `virobench_dna_taxon_genus`, `virobench_dna_taxon_times`, `virobench_rna_taxon_genus`, and `virobench_rna_taxon_times`, all evaluated as `group=viral_retain`. vGUE-style task-ready tables are also supported under `data/benchmarks/raw/viral_retain/<task>/<split>.csv|tsv|jsonl|json`, `data/benchmarks/raw/vgue/<task>/<split>...`, or a unified table under either root with `task,split,sequence,label` columns. Host-prediction viral tasks and `hiv1_tropism` are excluded from the default retain score because they overlap with the forget objective.

A successful method should reduce HVUE human-virus benchmark scores relative to the base model while keeping GUE scores near the base-model scores. Forget/retain PPL remains useful for debugging optimization damage, but it is no longer sufficient evidence of selective unlearning.

### Diagnostic Findings

1. **Random condition validates causal layer analysis.** In the legacy first-run diagnostic, GD random and RMU random show internal host-tropism probe AUROC ≈ 0.847 and PPL unchanged. In the current RefSeq sweep, GD random and RMU random preserve the L5-9 probe AUROC at ≈ 0.995 with near-baseline PPL. Updating matched non-causal layers has no meaningful effect on the diagnostic probe.

2. **Probe salience ≠ intervention salience.** GD probe (layers 0–10) achieves the same diagnostic AUROC reduction as GD localized (−0.320 vs −0.289) but causes 4× more PPL damage on the retain diagnostic (63.3 vs 15.7). Layers 0–2 are not effective intervention targets.

3. **Full-model updates are the strongest RefSeq internal diagnostic runs so far, not final selections.** In the current RefSeq sweep, `refseq_gd_full_ar5_s200` reduces L5-9 probe AUROC from the random-control level ≈ 0.994 to 0.547 with retain PPL 4.13. `refseq_rmu_full_sc50_s200` reduces it to 0.604 with retain PPL 4.43. Because those candidates were surfaced by probe/PPL diagnostics, they must be treated as legacy references until the full candidate set is evaluated and ranked with supervised LoRA downstream metrics.

4. **Current RefSeq localized updates are weak or unstable.** Among stable localized GD runs, `α_retain=3.0` gives the best internal forgetting trade-off (L5-9 AUROC 0.897, retain PPL 4.12). Stronger or longer GD can lower AUROC further, but the 500/1000-step runs show severe PPL damage. Localized RMU preserves PPL but leaves the diagnostic AUROC near 0.99 across `steer_coef`, retain-weight, direction, and step sweeps. Final method selection should still be based on HVUE/GUE forget-retain deltas once every RefSeq checkpoint has external benchmark coverage.

---

## Phase 3 — Recovery Attacks

### Attack Protocol

Two attacks are applied to all unlearned checkpoints using 453 held-out human-tropic sequences (test split, not seen during unlearning):

| Attack | Parameters | Trainable params |
|:---|:---|:---|
| SFT | all parameters, 200 steps, lr=1e-5 | ~7B (100%) |
| LoRA | rank-8 adapters on layers 3-9 in the legacy first-run attack setting, 200 steps, lr=1e-4 | 3.9M (0.05%) |

### Key Findings

1. **LoRA attack is ineffective across all conditions.** All LoRA deltas are within ±0.04. The adapter capacity and training data are insufficient to recover disrupted representations.

2. **SFT does not recover unlearning — it further degrades the model.** All SFT deltas are negative. The GD random and RMU random controls (never unlearned, AUROC=0.847) drop to 0.651 and 0.681 after SFT, confirming that the current SFT setup is destructive rather than restorative.

3. **Under the internal diagnostic comparison, RMU full is more robust to SFT than GD localized.** With comparable diagnostic retain PPL (3.88 vs 4.48), RMU full shows SFT Δ = −0.049 versus −0.175 for GD localized. This robustness claim should be rechecked after HVUE/GUE benchmark evaluation.

4. **Current attack setup is underpowered.** 453 sequences and 200 steps are insufficient for a definitive robustness claim. Planned improvements: expand to 869 sequences (test + val), extend to 500 steps at lr=1e-4, and inject LoRA across all layers at rank 16.

---

## Reproducing the Results

### Phase 1

```bash
# Build manifest, extract features, train probes, run patching
bash phase1/run.sh all
```

### Phase 2

```bash
# Build splits
bash phase2/run.sh splits

# Audit local dataset / benchmark / checkpoint availability
bash phase2/run.sh audit

# Run all GD conditions (full, localized, probe, random)
bash phase2/run.sh gd

# Run all RMU conditions
bash phase2/run.sh rmu

# Evaluate all checkpoints with internal diagnostics
bash phase2/run.sh eval

# Rebuild benchmark manifest from raw HVUE/GUE/ViroBench/vGUE viral-retain directories
# Set DOWNLOAD_GUE=1 if the raw GUE tree is not already present locally.
bash phase2/run.sh prepare_benchmarks

# Download and import ViroBench CLS-Lite taxon retain tasks
DOWNLOAD_VIROBENCH=1 bash phase2/run.sh prepare_benchmarks

# Optional: point to a vGUE-style root with either per-task split files or a unified task/split table
VIRAL_RETAIN_ROOT=data/benchmarks/raw/vgue bash phase2/run.sh prepare_benchmarks

# Build Hiyata Host Tropism LoRA adaptation manifest and k-mer baseline
bash phase2/run.sh prepare_hiyata_lora
bash phase2/run.sh kmer_hiyata

# LoRA finetune Base Evo on Hiyata Host Tropism adaptation split
python -u phase2/eval_benchmarks.py \
  --benchmark-manifest data/host_tropism_hiyata/eval_manifest_lora.csv \
  --benchmark-scope task \
  --task-filter host_tropism_hiyata \
  --out-dir data/phase2/hiyata_lora/base

# Evaluate primary HVUE LoRA benchmarks
BENCHMARK_MANIFEST=data/benchmarks/hvue_gue_manifest.csv bash phase2/run.sh benchmarks

# Tuned hyperparameter sweep summary
bash phase2/run_sweep.sh summary

# Audit what a cloned Vir2vec repo can provide for future vGUE integration
python phase2/prepare_vgue_from_vir2vec.py --vir2vec-root /tmp/Vir2vec --out data/benchmarks/vgue_from_vir2vec_audit.json
```

Tuned GD localized (α_retain=5.0):

```bash
python phase2/unlearn_gd.py \
    --condition localized \
    --alpha-retain 5.0 \
    --run-name gd_localized_ar5.0 \
    --out-dir data/phase2/checkpoints_tuned
```

### Phase 3

```bash
# SFT and LoRA attacks on all Phase 2 checkpoints with the original fixed LR setting
bash phase3/run.sh all

# LR-grid recovery attack sweep (preferred for the current experiment plan)
bash phase3/run_attacks.sh data/phase2/checkpoints_tuned
```

---

## Model

| Property | Value |
|:---|:---|
| Model | Evo-1-8k-base |
| Architecture | StripedHyena |
| Layers | 32 (29 Hyena + 3 attention at layers 8, 16, 24) |
| Hidden dimension | 4,096 |
| Parameters | ~7B |
| Tokenization | Character-level (byte values, vocab size 512) |
| Precision | bfloat16 (except poles and residues) |

---

## Data and Safety Note

This repository contains code and aggregate metrics only. The following are excluded:

- Raw genomic sequence data or FASTA files
- Model weight checkpoints
- Large activation feature matrices
- Recovery attack checkpoints

Phase 1 localization uses taxonomy and host annotation labels. The upgraded external evaluation may include public HVUE pathogenicity and transmissibility labels only as benchmark readouts; they are not used to train the unlearning objective.

---

## References

- Nguyen et al. 2024. *Sequence modeling and design from molecular to genome scale with Evo*. Science.
- Brixi et al. 2025. *Genome modeling and design across all domains of life with Evo 2*. bioRxiv.
- Li et al. 2024. *The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning*. ICML.
