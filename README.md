# Evo-1 Host-Tropism Representation Localization and Targeted Unlearning

This repository contains a three-phase pipeline for **capability localization and targeted machine unlearning** in the Evo-1-8k-base genomic foundation model (StripedHyena, 32 layers, ~7B parameters).

The research question is whether host-tropism information (human-tropic vs. non-human-tropic viral sequences) can be (1) localized to specific layers via probing and causal analysis, (2) selectively removed via targeted unlearning while preserving general sequence modeling ability, and (3) tested for robustness against fine-tuning recovery attacks.

All labels are based on taxonomy and host annotation only. No virulence, pathogenicity, or infectivity labels are used at any stage.

---

## Results Summary

### Phase 1 — Layer-wise Probe AUROC and Activation Patching

![Activation patching analysis](figures/patching_analysis.png)

**Figure:** (a) Layer-wise probe AUROC. (b) Activation patching causal effect |Δprob| per layer. (c) PPL delta per layer (flat — single-layer patching is compensated downstream).

- Layers 0–10 achieve probe AUROC 0.975–0.997, far above the k-mer baseline (0.851)
- Activation patching identifies layers 3–9 as the causal target region (layer 6: |Δprob| = 0.355, layer 8: 0.276)
- Layers 0–2 are linearly decodable but have near-zero patching effect — probe salience ≠ intervention salience
- PPL delta is flat across all layers (std < 0.0001), confirming that unlearning must target multiple layers simultaneously

### Phase 2 — Targeted Unlearning

![Phase 2 unlearning results](figures/phase2_results.png)

**Figure:** (a) Probe AUROC by layer after unlearning. (b) Forget–retain PPL trade-off (log scale).

| Method | Updated layers | AUROC L3–9 | Δ AUROC | Forget PPL | Retain PPL |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | — | 0.844 | — | ~4.2 | ~4.2 |
| GD full | all 32 | 0.524 | −0.320 | 31.2 | 37.9 |
| GD localized | 3–9 (patching) | 0.555 | −0.289 | 20.4 | 15.7 |
| GD probe | 0–10 (probe curve) | 0.524 | −0.320 | 137.5 | 63.3 |
| GD random | 7 random (11–30) | 0.847 | +0.003 | 4.2 | 4.2 |
| RMU full | all 32 | 0.700 | −0.144 | 4.5 | 4.48 |
| RMU localized | 3–9 | 0.765 | −0.079 | 4.4 | 4.42 |
| RMU random | 7 random (11–30) | 0.847 | +0.003 | 4.2 | 4.3 |

### Phase 3 — Recovery Attacks

![Phase 3 recovery attack results](figures/phase3_results.png)

**Figure:** (a) Full method × attack matrix heatmap. (b) Tuned comparison: GD localized (α_retain=5.0) vs RMU full under SFT and LoRA attacks.

Controlled comparison after hyperparameter tuning (both methods with retain PPL ≈ baseline):

| Method | After unlearning | Retain PPL | After SFT | SFT Δ | After LoRA | LoRA Δ |
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
│   ├── eval_unlearn.py             # Post-unlearning probe AUROC and PPL evaluation
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

**Key result:** Layers 3–9 are the causal target region. Layers 0–2 have high probe AUROC (0.975–0.995) but near-zero patching effect (|Δprob| < 0.02). Layers 11+ show numerically unstable activations in bfloat16 (L2 norm jumps from ~257 at layer 10 to ~1.8M at layer 11) and are excluded.

---

## Phase 2 — Targeted Unlearning

### Data Construction

- **Forget set**: human-tropic viral sequences (label=1), train split, 3,800 sequences
- **Retain set**: non-human-tropic viral sequences (label=0), train split, 3,814 sequences
- **Evaluation**: val + test split, probe AUROC and perplexity

### Gradient Difference (GD)

At each training step, one forget batch and one retain batch are sampled. The loss is:

$$\mathcal{L} = -\alpha_{\text{forget}} \cdot \mathcal{L}_{\text{forget}} + \alpha_{\text{retain}} \cdot \mathcal{L}_{\text{retain}}$$

where $\mathcal{L}_{\text{forget}}$ and $\mathcal{L}_{\text{retain}}$ are next-token cross-entropy losses. Maximizing $\mathcal{L}_{\text{forget}}$ degrades the model's ability to predict human-tropic sequences; minimizing $\mathcal{L}_{\text{retain}}$ preserves non-human-tropic sequence modeling. Gradient updates are restricted to the selected layer condition via `requires_grad` masking. Checkpoints store only the weight deltas of modified layers.

Four conditions are implemented:

| Condition | Updated layers | Basis |
|:---|:---|:---|
| `full` | all 32 layers | full-model baseline |
| `localized` | layers 3–9 | activation patching (causal) |
| `probe` | layers 0–10 | probe curve (alternative target) |
| `random` | 7 layers from 11–30 | matched negative control |

### RMU (Li et al., ICML 2024)

A frozen reference model is maintained. At each step:
- **Forget**: push hidden activations at layer 6 (strongest causal layer) toward a fixed random unit direction scaled by `steer_coef`
- **Retain**: constrain hidden activations at layer 6 to stay close to the reference model via MSE loss

Three conditions: `full`, `localized` (layers 3–9), `random`.

### Key Findings

1. **Random condition validates causal layer analysis.** GD random and RMU random show AUROC ≈ 0.847 and PPL unchanged. Updating 7 non-causal layers has no effect.

2. **Probe salience ≠ intervention salience.** GD probe (layers 0–10) achieves the same AUROC reduction as GD localized (−0.320 vs −0.289) but causes 4× more retain damage (PPL 63.3 vs 15.7). Layers 0–2 are not effective intervention targets.

3. **RMU full gives the best forget–retain balance.** AUROC −0.144 with retain PPL 4.48 (near baseline). GD localized gives stronger forgetting (−0.289) but higher retain damage (PPL 15.7).

4. **GD localized with α_retain=5.0** reduces retain PPL to 3.88 (near baseline) while maintaining AUROC drop of −0.220, selected as the tuned GD representative for Phase 3.

---

## Phase 3 — Recovery Attacks

### Attack Protocol

Two attacks are applied to all unlearned checkpoints using 453 held-out human-tropic sequences (test split, not seen during unlearning):

| Attack | Parameters | Trainable params |
|:---|:---|:---|
| SFT | all parameters, 200 steps, lr=1e-5 | ~7B (100%) |
| LoRA | rank-8 adapters on layers 3–9, 200 steps, lr=1e-4 | 3.9M (0.05%) |

### Key Findings

1. **LoRA attack is ineffective across all conditions.** All LoRA deltas are within ±0.04. The adapter capacity and training data are insufficient to recover disrupted representations.

2. **SFT does not recover unlearning — it further degrades the model.** All SFT deltas are negative. The GD random and RMU random controls (never unlearned, AUROC=0.847) drop to 0.651 and 0.681 after SFT, confirming that the current SFT setup is destructive rather than restorative.

3. **Under controlled comparison, RMU full is more robust to SFT than GD localized.** With comparable retain PPL (3.88 vs 4.48), RMU full shows SFT Δ = −0.049 versus −0.175 for GD localized. RMU achieves forgetting through representation misdirection without disrupting the generation objective, making it more stable under subsequent fine-tuning.

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

# Run all GD conditions (full, localized, probe, random)
bash phase2/run.sh gd

# Run all RMU conditions
bash phase2/run.sh rmu

# Evaluate all checkpoints
bash phase2/run.sh eval
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
# SFT and LoRA attacks on all Phase 2 checkpoints
bash phase3/run.sh all
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

All experiments use taxonomy and host annotation labels only. No virulence, pathogenicity, infectivity, or gain-of-function labels are used at any stage.

---

## References

- Nguyen et al. 2024. *Sequence modeling and design from molecular to genome scale with Evo*. Science.
- Brixi et al. 2025. *Genome modeling and design across all domains of life with Evo 2*. bioRxiv.
- Li et al. 2024. *The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning*. ICML.
