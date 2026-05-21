# Evo-1 Viral-Family Representation Localization

## Phase 1 Pilot: Layer-wise Probing of Evo-1 Hidden Representations

This repository contains a Phase 1 pilot pipeline for **capability localization in Evo-1 genomic foundation models**.

The goal of this stage is to identify where viral-family sequence information is most linearly accessible inside Evo-1’s hidden representations. This is not a viral classifier project and does not study virulence, infectivity, pathogenicity, host range, or viral generation. Instead, the focus is **taxonomy-level representation probing**: whether frozen Evo-1 activations can separate viral-family sequences from non-viral genomic backgrounds.

The current repository reports an initial **plasmid-negative pilot**. In this pilot, plasmid sequences are used as the first non-viral control set to validate the probing pipeline and produce an initial layer-wise localization result. Broader non-viral controls, especially bacteria-negative reruns, are handled separately and should be used before making stronger biological claims.

---

## Main Result

The current pilot result shows that the viral-family signal is already highly accessible in the earliest Evo-1 layers under the plasmid-negative control setting.

### Layer-wise Probe AUROC

![Layer-wise viral vs non-viral probe](docs/assets/probe_metrics.png)

**Figure:** Layer-wise validation AUROC and test AUROC across all 32 Evo-1 layers.

The strongest signal appears in the early layers:

- Layer 0 reaches **0.991425 test AUROC**
- Layer 1 reaches **0.999975 test AUROC**
- Layer 2 reaches **1.0 test AUROC**
- Layers 1–10 stay near the top of the AUROC curve
- Layer 12 is the main local validation AUROC valley
- Attention layers at 8, 16, and 24 do not form special AUROC peaks in this pilot

The current pilot suggests that viral-family information is highly linearly accessible in the earliest Evo-1 layers under the current plasmid-negative setup, especially Layers 1–10. This is an initial probing result and should be validated with broader non-viral controls and activation steering.

---

## Layer-wise Probe Metrics

Full layer-wise metrics are available in:

```text
results/probe_metrics_by_layer.csv
```

The table below matches the output columns of `probe_metrics_by_layer.csv`.

| layer | C | train_acc | train_auroc | val_acc | val_auroc | test_acc | test_auroc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.0 | 0.961875 | 0.992215612836895 | 0.9625 | 0.993949394939494 | 0.9525 | 0.991425 |
| 1 | 10.0 | 0.9946875 | 0.9998851560705563 | 0.995 | 0.9997749774977498 | 0.9925 | 0.999975 |
| 2 | 10.0 | 0.9984375 | 0.9999968749951172 | 0.9925 | 0.9997249724972497 | 0.9925 | 1.0 |
| 3 | 1.0 | 0.9975 | 0.9999925781134034 | 0.995 | 0.9997749774977497 | 0.9925 | 0.999925 |
| 4 | 1.0 | 0.9984375 | 0.9999992187487793 | 0.995 | 0.9998499849984999 | 0.995 | 0.999925 |
| 5 | 10.0 | 1.0 | 1.0 | 0.9975 | 0.9998249824982498 | 0.9925 | 0.99995 |
| 6 | 1.0 | 1.0 | 1.0 | 0.9925 | 0.9997499749974997 | 0.9925 | 0.999975 |
| 7 | 1.0 | 1.0 | 1.0 | 0.995 | 0.9998249824982498 | 0.995 | 0.999975 |
| 8 | 1.0 | 1.0 | 1.0 | 0.9925 | 0.9997249724972497 | 0.995 | 0.9999 |
| 9 | 1.0 | 1.0 | 1.0 | 0.995 | 0.9997999799979997 | 0.9925 | 0.999975 |
| 10 | 0.1 | 1.0 | 1.0 | 0.9925 | 0.9992249224922491 | 0.99 | 0.9999 |
| 11 | 0.1 | 1.0 | 1.0 | 0.9725 | 0.9959245924592458 | 0.9875 | 0.9947874999999999 |
| 12 | 0.1 | 1.0 | 1.0 | 0.9625 | 0.9843109310931093 | 0.975 | 0.9912875 |
| 13 | 0.1 | 0.996875 | 0.9999605468133543 | 0.96 | 0.9937743774377438 | 0.9725 | 0.9947 |
| 14 | 10.0 | 0.9946875 | 0.9998855466961667 | 0.9625 | 0.9944494449444945 | 0.975 | 0.9959250000000001 |
| 15 | 1.0 | 0.9921875 | 0.9997308589544671 | 0.9675 | 0.9941994199419942 | 0.975 | 0.99625 |
| 16 | 1.0 | 0.9921875 | 0.9997308589544671 | 0.9675 | 0.9941994199419942 | 0.975 | 0.99625 |
| 17 | 10.0 | 0.9940625 | 0.9998421872534176 | 0.9625 | 0.9938743874387439 | 0.975 | 0.9959 |
| 18 | 0.1 | 0.9903125 | 0.9996765619946282 | 0.9675 | 0.9948244824482448 | 0.9775 | 0.997125 |
| 19 | 1.0 | 0.996875 | 0.9999207030010985 | 0.96 | 0.9942494249424942 | 0.975 | 0.994975 |
| 20 | 1.0 | 0.994375 | 0.9998968748388669 | 0.955 | 0.9940494049404941 | 0.9725 | 0.995225 |
| 21 | 1.0 | 0.995 | 0.9997855465399165 | 0.965 | 0.9941744174417442 | 0.9725 | 0.9956750000000001 |
| 22 | 1.0 | 0.9940625 | 0.9998667966668698 | 0.9625 | 0.9942744274427443 | 0.9725 | 0.9958750000000001 |
| 23 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 24 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 25 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 26 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 27 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 28 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 29 | 0.1 | 0.9978125 | 0.9999503905474852 | 0.965 | 0.9940744074407442 | 0.975 | 0.994375 |
| 30 | 1.0 | 0.9940625 | 0.9997546871166986 | 0.96 | 0.9946744674467447 | 0.9775 | 0.9966250000000001 |
| 31 | 1.0 | 0.9940625 | 0.9997546871166986 | 0.96 | 0.9946744674467447 | 0.9775 | 0.9966250000000001 |

---

## GC + 1-gram Baseline Metrics

Baseline metrics are available in:

```text
results/gc_1gram_metrics.csv
```

The table below matches the output columns of `gc_1gram_metrics.csv`.

| C | train_acc | train_auroc | val_acc | val_auroc | test_acc | test_auroc |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0 | 0.6225 | 0.6697489702867573 | 0.615 | 0.668916891689169 | 0.64 | 0.6853634085213033 |

The GC + 1-gram baseline uses only GC content and A/C/G/T/N frequencies. No Evo representations are used in this baseline.

The baseline test AUROC is **0.6853634085213033**, while the Evo layer-wise probes reach much higher AUROC values in early layers under the current plasmid-negative pilot setting. This indicates that the frozen Evo-1 activations contain stronger viral-family separability than the simple 1-gram composition baseline in this pilot.

---

## Current Completed Work

This repository includes the completed Phase 1 pilot workflow:

- Evo-1 checkpoint loading and forward inference test
- RefSeq-based pilot manifest construction using plasmids as the initial non-viral control
- Layer-wise activation extraction from all 32 Evo-1 blocks
- Mask-aware mean pooling into 4096-dimensional sequence representations
- 32 layer-wise L2-regularized logistic probes
- Probe coefficient, intercept, selected C value, accuracy, and AUROC outputs
- GC + 1-gram logistic regression baseline
- Layer-wise AUROC visualization
- Full 32-layer metrics table
- Separate bacteria-negative rerun script prepared for broader non-viral validation

---

## Dataset

The current plotted results are based on the **plasmid-negative pilot manifest**.

| Item | Value |
|---|---:|
| Data source | NCBI RefSeq |
| Viral sequences | 2,000 |
| Initial non-viral control sequences | 2,000 plasmid sequences |
| Total sequences | 4,000 |
| Window length | 2,048 bp |
| Train / Val / Test | 3,200 / 400 / 400 |
| Split ratio | 80% / 10% / 10% |
| Random seed | 42 |
| Current pilot label definition | viral = 1, plasmid-control = 0 |

Exact duplicate checking was performed across the train, validation, and test splits. All 4,000 sequences are unique, so the current AUROC values are not inflated by exact duplicate sequence leakage.

The plasmid set should be interpreted as the current pilot negative control, not as the full definition of non-viral genomic background. Additional bacteria-negative reruns are prepared separately to evaluate whether the early-layer localization pattern remains stable under broader non-viral controls.

---

## Model

The probing pipeline uses Evo-1 as a frozen genomic foundation model.

| Model property | Value |
|---|---:|
| Model family | Evo-1 |
| Architecture | StripedHyena |
| Number of layers | 32 |
| Hidden dimension | 4,096 |
| Hyena layers | 29 |
| Attention layers | 8, 16, 24 |
| Input resolution | character / nucleotide-level tokenization |

Evo-1 is not a standard Transformer. Most layers are Hyena convolutional operators, with only three attention layers. This matters for interpretation: the strongest separability appears very early, and the attention layers do not form special AUROC peaks in the current pilot.

---

## Pipeline

The current pilot pipeline is organized into four main steps.

```text
Step 1: Build pilot manifest
RefSeq FASTA files
→ clean A/C/G/T/N
→ crop or sample 2048-bp windows
→ assign labels and splits
→ write manifest.csv

Step 2: Extract activations
DNA sequences
→ frozen Evo-1 forward pass
→ forward hooks on all 32 blocks
→ mask-aware mean pooling
→ 4096-dimensional vector per sequence per layer

Step 3: Train probes
Layer-wise features
→ L2 logistic regression
→ C-grid search over {0.1, 1, 10}
→ select best C by validation AUROC
→ save probe metrics and probe weights

Step 4: Plot results
probe_metrics_by_layer.csv
→ layer-wise validation/test AUROC curve
```

A separate `run_bacteria_rerun.sh` script is included for the bacteria-negative rerun. It writes to a separate output directory to avoid overwriting the current plasmid-negative pilot results.

---

## Method

For each sequence and each Evo-1 layer, the hidden state is a token-level representation:

$$
H_l \in \mathbb{R}^{T \times 4096}
$$

where \(T\) is the sequence length and 4096 is the hidden dimension.

I use mask-aware mean pooling to convert token-level hidden states into one fixed-length sequence representation:

$$
h_l = \frac{1}{T}\sum_{t=1}^{T} H_{l,t}
$$

This produces one 4096-dimensional vector per sequence per layer.

For each layer, I then train an L2-regularized logistic regression probe:

```text
Input: mean-pooled activation from layer l
Output: current pilot label, viral vs plasmid-control
Metric: AUROC and accuracy
C-grid: 0.1, 1, 10
Selection criterion: validation AUROC
```

The linear probe is intentionally simple. The goal is not to maximize classifier complexity, but to test whether viral-family information is linearly accessible in the frozen representation space under a given non-viral control setting.

---

## Interpretation

The main interpretation from the pilot result is:

> Viral-family separability is already highly linearly accessible in the earliest Evo-1 layers under the current plasmid-negative pilot setting.

This is notable because one might expect higher-level biological or taxonomic information to emerge in middle or later layers. Instead, Layer 0 reaches 0.991425 test AUROC, and the early layers remain high.

A plausible explanation is that Hyena convolutional layers can aggregate sequence composition and higher-order motif statistics very early. The GC + 1-gram baseline already shows that simple nucleotide composition carries some signal, but the gap between the baseline and the early Evo probes suggests that Evo is extracting richer sequence features than single-base frequencies.

However, because the current plotted result only uses plasmids as the negative control, this should be interpreted as an initial pilot localization result rather than a final biological conclusion about all non-viral genomic backgrounds. Future reruns with bacteria and other broader non-viral controls are necessary before making stronger claims.

The attention layers do not show special peaks in this pilot. Layer 8 is high, but it follows the broader early-layer trend. Layers 16 and 24 do not create clear new peaks in the current plotted results.

---

## Probe Weight Files

Each trained probe produces a `.npz` file containing:

| Field | Meaning |
|---|---|
| `coef` | 1 × 4096 linear direction in layer representation space |
| `intercept` | Logistic regression intercept |
| `C` | Selected regularization strength |

The probe coefficient vector can be interpreted as the linear direction that increases the viral-family logit under the current pilot label setup. These vectors are useful for analyzing the geometry of layer-wise separability and can be reused as candidate steering directions in follow-up experiments.

These probe directions should not be treated as causal viral-capability directions until they are validated with activation steering or ablation.

---

## Notes on Result Reliability

The early-layer results are the main conclusion of this pilot. The late-layer rows are included for completeness, but some late-layer metrics show exact duplication:

- Layers 23–29 have identical metrics.
- Layers 15–16 are duplicated.
- Layers 30–31 are duplicated.

These rows should not be used as the main basis for interpretation. The early-layer localization pattern, especially Layers 1–10, does not depend on the duplicated late-layer region.

Additional reliability work is still needed:

- bacteria-negative rerun for a broader non-viral control
- higher-order k-mer baselines
- shuffled-label control
- multiple random seeds
- group-level split by accession, genus, or family where possible
- activation steering validation
- late-layer feature extraction debugging

---

## Repository Structure

```text
evo1-viral-capability-localization/
├── README.md
├── run.sh
├── run_bacteria_rerun.sh
├── phase1/
│   ├── download_refseq.py
│   ├── extract_features.py
│   ├── train_probes.py
│   ├── baseline_gc_1gram.py
│   ├── plot_metrics.py
│   └── utils.py
├── results/
│   ├── probe_metrics_by_layer.csv
│   └── gc_1gram_metrics.csv
└── docs/
    └── assets/
        └── probe_metrics.png
```

---

## Key Files

| File | Description |
|---|---|
| `run.sh` | End-to-end current plasmid-negative Phase 1 pilot pipeline |
| `run_bacteria_rerun.sh` | Optional bacteria-negative rerun script using a separate output directory |
| `phase1/download_refseq.py` | Downloads RefSeq FASTA files and builds the manifest |
| `phase1/extract_features.py` | Extracts mean-pooled 32-layer Evo-1 activations |
| `phase1/train_probes.py` | Trains layer-wise L2 logistic probes |
| `phase1/baseline_gc_1gram.py` | Runs GC + 1-gram baseline |
| `phase1/plot_metrics.py` | Generates layer-wise AUROC visualization |
| `phase1/utils.py` | Shared utilities for manifest reading, sequence cleaning, batching, and feature writing |
| `results/probe_metrics_by_layer.csv` | Full 32-layer probe metrics from the current pilot |
| `results/gc_1gram_metrics.csv` | GC + 1-gram baseline metrics from the current pilot |
| `docs/assets/probe_metrics.png` | Layer-wise AUROC plot shown in README |

---

## Data and Safety Note

This repository is intended for code and aggregate result sharing.

It should not include:

- model checkpoints
- raw FASTA files
- raw genomic sequence data
- large activation feature chunks
- large `.npy` feature matrices
- private server paths
- recovery attack checkpoints
- sensitive generated sequences

The current repository focuses on Phase 1 representation probing and aggregate metrics.

---

## References

- Nguyen et al. 2024. *Sequence modeling and design from molecular to genome scale with Evo*. Science.
- Brixi et al. 2025. *Genome modeling and design across all domains of life with Evo 2*. bioRxiv.
