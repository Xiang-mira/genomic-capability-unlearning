# Baseline Capacity & Architecture Ceiling

**Question.** Every "a small supervised CNN matches the FM" claim in this project rested on
ONE architecture at 0.68M parameters (dilated conv stack, 5 blocks). That is an arbitrary
point, not a ceiling. If a better/larger baseline exists, our FM comparisons are unfair to
the baseline and any FM "win" inside that margin is void.

**Design.** 13 (architecture, capacity) cells spanning 0.04M -> 9.4M parameters across three
families, on 9 tasks. 2 seeds for splice/HVUE, 1 for ViroBench (20kb inputs). Early stopping
on dev, patience 6, max 30 epochs.

**Discipline.** Architecture AND capacity selected on **dev only**. Test is scored once per
cell; the dev-selected number and the oracle-best-over-ladder number are reported separately
so the reader can see the selection cost. For HVUE the dev carve is **group-disjoint** (via
`GroupShuffleSplit` on the homology cluster id) so architecture selection faces the same
homology holdout as test. ViroBench uses `min_count=1`, matching `virobench_frozen_probe.py`
exactly (173 classes / 46,389 train / 5,505 test).

Script: `scripts/viral_benchmark/capacity_sweep.py`. Raw: `scratchpad/multimodel/capacity_sweep/*.json`.

---

## Headline: the binding constraint is RECEPTIVE FIELD, not parameter count

Splice test MCC against effective receptive field (kernel 9, pad 4; each dilated conv adds
`8 x dilation`; U-Net reaches whole-sequence via strided down/up):

| arch | blocks | params | RF (bp) | acceptors | all | donors |
|:--|--:|--:|--:|--:|--:|--:|
| resnet | 3 | 0.27M | 57 | 0.4959 | 0.3115 | 0.5583 |
| resnet | 4 | 1.27M | 73 | 0.5201 | 0.3191 | 0.5410 |
| resnet | 5 | 6.08M | 89 | 0.4964 | 0.3090 | 0.5395 |
| resnet | 5 | **9.44M** | 89 | 0.5070 | 0.3356 | 0.5445 |
| dilated | 3 | 0.04M | 57 | 0.4422 | 0.2909 | 0.5103 |
| dilated | 4 | 0.15M | 121 | 0.5323 | 0.3092 | 0.5876 |
| dilated | 5 | **0.68M** *(incumbent)* | 249 | 0.6131 | 0.3537 | 0.6689 |
| dilated | 5 | 2.53M | 249 | 0.5945 | 0.3435 | 0.6738 |
| dilated | 6 | 9.33M | **505** | **0.9527** | **0.9528** | **0.9617** |
| unet | 3 | **0.26M** | **global** | **0.9514** | **0.9513** | **0.9630** |
| unet | 4 | 7.05M | global | 0.9466 | 0.9516 | 0.9637 |

Splice inputs are 600bp with the site centred, so the task **requires** near-full-sequence
context. Performance is monotone in RF and flat in parameters:

- **resnet 9.44M (RF 89) = 0.5070** vs **unet 0.26M (global) = 0.9514** — 36x fewer
  parameters, **+0.44 MCC**. Pure receptive-field effect.
- resnet gains nothing from 0.27M -> 9.44M (RF only 57 -> 89): flat at ~0.50-0.55.
- dilated jumps 0.3435 -> 0.9528 between 2.53M and 9.33M. This is **not** a capacity effect:
  the 6th block doubles RF from 249 to 505, crossing the 600bp requirement.
- U-Net is saturated at its **smallest** size (0.26M). More capacity adds nothing.

**Conclusion: "how big is the baseline" is the wrong question. "Does the baseline see the whole
input" is the right one.**

## Consequence 1 — the splice positive control PASSES; the published gap is a baseline artifact

Previously reported as a failure of our harness. It was a failure of the baseline architecture.

| | best baseline | published FM comparators |
|:--|--:|--:|
| acceptors | **0.9527** | 0.971-0.984 |
| all | **0.9528** | 0.971-0.984 |
| donors | **0.9637** | 0.971-0.984 |

Our harness reaches published-level performance once the baseline sees the whole input. The
residual FM margin is roughly **+0.02 to +0.03**, not the **+0.31 to +0.60** those papers
report against their CNN baselines. Verified no leakage: exact-duplicate and reverse-complement
train/test overlap are both **0** on all three tasks. Consistent with the field — SpliceAI, the
standard splice predictor, is a CNN.

*Caveat: the published numbers are EXTERNAL/PUBLISHED and fine-tuned; ours are supervised
baselines trained from scratch. This is not a like-for-like head-to-head and is not claimed as one.*

## Consequence 2b — FULL FINE-TUNE arm on splice: only 1 of 3 gLMs beats the baseline

`splice_sites_all` (3-class, 600bp, chromosome-disjoint). LR selected on **dev only** from
{1e-5, 3e-5, 1e-4}; linear warmup 10%; min_epochs 4; patience 3. All OUR RUN.

| model | best dev | dev-selected LR | TEST MCC | vs baseline 0.9528 |
|:--|--:|--:|--:|--:|
| **NT-v2-500M** | 0.9770 | 3e-5 | **0.9680** | **+0.0152** |
| GENA-LM | 0.8541 | 1e-4 * | 0.8294 | **−0.1234** |
| HyenaDNA-medium | 0.8475 | 3e-5 | 0.8498 | **−0.1030** |
| *frozen probe, best of 3* | — | — | 0.3636 | −0.5892 |
| published (EXTERNAL/PUBLISHED, FT) | — | — | 0.971–0.984 | +0.018–0.031 |

\* GENA-LM's dev-selected LR sits at the TOP EDGE of the {1e-5,3e-5,1e-4} grid, so 0.8294 is a
lower bound on its FT ceiling, not a converged optimum. NT-v2 (0.9665/0.9680/0.9685 across the
three LRs) and HyenaDNA (0.8127/0.8498/0.8387) both selected interior points.

1. **The positive control passes.** NT-v2 fine-tuned reaches 0.9680 against published
   0.971–0.984 — our harness reproduces published-level splice performance once the adaptation
   regime matches. The earlier "control failed" reading was a regime artifact: frozen probing
   costs **−0.59 MCC** relative to fine-tuning on this task.
2. **Even fully fine-tuned, 2 of 3 gLMs LOSE to a 9.3M-parameter dilated CNN** by 0.12–0.14 MCC.
   Only NT-v2 clears the baseline, and only by **+0.0152**. The published +0.31–0.60 gaps compare
   against CNN baselines whose receptive field could not span the 600bp input.

### GENA-LM weight-loading bug (affects the FT arm only)

`AutoModelForSequenceClassification` on `gena-lm-bert-base-t2t` silently discards **all 48
pretrained LayerNorm tensors** and randomly initialises 48 replacements: GENA-LM is pre-LN
(`pre_attention_ln`/`post_attention_ln`) while stock HF BERT is post-LN, so the names do not
match and HF fills the gap with fresh weights behind a warning. The model then collapses to the
majority class at every LR (dev MCC exactly 0.0000).

- **Frozen probes are UNAFFECTED** — they load via `AutoModel` + `trust_remote_code`, which
  matches the checkpoint exactly (48 pre-LN expected, 48 present, 0 fresh-init). Verified.
- Fixed by attaching our own mean-pool + linear head to `AutoModel`, which also makes the
  frozen-vs-FT comparison share an identical encoder.
- `assert_no_fresh_encoder_weights()` in `splice_finetune.py` now **aborts** on any load path
  that re-initialises pretrained tensors, listing them. This is the class of bug that silently
  becomes a published "model X lacks capability" claim.

## Consequence 2 — HVUE baselines were already at ceiling; the negative result is strengthened

| task | incumbent 0.68M | dev-selected | lift |
|:--|--:|--:|--:|
| Host_Tropism | 0.9425 | dilated 2.53M -> **0.9486** | +0.0061 |
| Pathogenecity | 0.9666 | resnet 1.27M -> **0.9718** | +0.0052 |
| Transmissibility | 0.9066 | dilated 0.68M -> **0.9066** | +0.0000 |

Searching 13 cells to 9.4M buys **at most +0.006 AUROC**. The incumbent baseline was already
essentially at its ceiling, so no FM comparison on HVUE was disadvantaged by baseline choice.
U-Net is *worse* here (0.89-0.90) — HVUE is a composition task, not a positional-motif task,
so global receptive field is not what it rewards. Mechanistically consistent with the splice result.

## Consequence 3 — on ViroBench the CNN family is not the binding baseline at all

| level | best CNN (dev-selected) | k-mer3-6 LR | best frozen FM |
|:--|--:|--:|--:|
| family (173 cls) | 0.1959 | **0.5738** | NT-v2 0.5853 |
| kingdom (18 cls) | 0.3795 | **0.560** | NT-v2 0.5778 |

Supervised CNNs are **~0.38 macro-F1 below k-mer LR** on 173-way taxonomy from 20kb. Taxonomy
is a compositional problem and k-mer LR dominates it. The correct ViroBench comparator is k-mer,
not CNN — which is what we used.

## Consequence 4 — Phase 3: NT-v2's ViroBench win is NOT statistically significant

Paired bootstrap, 2,000 resamples over the 5,505 shared test genomes, paired by taxid against
k-mer3-6 regenerated under the identical protocol (whole-genome counts, StandardScaler, C on dev).
Margins δ ∈ {0.01, 0.02, 0.03, 0.05} declared in the script **before** any CI was computed.

| run | FM | k-mer | Δ | 95% CI | P(Δ>0) |
|:--|--:|--:|--:|:--|--:|
| **nt_v2 W2048** | 0.5853 | 0.5738 | **+0.0115** | **[−0.0176, +0.0292]** | **0.714** |
| nt_v2 W1024 | 0.5149 | 0.5738 | −0.0589 | [−0.0822, −0.0345] | 0.000 |
| nt_v2 W512 | 0.4565 | 0.5738 | −0.1173 | [−0.1425, −0.0961] | 0.000 |
| gena_lm W2048 | 0.5268 | 0.5738 | −0.0470 | [−0.0678, −0.0278] | 0.000 |
| hyenadna W2048 | 0.4986 | 0.5738 | −0.0752 | [−0.1069, −0.0552] | 0.000 |
| lucavirus W2048 | 0.2802 | 0.5738 | −0.2936 | [−0.3235, −0.2734] | 0.000 |

**Kingdom level (18 classes):** NT-v2 W2048 Δ = **+0.0165**, CI **[−0.0847, +0.1121]**, not
significant — but note this CI is **not** equivalent at any pre-declared δ including 0.05. With
18 classes the macro-F1 bootstrap is far noisier than at family level. The honest reading is
**inconclusive / underpowered**, NOT "equivalent". Reporting it as equivalence would overclaim.

NT-v2 @ W2048 (family): **CI straddles zero**, P(Δ>0) = 0.71, and the difference is **statistically
equivalent to k-mer within δ = 0.03** (not within δ = 0.01 or 0.02). This was the last surviving
frozen-probe FM win on a defensible viral split, and it does not survive a paired test.

LucaVirus is **−0.29 to −0.32** below k-mer at every window. The previously-quoted "+0.079
LucaVirus advantage" is withdrawn as a head-to-head. **Caveat:** a discrepancy this large against
a published 0.649 more likely indicates a representation-extraction mismatch on our side (LucaVirus
is a joint gene/protein model with a shared 39-token vocab; naive last-layer mean-pooling in gene
mode may not be the intended read-out) than an error in their paper. Must be checked against their
reference implementation before publication.

---

## What changes in the paper

1. **Add the receptive-field analysis as a methodological contribution.** "Published FM-vs-CNN
   gaps on positionally-structured tasks are substantially a receptive-field artifact of the CNN
   baseline" is a defensible, mechanistically-explained, independently-useful claim.
2. **The splice positive control now passes** — restate it as harness validation, and report the
   residual FM margin as +0.02-0.03 rather than citing the published +0.31-0.60 gap.
3. **HVUE negatives stand and are stronger** — we can now state the baseline was at its ceiling.
4. **ViroBench: no FM advantage survives** a paired test at any level measured.
5. **Retract** the LucaVirus +0.079 claim pending the extraction check.

## Statistical power caveat

Family level (173 classes, 5,505 test genomes) supports an equivalence claim at δ = 0.03.
Coarser levels do not: fewer classes make macro-F1 bootstrap CIs much wider, so "not
significant" there means **underpowered**, not **equivalent**. Each level must be reported with
its own CI width rather than pooled into a single "no advantage" statement.

## Open items

- LucaVirus embedding extraction vs reference implementation (gates any LucaVirus claim).
- Fine-tuned splice arm, to separate "frozen probe is weaker" from "published numbers are FT".
- ViroBench `order` ladder complete: dev-selected resnet 0.282M -> macro-F1 0.2899, vs k-mer
  0.5987 -- CNNs remain far below k-mer at every taxonomic level measured.
- Paired bootstrap for the remaining ViroBench levels (kingdom/order/class/phylum).
