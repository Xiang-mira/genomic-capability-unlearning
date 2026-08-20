# What we now know, and what it means for the unlearning project

2026-08-19. Synthesis of the verification + multi-model experiments.

---

## 1. The central empirical claim of the project is a split artifact

The project's narrative rests on "model advantage collapses under composition-disjoint
evaluation." I rebuilt the splits with MMseqs2 90%-identity holdout and **no
baseline-performance gate** (3 holdout seeds per task):

| task | random | composition-cluster (what was used) | **identity-disjoint (mean of 3)** | drop vs random |
|:--|--:|--:|--:|--:|
| Host_Tropism | 0.9213 | 0.8034 | **0.9182** | −0.0031 |
| Pathogenecity | 0.9685 | 0.8044 | **0.9547** | −0.0138 |
| Transmissibility | 0.9238 | 0.7395 | **0.9175** | −0.0063 |

Under genuine homology holdout the k-mer baseline loses **0.003–0.014**. Under composition
holdout it loses **0.12–0.18**. The difference is the split, not the biology.

Cause, in `build_splits_v2.py`: clusters are built in **k-mer5-PCA space** (the baseline's own
feature space), and candidates are accepted only if `kmer_auroc(split) ≤ kmer_auroc(random) −
0.03` (`GATE = 0.03`). The homology-disjoint split *was* computed, gave 0.9131, failed that gate,
and was written off as `VALIDITY: INVALID` on all three tasks.

**These HVUE tasks are not meaningfully out-of-distribution under homology holdout.** The
baseline to beat is 0.92–0.95, not 0.80.

## 2. Against the best baseline, there is no qualified headroom

The right comparator is `max(k-mer3-6, supervised CNN)`. The CNN is 0.64M params, no
pretraining, trained from scratch — a baseline, not a model.

| task / split | best baseline | gLMs vs best |
|:--|:--|:--|
| Host_Tropism random | CNN 0.9502 | GENA-LM +0.0039 (noise), HyenaDNA **−0.0047** |
| Host_Tropism comp-disjoint | CNN 0.8407 | 5 wins, +0.006…+0.035 — **on the handicapped split** |
| Pathogenecity comp-disjoint | CNN 0.8436 | **all 3 lose**, −0.038…−0.095 |
| Transmissibility comp-disjoint | k-mer 0.7395 | 1 mixed, 1 loss |
| GUE `virus_covid` (9 cls) | k-mer 0.7282 | GENA-LM 0.607, NT-v2 0.679 — below |
| GUE `virus_species_40` (25 cls) | k-mer 0.4407 | running |

**One cell out of six, on the one split selected to depress the baseline.** Which baseline binds
also varies — CNN on two tasks, k-mer on four — so reporting k-mer alone understates the
baseline and is how the models came to look better than they are.

## 3. There is nothing in the frozen representation to remove

| | AUROC | vs best baseline |
|:--|--:|--:|
| HyenaDNA frozen probe (head only) | 0.5910 | **−0.2497** |
| Evo frozen probe (4 independent measurements) | — | −0.021 … −0.063 vs k-mer |

The localize-then-remove program requires the capability to be linearly decodable from the
representation. It is not — it is 0.25 AUROC below a baseline. **You cannot localize or excise
what is not there.** Consistent with this, every probe measurement peaks at the *shallowest*
layer (best layer 00 on two of three tasks; per-layer AUROC declining monotonically L0 0.906 →
L9 0.878), which flatly contradicts the "layers 3–9 causal, peak at layer 6" localization the
entire removal section was built on.

## 4. The capability is not gated by the model — which removes the biosecurity rationale

| Host_Tropism comp-disjoint | params | pretrained? | AUROC |
|:--|--:|:--|--:|
| HyenaDNA random-init, full FT | 7M | **no** | 0.8245 |
| CNN from scratch | **0.64M** | **no** | 0.8407 |
| HyenaDNA pretrained, full FT | 7M | yes | 0.8686 |
| best pretrained gLM (NT-v2) | 498M | yes | 0.8756 |

A **0.64M-parameter CNN trained from scratch on public data** reaches 0.8407 — above the k-mer,
above Evo's LoRA (0.8173), above random-init HyenaDNA. Pretraining buys +0.028 over the CNN on
the handicapped split and nothing detectable on the random split.

**This is the decisive point for the unlearning premise.** Removing a capability from open
weights is a meaningful safeguard only if the capability is hard to obtain otherwise. Here an
adversary with a laptop and the public training set gets equivalent performance. Excising it from
Evo protects nothing.

## 5. The unlearning results themselves are not interpretable

| claim | what the data says |
|:--|:--|
| RMU "best AUROC drop ~0.14" | **a 0.146 *increase*** (`delta_drop = 0.844 − score`); localized RMU 0.990 vs random-layer control 0.994 — **indistinguishable** |
| GD localized | best +0.035 on the joint probe; one run diverged to retain PPL 2.0e7 |
| One apparent success, BVBRC_CoV −14pp | on the task with the **highest** shortcut ceiling (0.9476) where the model was already ~0.19 *below* k-mer — RMU moved the needle only where there was no model-specific capability |
| Fresh-probe recovery to ~0.99 (flagship) | **no artifact anywhere** in the repo; my feature-space reconstruction recovers −6.8%…+8.5% of the above-chance margin, i.e. nothing |
| `internal_auroc_drop` column | holds **two incompatible quantities**; same method/condition appears both effective and ineffective |

---

## What this means — three options

**A. The unlearning project as scoped is not viable on these viral tasks.** There is no qualified
target: no model-specific headroom against the best baseline, nothing decodable in the frozen
representation to localize, and no biosecurity value in removal because the capability is
reproducible from scratch.

**B. The negative result plus the methodology *is* the paper — and it is a good one.** Not
"k-mer beats foundation models" (that claim is half wrong: on random splits the CNN, not the
k-mer, is the binding baseline, and the gLMs roughly tie it). The defensible claims are:

1. Apparent OOD advantage on viral genomic benchmarks is **highly sensitive to how the disjoint
   split is constructed**; building clusters in the baseline's feature space and selecting splits
   on baseline degradation manufactures the effect. Quantified: 0.12–0.18 vs 0.003–0.014.
2. The correct comparator is a **small supervised CNN**, not a k-mer; it binds on half the tasks
   and matches or beats every pretrained gLM tested on the non-disjoint splits.
3. **Frozen genomic representations are far below baseline** on these tasks (−0.25 AUROC), so
   representation-localization methods have no target.
4. Therefore **capability-removal from open genomic weights is not an effective biosecurity
   intervention for viral classification tasks** — the capability is not model-gated.
5. Methodological: the disjoint-split metric swings 0.014–0.045 across checkpoints *after*
   inner-validation plateaus, so single-checkpoint reporting on such splits is inadequate.

That is publishable, useful to the field, and the strongest thing this programme has produced.

**C. One viral candidate remains genuinely untested: taxonomy under genus-disjoint splits.**
ViroBench reports LucaVirus 75.88 vs BLAST 47.67 there (+28). Neither project has a valid
measurement: the student's 6 ViroBench runs diverged (F1 0.013–0.19), and the locking project's
`virobench_probe_results.csv` reports AUROC = F1 = MCC = **1.0000** on 100 classes with
`n_train == n_val == 4475` — leakage. `virus_species_40` (25-class, running now) is a proxy for
it. **Test that before abandoning the viral modality.** If it also fails, pivot to narrow
non-viral specificity tasks (kinase–substrate, enzyme–substrate, partner-specific PPI).

## Recommendation

Do B and C together. Write the negative + methodology now — it does not depend on any pending
run. In parallel, finish `virus_species_40` and download ViroBench for a proper genus-disjoint
taxonomy test. That single result decides whether the unlearning project has a viral target at
all.

Do **not** invest further in HVUE-based unlearning. Every axis is closed: no headroom, no
localizable representation, no biosecurity rationale, and the interventions that were tried are
indistinguishable from their random-layer controls.

## Corrections I made during this work

- My R1 sweep (k-mer budget vs Evo's LoRA) was **unmatched on training data** and therefore
  invalid; `logs_v2/CHECK_4MER.log` had already done it correctly.
- My "the gain is capacity not pretraining" claim was **refuted** by the random-init control
  (+0.034 for pretraining).
- I ran eight GPUs of models on the composition-disjoint split **before** checking how it was
  constructed. The split-selection gate should have been the first thing I looked at.
- The C-grid-boundary objection I raised was already tested by the student (`CHECK_LOWC.log`) and
  was wrong.
