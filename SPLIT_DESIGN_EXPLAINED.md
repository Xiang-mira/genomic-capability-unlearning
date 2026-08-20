# Composition-disjoint vs homology holdout: what each does, what was tested, what's correct

## The two splits, mechanically

**Composition-cluster-disjoint** (`build_splits_v2.py`, the one everything was run on):
1. compute the k-mer 3–6 frequency vector for every sequence (5,440 dims)
2. take the **k=5 block** (1,024 dims) and PCA it to 50 dims
3. **MiniBatchKMeans into 25 clusters in that k-mer-PCA space**
4. hold out whole clusters until ~30% of the data is held out
5. accept the split **only if the k-mer baseline's AUROC drops ≥ 0.03** vs a random split;
   scan 7 KMeans configs until one passes (`GATE = 0.03`)

**MMseqs2 identity-disjoint** (computed, then discarded):
1. MMseqs2 `easy-cluster` at 90% sequence identity / 90% coverage — alignment-based
2. hold out whole identity clusters until ~30%
3. (their version applied the same 0.03 gate; my rebuild removes it)

## Why they answer different questions

| | question it answers |
|:--|:--|
| composition-disjoint | *can the method generalise to sequences whose **nucleotide composition** is unlike anything in training?* |
| identity-disjoint | *can the method generalise to sequences with no close **homolog** in training?* |

Both are legitimate scientific questions. Only one of them is a fair way to compare a k-mer
baseline against a model.

## Why composition-disjoint is invalid *as a baseline comparison*

A k-mer logistic regression's **entire input is the composition vector**. Partitioning the data
by k-means in (a PCA of) that same vector space guarantees the held-out set sits in a region of
the k-mer feature space with **no training support**. The baseline is forced to extrapolate
outside its own feature support, by construction. Any method that also uses order, position, or
longer-range structure is less affected.

The analogy: comparing a linear model in feature space *X* against a nonlinear model, and
choosing the test set to be the points farthest from the training data *in X*. The nonlinear model
wins by construction. What you have measured is "how much does each method rely on k-mer
composition" — a real and interesting quantity — **not** "which method has more capability."

There is a second, independent problem: even granting composition-splitting as a legitimate
stressor, **selecting among candidate splits by which one hurts the baseline most is selection on
the outcome.** `GATE = 0.03`, scan 7 configs, keep the first that passes. That is not a
pre-registered split; it is a search for an unfavourable one.

## What the numbers show

k-mer3-6 baseline, same data pool, MMseqs2 99%-deduplicated first, no gate on my rebuild
(3 holdout seeds averaged):

| task | random | composition-cluster | identity-disjoint @90% |
|:--|--:|--:|--:|
| Host_Tropism | 0.9213 | **0.8034** | **0.9182** |
| Pathogenecity | 0.9685 | **0.8044** | **0.9547** |
| Transmissibility | 0.9238 | **0.7395** | **0.9175** |
| **mean drop vs random** | — | **−0.15** | **−0.008** |

Same sequences, same baseline, same ~30% holdout fraction. The 0.15 drop is the split geometry.

## But the identity split as run is *also* inadequate — it's too weak

The pool is **already deduplicated at 99% identity** before splitting. Clustering at **90%** then
removes only near-duplicates that survived that. So "90% identity-disjoint" barely changes the
distribution — which is exactly why the baseline drops only 0.008. It is not a meaningful OOD
test either; it is close to a random split with near-duplicates removed.

So neither split as executed is the right one:

| split as run | problem |
|:--|:--|
| composition-cluster @ k-mer5-PCA + GATE 0.03 | circular w.r.t. the baseline, and outcome-selected |
| identity-disjoint @ 90% after 99% dedup | too permissive — barely OOD, drop 0.008 |

## What is actually correct

**Taxonomic holdout — genus- or family-disjoint — with a strict identity threshold as a
secondary check.** Reasons:

1. **Model-agnostic.** Taxonomy is defined by external annotation, not by any method's feature
   space. It cannot privilege or penalise k-mer, CNN, or a gLM a priori.
2. **Biologically meaningful.** It answers the question the biosecurity framing actually needs:
   *does this work on a novel virus with no close relative in the training data?*
3. **Field convention.** ViroBench uses genus-disjoint and temporal splits; PHIStruct used CD-HIT
   at 40% identity; the student's own ViroBench analysis used leave-one-family-out.
4. **Not selectable.** There is one genus partition, so there is nothing to scan and no gate to
   apply.

For an identity-based check, use **50–70%**, not 90%, and report the threshold. I have a sweep
running now at 90 / 70 / 50 / 30% to show how the baseline degrades as a function of threshold —
that curve is the honest way to characterise how OOD a split really is.

## The relevant result already exists, and it is a clean negative

The locking project's `exp3_virobench/host_lofo_probe.csv` did **leave-one-family-out** on
ViroBench host prediction — the strictest split in either project, and model-agnostic:

| model | AUROC | k-mer | residual | 95% CI |
|:--|--:|--:|--:|:--|
| pretrained | 0.5934 | 0.6910 | **−0.0973** | [−0.1178, −0.0775] |
| locked_no_ft | 0.5991 | 0.6910 | −0.0916 | [−0.1119, −0.0714] |
| unlocked_ft | 0.5844 | 0.6910 | −0.1064 | [−0.1271, −0.0855] |
| M_a300k | 0.5213 | 0.6910 | −0.1698 | [−0.1907, −0.1481] |

All four CIs exclude zero; the k-mer wins by 0.09–0.17. This was run at `kmer_k = 4`, so a
kmer3-6 baseline would make the negative *larger*. **On the one split that is both strict and
model-agnostic, the model loses decisively.**

## Bottom line

- What was tested: a split built in the baseline's own feature space and chosen because it hurt
  the baseline. Every "model beats k-mer" number from HVUE inherits that.
- The discarded alternative was thrown out for the wrong reason (it didn't hurt the baseline
  enough), but it was also too weak to be the answer.
- The correct design is **genus/family-disjoint**, with an identity-threshold sweep to report how
  OOD the split is.
- Under that design, the existing leave-one-family-out evidence says the model **loses** by
  0.09–0.17 AUROC with CIs excluding zero.
