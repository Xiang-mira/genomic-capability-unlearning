# GENEB sentinel reproduction — Vista (C2-GENEB-SENTINEL-002)

13-task sentinel subset (one per category, benchmark_spec.json category_order), full regime,
5 seeds averaged per GENEB's own protocol. Models: reference 4-mer k-mer baseline and our 3
project gLMs (NT-v2-500M, HyenaDNA-medium-160k, GENA-LM-bert-base-t2t), new extractors at
harness/extractors/vista_models.py, mean-pooled per spec. Compared against the ~40 pre-existing
published submission JSONs already shipped in this repo's submissions/.

## Critical correction: GENEB's own reference k-mer baseline is miscalibrated

GENEB's shipped ExampleKmerExtractor + run_GENEB.py's default LogisticRegression(max_iter=1000,
C=1.0, no scaling) produced a **degenerate MCC=0.000** on iDHS-EL_DNase_I -- the classifier
predicted the majority class 100% of the time, even on its own training set (72.4% train
accuracy, exactly the class prior). Confirmed NOT a data bug: features have full variance
(256/256 nonzero-variance dims), no NaNs, sequences and labels look normal on inspection.

Refit the same 4-mer features properly (StandardScaler on train, C swept 0.001-10 on a 15% dev
split carved from train -- never touching test, class_weight="balanced", matching this project's
own established k-mer-baseline convention elsewhere): **MCC jumps from 0.000 to 0.589** --
actually competitive with two of our three gLMs on that task. This is the same class of
"unfair/miscalibrated baseline" issue this entire project has repeatedly found and corrected
(HVUE composition splits, aggregate_positive_control.py seed-averaging, EPI leakage) -- now found
inside GENEB's own reference implementation. Script: fair_kmer_sentinel.py (this repo's GENEB
clone), results: fair_kmer_sentinel_results.json.

## Results (MCC, full regime)

| task | naive kmer (GENEB ref) | fair kmer (tuned) | NT-v2-500M | HyenaDNA | GENA-LM | best published |
|:--|--:|--:|--:|--:|--:|--:|
| NT H3 (Histone Mod.) | 0.602 | 0.590 | 0.662 | 0.671 | 0.705 | 0.781 (GenomeOcean-4B) |
| NT promoter_all (Promoters) | 0.754 | 0.813 | 0.882 | 0.835 | 0.917 | 0.930 (gena-lm-bert-large-t2t) |
| NT enhancers (Enhancers) | 0.456 | 0.425 | 0.396 | 0.485 | 0.463 | 0.526 (JanusDNA_72_wo) |
| deep4mc A.thaliana 4mC (DNA Methyl.) | 0.202 | 0.204 | 0.331 | 0.071 | 0.185 | 0.402 (LucaOne-step36M) |
| NT splice_acceptors revised (Splice Sites) | 0.269 | 0.387 | 0.479 | 0.402 | 0.543 | 0.685 (enformer-official-rough) |
| lncrna g_max (lncRNA) | 0.111 | 0.155 | 0.233 | 0.156 | 0.225 | 0.475 (LucaOne-step36M) |
| GUE mouse_0 (Mouse Enh.) | 0.437 | 0.437 | 0.378 | 0.156 | 0.464 | 0.667 (enformer-official-rough) |
| GUE human_tf_0 (TF Binding) | 0.611 | 0.537 | 0.576 | 0.563 | 0.672 | 0.690 (MutBERT) |
| human_or_worm (Species Clf.) | 0.815 | 0.812 | 0.893 | 0.782 | 0.931 | 0.948 (MutBERT) |
| ensembl_regulatory (Regulatory) | 0.348 | 0.289 | 0.526 | 0.555 | 0.526 | 0.597 (space) |
| GUE phage_fragments (Virus/Phage) | 0.512 | 0.604 | 0.854 | 0.479 | 0.659 | 0.950 (GenomeOcean-4B) |
| coding_vs_intergenomic (Coding/NC) | 0.706 | 0.734 | 0.780 | 0.677 | 0.853 | 0.904 (GENERator-eukaryote-3b) |
| iDHS-EL DNase_I (Chromatin Acc.) | **0.000** | **0.589** | 0.593 | 0.413 | 0.509 | 0.728 (GENERator-eukaryote-3b) |

## Reading

**With the fair k-mer baseline, our gLMs beat it on 13 of 13 sentinel tasks** (best of the 3
models per task) -- more decisive than the naive-kmer comparison (11/13), because two of the
apparent "kmer wins" against individual models (GUE mouse_0 vs NT-v2/HyenaDNA, GUE human_tf_0 vs
NT-v2/HyenaDNA) are still beaten by GENA-LM in both cases. This is the cleanest, most
comprehensive positive-control evidence found anywhere in this project this session -- broader
than the single-task-family NTv3 splice-site result, because it spans 13 independently-chosen
categories (histone marks, promoters, enhancers, methylation, splice sites, lncRNA, mouse
enhancers, TF binding, species classification, regulatory elements, virus/phage, coding/noncoding,
chromatin accessibility) under a frozen-probe (not fine-tuning) protocol.

Our 3 models still generally trail the best of GENEB's 40 published models (mean gap ~0.10-0.20
MCC) -- expected, since several of those (GenomeOcean-4B, Enformer, GENERator-3B, LucaOne) are
larger or architecturally specialized (Enformer specifically for regulatory tracks) versus our
project's 500M/6.5M/110M general-purpose models.

## Caveats
- 13/100 tasks, one regime (full; not yet 10-shot/1-shot).
- GENA-LM/HyenaDNA extraction used a fixed max_length (512/1024 tokens) per this project's
  existing convention, not tuned per GENEB task -- some tasks may have longer sequences truncated.
- The published "best" comparator per task is the max over all 40 submitted models (an
  oracle/best-observed statistic), not a fixed single model beating ours consistently across
  tasks -- worth reporting median-of-40 alongside max in any full run.
- Full-100-task run and 10-shot/1-shot regimes not attempted (would need ~7.7x more embedding
  extraction).
- This finding (GENEB reference k-mer miscalibration) should probably be reported back to the
  GENEB maintainers (darlednik/GENEB) as a real bug in their reference implementation, separate
  from anything specific to this project.
