# EPI (enhancer-promoter interaction) — baseline vs published pretrained-embedding numbers

Data: `/scratch/10906/arisk/biojepa_data/gue/GUE/EPI/{cell}` — TargetFinder/SPEID/EPIVAN lineage
(6 cell lines: GM12878, HeLa-S3, HUVEC, IMR90, K562, NHEK; enhancer 3000bp + promoter 2000bp,
~10k train / 2k test per cell line). This is NOT part of DNABERT-2's official 28-dataset GUE
benchmark — it was fetched separately (`dl_epi.sh`) and just lives under the same directory tree.

Baseline: `epi_baselines.py` — k-mer features computed separately per anchor and concatenated
(k in {3,4,5} and {3,4,5,6}) into logistic regression, plus a siamese dilated-CNN (shared-weight
trunk applied to each anchor independently, pooled representations concatenated before a small
head), same methodology as the rest of this positive-control work: no baseline-performance gate,
report best(k-mer, CNN).

Splits: `official` (released train/test, dev carved from train) and `random` (pool + reshuffle
70/15/15) — same two-split policy used everywhere else in this work. **No chromosome-disjoint
variant exists for this benchmark family**: every paper in this lineage (TargetFinder, SPEID,
EPIVAN, EPI-DLMH, EPIPDLF, EPINTLM) uses random/stratified-random splits; BENGI (Moore et al.,
*Genome Biology* 2020) and LOCO-EPI explicitly critique this as leaking nearby genomic pairs across
train/test and inflating every method's reported performance, ours included.

Published comparators (frozen/lightly-tuned pretrained-embedding methods — no fully fine-tuned
DNABERT-2/NT/HyenaDNA/Caduceus result exists on this exact benchmark):
- **EPIPDLF** (static DNABERT embedding matrix, not fine-tuned)
- **EPINTLM** (Nucleotide Transformer k-mer embeddings, briefly fine-tuned)

## Results (test AUROC)

`our baseline` = max(k-mer3-5, k-mer3-6, **mean-over-3-seeds** CNN) AUROC — corrected from an
earlier version of this table that took the max CNN seed instead of the mean (an optimistic-
selection bias against a single-point published number; caught in audit, see
`aggregate_positive_control.py`'s docstring for the same fix applied to the GUE/NT table). The
correction is small here (all margins were already large) but is applied for consistency.

| cell line | our baseline (official split, mean-of-3-seed CNN) | EPIPDLF | EPINTLM | best published | our margin vs best published |
|:--|--:|--:|--:|--:|--:|
| GM12878 | 0.9655 | 0.939 | 0.949 | 0.949 | **+0.017** |
| HeLa-S3 | 0.9772 | 0.964 | 0.970 | 0.970 | **+0.007** |
| HUVEC | 0.9829 | 0.935 | 0.935 | 0.935 | **+0.048** |
| IMR90 | 0.9879 | 0.936 | 0.909 | 0.936 | **+0.052** |
| K562 | 0.9670 | 0.943 | 0.947 | 0.947 | **+0.020** |
| NHEK | 0.9886 | 0.993 | 0.985 | 0.993 | -0.004 (tied) |

Additionally, this session's split-leakage audit directly measured (not just cited) this
benchmark's leakage two ways. Exact-string matching found promoter sequences overlapping 42-62%
between train and test (enhancers 5.9-8.6%), while exact (enhancer,promoter) *pair* duplication is
0%. A second, stricter check using MMseqs2 (>=90% identity clustering, same method
`build_identity_splits.py` uses for HVUE) found leakage is substantially **worse** than exact
matching suggested: **enhancers 39-46%** and **promoters 67-80%** of test anchors fall in a cluster
that also contains a train anchor, across all 6 cell lines (`reports/mmseqs_leakage_check.csv`).
This independently confirms BENGI/LOCO-EPI's critique by direct measurement, and shows the
leakage is more severe than a naive exact-duplicate check would lead you to believe.

## Verdict

**Not a positive control — the opposite.** Our simple k-mer/CNN baseline matches or clearly
*beats* both published pretrained-embedding numbers on 5 of 6 cell lines, on the official
(non-disjoint) split. This isn't a failure of our methodology; it independently corroborates
BENGI/LOCO-EPI's published critique that this exact benchmark family's random splits leak enough
genomic-locus information that any reasonably capable model saturates it — including simple ones.

Net effect for the paper: this is a *second*, independently-sourced confirmation (this time from
pre-existing literature, not just our own split-construction critique) that apparent model
advantages on regulatory-genomics benchmarks can be split artifacts rather than real capability —
same shape as the viral-task and HVUE-composition-split findings elsewhere in this repo. It should
be reported alongside the splice-site positive control specifically *because* it shows the harness
doesn't just always find "no advantage" for uninteresting reasons — here the literature itself
already flags why the numbers are unreliable, and our result lines up with that flag rather than
contradicting it.
