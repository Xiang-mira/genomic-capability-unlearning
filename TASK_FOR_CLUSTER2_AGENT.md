# Task for the Cluster 2 (Vista) agent — GENEB provenance + missing artifacts

**Blocking:** action A1 (CNN ladder on the 13 GENEB sentinel tasks) cannot start on Cluster 1 until
these are resolved. No GPU time needed for any of it.

---

## T1 — Is GENEB's splice task the same data as ours? (highest priority)

Our headline comparison is currently unverified and may be cross-dataset:

> our CNN ladder **0.9527** vs GENEB fair k-mer **0.387** vs GENEB best-of-40 published **0.685**

Our splice source is `/data/nvidia/data/ntv3/splice_sites_acceptors/{train,test}.parquet` on
Cluster 1: **n_train 30,000 / n_test 3,000**, seqlen **600bp**, columns `sequence,name,label,task`,
train balance `{1: 15031, 0: 14969}`. Our k-mer on it: **kmer3-5 0.3945, kmer3-6 0.4131** — close to
GENEB's fair k-mer 0.387, which is suggestive but not proof.

For the GENEB `NT splice acceptors` task, report:
- n_train, n_test, sequence length distribution, label balance
- **sorted SHA-256 of the sequence column**, train and test separately, so we can compare exactly
- the count of sequences shared with our set (send us the hash list if easier)

**If they are not the same data, say so plainly — we will pull the 0.9527-vs-0.685 comparison out of
`reports/PAPER_DESIGN.md`, `RESEARCH_PLAN.md` and the student briefing.**

## T2 — Which hidden layer do the GENEB probes use?

Neither the GENEB README nor the paper states it. Read it out of the code and report:
- the layer index used for precomputed embeddings (final? fixed intermediate? swept?)
- the pooling (mean / CLS / other)

This decides comparability. Our layer-swept probes gain **+0.030 (NT-v2)** and **+0.040 (HyenaDNA)**
over reading `hidden_states[-1]`, which is larger than most margins in the GENEB table. If GENEB
fixes the final layer, our numbers are **not** comparable to their leaderboard and we must say so.

## T3 — Exact task identifiers for the 13 sentinel tasks

We only have display labels from your handoff. Send the **identifiers as they appear in the dataset**,
with the GENEB category each belongs to. Our current mapping, for you to confirm or correct:

| # | category | our label for it |
|--:|:--|:--|
| 1 | Histone modifications | NT H3 |
| 2 | Promoters | NT promoter_all |
| 3 | Enhancers | NT enhancers |
| 4 | DNA methylation | deep4mc A.thaliana 4mC |
| 5 | Splice sites | NT splice acceptors |
| 6 | lncRNA | lncrna g_max |
| 7 | Virus / phage | GUE phage_fragments |
| 8 | Regulatory elements | ensembl_regulatory |
| 9 | TF binding | GUE human_tf_0 |
| 10 | Chromatin accessibility | iDHS-EL DNase_I |
| 11 | Mouse enhancers | GUE mouse_0 |
| 12 | Coding / non-coding | coding_vs_intergenomic |
| 13 | Species classification | human_or_worm |

Also: **why these 13 specifically** — was it purely one-per-category, or was there a further
criterion (size, difficulty, prior results)?

## T4 — Push the artifacts you reported as pushed (they are not on any ref)

Your handoff §10 says these were *"committed and pushed on `viral-benchmark-continuation`"*. They are
**not present on any reachable ref** — `git ls-remote` shows only `main`,
`viral-benchmark-continuation` and `xiang/viral-benchmark-continuation-a1`, and none contains them:

- `scripts/geneb/` — especially `fair_kmer_sentinel.py` (defines task loading + the fair refit)
- `reports/geneb_sentinel_results.md`
- `reports/geneb_fair_kmer_sentinel_results.json`
- `scripts/viral_benchmark/virobench_kmer_predictions.py` (the per-example `.npz` emitter)

Plus two version pins we need in writing:
- the **git commit hash** of the GENEB clone at `/scratch/10906/arisk/GENEB`
- the **pinned dataset version** from `benchmark/benchmark_spec.json` (HF `darlednik/geneb-tasks`)

## T5 — Dev split: confirm what you used

The public GENEB ships **`split` ∈ {train, test} only — no dev/validation split**. So the fair-k-mer
refit's C sweep must have used a carve from train. Confirm:
- how dev was carved (fraction, seed, random or group-disjoint)
- whether the same carve was used for the frozen probes
- confirm test was never touched for any selection

If the carve was random on a positionally-structured task, flag it — that is the same protocol debt
we currently have live on our splice runs, and it has to be identical across k-mer / CNN / probe or
the three are not comparable.

## T6 — Two corrections from Cluster 1 to fold into your reports

1. **Your §4 NTv3 splice gap of +0.598 / +0.352 / +0.308 is a receptive-field artifact.** Your
   baselines (0.373 / 0.619 / 0.676) reproduce our *incumbent* CNN (0.354 / 0.613 / 0.669). A
   13-cell architecture × capacity ladder reaches **0.9528 / 0.9527 / 0.9637** — the real gap is
   **+0.02 to +0.03**. Mechanism: ResNet at 9.44M params with RF 89bp scores 0.336 on a 600bp input;
   a U-Net at 0.26M with global RF scores 0.951. Please retract the +0.598 framing before it goes in
   a draft.
2. **Report the GENEB sentinel per model, not best-of-3.** Best-of-3 is a max-over-models statistic —
   the same bias you flag in your own §1.2. Per model (±0.005 tie band): GENA-LM 11W/0T/2L
   (mean +0.083), NT-v2 10W/1T/2L (+0.078), HyenaDNA 6W/1T/6L (**−0.026**). It survives for 2 of 3
   models, so the claim holds — but it is model-dependent, and HyenaDNA is a coin flip.

---

**Priority: T1 > T2 > T4 > T3 > T5 > T6.** T1 alone decides whether our top-priority experiment is
even well-posed.
