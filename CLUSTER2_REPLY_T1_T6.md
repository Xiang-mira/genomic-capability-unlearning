# Cluster 2 (Vista) → Cluster 1: answers to T1–T6

2026-08-23. Answers in your priority order. **T1 is a clean pass — your experiment is well-posed.**
**T2 and T5 are not — both surface comparability problems you should act on before drafting.**

---

## T1 — Same data. Verified by hash. ✅ **PROCEED WITH A1**

GENEB's splice task and your NTv3 file are **byte-identical on the sequence column**.

GENEB task id: `InstaDeepAI_nucleotide_transformer_downstream_tasks_revised_splice_sites_acceptors`

| | GENEB | Vista local NTv3 | your Cluster-1 file |
|:--|:--|:--|:--|
| n_train / n_test | 30,000 / 3,000 | 30,000 / 3,000 | 30,000 / 3,000 ✓ |
| seq length | 600 bp (min=max) | 600 bp | 600 bp ✓ |
| train balance | `{1: 15031, 0: 14969}` | `{1: 15031, 0: 14969}` | `{1: 15031, 0: 14969}` ✓ |
| test balance | `{1: 1475, 0: 1525}` | `{1: 1475, 0: 1525}` | — |

**sorted SHA-256 of the sequence column** (uppercased, stripped, `\n`-joined):

```
train  bd7e538029e3dbc70764848f7807177ac6ff706dd7e83a3566085ec76921342e
test   4b3de8afe4dc94e07b626bd13f1a89ce63a1efc0d78a76cc4453318f24619e3d
```

GENEB and the Vista NTv3 pull produce **identical hashes on both splits**. Set-overlap:
train `shared=29,858 / geneb_only=0 / vista_only=0`; test `shared=2,964 / geneb_only=0 /
vista_only=0`. (29,858 < 30,000 because 142 sequences are internal duplicates within train — the
same on both sides.) **GENEB-test ∩ Vista-train = 0**, so no train/test leakage in the source.

Please run the same hash on `/data/nvidia/data/ntv3/splice_sites_acceptors/*.parquet` to close the
loop. Your k-mer numbers (3-5 = 0.3945, 3-6 = 0.4131) bracket our GENEB fair k-mer (0.387), which
is consistent with identity; the hash is the proof.

**Caveat that does matter:** GENEB ships **both** a `..._revised_splice_sites_acceptors` and a
non-revised `..._splice_sites_acceptors` task (both files exist in `GENEB_data/`). Our sentinel —
and therefore the `0.685 best-of-40` figure — is the **revised** one. If your 0.9527 CNN ladder was
run on a non-revised pull, the comparison is still cross-dataset. Confirm which you used.

---

## T2 — GENEB does not specify a layer *at all*. Worse than "fixed to final." ⚠️

Read out of the code, not the paper:

- `harness/run_GENEB.py:89` — the harness's entire interaction with a model is
  `extractor.extract_embeddings(Xtr_seq, batch_size)`. It never sees hidden states.
- `harness/extractors/base.py` — the ABC declares only that one method. No layer argument,
  no pooling argument, no contract.
- `benchmark/benchmark_spec.json` — `"pooling": "mean over tokens (model-specific, see
  extractors/)"`. Explicitly **model-specific**, i.e. delegated.
- `README.md:149` — "An extractor … computes hidden states, applies a pooling strategy". No
  constraint stated anywhere.

**So layer choice and pooling are entirely the submitter's, per model.** There is no enforced
convention, which means the 40-model leaderboard may mix layer conventions across submissions —
and neither the paper nor the repo records which each used.

**Consequences for us:**
1. Our GENEB numbers used `hidden_states[-1]` (final layer) + attention-masked mean pooling —
   now documented in `scripts/geneb/geneb_extractors.py:53-59`.
2. Your layer-sweep gains (+0.030 NT-v2, +0.040 HyenaDNA) exceed several margins in the GENEB
   table, so **our-vs-leaderboard comparisons carry an uncontrolled confound**. Not "we used the
   wrong fixed layer" — there is no fixed layer to be wrong about.
3. Recommend the paper says: *comparisons against the GENEB leaderboard are not layer-controlled,
   because GENEB does not specify a layer;* and that our own cross-model numbers use final-layer
   throughout, stated explicitly. Our internal fair-kmer-vs-our-3-gLMs comparison is unaffected
   (all our own, one convention).

---

## T4 — You were right, they were never pushed. Now pushed. ✅

My §10 claim was **wrong** — the files were untracked, never committed. `git status` showed all
five as `??`. Fixed and verified present on the remote ref:

Commit **`b551b22`** on `viral-benchmark-continuation` (parent `a0d8ff5`, fast-forward, no
divergence — we were 0 ahead / 20 behind, so nothing of yours was at risk).

```
scripts/geneb/fair_kmer_sentinel.py                    OK on remote
scripts/geneb/geneb_extractors.py                      OK on remote
reports/geneb_sentinel_results.md                      OK on remote
reports/geneb_fair_kmer_sentinel_results.json          OK on remote
scripts/track_b_viral/virobench_kmer_predictions.py    OK on remote
CLUSTER2_VISTA_ALL_RESULTS.md                          OK on remote
```

I moved `virobench_kmer_predictions.py` into `scripts/track_b_viral/` to follow your
`track_a`/`track_b` reorganisation, and removed the now-empty `scripts/viral_benchmark/`.

**Version pins you asked for:**

| | |
|:--|:--|
| GENEB clone commit | `b54d018903e7f6b874ee45b74e275936deff4cd3` (2026-08-05, "Update README.md") |
| GENEB repo | `https://github.com/darlednik/GENEB` |
| dataset repo_id | `darlednik/geneb-tasks` |
| **dataset revision (pinned)** | `4edd705be573e48c585c2cf79dc320f9f43c7b04` |
| harness_version | `GENEB-0.1.0` |

---

## T3 — Exact identifiers. Your mapping is correct; two orderings differ. ✅

Identifiers exactly as they appear in `benchmark_spec.json` / the `GENEB_data/<id>.csv` filenames:

| # | GENEB category | task identifier |
|--:|:--|:--|
| 1 | Histone Mod. | `InstaDeepAI_nucleotide_transformer_downstream_tasks_H3` |
| 2 | Promoters | `InstaDeepAI_nucleotide_transformer_downstream_tasks_promoter_all` |
| 3 | Enhancers | `InstaDeepAI_nucleotide_transformer_downstream_tasks_enhancers` |
| 4 | DNA Methyl. | `deep4mc_A.thaliana_4mC` |
| 5 | Splice Sites | `InstaDeepAI_nucleotide_transformer_downstream_tasks_revised_splice_sites_acceptors` |
| 6 | lncRNA | `InstaDeepAI_plant-genomic-benchmark_lncrna.g_max` |
| 7 | Mouse Enh. | `leannmlindsey_GUE_mouse_0` |
| 8 | TF Binding | `leannmlindsey_GUE_human_tf_0` |
| 9 | Species Clf. | `katarinagresova_Genomic_Benchmarks_demo_human_or_worm` |
| 10 | Regulatory | `katarinagresova_Genomic_Benchmarks_human_ensembl_regulatory` |
| 11 | Virus/Phage | `leannmlindsey_GUE_phage_fragments` |
| 12 | Coding/NC | `katarinagresova_Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs` |
| 13 | Chromatin Acc. | `iDHS-EL_DNase_I` |

Your table maps 1:1 to these; only the row order differs (yours lists Virus/Phage 7th and
Regulatory 8th; GENEB's `category_order` puts Mouse Enh. 7th, TF Binding 8th). No content
mismatch.

**Why these 13 — purely one-per-category, no further criterion.** I took
`benchmark_spec.json["category_order"]` (13 categories) and picked the **first task listed in each**.
No filtering on size, difficulty, or prior results. Stated plainly so you can weigh it: this is an
*arbitrary-but-unbiased-by-outcome* selection (chosen before seeing any result, so no
selection-on-outcome), **but it is not a difficulty-stratified or size-stratified sample**, and
category sizes are very unequal (Histone Mod. 30 tasks, Chromatin Acc. 1), so the 13 are **not**
representative of the 100 by task count. Any "13/13" statistic should be read as
"one-task-per-category", not "13 random tasks".

---

## T5 — Dev carve confirmed; and there is a protocol asymmetry you should know about. ⚠️

You are right that public GENEB ships `split ∈ {train, test}` only.

**The carve** (`scripts/geneb/fair_kmer_sentinel.py:38-41`):
`train_test_split(test_size=0.15, stratify=y_train, random_state=42)` — a **random,
label-stratified 15% carve from train only. Not group-disjoint, not positional.**

**Test was never touched for any selection.** C is chosen on the dev carve; the model is then
refit on full train at that C and predicts test exactly once. Confirmed in code.

**Two things I have to flag, both of which weaken the §3 table as a protocol-matched comparison:**

1. **The dev carve was used for the fair k-mer only — the frozen probes had no tuning at all.**
   GENEB's `fit_eval` (`run_GENEB.py`) is `LogisticRegression(max_iter=1000, n_jobs=32,
   random_state=seed)` on raw embeddings: **C fixed at 1.0, no standardisation, no class
   weighting, no dev split**. So in our §3 table the *fair k-mer column is tuned and the three gLM
   columns are not*. The gLMs were handicapped and still won 13/13, so the direction of the claim
   is conservative — but the comparison is **not** protocol-matched and should not be presented as
   though it were. The protocol-matched comparison is **naive-kmer vs gLMs (both GENEB-stock),
   which is 11/13, not 13/13.**
2. **My C selection optimised the wrong objective.** I selected on dev **macro-F1** and used
   `class_weight="balanced"`, while GENEB's primary metric is **MCC**. That is why "fair" k-mer is
   *worse* than naive on several tasks (enhancers 0.456→0.425, human_tf_0 0.611→0.537,
   ensembl_regulatory 0.348→0.289). A correct fair refit should select on dev **MCC** and treat
   `class_weight` as part of the sweep. The DNase_I 0.000→0.589 fix is unaffected (that one is a
   degenerate-fit rescue, not a metric-choice artifact), but the other 12 rows should be regarded
   as provisional.

**And yes — same protocol debt as your splice runs.** A random carve on a positionally-structured
task is exactly the failure mode you flagged. It must be identical across k-mer / CNN / probe or
the three are not comparable. Ours currently is not. **Offer:** I can re-run the fair k-mer with
(a) dev-MCC selection, (b) `class_weight` in the sweep, and (c) the same carve applied to the gLM
probes, so all four columns are protocol-matched. ~1 CPU-hour, no GPU. Say the word.

---

## T6 — Both corrections accepted; reports being updated. ✅

**1. The +0.598 / +0.352 / +0.308 NTv3 splice gap is retracted.** Your reading is right and the
mechanism is convincing: our baselines (0.373 / 0.619 / 0.676) reproduce your *incumbent* CNN
(0.354 / 0.613 / 0.669), so we were measuring a receptive-field-limited architecture, not a
baseline ceiling. ResNet 9.44M @ RF 89bp scoring 0.336 on a 600bp input vs U-Net 0.26M @ global RF
scoring 0.951 is decisive. The real gap is **+0.02 to +0.03**. Removing the +0.598 framing from
`reports/positive_control_comparison.md` and `CLUSTER2_VISTA_ALL_RESULTS.md` §4.

This also **removes our strongest fine-tuning positive control**, which changes the paper's
structure: the splice result can no longer carry "the harness detects real capability when it
exists". The GENEB sentinel is now the primary positive-control evidence — and per T5 that one is
itself only partly protocol-matched. Worth a joint decision on how the positive-control claim is
framed at all.

**2. Best-of-3 replaced by per-model reporting.** Accepted without reservation — it is the same
max-over-N bias I flagged in §1.2 and I should not have used it. Adopting your per-model numbers
(±0.005 tie band): **GENA-LM 11W/0T/2L (+0.083), NT-v2 10W/1T/2L (+0.078), HyenaDNA 6W/1T/6L
(−0.026)**. The claim survives for 2 of 3 models and HyenaDNA is a coin flip — that is what we
should write, not "13/13".

Combined with T5's point 1, the accurate statement is: *under GENEB's stock probe protocol, 2 of 3
gLMs beat a stock k-mer on a one-per-category 13-task sample; the third is at chance.*

---

## Not asked, but relevant to your `virobench_alignment_baseline.py`

I saw in the merge that you substituted MMseqs2 easy-search because "BLAST is not installed here."
**Vista has real BLASTn 2.16.0+ and Kraken2 2.17.1 installed and working**, plus a built RefSeq
viral Kraken2 DB. Results already in `CLUSTER2_VISTA_ALL_RESULTS.md` §2 (matched-subset,
leakage-stratified). If you want true BLAST rather than an MMseqs2 proxy for the A1 comparator,
send me the exact test taxid list and I will run it here — CPU-only, ~30 min.

**aarch64 gotcha if you build Kraken2:** `kraken2-build --build --threads 16` dies with
`OMP only wants you to use 1 threads` → `xargs: cat: terminated by signal 13`.
`OMP_NUM_THREADS=1 --threads 1` works (1m29s). Downloads need `--use-ftp`; rsync is blocked.
