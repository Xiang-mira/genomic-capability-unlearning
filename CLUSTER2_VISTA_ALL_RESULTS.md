# Cluster 2 (Vista, 8× GH200) — complete results handoff

2026-08-23. Everything Cluster 2 has run, in one document. Supersedes
`CLUSTER_HANDOFF_FROM_VISTA.md` (earlier, partial). All raw artifacts under
`/scratch/10906/arisk/genomic_unlearning_pc/` and `/scratch/10906/arisk/GENEB/`;
reports committed to `reports/` on `viral-benchmark-continuation`.

**Read §1 first — it contains three corrections that change conclusions on both clusters.**

---

## 1. Corrections that affect Cluster 1's numbers too

### 1.1 Single-seed "wins" do not survive 3-seed averaging (HVUE)
Your scoreboard lists *"HVUE identity-disjoint hsd0: 8 losses, 1 win of +0.0012"*. We rebuilt
hsd0 from scratch and got a near-identical single-seed result (NT-v2 Host_Tropism **+0.0023**).
Extending to seeds 42/43/44 collapses it to **+0.0003 — a dead tie**. If your +0.0012 is
single-seed, it very likely does not survive either. **Every one of our 9 (model × task) cells on
HVUE is now a loss or a tie; zero wins.** Recommend re-running with `--seeds 42 43 44`.

### 1.2 `max()` over seeds vs a single published number is an optimistic-selection bias
`aggregate_positive_control.py` originally took `max()` over 3 CNN seeds before comparing against
single-point competitor numbers. That artifact alone manufactured the *only* two "baseline beats
FM" rows in a 30-row table (GUE Core Prom. TATA, TF Human 3). Both flip to ties/FM-wins under the
mean. Fixed; the aggregator now reports `n_seeds` and `cnn_seed_std` per row so no cell can be
silently n=1 again. **Guardrail #6 in your own doc — worth auditing anywhere it wasn't applied.**

### 1.3 GENEB's own reference k-mer baseline is miscalibrated
GENEB ships `ExampleKmerExtractor` + `LogisticRegression(max_iter=1000)` with **no feature
scaling, no C tuning, no class weighting**. On `iDHS-EL_DNase_I` this produces a **degenerate
MCC = 0.000** — majority-class prediction 100% of the time, including on its own training set
(72.4% train acc = exactly the class prior). Not a data bug (256/256 dims have nonzero variance,
no NaNs). Refit fairly (StandardScaler, C swept 0.001–10 on a dev split carved from train,
`class_weight="balanced"`): **MCC 0.000 → 0.589**. Any GENEB analysis using their reference k-mer
as "the baseline" is comparing against a broken baseline. Script:
`scripts/geneb/fair_kmer_sentinel.py`.

---

## 2. C2-KRAKEN-001 — Kraken2 + BLAST on ViroBench (P0, **the load-bearing baseline**)

Your doc: *"Kraken2 … will tell us whether our k-mer's 0.570 is a real ceiling or an artifact of
our implementation."* **Answer: partly our method — there IS modest headroom above our k-mer on
macro-F1, though our k-mer wins on accuracy. And Kraken2 carries a large reference-leakage
advantage that must be stratified out before any comparison is meaningful.**

ViroBench `--mod ALL --split times --level family`, 173 classes. All three methods scored on
**exactly the same examples** (k-mer's family-filtered test set, n=5,505), stratified by whether
the test taxid is present verbatim in Kraken2's RefSeq viral reference DB.

**⚠ Methodological note:** macro-F1 is **not** comparable across different example subsets — the
class set changes, which changes the macro denominator. An earlier version of this analysis
compared Kraken2 on 4,931 examples against k-mer on 4,667 and wrongly concluded k-mer beat
Kraken2 on the clean subset. On matched examples it does not. Only the matched table below is
valid.

| subset | n | families | method | macro-F1 | accuracy |
|:--|--:|--:|:--|--:|--:|
| **LEAKED** (taxid in Kraken2 DB, 15.2%) | 838 | 107 | k-mer3-6 (train-only) | 0.5997 | 0.7888 |
| | | | **Kraken2 (RefSeq DB)** | **0.9282** | **0.9809** |
| | | | BLASTn (train-only) | 0.5951 | 0.6313 |
| **CLEAN** (taxid not in DB, 84.8%) | 4,667 | 159 | k-mer3-6 (train-only) | 0.5654 | **0.8528** |
| | | | **Kraken2 (RefSeq DB)** | **0.6269** | 0.8119 |
| | | | BLASTn (train-only) | 0.6190 | 0.7851 |

Two separate findings, both real:

**(a) Kraken2's reference leakage is large and must be disclosed.** 838 / 5,505 (15.2%) of
ViroBench test taxids sit verbatim in Kraken2's RefSeq viral DB. On those, Kraken2 scores 0.928
macro-F1 / 0.981 accuracy — essentially a lookup, not classification. On the clean 84.8% it drops
to 0.627 / 0.812. Our k-mer and BLAST only ever see the ViroBench training split, so any
un-stratified comparison against Kraken2 is unfair to them. (BLAST shows no such pattern —
0.595 leaked vs 0.619 clean — confirming it as a fair train-only comparator.)

**(b) Even on the clean subset, both alignment methods beat our k-mer on macro-F1** — Kraken2
0.627 and BLAST 0.619 vs k-mer 0.565 (+0.06 / +0.05) — **while our k-mer beats both on accuracy**
(0.853 vs 0.812 / 0.785). This is interpretable, not contradictory: macro-F1 weights every family
equally, and a single close reference match resolves a *rare* family that k-mer has almost no
training signal for; accuracy is dominated by *abundant* families, where k-mer's learned
composition signal is stronger. So the honest answer to your question is that our k-mer is not
the ceiling on macro-F1 (reference methods add ~0.06), but it is not badly implemented either,
and it is the stronger method on accuracy.

Full-test (unstratified, for reference only): k-mer 0.5703 macro-F1 / 0.8431 acc; Kraken2 0.6403 /
0.8273 (n=5,832); BLASTn 0.4235 / 0.7190 (n=5,832). ViroBench published, EXTERNAL: BLAST 0.477,
LucaVirus 0.759 — our BLAST is in their ballpark, an independent protocol sanity check.

**Split integrity:** the temporal split is genuinely clean (train ≤ 2017-10-21, test ≥ 2020-02-03,
zero date overlap — verified), so (a) is purely Kraken2's *external reference* advantage, not a
flaw in the ViroBench split.

Artifacts: `virobench_blast_results.tsv`, `virobench_kraken2_results.tsv`,
`ALL_times_family__kmer36_predictions.npz` (per-example), `kraken_viral/` DB.
**aarch64 gotcha:** kraken2 `build_db` needs `OMP_NUM_THREADS=1 --threads 1` (see §9).

## 3. C2-GENEB-SENTINEL-002 — 13-task sentinel, 4 models

One task per category, full regime, 5 seeds (GENEB protocol). Our 3 project gLMs + k-mer, vs the
~40 published submissions already shipped in the GENEB repo. Full table:
`reports/geneb_sentinel_results.md`.

| task (category) | naive kmer (GENEB ref) | **fair kmer** | NT-v2 | HyenaDNA | GENA-LM | best published |
|:--|--:|--:|--:|--:|--:|--:|
| NT H3 (Histone) | 0.602 | 0.590 | 0.662 | 0.671 | 0.705 | 0.781 |
| NT promoter_all | 0.754 | 0.813 | 0.882 | 0.835 | 0.917 | 0.930 |
| NT enhancers | 0.456 | 0.425 | 0.396 | 0.485 | 0.463 | 0.526 |
| deep4mc A.thaliana 4mC | 0.202 | 0.204 | 0.331 | 0.071 | 0.185 | 0.402 |
| NT splice acceptors | 0.269 | 0.387 | 0.479 | 0.402 | 0.543 | 0.685 |
| lncrna g_max | 0.111 | 0.155 | 0.233 | 0.156 | 0.225 | 0.475 |
| GUE mouse_0 | 0.437 | 0.437 | 0.378 | 0.156 | 0.464 | 0.667 |
| GUE human_tf_0 | 0.611 | 0.537 | 0.576 | 0.563 | 0.672 | 0.690 |
| human_or_worm | 0.815 | 0.812 | 0.893 | 0.782 | 0.931 | 0.948 |
| ensembl_regulatory | 0.348 | 0.289 | 0.526 | 0.555 | 0.526 | 0.597 |
| GUE phage_fragments | 0.512 | 0.604 | 0.854 | 0.479 | 0.659 | 0.950 |
| coding_vs_intergenomic | 0.706 | 0.734 | 0.780 | 0.677 | 0.853 | 0.904 |
| iDHS-EL DNase_I | **0.000** | **0.589** | 0.593 | 0.413 | 0.509 | 0.728 |

**Against the *fair* k-mer, best-of-our-3-gLMs wins 13/13.** Under a frozen-probe protocol, across
13 independently-chosen categories. This is the broadest positive-control evidence either cluster
has produced — wider than the NTv3 splice result, which is a single task family.

Our 3 models still trail the best-of-40 published (~0.10–0.20 MCC) — expected, several of those
are much larger or task-specialized (GenomeOcean-4B, Enformer, GENERator-3B, LucaOne).
**Note:** "best published" is a max-over-40 oracle statistic, not one model beating us
consistently — report median-of-40 alongside it in any full run (your guardrail).

Not done: 87 remaining tasks, 10-shot/1-shot regimes.

---

## 4. Positive controls on GUE / NTv3 / EPI (42 task-splits)

`reports/positive_control_comparison.md`, `reports/epi_comparison.md`. Baseline =
max(k-mer3-5, k-mer3-6, **mean-over-3-seeds** CNN).

**NTv3 splice = the clean fine-tuning positive control** (chromosome-disjoint, 0% overlap verified):

| task | our baseline MCC | best published | gap |
|:--|--:|--:|--:|
| NT Splice All | 0.373 | 0.971 (NTv2) | **+0.598** |
| NT Splice Acceptor | 0.619 | 0.971 (GJ-B) | **+0.352** |
| NT Splice Donor | 0.676 | 0.984 (GJ-B) | **+0.308** |

Everything else is small: mean gap GUE **0.055**, NT **0.140**; 11/30 tasks have the baseline
within 0.05 MCC of the best published model.

**EPI (6 cell lines) — our baseline beats published pretrained-embedding methods on 5/6**
(EPIPDLF/EPINTLM), margins +0.007…+0.052 AUROC. Cause measured directly, not assumed: MMseqs2
≥90% identity shows **promoters 67–80% and enhancers 39–46% train→test overlap** in every cell
line. Independently corroborates BENGI / LOCO-EPI. Exact-duplicate matching *understated* this
(42–62% promoter) — the identity-clustering number is the honest one.

---

## 5. HVUE core — rebuilt and independently verified (was thought unreachable)

Data was **not** actually missing: a local clone at `/work/10906/arisk/ls6/evo-locking/data/hvue/`
has it (Host_Tropism 47,194 / Pathogenecity 134,066 / Transmissibility 458,756 rows, **every
sequence exactly 1000 bp** — matches your context audit). `paths.py`'s `/home/nvidia/...` defaults
were the only thing broken.

Rebuilt `identity_disjoint_hsd0` from scratch (MMseqs2 90%, no gate). Reproduces your numbers:

| task | our k-mer | HANDOFF | our CNN (3 seeds, best LR) | HANDOFF CNN |
|:--|--:|--:|--:|--:|
| Host_Tropism | 0.9171 | 0.9131 | 0.9491 | 0.9482 |
| Pathogenecity | 0.9558 | (gate-failed) | 0.9715 | 0.9667 |
| Transmissibility | 0.9224 | (gate-failed) | 0.9340 | 0.9202 |

gLM excess vs best(k-mer, CNN), **mean over seeds**:

| task | NT-v2 (3 seeds) | HyenaDNA (3 seeds) | GENA-LM (1 seed) |
|:--|--:|--:|--:|
| Host_Tropism | **+0.0003 (tie)** | −0.0119 | −0.0106 |
| Pathogenecity | −0.0283 | −0.0164 | −0.0074 |
| Transmissibility | −0.0325 | −0.0373 | −0.0177 |

---

## 6. ViroBench — full P3 spec + the matched-context correction

**We confirm your Part A finding.** Our earlier claim that HyenaDNA's +0.236 over the CNN at 20 kb
showed "real capability masked by context" was **wrong** — we never ran the k-mer at matched
context. Your matched-context table closes it; `reports/virobench_glm_comparison.md` now marks
that reading **retracted**. No disagreement.

Full spec (`--mod ALL --min_count 1`, all 5 levels, `times`), k-mer3-6 whole-genome vs CNN 20 kb,
3 seeds — `reports/virobench_full_spec_baselines.md`:

| level | classes | k-mer3-6 | CNN |
|:--|--:|--:|--:|
| kingdom | 18 | 0.560 | 0.294 |
| phylum | 28 | 0.555 | 0.325 |
| class | 45 | 0.520 | 0.327 |
| order | 67 | 0.599 | 0.238 |
| family | 173 | 0.570 | 0.208 |

This removes the `min_count≥10` / DNA-only filter that made the earlier numbers explicitly
non-comparable to ViroBench's published 47.67 / 75.88. **Still unverified:** which taxonomic level
their headline figures refer to — treat the cross-paper comparison as tentative until checked.

gLMs on DNA/family/times (earlier subset, `reports/virobench_glm_comparison.md`): NT-v2 0.887
(3 seeds, ±0.008), HyenaDNA 0.878, GENA-LM 0.638, all below whole-genome k-mer 0.948. Frozen
probes collapse (0.049–0.486).

---

## 7. Known gaps / pending

- §2 is now complete (matched-subset stratification done). Script:
  `scripts/viral_benchmark/virobench_kmer_predictions.py` — also emits per-example `.npz`, which
  `virobench_baselines.py` does **not**; worth adding on your side per the reproducibility spec,
  since the matched-subset analysis in §2 is impossible without per-example predictions.
- GENEB: 87/100 tasks, and 10-shot/1-shot regimes, not run.
- C2-GENEB-KMER-003 / C2-GENEB-LEAK-004 not started (LEAK-004 should use `-c 0.3` per your note).
- C2-VIROBENCH-LEVELS-005, C2-EVO-FULLFT-006 not started.
- ProteinGym (C2) and antibody escape (C3) remain yours — no local data here.

## 8. Bugs found in shared code (check if they bite you)

1. `build_identity_splits.py` — **missing `import paths as P` entirely**; instant `NameError`.
   Every sibling script has the shim. Fixed.
2. Same file — `df.groupby("label", group_keys=False).apply(lambda g: g.sample(...))` silently
   drops the `label` column under pandas ≥ 2.2 (`include_groups` default change) →
   `AttributeError` two lines later. Rewrote as an explicit per-group loop. Fixed.
3. `hvue_glm.py` — hardcoded `KMER[(task, split)]` dict has no `identity_disjoint_hsd*` entry, so
   it raises `KeyError` **after training completes**, discarding the result. Must pass
   `--kmer_json` with a flat `{"{task}__{split}": [value]}` file.
4. `run_baselines.sh` / `run_epi_baselines.sh` — skip-if-exists caches on *file existence*, not
   seed count. A 1-seed pilot silently masqueraded as a finished 3-seed run for two cells
   (including the largest gap in the whole table). Both re-run.
5. `virobench_baselines.py` — output filename omits `--kmer_cap`/`--cnn_len`, so context-ladder
   runs overwrite each other (you already flagged this; confirming it's real).

## 9. Environment notes (aarch64 / GH200)

- **mmseqs2**: needs `pixi workspace channel add https://conda.anaconda.org/bioconda/` first.
- **kraken2 on aarch64**: `--threads 16` crashes in `build_db` (`OMP only wants you to use 1
  threads` → `xargs: cat: terminated by signal 13`). **`OMP_NUM_THREADS=1 --threads 1` works**
  (build took 1m29s). Also: `--download-library`/`--download-taxonomy` need `--use-ftp`; rsync is
  blocked from compute nodes (HTTPS/FTP fine).
- **transformers**: must pin `<5` (NT-v2/HyenaDNA/GENA-LM custom modeling code breaks:
  `ImportError: find_pruneable_heads_and_indices`), which needs `huggingface_hub<1.0`. This
  conflicts with `datasets`' conda-pinned `huggingface-hub==1.28.0` — we built a **separate** env
  at `/scratch/10906/arisk/virobench-glm-env` rather than break the shared one.
- **NT-v2 `modeling_esm.py`**: no gradient-checkpointing support (raises even though the base
  class exposes the method) + eager attention. Full-FT at 2048 tok / bs 8 OOMs a 96 GB GPU; we
  capped to 1024 tok. Your LoRA-at-2048 attempt is the better fix.
- **Vista topology**: 8 nodes × 1 GPU, *not* 8 GPUs on one node. `cuda:1..7` don't exist locally;
  dispatch with `srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --ntasks=1 -w <node>`.

## 10. Repo state

Committed and pushed on `viral-benchmark-continuation` (auto-commit is active here, so the tree
should already match origin). New this round: `reports/geneb_sentinel_results.md`,
`reports/geneb_fair_kmer_sentinel_results.json`, `reports/virobench_full_spec_baselines.md`,
`reports/hvue_real_data_verification.md`, `reports/mmseqs_leakage_check.csv`,
`scripts/geneb/`, `scripts/viral_benchmark/virobench_glm.py`,
`scripts/viral_benchmark/virobench_kmer_predictions.py`, `scripts/positive_control/*`.
GENEB clone lives at `/scratch/10906/arisk/GENEB` (separate upstream remote, not pushed there).
