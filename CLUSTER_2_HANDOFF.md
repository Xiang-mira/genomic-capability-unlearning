# Cluster 2 handoff — executable task list

2026-08-22. Cluster 1 = 8× A100-80GB (this repo's host). Cluster 2 = your 8× A100.
Every command below was run at least once on Cluster 1; none are placeholders.
**Nothing here overlaps with what Cluster 1 is running** (see §0).

## 0. What Cluster 1 has taken — do not duplicate

| GPU | job | ETA |
|:--|:--|:--|
| 0,1,2 | `virobench_frozen_probe.py --model lucavirus` at W=2048/1024/512 | ~3.5 h each |
| 3,4,5 | same frozen probe for nt_v2_500m / hyenadna / gena_lm, W=2048 then 1024 | ~7 h each |
| 6,7 | `virobench_glm.py --mod ALL` full-FT for nt_v2_500m / hyenadna (already in flight) | ~6 h |

Cluster 1 also owns: ProteinGym supervised (C2 in the old doc — data is local here),
antibody escape completion (data local here), and the splice positive control.

---

## 1. Environment (verified on Cluster 1)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r scripts/viral_benchmark/requirements.txt   # torch>=2.4, transformers>=4.48,<5, peft, sklearn, pyarrow
export VB_ROOT=/scratch/$USER/viral-bench
export VB_OUT=$VB_ROOT/results
export VB_VIRO_DIR=$VB_OUT/virobench
export VB_GUE_DIR=$VB_OUT/gue_viral
export VB_MMSEQS=$(which mmseqs)
export HF_HOME=$VB_ROOT/hf_cache
cd scripts/viral_benchmark
python download_data.py --what both      # GUE viral + ViroBench, ~3.7 GB
```
Known env traps: `transformers>=5` breaks NT-v2/HyenaDNA/GENA-LM custom modeling code
(`ImportError: find_pruneable_heads_and_indices`) — pin `<5`. `huggingface_hub<1.0`.
mmseqs2 on aarch64 needs the bioconda channel added first.

Launch pattern (bare `nohup &` gets SIGTERM'd when the parent exits):
```bash
setsid env CUDA_VISIBLE_DEVICES=$G nohup python -u <script> <args> > $LOG 2>&1 </dev/null & disown
```
Or use the sequential queue runner: `scratchpad/multimodel/queues/runner.sh <gpu> <ABSOLUTE-queue-file>`.

---

## 2. Task list

### C2-KRAKEN-001 — Kraken2 + BLAST baselines on ViroBench taxonomy
**Purpose:** Kraken2 is a k-mer classifier by design and is ViroBench's own comparator. It tells us
whether our k-mer's 0.570 is a real ceiling or an artifact of our implementation. This is the single
most load-bearing missing baseline.
**Priority:** P0 · **GPU:** CPU-only (16+ cores, ~100 GB disk for the viral DB) · **ETA:** 6–10 h
· **GPU-hours:** 0
```bash
# install
conda install -c bioconda kraken2 blast
# build viral DB
kraken2-build --download-library viral --db $VB_OUT/kraken_viral --threads 16
kraken2-build --build --db $VB_OUT/kraken_viral --threads 16
```
Then classify `$VB_VIRO_DIR/ALL_taxon_times__{train,test}_seq.jsonl` genomes, map Kraken2's
assigned taxid to the `family` label via the CSV's taxonomy columns, and score macro-F1 over the
**same 173-class filtered label set** Cluster 1 uses (`--mod ALL --split times --level family
--min_count 1`). Same for BLASTn nearest-hit label transfer against the train set.
**Success:** `reports/virobench_alignment_baselines.md` with macro-F1, micro-F1, accuracy, MCC and
per-class F1 for Kraken2 and BLASTn, plus per-example predictions as `.npz`.
**Depends on:** nothing.

### C2-GENEB-SENTINEL-002 — reproduce a GENEB sentinel subset
**Purpose:** before any cross-paper GENEB claim, verify our probe protocol reproduces their
published frozen-probe numbers. Phase 6 of the master plan.
**Priority:** P0 · **GPU:** 8 independent 1-GPU jobs · **ETA:** ~12 h wall · **GPU-hours:** ~60
Pick 5–8 models × 8–15 tasks spanning architecture, size, task category and difficulty.
Use `scripts/viral_benchmark/virobench_frozen_probe.py` as the reference implementation of the
protocol (frozen mean-pooled embeddings → standardised LR head, `C` selected on dev only, test
scored once, per-example predictions saved). Port its `embed()`/`agg_predict()` to GENEB's loader.
**Critical:** mirror GENEB's exact standardisation, LR hyperparameters, shot regime and seed
convention. Deviations are the finding, so record them rather than smoothing them over.
**Success:** table of ours-vs-published per (model, task) with deviation; `reports/geneb_sentinel_reproduction.md`.
**Depends on:** nothing. **If deviation >0.05 on >20% of cells:** stop and report before scaling.

### C2-GENEB-KMER-003 — k-mer probe sweep across GENEB tasks
**Purpose:** the primary GENEB analysis — `FM frozen embedding → LR` vs `k-mer → the same LR
protocol`. Yields `A_t = best_FM_t − kmer_t` per task.
**Priority:** P1 · **GPU:** CPU-heavy, 1 GPU optional · **ETA:** ~1 day · **GPU-hours:** <5
Reuse the k-mer featuriser in `virobench_baselines.py` (`kmer_feats`, vectorised base-4, k=3–6).
`C` **must** be validation-selected — GENEB's convention, not test.
**Success:** per-task and per-category tables; distribution of FM−k-mer gaps; fraction of models
beating k-mer and beating it by δ ∈ {0.01,0.02,0.03,0.05}; label the max-over-40-models statistic
explicitly as **oracle/best-observed**, and report median and fraction-above-baseline alongside.
**Depends on:** C2-GENEB-SENTINEL-002 passing.

### C2-GENEB-LEAK-004 — GENEB split-relatedness audit
**Purpose:** Phase 7. Test whether large apparent FM gains concentrate in high-overlap tasks.
**Priority:** P1 · **GPU:** CPU-only · **ETA:** ~8 h · **GPU-hours:** 0
Start with exact duplicates, then MMseqs2 ≥90%. Use `scripts/positive_control/mmseqs_leakage_check.py`
as the template — but **measure with `-c 0.3`, not `-c 0.9`**. Cluster 1 proved `-c 0.9` is blind to
50% overlap (0/150 caught in a direct test), and that our own HVUE "identity-disjoint" splits retain
33–82% of test sequences with a ≥90%-identity match over ≥50% of their length. Choose the
biologically meaningful disjointness per task family; do not apply one definition indiscriminately.
**Success:** `reports/geneb_leakage.csv` + a scatter of FM advantage vs train/test relatedness.

### C2-VIROBENCH-LEVELS-005 — gLM frozen probes at all 5 taxonomic levels
**Purpose:** Cluster 1 is only doing `family`. ViroBench's published figure may be a 5-level mean,
so the other four levels are needed to compare on their aggregation.
**Priority:** P2 · **GPU:** 4 independent 1-GPU · **ETA:** ~10 h · **GPU-hours:** ~40
```bash
for LVL in order class phylum kingdom; do
  python -u virobench_frozen_probe.py --model lucavirus --mod ALL --split times \
      --level $LVL --min_count 1 --window 2048 --max_win 16 --bs 16
done
```
**Depends on:** Cluster 1's family-level result landing first (so the protocol is confirmed).

### C2-EVO-FULLFT-006 — Evo full fine-tuning on HVUE
**Priority:** P3 · **GPU:** 4–8×A100 FSDP · **ETA:** ~1 day · **GPU-hours:** ~100
Only worth it for regime completeness. Evo LoRA at lr 3e-4 already reaches 0.8742 (vs its
published 0.8173), so Evo was under-tuned, not weak. Low evidentiary value — do last, or skip.

---

## 3. Allocation table

| Cluster | GPU(s) | Task | ETA | GPU-h | Dependency | Priority |
|:--|:--|:--|--:|--:|:--|:--|
| C1 | 0,1,2 | LucaVirus frozen probe, 3 windows | 3.5 h | 10 | — | P0 |
| C1 | 3,4,5 | NT-v2 / HyenaDNA / GENA-LM frozen, 2 windows | 7 h | 42 | — | P0 |
| C1 | 6,7 | ViroBench full-FT (in flight) | 6 h | 12 | — | P1 |
| C1 | next | splice positive control (own pipeline) | 4 h | 8 | — | P0 |
| C1 | next | deduped virus_covid rerun | 3 h | 6 | dedup done | P1 |
| C1 | next | ProteinGym supervised, 22 viral assays | 2 d | 40 | — | P1 |
| **C2** | **CPU** | **C2-KRAKEN-001** | **10 h** | **0** | — | **P0** |
| **C2** | **8×1** | **C2-GENEB-SENTINEL-002** | **12 h** | **60** | — | **P0** |
| C2 | CPU+1 | C2-GENEB-KMER-003 | 1 d | 5 | 002 | P1 |
| C2 | CPU | C2-GENEB-LEAK-004 | 8 h | 0 | — | P1 |
| C2 | 4×1 | C2-VIROBENCH-LEVELS-005 | 10 h | 40 | C1 family | P2 |
| C2 | 4–8 | C2-EVO-FULLFT-006 | 1 d | 100 | — | P3 |

Start **C2-KRAKEN-001 (CPU) and C2-GENEB-SENTINEL-002 (all 8 GPUs) simultaneously** — they don't
contend for the same resource. Critical path on Cluster 2 is GENEB sentinel → k-mer sweep, ~2 days.

---

## 4. Reproducibility requirements (both clusters)

Every run records: git commit + uncommitted diff, `pip freeze`, hostname, GPU type, CUDA version,
full command, config, seed, dataset checksum, split checksum, checkpoint id, start/end time, raw
per-example predictions, metrics, stdout/stderr. Timestamped run directory; **never overwrite**.

Known bug to avoid: `virobench_baselines.py`'s output filename omits `--kmer_cap`/`--cnn_len`, so
context-ladder runs silently overwrite each other. Add the cap to the filename before any sweep.

## 5. Result ingestion

Write to `reports/` and `$VB_OUT/<task>_results/`, commit to branch
`viral-benchmark-continuation`, push. Cluster 1 merges and folds into `ANALYTICAL_RESULTS.md`.
Flag any result that contradicts the current narrative immediately rather than at the end.

## 6. Scientific guardrails

1. Never select splits, baselines, thresholds or hyperparameters on the test set.
2. Do not write "no capability" — write "no detectable model-specific advantage over the evaluated
   comparator under this task/evaluation regime."
3. Keep frozen probing, LoRA, full FT and external published numbers strictly separate. Mark
   external cells **EXTERNAL/PUBLISHED**, never OUR RUN.
4. Report CIs and effect sizes, not just significance; preserve per-seed results.
5. Report effective context in bp for every row; cap baselines to match.
6. Use the seed **mean**, never best-of-N, against a single published number.
7. If LucaVirus or GENEB shows genuine FM advantage, report it as a positive result.
