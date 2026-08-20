# Split of work: what this cluster is running, what the other cluster should run

2026-08-20. 8× A100-80GB here, fully allocated. Everything below uses
`scripts/viral_benchmark/` (path-portable via `VB_*` env vars — see README).

---

## PART A — The finding that reframes the ViroBench result

The other cluster's `reports/virobench_glm_comparison.md` concluded that on ViroBench taxonomy
"the binding constraint on this task is context length, not model capability", based on
HyenaDNA (0.8784 @ 20 kb) beating the CNN (0.6421 @ 20 kb) by +0.236 while both lost to a
whole-genome k-mer (0.9475 @ 43.5 kb).

**That comparison omitted the k-mer at matched context.** I ran it:

| effective context | k-mer3-5 (capped) | best gLM at that context | winner |
|--:|--:|:--|:--|
| 3.1 kb | **0.8469** | GENA-LM 0.6378 | **k-mer +0.209** |
| 6.1 kb | 0.8662 | NT-v2 **0.8867** | NT-v2 +0.021 |
| 20 kb | **0.9297** | HyenaDNA 0.8784 | **k-mer +0.051** |
| whole 43.5 kb | **0.9475** | — | — |

The k-mer scales with context the same way the models do. It wins at 3.1 kb and 20 kb and loses
only narrowly at 6.1 kb. **So the k-mer advantage is not a context artifact** — the CNN is simply
the wrong comparator on this task, and `best(k-mer, CNN)` must be evaluated *at each model's own
context* before any capability claim.

The other cluster's reading #1 (k-mer wins, no qualified target) stands. Reading #2
("gLMs show real capability the CNN does not") is true against the CNN but does not survive
against the k-mer at matched context.

---

## PART B — Running HERE now (8 GPUs, sequential queues)

Queue runner: `scratchpad/multimodel/queues/runner.sh <gpu> <abs-queue-file>`; status in
`scratchpad/multimodel/logs_matched/queue_gpu*.status`.

| GPU | jobs | question it answers |
|:--|:--|:--|
| 0 | HyenaDNA + GENA-LM on HVUE identity-disjoint **hsd1, hsd2**, 3 tasks × 3 seeds | is the identity-split result stable across split seeds? (CNN already has all 3; gLMs only had hsd0) |
| 1 | NT-v2 on identity **hsd1/hsd2**; then **Evo LoRA on identity_disjoint_hsd0** | Evo has no number on any defensible split |
| 2 | ViroBench baselines @ **20 kb** cap (finishing) | matched-context baseline vs HyenaDNA |
| 3 | **HyenaDNA at 49,152 bp** (full genome), 3 seeds | the one model that can match the k-mer's context — decisive test |
| 4 | NT-v2 **LoRA @ 2048 tok (~12 kb)** | full-FT OOM'd at 6.1 kb; LoRA should reach its nominal window |
| 5 | GENA-LM ViroBench, **3 seeds** | was n=1 |
| 6 | ViroBench baselines @ **whole-genome + 12 kb**, then **genus split** @ 20 kb + whole | completes the context ladder; genus split as secondary reference |
| 7 | GUE `virus_species_40` baselines @ **3.1 kb and 5 kb**; then **frozen-probe** regime for 3 gLMs | matched-context baseline for GENA-LM; probe arm on GUE |

**Context audit (verified, not assumed):**

| benchmark | sequence length | context matched? |
|:--|--:|:--|
| HVUE | exactly 1000 bp, all 916,086 rows | **yes, inherently** |
| GUE `virus_covid` | 999 bp | **yes** |
| GUE `virus_species_40` | 5000 bp | yes for HyenaDNA/NT-v2; **GENA-LM sees ~3.1 kb** → being fixed on GPU 7 |
| ViroBench | median 43.5 kb, max 1.4 Mb | **no** — the entire Part A problem |

---

## PART C — For the OTHER cluster: what to run there

Ordered by value. None of these overlap with Part B.

### C1. ViroBench full P3 spec — the biggest remaining gap
`virobench_baselines.py` + `virobench_glm.py`
- `--mod ALL` (46,651 train, not the 7,600 DNA subset)
- **all 5 taxonomic levels**: `--level {family,order,class,phylum,kingdom}`
- `--min_count 1` (no class filter — the current numbers filter to ≥10 and are therefore
  **not comparable to ViroBench's published 47.67 / 75.88**; fixing this is what makes the
  cross-paper comparison legitimate)
- both splits, but treat `times` as primary (`genus` has **82–84% genus overlap** — verified
  independently on both clusters, it is a record-level holdout despite the name)
- **add BLAST and Kraken2** as the alignment baselines ViroBench itself uses. Kraken2 especially:
  it is a k-mer classifier by design, so it is the honest SOTA comparator for taxonomy and will
  tell you whether 0.95 is a ceiling or just our implementation.
- run every baseline **capped at each model's context**, per Part A.

### C2. Supervised single-variant effect — the only untouched modality
Not yet started anywhere. The 95-scorer ProteinGym leaderboard is **entirely zero-shot**; nobody
has tested whether supervised adaptation beats MSA methods.
- 22 viral assays in `/data/nvidia/proteingym/DMS_ProteinGym_substitutions`
- ESM-2 650M / 3B + ESM-1v (all cached), regimes frozen-ridge / LoRA / full FT
- splits: ProteinGym **`contiguous`** and **`modulo`** (position-disjoint), not random
- comparators: published ESCOTT / GEMME / S3F_MSA from
  `evo-locking/results/pg_all_scorers_viral.csv` **plus a supervised one-hot + biochemical
  control** — that control beat ESM-2 in the escape arm and beat the gLMs on HVUE
- do **not** add more zero-shot pLMs: ESM-2 8M→15B, ESM3, ESMC, SaProt, ProSST, Progen3 and
  xTrimoPGLM-100B are already on the leaderboard, and MSA methods still hold the top 15
  (best sequence-only, VESPA, ranks 16/95)

### C3. Antibody escape completion
`run_escape_regime.py` in `glm-locking/experiments/exp5_regimes/`
- RABV_G LoRA at matched *n* is **done**: 4/5 folds lose to the GBT (margins −0.003…−0.025,
  one win +0.062). Frozen arm was already 20/20 losses.
- remaining: LASV_GP / SARS2_spike / H5_HA LoRA (were mid-run when stopped), then
  `--regime full` for all 4 antigens
- **must use** `--max_train 120000 --epochs 8`; the original defaults gave the model 800–3,200
  rows against the baseline's 39,896 with `--epochs 2`, producing ρ=0.04 vs frozen 0.30
- gate: discard any FT run that scores below its own frozen arm

### C4. NT-v2 context ceiling
NT-v2 full-FT OOM'd above ~6.1 kb because its `modeling_esm.py` uses eager attention with no
gradient-checkpointing support. Either patch checkpointing in or use LoRA (being tried here on
GPU 4). Worth resolving because NT-v2 is the only gLM that beat the k-mer at matched context.

### C5. HVUE Evo full fine-tuning
No valid Evo full-FT number exists anywhere — the locking project's runs had `pre_auroc ≈ 0.50`
and never converged. Needs FSDP across several GPUs, or a last-N-blocks approximation. Only
matters if you want to claim Evo was tested in every regime; LoRA at lr 3e-4 already reaches
0.8742 (vs its published 0.8173), so Evo is **not** a weak model — it was under-tuned.

---

## PART D — Rules that must hold in both clusters

1. **Never select a split, hyperparameter or checkpoint on a criterion involving the baseline's
   performance, and never build a split in the baseline's feature space.** The HVUE composition
   splits did both (`GATE = 0.03`, clusters in k-mer5-PCA space) and it moved headline numbers by
   0.12–0.18 AUROC.
2. **`best(k-mer, CNN)` at the model's own context** is the baseline. Not k-mer alone (the CNN
   binds on HVUE), not CNN alone (the k-mer binds on ViroBench and GUE).
3. **Report the effective context in bp for every row**, and cap baselines to match when
   sequences exceed any model's window.
4. **Report AUROC and MCC** (they disagree in sign on at least one cell), and macro-F1 for
   multiclass.
5. **Evaluate the reported split at every checkpoint**, report the mean over the last K —
   single-checkpoint reporting swings 0.014–0.045 on disjoint splits.
6. **Use the mean over seeds, not the best**, when comparing against a single published number.

---

## PART E — Current scoreboard (for context when interpreting new runs)

Cells where any gLM beats `best(k-mer, CNN)`:

| benchmark / split | outcome |
|:--|:--|
| HVUE identity-disjoint hsd0, 3 tasks | **8 losses, 1 win of +0.0012** |
| HVUE random, 3 tasks | 1 marginal win (+0.0011), 2 losses |
| HVUE composition-cluster (invalid split) | 6 wins — the only place gLMs win |
| GUE `virus_covid` | all 3 lose (−0.046 … −0.31) |
| GUE `virus_species_40` | NT-v2 +0.009, others lose |
| ViroBench taxonomy, matched context | k-mer wins at 3.1 kb and 20 kb; NT-v2 +0.021 at 6.1 kb |
| ViroBench host, leave-one-family-out | all lose, −0.097…−0.170, CIs exclude zero |
| Antibody escape, frozen | GBT wins **20/20** folds |
| Antibody escape, LoRA matched *n* (RABV_G) | GBT wins 4/5 |
| ProteinGym viral, 25 assays | MSA methods hold top 15; best sequence-only ranks 16/95 |

**Positive control exists and passes**, so the harness is not broken: on NT splice tasks the
published gLMs beat the best classical baseline by **+0.30 to +0.60 MCC**
(`reports/positive_control_comparison.md`). The viral negatives are not a measurement artifact.
