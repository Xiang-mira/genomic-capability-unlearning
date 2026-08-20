# Handoff: viral capability benchmarking for the unlearning project

Written 2026-08-19. All runs **stopped**; 8× A100 free, 0 compute apps.
Everything below is reproducible from the scripts named in each section.

---

## 0. Where things live

| path | contents |
|:--|:--|
| `/data/nvidia/genomic-capability-unlearning` | student's repo (benchmark qualification + unlearning). `/home/nvidia/genomic-capability-unlearning` is the same dir. |
| `/home/nvidia/glm-locking` | weight-locking source: `scripts/`, `experiments/`, `src/` |
| `/data/nvidia/evo-locking/results` | weight-locking outputs (4 TB) |
| `/data/nvidia/escape_screen` | antibody-escape data, splits, results |
| `/data/nvidia/proteingym` | 217 DMS assays (22 viral), zero-shot scores |
| `scratchpad/multimodel/` | **all new work from this session** — scripts, results, logs |

Python: `/home/nvidia/miniconda3/envs/evo/bin/python` (torch 2.6, transformers 4.48.1, peft 0.20).
Set `HF_HOME=/data/nvidia/hf_cache`. Cached models: Evo-1-8k, ESM-2 8M/150M/650M/3B, ESM-1v ×5,
HyenaDNA-medium-160k, GENA-LM-bert-base-t2t, NT-v2-500M.

**Launch pattern that works** (bare `nohup &` gets SIGTERM'd when the wrapper exits):
```bash
setsid env CUDA_VISIBLE_DEVICES=<g> nohup <py> -u <script> <args> > /abs/path/log 2>&1 </dev/null &
disown -a
```

---

## 1. Runs that were in flight when stopped

| script | args | state at stop | resumable? |
|:--|:--|:--|:--|
| `evo_lora_fixed.py` | `--task Host_Tropism --split cluster_disjoint --seeds 42 43 44 --lrs 3e-4 1e-3 1e-4 --max_steps 8000` | **2 of 9 runs done** (s42, s43 @ lr3e-4). ~10 h/run. | yes — skips existing JSONs in `evo_results/` |
| `run_escape_regime.py` (glm-locking `experiments/exp5_regimes/`) | `--antigen {RABV_G,LASV_GP,SARS2_spike,H5_HA} --regime lora --max_train 120000 --epochs 8` | **RABV_G complete (5/5 folds)**. LASV_GP ep1/8, SARS2_spike ep7/8, H5_HA ep1/8. | per-antigen; restart the 3 unfinished |
| `build_idsweep.py` | — | **complete** (24 splits, `splits_idsweep/`) | n/a |
| `virobench_prep_baselines.py` | `--mod DNA --split {times,genus} --level family` | **both complete** | n/a |

---

## 2. Scripts written this session (all in `scratchpad/multimodel/`)

| script | purpose | key args |
|:--|:--|:--|
| `glm_finetune.py` | HVUE binary tasks, any HF gLM. Regimes probe/lora/full, random-init control, **test-set eval every epoch** (trajectory saved) | `--model {hyenadna,gena_lm,nt_v2_500m} --regime {probe,lora,full} --random_init --tasks --splits --seeds --lrs --split_dir --kmer_json` |
| `cnn_baseline.py` | 0.64M dilated-CNN baseline, HVUE binary | `--tasks --splits --seeds --lrs --split_dir` |
| `glm_gue_viral.py` | GUE viral multiclass (`virus_covid` 9-cls, `virus_species_40` 25-cls) | `--model --task --regime --seeds --lrs --maxlen` |
| `gue_viral_baselines.py` | k-mer3-5/3-6 + CNN on GUE viral | `--task --maxlen --cap` |
| `virobench_prep_baselines.py` | joins ViroBench CSV+jsonl, builds taxonomy task, k-mer + CNN | `--mod {DNA,ALL,RNA} --split {genus,times} --level {family,order,class,phylum,kingdom} --min_count --kmer_cap --cnn_len` |
| `build_ungated_splits.py` | rebuilds HVUE MMseqs2 identity-disjoint splits **with no baseline gate** | — (edit `TASKS`, thresholds) |
| `build_idsweep.py` | same, sweeping identity 90/70/50/30% | — |
| `evo_lora_fixed.py` | Evo-1-8k LoRA with extended LR, real early stopping, ev-trajectory | `--task --split --seeds --lrs --max_steps --val_every --patience --min_delta` |
| `verify_hvue_model_vs_kmer.py` | reproduces + fixes the student's decisive script | — |
| `verify_ft_vs_kmer.py` | matched k-mer on the exact FT test splits | — |
| `verify_fresh_probe.py` | feature-space rank-k difference-of-means erasure + fresh probe | — |
| `verify_phistruct_bootstrap.py`, `phistruct_recheck.py` | PHIStruct re-analysis | — |
| `selection_bias.py`, `gate_power.py` | EvoMIL protocol / gate power simulations | — |

Results: `glm_results/` (99), `cnn_results/` (63), `gue_results/` (2), `gue_glm_results/` (19),
`virobench_results/` (2), `evo_results/` (2), `splits_ungated/` (11), `splits_idsweep/` (25).

---

## 3. Findings, with the caveat attached to each

### 3.1 The split used for every published HVUE number is baseline-hostile

`build_splits_v2.py` builds clusters in **k-mer5-PCA space** and accepts a candidate split only
if `kmer_auroc(split) ≤ kmer_auroc(random) − 0.03` (`GATE = 0.03`), scanning 7 KMeans configs.
The MMseqs2 identity-disjoint alternative was computed, gave 0.9131 on Host_Tropism, **failed
that gate, and was recorded `VALIDITY: INVALID`** on all three tasks.

k-mer3-6 AUROC, no gate, my rebuild (`build_idsweep.py`, 2 holdout seeds each):

| task | random | id90 | id70 | id50 | id30 | composition-cluster |
|:--|--:|--:|--:|--:|--:|--:|
| Host_Tropism | 0.9213 | 0.9126 | 0.9138 | 0.9066 | 0.9130 | **0.8034** |
| Pathogenecity | 0.9685 | 0.9530 | 0.9222 | 0.9301 | 0.9307 | **0.8044** |
| Transmissibility | 0.9238 | 0.9075 | 0.8227 | 0.8481 | 0.8665 | **0.7395** |

Homology stringency from 90% down to **30%** barely moves the baseline. The composition split
sits below every homology threshold. **It is not a stricter OOD test; it is a differently-biased
one.** Every "model beats k-mer" HVUE number inherits this.

### 3.2 On the defensible split, a 0.64M CNN wins 8 of 9 cells

MMseqs2 90%-identity-disjoint, no gate, 3 seeds. CNN beats k-mer on all 9 task×split combos, so
it is the binding baseline everywhere.

| task (hsd0) | k-mer | **CNN 0.64M** | GENA-LM | HyenaDNA | NT-v2 | verdict |
|:--|--:|--:|--:|--:|--:|:--|
| Host_Tropism | 0.9194 | **0.9482** | 0.9372 | 0.9438 | 0.9494 | NT-v2 +0.0012; others lose |
| Pathogenecity | 0.9479 | **0.9667** | 0.9494 | 0.9234 | 0.9409 | all lose (−0.017…−0.043) |
| Transmissibility | 0.9085 | **0.9202** | 0.9037 | 0.8475 | 0.8694 | all lose (−0.016…−0.073) |

MCC agrees in every cell. CNN across hsd0/1/2 is stable (e.g. Host_Tropism 0.9482/0.9482/0.9516).

### 3.3 Evo was under-tuned, not weak — this corrects earlier claims

`evo_lora_fixed.py`, lr **3e-4** (the original grid stopped at 1e-4 and selected it in every cell):

| seed | ev AUROC | k-mer | excess | MCC | early-stopped? | best_step |
|:--|--:|--:|--:|--:|:--|--:|
| 42 | **0.8849** | 0.8034 | **+0.0815** | 0.5613 | yes | 5250 |
| 43 | **0.8636** | 0.8034 | **+0.0602** | 0.5502 | yes | 4750 |

vs the published 0.8173. Both converged (the original 18 runs all hit the step cap, 0 early-stopped).
So **Evo ≈ NT-v2** once tuned. Any claim that Evo is specifically poor is wrong. Third seed and
lr 1e-3 not run.

### 3.4 Frozen representations are far below baseline

HyenaDNA head-only probe: **0.5910 AUROC, −0.2497 vs best baseline**, MCC 0.1430.
Evo frozen probe, 4 independent measurements: −0.021…−0.063 vs k-mer.
→ representation-localisation methods have no target to localise.

### 3.5 Viral taxonomy is a k-mer task

`virobench_prep_baselines.py`, DNA subset, family level, whole-genome k-mer vs 20 kb CNN:

| split | classes | k-mer3-6 | CNN | k-mer − CNN |
|:--|--:|--:|--:|--:|
| `times` (temporal) | 44 | **0.9527** | 0.6774 | **+0.2753** |
| `genus` | 69 | **0.9252** | 0.7310 | **+0.1942** |

**CAVEAT — do not quote against ViroBench's 47.67/75.88.** I filtered to families with ≥10 train
examples *and* present in test (69 of 152) and used DNA (7,600) not ALL (46,651). That inflates
macro-F1. The internal k-mer-vs-CNN comparison is valid; the cross-paper one is not yet.

Also: GUE `virus_species_40` (25-cls, random split) — k-mer 0.4407 vs HyenaDNA 0.4329,
NT-v2 0.4498, GENA-LM 0.3643. GUE `virus_covid` (9-cls) — k-mer **0.7282** vs GENA-LM 0.6671,
NT-v2 0.6823, HyenaDNA 0.4217.

### 3.6 ViroBench's "genus-disjoint" split is not genus-disjoint

Verified on the downloaded metadata (`virobench/*_taxon_*__{train,test}.csv`):

| split | taxid | species | **genus** | dates |
|:--|--:|--:|--:|:--|
| `taxon/genus` (their **G-split**) | 0% | 18–20% | **82–84% overlap** | fully overlapping |
| `taxon/times` (their **T-split**) | 0% | 1–2% | 50–80% | train ≤2017 / test ≥2020, zero overlap |

Only taxid is held out in the G-split. Their headline +28-point LucaVirus-over-BLAST claim is on
a record-level holdout. **The `times` split is the strict one.** Confirm against their paper text
before asserting publicly.

### 3.7 Escape: the FT arm now works and the negative holds

`run_escape_regime.py --regime lora --max_train 120000 --epochs 8` (ESM-2 3B). Previously the
model got 800–3,200 rows vs the baseline's 39,896 and `--epochs 2`, giving ρ=0.04 vs frozen 0.30.
With the full fold (`n_train_model == n_train`):

| RABV_G fold | model ρ | GBT ρ | margin |
|:--|--:|--:|--:|
| 17C7 | 0.2989 | 0.3245 | −0.0254 |
| CR4098 | 0.4595 | 0.4801 | −0.0211 |
| CR57 | 0.2141 | 0.2172 | −0.0034 |
| CTB012 | 0.1578 | 0.0953 | **+0.0622** |
| RVA122 | 0.4378 | 0.4549 | −0.0172 |

4/5 lose. Frozen arm (pre-existing, complete): GBT beats ESM-2 3B **20/20 folds**.

### 3.8 Methodological: the disjoint-split metric is checkpoint-unstable

With test-set eval at every epoch: range **0.014–0.045** over the last 8 epochs after inner-val
plateaus (full-training range 0.17–0.27). Inner-val and test differ by 0.12–0.26 on the
composition split but are near-equal on identity splits. Single-checkpoint reporting on
composition-disjoint splits is inadequate.

### 3.9 Errors in the existing repos (all verified)

- `internal_auroc_drop = 0.844 − score`, so RMU's reported "0.14 drop" is a **0.146 increase**;
  localized RMU 0.990 vs random-layer control 0.994 — indistinguishable. Column also holds two
  incompatible quantities (11 rows use `internal_mean_drop`).
- Fresh-probe recovery (~0.99) — **no artifact anywhere** in the repo.
- `hvue_composition_confound/SUMMARY.md` is **stale v1** and contradicts `results_v2.csv`
  (says Host_Tropism +0.051 / Pathogenecity +0.047 "GENUINE"; v2 says +0.0139 / −0.0046).
- `virobench_probe_results.csv` reports AUROC=F1=MCC=**1.0000** on 100 classes with
  `n_train == n_val == 4475` → leakage.
- `experiments/FINDINGS.md` is an unfilled template.
- `hvue_lora_ci.csv` / `hvue_lora_metrics_table.md` use `KMER_K=4` at `N_TRAIN=3000` (of 458,756);
  superseded by `results_v2.csv`.
- Student's 44-task FT stack: median **0.286 epochs**; `bvbrc_cov` at `val_loss 0.6899` ≈ ln 2;
  0.372 AUROC below its own frozen probe. Void.

---

## 4. What still needs testing, in priority order

**P1 — finish Evo (≈20 GPU-h).** `evo_lora_fixed.py`, seed 44 @ lr3e-4 plus 3 seeds @ lr1e-3.
Decides whether Evo matches NT-v2. Also run Evo on the **identity-disjoint** splits
(`--split identity_disjoint_hsd0 --split_dir splits_ungated`) — currently no Evo number exists
on a defensible split.

**P2 — finish escape (≈2 GPU-days).** LASV_GP / SARS2_spike / H5_HA LoRA (restart), then
`--regime full` for all 4 antigens. 20 folds total per regime. Gate: discard any FT run below its
own frozen arm.

**P3 — ViroBench done properly (≈1 GPU-day).** Reproduce their exact protocol: `--mod ALL`
(46,651), all 5 levels Kingdom→Family, `--min_count 1` (no class filter), both splits. Only then
is comparison to 47.67/75.88 legitimate. Then run the 4 gLMs — but **record effective context**:
median genome 43 kb, so k-mer sees all, HyenaDNA 160 kb, NT-v2 ~12 kb, GENA-LM ~2–4 kb. Add BLAST
and Kraken2 as the alignment baselines ViroBench uses.

**P4 — supervised single-variant (≈2 GPU-days). NOT YET STARTED.** 22 viral ProteinGym assays
(`/data/nvidia/proteingym/DMS_ProteinGym_substitutions`), ESM-2 650M/3B + ESM-1v (cached),
regimes frozen-ridge / LoRA / full, splits = ProteinGym **contiguous** and **modulo**
(position-disjoint, not random). Comparators: published ESCOTT / GEMME / S3F_MSA from
`evo-locking/results/pg_all_scorers_viral.csv` **plus a supervised one-hot+biochemical baseline**
— that control is what beat ESM-2 in the escape arm and beat the gLMs in HVUE.
Note: the 95-scorer leaderboard is **entirely zero-shot**; supervised adaptation is the open
question. Do **not** add more zero-shot pLMs — ESM-2 8M→15B, ESM3, ESMC, SaProt, ProSST, Progen3,
xTrimoPGLM-100B are already on it, and MSA methods still hold the top 15 (best sequence-only,
VESPA, ranks 16/95).

**P5 — gLMs on the identity-disjoint splits at hsd1/hsd2** (~8 GPU-h). Currently only hsd0 has
model runs; CNN has all three. Cheap seed-robustness for the headline table.

**P6 — lineage/temporal split for GUE `virus_covid`** (~4 GPU-h). SARS-CoV-2 variants are
temporally nested; GUE's random split is in-distribution only. Needs collection-date metadata.

**Do not run:** non-viral GUE/NT tasks (outside the claim); more zero-shot pLMs; DNABERT-2
(`config_class` conflict with transformers 4.48); further HVUE composition-split experiments.

---

## 5. Two rules that must apply to every new run

1. **Never select a split, hyperparameter, or checkpoint on a criterion involving the baseline's
   performance, and never build a split in the baseline's feature space.** Both happened here and
   both moved headline numbers by 0.03–0.15 AUROC.
2. **Every model needs a supervised non-pretrained control of comparable capacity** (CNN for DNA;
   one-hot/biochemical ridge or GBT for protein), not only a k-mer or conservation baseline. That
   control has now beaten the foundation model in three modalities.

3. **Set and report the sequence-length cap per dataset.** `hvue_cnn.py`/`hvue_glm.py` default
   to 1000 bp and `virobench_baselines.py` to `--cnn_len 20000`. Verified as of 2026-08-20:

   | dataset | sequence length | cap used | truncated |
   |:--|--:|--:|--:|
   | HVUE (all 916,086 rows, all 3 tasks, all splits) | **exactly 1000 bp** | 1000 | **0%** |
   | GUE `virus_covid` | 999 bp | 1000 | 0% |
   | GUE `virus_species_40` | 5000 bp | 5000 (passed explicitly) | 0% |
   | ViroBench genomes | median **43 kb**, max 1.4 Mb | CNN 20 kb | **~54%** |

   So HVUE and GUE are clean — the k-mer and the CNN saw identical input. **ViroBench is not:**
   the k-mer reads the whole genome while the CNN reads ~46% and GENA-LM ~7%. Part of the
   ViroBench k-mer margin (§3.5) is therefore context, not capability. Re-run the CNN at >=43 kb
   and report matched context before drawing a capability conclusion there. Leaving `--maxlen`
   at its default silently creates this confound whenever sequences exceed it.

Plus: report AUROC **and** MCC (they disagree in sign on at least one cell); report the effective
context per model when sequences exceed any model's window; evaluate the reported split at every
checkpoint and report the mean over the last K.

---

## 6. Bottom line for the unlearning project

No viral task tested provides qualified model-specific headroom against the best non-foundation
baseline on a defensible split. Frozen representations sit 0.02–0.25 AUROC *below* baseline, so
there is nothing localisable to excise. And a 0.64M CNN trained from scratch on public data
matches or beats every pretrained gLM on 8 of 9 cells — so removing the capability from open
weights would not deny it to an adversary.

The publishable output is the negative plus the methodology (split-construction sensitivity, the
supervised-CNN comparator, checkpoint instability on disjoint splits). The one remaining chance of
a positive viral target is ViroBench taxonomy under its **temporal** split, run at their full
protocol — but §3.5 suggests a k-mer will win there too.
