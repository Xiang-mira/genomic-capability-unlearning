# Results

Every result this repository currently supports, with an explicit provenance
status for each number. This is one of only two documents in the repository; the
practical guide — setup, stages, methods, extension — is [../README.md](../README.md).

**Provenance labels used throughout:**

| Label | Meaning |
|:--|:--|
| `verified` | the number was re-read from a checked-in artifact in this repository |
| `unverified` | the number appears only in prose; its source artifact was produced on the original host and is **not** in the repository |

Treat `unverified` numbers as provisional. They are reported because they are the
project's recorded conclusions, not because they can be audited here.

**Contents**

- [Summary](#summary)
- [The causal chain being tested](#the-causal-chain-being-tested)
- [1. Stage 2 — Unlearning trade-off (44-task benchmark)](#1-stage-2--unlearning-trade-off-44-task-benchmark)
- [2. Stage 2 — Sweep coverage](#2-stage-2--sweep-coverage)
- [3. Stage 2 — Internal diagnostics](#3-stage-2--internal-diagnostics)
- [4. Stage 1 — Localization](#4-stage-1--localization)
- [5. Stage 0 — Benchmark qualification](#5-stage-0--benchmark-qualification)
- [6. Stage 3 — Relearning attacks](#6-stage-3--relearning-attacks)
- [Measurement caveats](#measurement-caveats)
- [Open work](#open-work)
- [Artifact index](#artifact-index)
- [Missing artifacts](#missing-artifacts)

---

## Summary

1. **Unlearning has not reached a usable operating point.** Across 90 checked-in
   runs, target removal and general genomic capability trade off roughly
   one-for-one. Best forgetting costs 0.144 GUE AUROC; best retention forgets
   nothing measurable.
2. **No target capability benchmark qualified.** All four studies found a
   conventional baseline that matched or beat the foundation model under the
   intended held-out protocol.
3. **Localization is real but does not equal separability.** Activation patching
   isolates a causal span (layers 3-9, primary layer 6) that does not coincide
   with the layers where linear probes score highest (0-2).
4. **The negative control works.** Updating matched parameter counts in
   non-causal layers 11-30 leaves the target capability essentially untouched
   (ΔAUROC ≈ +0.002), so the observed forgetting is not generic damage.

---

## The causal chain being tested

Every artifact in this document is meant to support one link in a single claim:

`target knowledge → model representation → probe measurement → unlearning → downstream target behaviour → general capability retention`

| Objective | Scientific claim | Status |
|:--|:--|:--|
| **1. Target validity** | the capability is real, model-internal information, not a dataset shortcut | **failed** — a full-sequence composition baseline matches Evo on host tropism ([§5d](#5d-hvue--evo--partial)) |
| **2. Probe metric validity** | frozen probe scores predict downstream supervised behaviour | **not established** — probe and downstream rankings disagree ([§3](#3-stage-2--internal-diagnostics)); tooling in `phase2/probe_vs_sft.py` |
| **3. Localization** | the capability is carried by an identifiable causal layer span | **supported** — patching isolates layers 3-9 and the random control is clean ([§4b](#4b-activation-patching--causal-layers)) |
| **4. GD vs RMU trade-off** | methods can be compared by removal versus collateral damage | **measured, and the answer is negative** ([§1](#1-stage-2--unlearning-trade-off-44-task-benchmark)) |
| **5. Attack robustness** | removal survives a motivated relearning adversary | **unmeasured** ([§6](#6-stage-3--relearning-attacks)) |

Because objective 1 failed, objective 4's numbers describe removal of something
the model does not distinctly possess. That is the single most important caveat
in this document.

Target-validity datasets and their intended roles:

- `hiyata/Virus-Host-Genomes` — primary controlled-split evidence; carries
  `family` and `genus`, so real family/genus-held-out splits are definable.
- `data/host_tropism/manifest.csv` — legacy continuity with Stage 1 layer
  localization. It has `virus_tax_id`, `virus_name`, `source` but no lineage
  table, so a `--group-key virus_tax_id` split is species-like holdout, **not**
  family-held-out.
- `duttaprat/HVUE` Host Tropism — external downstream validation, not the only
  target-validity dataset.

Controlled splits that should each be run: `random` (learnability only),
`taxonomy` (held-out taxon), `homology` (sequence-cluster holdout),
`within_group` (between-group taxonomy shortcut). Success criterion: the target
remains meaningfully predictable under at least one *controlled* split, not only
under `random`.

## 1. Stage 2 — Unlearning trade-off (44-task benchmark)

**Status: `verified`.** Two things to know first:

> **1. The GD objective mismatch is RESOLVED.** For a period, `unlearn_gd.py`
> contained a probe-guided representation objective rather than gradient
> difference, so the `lora_gd_*` rows below were not reproducible from the code.
> Classic gradient difference has been restored to
> [../phase2/unlearn_gd.py](../phase2/unlearn_gd.py) and the other objective moved
> to [../phase2/unlearn_probe_repr.py](../phase2/unlearn_probe_repr.py) (method key
> `probe_repr`). **These rows are reproducible again.** Details:
> [§3 internal diagnostics](#3-stage-2--internal-diagnostics).
>
> **2. None of the 90 checked-in runs records its code version.** Zero `meta.json`
> files contain `commit_hash` — `phase2/run_metadata.py` was added after all
> checked-in results were produced. So no artifact in this repository can be tied
> to the code that generated it. Provenance capture works for *new* runs; it was
> not retrofitted.

RMU was never affected by the first issue; `unlearn_rmu.py`'s objective is
unchanged. Both methods remain subject to the second.

The completed headline experiment. Four unlearned checkpoints against the same
base model on the same 44 downstream tasks, one fixed supervised LoRA protocol
per model, best task checkpoint chosen on validation, paired task-level deltas.

**Coverage:** 33 GUE tasks (general genomic retain) + 5 HVUE tasks (target
forget) + 6 ViroBench tasks (non-target viral retain).

**Base model group scores:**

| Group | Tasks | Metric | Base score |
|:--|--:|:--|--:|
| `hvue_forget` | 5 | AUROC | 0.7777 |
| `gue_retain` | 33 | AUROC | 0.7638 |
| `viral_retain` | 6 | macro-F1 (see caveats) | 0.0451 |

**Results** — forget drop higher is better, retain delta closer to zero is better:

| Checkpoint | Method | Balanced forget | HVUE drop [95% CI] | GUE delta [95% CI] | Viral delta | Selection score |
|:--|:--|--:|:--|:--|--:|--:|
| `lora_gd_full_ar3_s200` | GD | 0.2066 | **0.1981** [0.0885, 0.3117] | **-0.1443** [-0.1855, -0.1073] | -0.0266 | 0.0357 |
| `lora_gd_full_ar5_s500` | GD | 0.0863 | 0.0840 [0.0152, 0.1466] | -0.1047 [-0.1346, -0.0762] | -0.0218 | -0.0402 |
| `lora_rmu_full_sc200_s200` | RMU | 0.0310 | 0.0363 [0.0002, 0.0922] | -0.0442 [-0.0631, -0.0282] | -0.0058 | -0.0189 |
| `lora_rmu_full_sc50_s200` | RMU | 0.0045 | 0.0057 [-0.0064, 0.0185] | +0.0012 [-0.0124, 0.0190] | -0.0066 | -0.0021 |

**Interpretation.**

- Gradient difference produces the strongest target reduction but is **not
  selective**: an 0.198 HVUE drop costs 0.144 of GUE retain AUROC.
- Weak RMU (`sc50`) preserves GUE (+0.0012, CI straddling zero) but its forget
  drop of 0.0057 has CI `[-0.0064, +0.0185]` — **statistically indistinguishable
  from no forgetting at all**.
- Raising RMU strength (`sc200`) buys forgetting (0.0363) but GUE begins to fall
  (-0.0442), and the forget CI lower bound is 0.0002 — barely positive.
- No checkpoint reaches the high-forget / low-damage region. The frontier here is
  roughly linear, which is the central negative result.

**Per-task HVUE base AUROC** (the forget targets are not equally learnable):

| Task | Base AUROC | Test rows |
|:--|--:|--:|
| `hvue_human_transmissibility_orthomyxoviridae` | 0.9455 | 63,345 |
| `hvue_human_host_tropism` | 0.8911 | 10,852 |
| `hvue_human_transmissibility_coronaviridae` | 0.7496 | 26,872 |
| `hvue_human_virus_pathogenicity_cini` | 0.7358 | 545 |
| `hvue_human_virus_pathogenicity_bvbrc_cov` | 0.5666 | 19,747 |

`bvbrc_cov` is near chance at baseline, so "forgetting" it is not meaningful.

**Artifacts**

- [../results/full_benchmark_summary.csv](../results/full_benchmark_summary.csv) — compact table
- `data/phase2/full_benchmarks_lora_optimized_s600/full_rankings.csv` — full ranking with CIs
- `data/phase2/full_benchmarks_lora_optimized_s600/<run>/eval_benchmarks.csv` — per task
- `data/phase2/full_benchmarks_lora_optimized_s600/<run>/eval_benchmarks_summary.json` — group means
- `data/phase2/full_benchmarks_lora_optimized_s600/<run>/logs/<task>.jsonl` — 44 per-task training logs per run
- `data/phase2/full_benchmarks_lora_optimized_s600/manifest_audit.json` — input row counts, schemas, SHA-256
- `data/phase2/checkpoints_lora_grid/<run>/meta.json`, `log.json` — unlearning config and history
- Figure generator: [../tools/plot_full_benchmark_results.py](../tools/plot_full_benchmark_results.py)
- Figures: [../figures/full_benchmark_tradeoff.png](../figures/full_benchmark_tradeoff.png),
  [../figures/full_benchmark_target_vs_retain.png](../figures/full_benchmark_target_vs_retain.png),
  [../figures/full_benchmark_selection_score.png](../figures/full_benchmark_selection_score.png)

### Lean pilot (subsampled manifest)

**Status: `verified`.** Independent confirmation of the same trade-off on a
134,254-row manifest, 49 HVUE and 231 GUE task-layer pairs:

| Run | HVUE drop [95% CI] | GUE delta [95% CI] |
|:--|:--|:--|
| `gd_full_ar5` | 0.1006 [0.0854, 0.1156] | -0.1010 [-0.1080, -0.0938] |
| `rmu_full_sc200` | 0.0162 [0.0101, 0.0226] | -0.0221 [-0.0264, -0.0179] |

Artifact: `data/phase2/benchmark_pilot_lean/pilot_rankings.csv`. The GD ratio
(forget 0.101 / cost 0.101) reproduces the full-suite finding almost exactly.

---

## 2. Stage 2 — Sweep coverage

**Status: `verified`** (counts read from checked-in files).

| Quantity | Count |
|--:|:--|
| Unlearning runs with `meta.json` | **90** |
| Runs with internal evaluation (`eval_auroc.csv`) | **78** |
| Runs aggregated in `task2_runs.csv` | 38 |
| Sweep directories under `data/phase2/` | 8 checkpoint roots |

Method × condition coverage across all 90 runs:

| Method | Condition | Runs |
|:--|:--|--:|
| RMU | full | 49 |
| RMU | localized | 24 |
| probe null-space projection | localized | 10 |
| gradient difference | full | 7 |

And within the 38-run aggregated table (which includes the `random` controls):

| Method | Condition | Runs |
|:--|:--|--:|
| gradient difference | localized | 17 |
| RMU | localized | 10 |
| gradient difference | full / random | 3 / 3 |
| RMU | full / random | 2 / 2 |
| probe null-space | localized | 1 |

Sweep definitions: [../phase2/sweep_configs/](../phase2/sweep_configs/) —
RMU full-layer scan, RMU L6/L8 tuning, RMU/LoRA Pareto search, RMU localized
(nonhuman and joint-probe directions), LoRA full grid, and three projection
variants.

Artifacts: `data/phase2/results/task2_runs.csv` (38 rows × 38 columns),
`task2_best_checkpoints.csv`, `task2_summary.json`, plus
`data/phase2/checkpoints_*/<run>/{meta.json,log.json,eval_auroc.csv,eval_ppl.json,eval_representation.csv}`.

### Gate thresholds used for screening

From `data/phase2/results/task2_summary.json` (`verified`):

| Gate | Value |
|:--|--:|
| internal probe AUROC drop threshold | 0.05 |
| minimum GUE retain delta | -0.02 |
| maximum retain PPL increase | 0.30 |
| base internal AUROC (layers 3-9) | 0.844 |
| base retain PPL | 4.2 |
| runs screened / passing | 38 / 7 |

### Training-free projection baseline

**Status: `verified`** (`data/phase2/results/task2_best_checkpoints.csv`).

`probe_nullspace_joint_l5_l9` — no gradient steps at all:

| Metric | Value |
|:--|--:|
| internal AUROC drop (layers 3-9) | 0.632 |
| host-tropism internal AUROC | 0.211 |
| HVUE forget drop | 0.226 |
| GUE retain delta | -0.0186 |
| retain PPL | 4.279 (base 4.2) |
| retain representation cosine vs base | 0.9855 |
| selection score | **0.207** |

This is the strongest selection score in that table, and it is **training-free**.
Any new method should be compared against it first
([README § Step 4](../README.md#step-4--validate-scientifically)). Note the caveat
in [Measurement caveats](#measurement-caveats): this HVUE drop was measured on
the pilot-scale protocol, not the 44-task suite, so it is not directly
comparable to the table in section 1.

---

## 3. Stage 2 — Internal diagnostics

**Status: `unverified`** — the source evaluation outputs for this specific
comparison table were not committed. Individual `eval_auroc.csv` /
`eval_ppl.json` files for 78 runs *are* checked in and can be used to rebuild an
equivalent table.

Host-tropism probe AUROC, mean over layers 3-9, at steps=200, lr=1e-5,
batch=2, max_length=512:

| Method | Condition | Probe AUROC | Δ vs base | forget PPL | retain PPL |
|:--|:--|--:|--:|--:|--:|
| base | — | 0.844 | — | ~4.2 | ~4.2 |
| GD | localized | 0.555 | -0.289 | 20.38 | 15.70 |
| GD | full | 0.524 | -0.321 | 31.21 | 37.90 |
| GD | **random** | 0.846 | **+0.002** | 4.20 | 4.23 |
| RMU | full | 0.700 | -0.144 | 4.45 | 4.48 |
| RMU | localized | 0.765 | -0.079 | 4.39 | 4.42 |
| RMU | **random** | 0.846 | **+0.002** | 4.21 | 4.26 |

**What this supports:**

- **The negative control holds.** Both `random` conditions move probe AUROC by
  +0.002 and leave perplexity at baseline. Forgetting depends on *touching the
  causal layers*, not on parameter count.
- **GD damages the generation objective.** Retain PPL rises from 4.2 to 15.7
  (localized) and 37.9 (full). RMU, operating in representation space, keeps
  retain PPL at 4.4-4.5.
- **RMU under-forgets in the localized condition** (-0.079).
- **Layers 0-2 are untouched by localized interventions** and hold AUROC at
  0.86-0.88, confirming they are not the intervention target.

Note the ordering inversion versus section 1: on *internal probe* metrics GD
localized looks best, while on *downstream* metrics full-model GD/RMU LoRA runs
were selected. This disagreement is precisely why the project moved to
downstream-first evaluation
([Causal-chain claims](#the-causal-chain-being-tested)).

Full method comparison and reasoning:
[§3 internal diagnostics](#3-stage-2--internal-diagnostics).

---

## 4. Stage 1 — Localization

### 4a. Viral-vs-plasmid pilot probes

**Status: `verified`.** 2,000 viral + 2,000 plasmid RefSeq sequences, 2,048 bp
windows, seed 42.

| Model | Test AUROC |
|:--|--:|
| Evo-1 linear probe, best layer (layer 2) | **1.0000** |
| Evo-1 linear probe, layer 0 | 0.9914 |
| Evo-1 linear probe, layers 1-10 | 0.9999-1.0000 |
| GC + 1-gram logistic regression baseline | **0.6854** |

Artifacts: [../results/probe_metrics_by_layer.csv](../results/probe_metrics_by_layer.csv)
(32 layers), [../results/gc_1gram_metrics.csv](../results/gc_1gram_metrics.csv),
[../figures/meeting_phase1_refseq_target.png](../figures/meeting_phase1_refseq_target.png),
`docs/assets/probe_metrics.png`.

Evo representations are far more linearly separable than composition on this
task. Note this is a *pilot* discrimination task (viral vs plasmid), not the
host-tropism capability that Stage 2 targets.

### 4b. Activation patching — causal layers

**Status: `unverified`** — `data/family_targets/coronaviridae/` including
`patching_layer_summary.csv` is not in the repository. The derived figure
[../figures/patching_analysis.png](../figures/patching_analysis.png) is.

Host-tropism (human vs non-human) patching, n_pairs=16, max_length=512, test
split, Evo-1-8k-base:

| Layer | nh→h \|Δprob\| | h→nh \|Δprob\| | Mean | Probe test AUROC |
|--:|--:|--:|--:|--:|
| **6** | **0.441** | **0.269** | **0.355** | 0.853 |
| **8** | 0.286 | 0.266 | 0.276 | 0.801 |
| 4 | 0.176 | 0.154 | 0.165 | 0.855 |
| 9 | 0.170 | 0.130 | 0.150 | 0.838 |
| 3 | 0.116 | 0.139 | 0.127 | 0.854 |
| 5 | 0.155 | 0.072 | 0.114 | 0.849 |
| 10 | 0.000 | 0.063 | 0.031 | 0.812 |
| 2 | 0.018 | 0.016 | 0.017 | 0.859 |
| 1 | 0.007 | 0.015 | 0.011 | 0.865 |
| 0 | 0.0001 | 0.0003 | 0.0002 | **0.870** |

**Three findings that shaped the whole project:**

1. **Probe AUROC ≠ causal importance.** Layer 0 has the *highest* probe AUROC
   (0.870) and effectively *zero* causal effect (0.0002). Selecting intervention
   targets by probe salience would have targeted the wrong layers.
2. **Single-layer patching does not change the output.** Patched perplexity was
   identical to four decimal places across all layers (Δppl 0.211 for nh→h,
   1.809 for h→nh, std ≤ 0.0001) — later layers reconstruct the feature.
   Interventions must cover the whole causal span, not one layer.
3. **Layers 11+ are numerically unusable.** Activation L2 norm jumps from ≈257
   at layer 10 to ≈1.8×10⁶ at layer 11 and ≈7.3×10⁷ at layer 13 in bfloat16
   (StripedHyena long-convolution accumulation). Probes there collapse to
   0.60-0.69 and patching effects go to zero. All probing and intervention is
   restricted to layers 0-10, with random controls drawn from 11-30.

Selected localized span: **layers 3-9**, primary target layer **6**. The default
fallback in [../phase2/utils.py](../phase2/utils.py) is the narrower `[5,6,7,8,9]`
with primary layer 6.

**Intervention target priority derived from this:** layer 6 (strongest causal
effect), then layer 8 (balanced bidirectional), then layers 3-5 and 9 (moderate).
Layers 0-2 should be *preserved* — destroying early surface features would damage
general genomic capability without removing the target.

**Three pitfalls this analysis rules out:** do not unlearn a single layer (high
risk of being routed around); do not trust probe output in layers 11+; high probe
AUROC does not mean causal importance, so the optimization signal should come from
layers 3-9.

`n_pairs=16` is small. The qualitative findings above were stable, but the
ordering *among* the moderate layers 3-5 and 9 was never established with
confidence intervals. Raise `n_pairs` before treating a per-layer ranking as
settled.

Full analysis: [§4 Stage 1](#4-stage-1--localization).

---

## 5. Stage 0 — Benchmark qualification

Four completed studies. **All negative.** Protocol and decision rules:
[README § Stage 0](../README.md#stage-0--benchmark-qualification). Task-level guidance:
[§5 code entrypoints](#per-task-code-entrypoints).

| Benchmark | Model | Final status | Strongest comparator | Provenance |
|:--|:--|:--|:--|:--|
| HVUE | Evo-1 | `UNQUALIFIED` | full-sequence k-mer / composition | partial |
| ProteinGym | ESM2 | `UNQUALIFIED` | evolutionary predictors | `unverified` |
| PHIStruct | SaProt | `PHISTRUCT_FAILURE_NOT_STATISTICALLY_RESOLVED` | BLASTp | `verified` |
| EvoMIL | ESM-1b | `NO_QUALIFYING_HEADROOM` | AA 3-mer TF-IDF | `unverified` |

### Per-task code entrypoints

| Task | Controller(s) | Result root |
|:--|:--|:--|
| HVUE / Evo | `phase2/prepare_benchmarks.py`, `phase2/eval_benchmarks.py`, `phase2/eval_kmer_baseline.py`, `phase2/run_hvue_complete_selection_and_full.sh`, `phase2/run_hvue_pipeline_watchdog.sh`, `phase2/aggregate_hvue_lora.py` | `data/phase2/full_benchmarks_lora_optimized_s600/` |
| ProteinGym / ESM2 | `phase2/proteingym_esm2_qualification.py`, `phase2/proteingym_esm2_top20_expansion.py` | **absent** (see [Missing artifacts](#missing-artifacts)) |
| PHIStruct / SaProt | `phase2/phistruct_qualification.py`, `phase2/phistruct_failure_audit_evomil_controller.py` | `data/phase2/phistruct_qualification/phistruct_failure_audit/` |
| EvoMIL / ESM-1b | `phase2/evomil_esm1b_qualification.py`, `phase2/signed_bootstrap.py` | `data/phase2/evomil_qualification/` (registry only) |
| VPF-PLM | `phase2/vpf_plm_qualification.py`, `phase2/vpf_plm_compat.py` | none — in progress |

Shared: `phase2/signed_bootstrap.py`, tested by `tests/test_signed_bootstrap.py`
and `tests/test_evomil_esm1b_qualification.py`.

To inspect one task quickly: read its subsection below, open the controller, then
read the listed summary `.json` or `.md` first — subject to the provenance labels.

### 5a. PHIStruct / SaProt — `verified`

Bacteriophage receptor-binding-protein host prediction. Every number below was
re-read from
`data/phase2/phistruct_qualification/phistruct_failure_audit/audit_summary.json`
and `paired_bootstrap_summary.json`.

| Quantity | Value |
|:--|--:|
| Dataset | 7,627 RBPs / 3,350 phages / 7 host genera |
| Splits (train / val / test) | 7,218 / 92 / 317 |
| SaProt test macro-F1 | 0.454732 |
| BLASTp test macro-F1 | **0.475180** |
| Observed delta (SaProt − BLASTp) | **-0.020448** |
| Bootstrap mean / median delta | -0.018095 / -0.016958 |
| 95% CI | **[-0.113066, +0.071794]** |
| P(delta > 0) | 0.3594 |
| P(delta < 0) | 0.6406 |
| Valid / attempted replicates | 10,000 / 15,778 (5,778 invalid) |
| Bootstrap unit | `phage_id` |
| BLASTp test accuracy / hit rate | 0.5773 / 0.9937 |
| HMMER sanity macro-F1 | 0.0105 (sanity floor, passes) |

Excluding tiny classes does not rescue it: macro-F1 without tiny classes is
0.6108 (SaProt) vs 0.6384 (BLASTp), delta **-0.0276**.

Per-genus BLASTp accuracy is highly uneven (Enterococcus and Klebsiella 1.00,
Acinetobacter 0.465, Enterobacter 0.00 on a single query), so the macro average
is fragile — which is exactly why the bootstrap CI is wide.

**Conclusion:** SaProt beats simple and structure-only baselines but shows no
statistically reliable gain over BLASTp. Unresolved rather than a clean negative.

Artifacts: `data/phase2/phistruct_qualification/phistruct_failure_audit/` —
`audit_summary.{json,md}`, `paired_bootstrap_summary.json`,
`paired_bootstrap_samples.csv`, `per_genus_{blast,saprot}_metrics.csv`,
`per_genus_comparison.csv`, `{blast,saprot}_confusion_matrix.csv`,
`blast_hit_audit.csv`, `blast_hit_summary.json`,
`leave_one_genus_out_sensitivity.csv`, `hmmer_sanity_audit.json`,
`controller_status.json`.

### 5b. EvoMIL / ESM-1b — `unverified`

Viral host prediction from proteome-level ESM-1b embeddings + multiple-instance
learning.

| Quantity | Value |
|:--|--:|
| Strongest baseline | `logistic_regression:aa_3mer_tfidf` |
| Baseline macro-F1 | **0.841270** |
| Best ESM-1b + MIL macro-F1 | 0.782246 |
| Observed delta (model − baseline) | **-0.059024** |
| Bootstrap 95% CI | [-0.204111, +0.056453] |
| Seeds with positive delta | **0 / 5** |

The AA 3-mer baseline beats every ESM-1b seed.

**What is actually checked in:** only
`data/phase2/evomil_qualification/evomil_experiment_registry.json` and
`evomil_external_assets.json`. The registry records
`final_status: NO_QUALIFYING_HEADROOM`, `current_stage: complete`, and asset
provenance (official repo `liudan111/EvoMIL` at revision
`78405057afa30cb5b66777bd5cda88c4fa85c55d`, VHDB table SHA-256
`56c92ca1…ee4ff3`, paper DOI `10.1371/journal.pcbi.1012597`).

**Important caveat visible in the registry:** it also records
`formal_evomil_started: false` with three `formal_blockers` — official viral
protein/proteome FASTA assets absent from the checkout, official precomputed
ESM-1b embeddings absent, and the strict dataset adapter / proteome-cluster split
/ leakage audit / mandatory baselines not implemented. So the recorded status was
reached without the full formal protocol. The metric table above has **no
checked-in source file** anywhere on this machine. Treat both the numbers and the
finality of the status as provisional.

### 5c. ProteinGym / ESM2 — `unverified`

Mutation-effect prediction under strict position-held-out evaluation.

| Quantity | Value |
|:--|--:|
| Static candidate assays screened | 20 |
| Pilot assays selected | 3 |
| Preliminarily qualified task-model pairs | **0** |
| Assays advanced to LoRA | 1 (`CCDB_ECOLI_Adkar_2012`) |
| LoRA mean test excess (3 seeds) | **-0.218436** |

Pilot assays and the baseline that beat each:

| Assay | Strongest public baseline |
|:--|:--|
| `CCDB_ECOLI_Adkar_2012` | `public_evolutionary:VESPAl` |
| `GAL4_YEAST_Kitzman_2015` | `public_evolutionary:VESPA` |
| `MET_HUMAN_Estevam_2023` | `public_evolutionary:S2F_MSA` |

Key finding: random-split gains **do not transfer** to position-held-out
evaluation, and the one assay that advanced to LoRA was unstable across three
formal seeds.

**`data/phase2/protein_48h_esm2_qualification/` does not exist in this
repository.** The eight result files referenced by the historical README
(`protein_48h_summary_report.{json,md}`,
`protein_48h_evolutionary_baseline_report.{json,md}`,
`protein_48h_lora_qualification_evidence.json`,
`protein_48h_candidate_ranking.csv`, `protein_48h_esm2_pilot_metrics.csv`,
`protein_48h_lora_metrics.csv`) are all absent. The controllers that produced
them are present and tested
([../phase2/proteingym_esm2_qualification.py](../phase2/proteingym_esm2_qualification.py),
[../phase2/proteingym_esm2_top20_expansion.py](../phase2/proteingym_esm2_top20_expansion.py),
9 + 3 passing tests), so the study is re-runnable.

### 5d. HVUE / Evo — partial

Not packaged as a standalone `*_qualification/` directory; its evidence is the
44-task benchmark stack in section 1 (`verified`) plus k-mer/composition control
audits.

The `UNQUALIFIED` verdict rests on the finding that the apparent host-tropism and
pathogenicity advantage largely disappears under stronger full-sequence
composition controls. The specific comparison that establishes this is
`unverified` here; the strong-baseline k-mer AUROC constants embedded in
[../phase2/lora_subspace_targeting.py](../phase2/lora_subspace_targeting.py)
are indicative:

| Constant (task: `hvue_human_host_tropism`) | Value |
|:--|--:|
| `STRONG_MATCHED_INPUT_KMER_AUROC` | 0.855455 |
| `STRONG_MATCHED_INPUT_KMER_MCC` | 0.599193 |
| `FULL_SEQUENCE_STRONG_AUROC` | **0.893001** |

Both are non-foundation baselines, grouped as `primary_baselines` in that
module. The first is fitted on the *same truncated input* the model sees; the
second uses the *whole* sequence.

Compare with the Evo base HVUE host-tropism AUROC of **0.8911** from section 1
(also on truncated 512-token input):

- against the **matched-input** baseline, Evo is ahead by about +0.036;
- against the **full-sequence** baseline (0.8930), Evo is **level or behind**.

That is the substance of the HVUE verdict: the apparent advantage is a
consequence of handicapping the baseline to the model's context window, and it
does not survive letting the conventional baseline see the full sequence. Both
sides of the comparison are reproducible from checked-in files.

### 5e. VPF-PLM — in progress, do not report

[../phase2/vpf_plm_qualification.py](../phase2/vpf_plm_qualification.py) and
[../phase2/vpf_plm_compat.py](../phase2/vpf_plm_compat.py) implement a fifth
candidate. As of the last recorded status (13 August 2026) it was still running
as a detached controller and **must not be reported as a completed result**. No
result artifacts are checked in.

---

## 6. Stage 3 — Relearning attacks

**Status: no checked-in results.**

The attack harness is implemented and wired
([../phase3/attack_sft.py](../phase3/attack_sft.py),
[../phase3/attack_lora.py](../phase3/attack_lora.py),
[../phase3/run_attacks.sh](../phase3/run_attacks.sh),
[../phase3/aggregate_attack_results.py](../phase3/aggregate_attack_results.py)),
including LR-grid sweeps with adversary-optimal LR selection and a richer recipe
distribution in [../phase2/next_steps_common.py](../phase2/next_steps_common.py).
No `data/phase3/` results are in the repository, and
[../figures/phase3_results.png](../figures/phase3_results.png) has no
corresponding checked-in table.

This is the largest open gap. Because no checkpoint reached an acceptable
forget/retain point (section 1), attack robustness was never the binding
constraint — but it must be measured before any removal claim.

Related in-flight work recorded in
[../logs/](../logs/): `experiment3_host_tropism_relearning_*` runs (July-August
2026) via [../phase2/run_experiment3_host_tropism_relearning.py](../phase2/run_experiment3_host_tropism_relearning.py)
and [../phase2/experiment3_artifact_retention.py](../phase2/experiment3_artifact_retention.py).
Logs only — treat as unfinished.

---

## Measurement caveats

Read these before quoting any number above.

**0. Provenance.** No checked-in run records a commit hash. The GD objective
mismatch that previously affected these rows has been fixed. See the boxed note
in [section 1](#1-stage-2--unlearning-trade-off-44-task-benchmark).

**1. The `viral_retain` group is degenerate and not comparable to the others.**
`eval_benchmarks.summarize` picks the first available metric in the order
`auroc, mcc, f1, accuracy, pearson, r2`. All 5 HVUE and all 33 GUE tasks report
AUROC. All 6 ViroBench tasks are multiclass, report **no** AUROC, and therefore
fall through to **macro-F1**. Base macro-F1 on those tasks is 0.0127-0.1017
(mean 0.0451) while accuracy is 0.31-0.65 — the LoRA head collapses onto a few
classes on many-class taxon prediction.

Consequences: the `viral_delta` column is a change on a collapsed macro-F1 scale,
**not** an AUROC change like the other two columns; it is not interpretable as
"non-target viral capability retained". It does **not** enter `selection_score`
(`retain_penalty = max(0, -gue_delta)` only), so the ranking is metric-consistent.
Anyone continuing this work should either fix the ViroBench task heads or drop
the group.

**2. `gue_retain` includes near-chance tasks.** Base AUROC ranges 0.4886
(`gue_epi_k562`) to 0.9803 (`gue_prom_300_notata`); five epigenetic-mark tasks
sit below 0.57. Averaging over 33 tasks of very different difficulty means the
GUE delta is dominated by the learnable ones. `gue_emp_h4ac` reports a
mean_score of exactly 0.0 in the earlier `base_benchmarks` run — inspect before
reusing that run.

**3. Frozen probe AUROC is a diagnostic, never a conclusion.** A fixed probe can
be defeated by rotating the representation without deleting the information. Use
`--fresh-probe`. Section 3's numbers are fixed-probe.

**4. Section 2 and section 1 scores are not directly comparable.** The
projection baseline's HVUE drop of 0.226 was measured on the pilot-scale
protocol; the section 1 table is the 44-task suite. Do not put them in one
ranking.

**5. Two HVUE Caliciviridae tasks are excluded** from the final external
manifest (`hvue_human_virus_pathogenicity_bvbrc_calici`,
`hvue_human_transmissibility_caliciviridae`, 111,071 rows). They ship with only
`sequence,label`, are already restricted to one family, and therefore admit no
definable family-held-out shortcut check.

**6. The runnable suite reports 0 viral-retain tasks.** Per
[README § Cost](../README.md#cost) the final
runnable suite is 38 tasks / 1,670,176 rows: 2 primary forget (72,930 rows),
3 secondary forget (732,085), 33 GUE retain (865,161), **0** viral retain.
vGUE/Vir2vec was audited and found to provide accession splits but no task-level
`sequence,label` table (`data/benchmarks/vgue_from_vir2vec_audit.json`), so it
was never integrated.

---

**7. Recorded data-availability constraints.** Two are load-bearing.

*Caliciviridae shortcut check is undefinable.* The public HVUE tasks
`BVBRC Calici pathogenicity` and `Calici transmissibility` ship with only
`sequence,label` — no family/genus/species/accession columns — and both are
already restricted to within `Caliciviridae`, so a strict family-held-out split
does not exist even in principle. They are excluded from the final external
manifest (111,071 rows). Making a shortcut audit possible requires importing
external taxonomy metadata and switching to genus- or species-held-out.

*vGUE / Vir2vec was never integrated.* Audited into
`data/benchmarks/vgue_from_vir2vec_audit.json`: the public Vir2vec repository
provides train/validation/test **accession** splits (BV-BRC, GISAID, HBVdb,
LANL-HIV-DB, NCBI Virus) plus embedding scripts, but **no** unified task table in
the `benchmark,task,split,sequence,label` form `eval_benchmarks.py` consumes. So
Vir2vec is a viable upstream source, but "vGUE is integrated" was never true.
ViroBench CLS-Lite became the viral-retain proxy instead — and turned out
degenerate under this protocol (caveat 1). `HIV-1 Tropism` is excluded because it
overlaps the forget objective.

**8. `final_benchmark_plan.md` was stale and has been removed.** Its evaluation
commands passed `--layers`, `--auto-batch-size`, `--probe-jobs` and
`--feature-cache-dir` to `eval_benchmarks.py`, which has none of those flags —
they belong to the legacy frozen-probe evaluator. Its two durable contents are
preserved: the group-relabelling rule (primary vs secondary forget, Calici
exclusions) and the checkpoint-scoping rule, both now in
[README § Stage 2b](../README.md#stage-2b--evaluation).

## Open work

Ordered by what would most change the conclusions.

### 1. Requalify the target, or pick a different one

Objective 1 of [the causal chain](#the-causal-chain-being-tested) failed: the
full-sequence composition baseline is level with or ahead of Evo on host tropism
([§5d](#5d-hvue--evo--partial)). Removing a capability the model does not
distinctly possess is not a well-posed problem, and every §1 number inherits that
caveat. Run the Stage 0 gate on a target that is not reducible to composition or
homology before spending more GPU time on methods.

### 2. Measure gradient-space interference between forget and retain

**Currently impossible with the shipped code**, and it is the most direct
explanation of the central negative result. Every method calls `.backward()` once
on the combined loss:

```python
loss = weighted_forget + weighted_retain
loss.backward()                       # g_forget and g_retain are summed here
```

so the two gradients are never separated. The only gradient-space quantity ever
recorded is `clip_grad_norm_` on the *combined* gradient.

This matters because the §1 result — forgetting and retention trading roughly
one-for-one — is exactly what you would expect if
`cos(g_forget, g_retain) ≈ +1` in the trainable subspace. If the objectives are
aligned, ascending the forget loss *necessarily* descends the retain loss, and no
amount of `alpha_retain` tuning escapes it. **That hypothesis can be tested
without ever looking at logits or task scores.** Two backward passes per step:

```python
# forget gradient alone
optimizer.zero_grad(set_to_none=True)
L_forget.backward(retain_graph=True)
g_f = torch.cat([p.grad.flatten() for p in trainable if p.grad is not None]).clone()

# retain gradient alone
optimizer.zero_grad(set_to_none=True)
L_retain.backward()
g_r = torch.cat([p.grad.flatten() for p in trainable if p.grad is not None]).clone()

cos       = torch.nn.functional.cosine_similarity(g_f, g_r, dim=0)   # per layer too
norm_ratio = g_f.norm() / g_r.norm().clamp(min=1e-12)
conflict   = (cos < 0)                                               # PCGrad's criterion
```

Interpretation:

| `cos(g_f, g_r)` | Meaning | Implication |
|:--|:--|:--|
| ≈ +1 | the two objectives want the same update direction | selective removal is **impossible** in that subspace; the one-for-one frontier is structural |
| ≈ 0 | orthogonal | selective removal is possible; damage is an optimizer artifact, not a constraint |
| ≈ -1 | directly opposed | the retain anchor is actively fighting the forget term; expect oscillation |

Log it per layer, not just globally — the interesting claim is that
`cos` is high in the causal layers 3-9 and low elsewhere, which would be a
mechanistic explanation of why localized intervention did not help.

It is also directly actionable. If the gradients conflict, project before
stepping (gradient surgery / PCGrad):

```python
if torch.dot(g_f, g_r) < 0:
    g_f = g_f - (torch.dot(g_f, g_r) / g_r.dot(g_r).clamp(min=1e-12)) * g_r
```

That is a new method variant, not just a diagnostic — it would slot in as a
`gd_surgery` entry in `METHOD_SCRIPT` with no other pipeline changes.

Cost: roughly 2× the backward cost per step, which is negligible next to the
32-36 GPU-hours a downstream evaluation takes. Suggested implementation point: a
`--grad-diagnostics` flag on `phase2/unlearn_gd.py` writing per-layer `cos`,
`norm_ratio` and conflict fraction into the existing `log.json` records, so
`phase2/plot_convergence_diagnostics.py` picks them up for free.

### 3. Run Stage 3

No relearning-attack results are checked in ([§6](#6-stage-3--relearning-attacks)).
Because no checkpoint reached an acceptable operating point, attack robustness was
never the binding constraint — but no removal claim is meaningful without it.

### 4. Fix or drop the `viral_retain` group

As implemented it contributes an uninterpretable macro-F1 column
([Measurement caveats](#measurement-caveats)). Either fix the ViroBench task
heads so they report AUROC, or remove the group. vGUE/Vir2vec integration needs
task-level `sequence,label` tables plus an accession-to-task mapping.

### 5. Establish probe-metric validity, or stop reporting probes

`phase2/probe_vs_sft.py` exists to test whether probe changes predict supervised
fine-tuning changes, on identical rows, over seeds `42,43,44`, reporting Pearson
and rank consistency between probe degradation and SFT degradation. It has not
been run to conclusion. Until it is, frozen-probe numbers are diagnostics.

### 6. Retrofit provenance, or accept the gap

Zero of 90 archived `meta.json` files carry a `commit_hash`. New runs are covered
by `build_run_metadata`; existing artifacts cannot be tied to their code version
and never will be.

### 7. Consolidate `probe_repr` and `probe_guided`

`probe_guided --forget-objective component_zero` computes the same forget loss as
`probe_repr`. They differ in what they train (probe target layers vs
`--condition` layers) and in retain terms (`hidden_mse`/`output_kl`/`ce` vs
`hidden_mse` + cosine). One script with explicit options for both axes would
remove the duplication.

### 8. Recorded next steps that remain open

- Run a true family-held-out split on the primary `host_tropism` dataset or on a
  mixed-family task such as CINI; use genus- or species-held-out for
  single-family Calici tasks.
- Add taxonomy metadata to the two excluded Calici tasks to make a shortcut audit
  definable at all.
- Save intermediate checkpoints (`--save-steps 100,200,500,1000`) for
  representative runs and evaluate trajectories, rather than only final states.
- RMU localized under-forgets; try larger `steer_coef` or more steps — but it
  only counts if GUE retain holds.
- GD can try larger `alpha_retain`, or the new `--forget-loss-cap`, for the same
  reason.

## Artifact index

Checked-in result artifacts, by directory:

| Directory | Files | Contents |
|:--|--:|:--|
| `data/phase2/full_benchmarks_lora_optimized_s600/` | 240 | headline 44-task benchmark: base + 4 checkpoints, per-task logs, rankings, manifest audit |
| `data/phase2/checkpoints_rmu_tuning/` | 124 | 24 RMU tuning runs |
| `data/phase2/benchmark_pilot_lora/` | 119 | 13 LoRA-grid pilot evaluations with per-task logs |
| `data/phase2/checkpoints_projection_opt/` | 91 | 10 projection runs |
| `data/phase2/checkpoints_layer_scan/` | 86 | 17 single-layer RMU scan runs |
| `data/phase2/checkpoints_rmu_localized_joint_probe/` | 61 | 12 joint-probe-direction RMU runs |
| `data/phase2/checkpoints_rmu_localized_nonhuman/` | 61 | 12 nonhuman-direction RMU runs |
| `data/phase2/checkpoints_rmu_pareto/` | 55 | RMU/LoRA Pareto search |
| `data/phase2/checkpoints_lora_grid/` | 25 | 12 LoRA candidate configs (meta + log) |
| `data/phase2/phistruct_qualification/` | 14 | PHIStruct failure audit (fully verified) |
| `data/phase2/benchmark_pilot_lean/` | 11 | lean pilot for 2 candidates + rankings |
| `data/phase2/checkpoints_tuned/` | 9 | selected tuned checkpoints |
| `data/phase2/final_fast_eval/` | 6 | fast-eval runs |
| `data/phase2/base_benchmarks/` | 3 | 39-task base evaluation |
| `data/phase2/results/` | 3 | `task2_runs.csv`, `task2_best_checkpoints.csv`, `task2_summary.json` |
| `data/phase2/evomil_qualification/` | 2 | registry + asset provenance only |
| `results/` | 3 | compact top-level tables |
| `figures/` | 10 | published figures |
| `logs/` | ~120 | historical run logs (provenance, not results) |

Per-run files: `meta.json` (config + provenance), `log.json` (training history),
`eval_auroc.csv`, `eval_ppl.json`, `eval_representation.csv`,
`eval_benchmarks.csv`, `eval_benchmarks_summary.json`,
`eval_benchmarks_progress.json`, `logs/<task>.jsonl`.

Deliberately excluded from git: the 2.4 GB full sequence manifest and 339 MB
pilot manifest, Evo base model files, all unlearning model weights, discarded
downstream task checkpoints, and raw benchmark corpora. Sizes, row counts,
schemas, and SHA-256 hashes are recorded in
`data/phase2/full_benchmarks_lora_optimized_s600/manifest_audit.json`.

---

## Missing artifacts

Documented results whose source files are **not** in this repository. Listed so
nobody wastes time looking for them, and so they can be regenerated.

| Missing | Referenced by | Regenerate with |
|:--|:--|:--|
| `data/phase2/protein_48h_esm2_qualification/` (8 files) | section 5c | `phase2/proteingym_esm2_qualification.py` |
| `evomil_summary_report.*`, `evomil_bootstrap_*`, `evomil_kmer_baselines.csv`, `evomil_model_results.csv`, `evomil_split_audit.json`, `evomil_preprocessing_audit.json`, `evomil_reproduction_sanity_report.json` | section 5b | `phase2/evomil_esm1b_qualification.py` |
| `phistruct_qualification/summary_report.*`, `baseline_results.csv`, `plm_results.csv`, `per_genus_metrics.csv` (top level) | section 5a | `phase2/phistruct_qualification.py` (the failure-audit subdirectory **is** present) |
| `data/family_targets/coronaviridae/` — probes, `localized_layers.json`, `patching_layer_summary.csv`, `patching_by_layer.csv` | section 4b, all of Stage 2 | `bash phase1/run.sh all` |
| `data/phase2/splits/` | section 3 | `bash phase2/run.sh splits` |
| `data/benchmarks/` manifests | sections 1, 5 | `bash phase2/run.sh prepare_benchmarks` |
| `data/phase3/` attack results | section 6 | `bash phase3/run.sh all` |
| `data/phase2/stage1_baseline_alignment_20260729/base_calibration_27run_unified.csv` | 2 skipped tests | `phase2/stage1_baseline_alignment.py` |

The original host paths recorded in checked-in provenance
(`/home/teacher1/UT-project1/project1/...`) confirm these were produced on a
different machine and never committed.
