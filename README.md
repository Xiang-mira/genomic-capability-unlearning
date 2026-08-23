# Viral capability benchmarking — continuation branch

Branch: `viral-benchmark-continuation`. Everything needed to resume the experiments on a
different cluster. **Start here:** [reports/PAPER_OUTLINE.md](reports/PAPER_OUTLINE.md) (thesis + section plan),
then [reports/CROSS_CLUSTER_SYNTHESIS.md](reports/CROSS_CLUSTER_SYNTHESIS.md) (both clusters
reconciled, with corrections), then [HANDOFF.md](HANDOFF.md) (full state and to-do list).

## What this branch is for

Deciding whether any viral benchmark gives a biological foundation model **reproducible,
model-specific headroom over the strongest non-foundation baseline on a defensible split**.
That question gates the unlearning work: removing a capability is only meaningful if the
capability is real and model-specific.

Current answer, across 3 HVUE tasks × 3 split families, 2 GUE viral tasks, ViroBench taxonomy
(5 levels), 5 gLM architectures and 4 baseline families: **no** — and on two of the three HVUE
tasks the benchmark cannot answer the question at all. See
[OUTCOME_FOR_UNLEARNING.md](OUTCOME_FOR_UNLEARNING.md).

**The negative result is viral-specific and positively controlled.** The same models, harness and
protocols DO show advantages on non-viral tasks: GENA-LM wins 11/13 GENEB categories and NT-v2
10/13 against a fairly-tuned k-mer, and fine-tuned NT-v2 reaches 0.9680 MCC on NT splice against a
published 0.971–0.984. We are not claiming gLMs lack capability; we are claiming the *viral*
evidence does not survive its baselines and splits.

## The four findings that matter most methodologically

1. **[SPLIT_DESIGN_EXPLAINED.md](SPLIT_DESIGN_EXPLAINED.md)** — the splits every previously
   published number used were built in the k-mer baseline's own feature space *and* selected on
   the condition that the baseline lose ≥0.03 AUROC. Use the identity-disjoint splits from
   `build_identity_splits.py` instead, or taxonomic holdout.
2. **The baseline is not just a k-mer.** A supervised CNN beats the k-mer on 9 of 9 HVUE
   task×split cells and beats every pretrained gLM on 8 of 9. Any comparison must report
   `max(k-mer, CNN)`.
3. **[reports/BASELINE_CAPACITY_CEILING.md](reports/BASELINE_CAPACITY_CEILING.md) — baseline
   RECEPTIVE FIELD, not capacity, dominates the reported gaps.** On 600bp splice, ResNet at 9.44M
   params (RF 89bp) scores 0.336 MCC while a U-Net at 0.26M (global RF) scores 0.951 — 36× fewer
   parameters, +0.62 MCC. Published FM-vs-CNN splice gaps of +0.31–0.60 shrink to **+0.02–0.03**
   against a baseline that can actually see the whole input. Our own incumbent 0.68M dilated CNN
   was at ceiling on HVUE (≤+0.006 from a 13-cell search) but badly under-powered on splice.
4. **Homology saturation: measure PARTIAL overlap with `easy-search`, not `easy-cluster`.**
   `-c 0.9` requires 90% *bidirectional* coverage and is blind to a test sequence sharing half its
   length at high identity. Measured: HVUE Pathogenecity **80.5%** and Transmissibility **83.2%** of
   test rows have a train hit at ≥90% id over ≥50% length (ViroBench: **2.1%**). Refiltering at
   ≥70%/≥30% leaves Pathogenecity **96 of 5,194** test rows and Transmissibility **60 of 4,956** —
   those two tasks cannot support a homology-clean evaluation at all, for anyone.

## Layout

```
scripts/viral_benchmark/     portable harnesses (see below)
HANDOFF.md                   full state, results, caveats, to-do list
OUTCOME_FOR_UNLEARNING.md    synthesis: what this means for unlearning
SPLIT_DESIGN_EXPLAINED.md    composition- vs homology-disjoint splits
phase2/                      original benchmark-qualification code (student's)
tests/                       regression tests for phase2/
docs/, reports/              original project write-ups (historical)
```

Data, model weights, logs, figures and all run outputs are gitignored. Nothing large is tracked.

## Quickstart on a new cluster

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r scripts/viral_benchmark/requirements.txt

export VB_ROOT=/scratch/$USER/viral-bench
export VB_OUT=$VB_ROOT/results
export VB_HVUE_DIR=$VB_ROOT/data/hvue        # {task}_{train,validation,test}.parquet
export VB_MMSEQS=$(which mmseqs)
export HF_HOME=$VB_ROOT/hf_cache

cd scripts/viral_benchmark
python download_data.py --what both          # GUE viral + ViroBench (~3.7 GB)
python build_identity_splits.py              # HVUE identity-disjoint splits (CPU, needs mmseqs)
bash run_examples.sh                         # reference launches, one GPU per job
```

`paths.py` resolves every path from `VB_*` environment variables — no absolute paths are
hard-coded in any script.

## Scripts

| script | what it does | notes |
|:--|:--|:--|
| `paths.py` | env-var path resolution for all scripts | edit defaults or export `VB_*` |
| `download_data.py` | fetches GUE viral + ViroBench from HF | `--what {gue,virobench,both}` |
| `build_identity_splits.py` | MMseqs2 identity-disjoint HVUE splits, **no baseline gate**, + k-mer reference | needs `mmseqs`; writes `kmer_baselines.json` |
| `hvue_glm.py` | any HF gLM on HVUE binary tasks | `--regime {probe,lora,full}`, `--random_init`, `--split_dir`, test eval every epoch |
| `hvue_cnn.py` | dilated-CNN baseline on HVUE | the binding baseline — always run it |
| `capacity_sweep.py` | 13-cell architecture × capacity ladder (dilated / U-Net / ResNet, 0.04M–9.4M) | **run before any baseline-vs-FM claim**; dev-only selection, group-disjoint dev for HVUE |
| `splice_finetune.py` | full fine-tune on NT splice; the FT positive control | LR sweep + warmup; `assert_no_fresh_encoder_weights()` aborts on silent weight re-init |
| `splice_positive_control.py` | frozen-probe arm of the same control | frozen probing costs −0.59 MCC vs FT here |
| `virobench_frozen_probe.py` | ViroBench frozen probe, their protocol | `--layer` — **required for LucaVirus**, whose final LN collapses the representation (53× less between-sequence variance at layer 12 than 11) |
| `hvue_evo_lora.py` | Evo-1-8k LoRA, extended LR, real early stopping | needs `glm-locking` on `PYTHONPATH` (`VB_LOCK_ROOT`) |
| `gue_baselines.py` | k-mer3-5/3-6 + CNN on GUE viral multiclass | |
| `gue_glm.py` | gLMs on GUE viral multiclass | |
| `virobench_baselines.py` | builds ViroBench taxonomy task, k-mer + CNN | `--level`, `--min_count`; **records effective context** |

Supported gLMs: `hyenadna` (LongSafari/hyenadna-medium-160k-seqlen-hf), `gena_lm`
(AIRI-Institute/gena-lm-bert-base-t2t), `nt_v2_500m`
(InstaDeepAI/nucleotide-transformer-v2-500m-multi-species). DNABERT-2 is excluded — its
`config_class` conflicts with transformers 4.48.

## Rules for any new run

1. Never select a split, hyperparameter or checkpoint using the baseline's performance, and never
   build a split in the baseline's feature space.
2. Every model needs a supervised **non-pretrained** control of comparable capacity (CNN for DNA,
   one-hot/biochemical ridge or GBT for protein) — not only a k-mer or conservation baseline.
3. Report AUROC **and** MCC; they disagree in sign on at least one cell.
4. Report each model's **effective context** when sequences exceed any model's window
   (ViroBench genomes are median 13.7 kb over ALL, 43 kb for the DNA subset; GENA-LM sees ~2–4 kb).
   For CNN baselines also report the **receptive field in bp** — see finding 3 above.
5. Evaluate the reported split at **every** checkpoint and report the mean over the last K —
   single-checkpoint reporting on disjoint splits swings 0.014–0.045 AUROC.
6. **Verify no pretrained tensor is silently re-initialised.** `AutoModelForSequenceClassification`
   on GENA-LM discards all 48 pretrained LayerNorms (pre-LN checkpoint vs post-LN HF class) and the
   model then collapses to the majority class at every LR — dev MCC exactly 0.0000. Frozen probes
   via `AutoModel` are unaffected. Use the guard in `splice_finetune.py`.
7. **For negative claims, pre-declare the equivalence margin δ before computing any CI**, and
   report per-level CI width. On ViroBench only the family level (173 classes) has the power to
   support an equivalence claim; coarser levels are *underpowered*, not equivalent.

## Highest-value work not yet started

**Supervised single-variant effect on viral DMS.** The 95-scorer ProteinGym leaderboard is
entirely zero-shot; nobody has tested whether *supervised* adaptation beats MSA methods. 22 viral
assays, ESM-2 650M/3B + ESM-1v, position-disjoint (`contiguous`/`modulo`) splits, against
published ESCOTT/GEMME/S3F_MSA **plus** a supervised one-hot+biochemical control. See HANDOFF §4 P4.
