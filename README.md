# Viral capability benchmarking — continuation branch

Branch: `viral-benchmark-continuation`. Everything needed to resume the experiments on a
different cluster. Read **[HANDOFF.md](HANDOFF.md)** first — it has the full state, results,
caveats and the prioritised to-do list.

## What this branch is for

Deciding whether any viral benchmark gives a biological foundation model **reproducible,
model-specific headroom over the strongest non-foundation baseline on a defensible split**.
That question gates the unlearning work: removing a capability is only meaningful if the
capability is real and model-specific.

Current answer, across 3 HVUE tasks × 3 split families, 2 GUE viral tasks, ViroBench taxonomy,
4 gLM architectures and 3 baseline families: **no**. See
[OUTCOME_FOR_UNLEARNING.md](OUTCOME_FOR_UNLEARNING.md).

## The two things that matter most methodologically

1. **[SPLIT_DESIGN_EXPLAINED.md](SPLIT_DESIGN_EXPLAINED.md)** — the splits every previously
   published number used were built in the k-mer baseline's own feature space *and* selected on
   the condition that the baseline lose ≥0.03 AUROC. Use the identity-disjoint splits from
   `build_identity_splits.py` instead, or taxonomic holdout.
2. **The baseline is not just a k-mer.** A 0.64M-parameter supervised CNN beats the k-mer on
   9 of 9 task×split cells and beats every pretrained gLM on 8 of 9. Any comparison must report
   `max(k-mer, CNN)`.

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
| `hvue_cnn.py` | 0.64M dilated-CNN baseline on HVUE | the binding baseline — always run it |
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
   (ViroBench genomes are median 43 kb; GENA-LM sees ~2–4 kb of that).
5. Evaluate the reported split at **every** checkpoint and report the mean over the last K —
   single-checkpoint reporting on disjoint splits swings 0.014–0.045 AUROC.

## Highest-value work not yet started

**Supervised single-variant effect on viral DMS.** The 95-scorer ProteinGym leaderboard is
entirely zero-shot; nobody has tested whether *supervised* adaptation beats MSA methods. 22 viral
assays, ESM-2 650M/3B + ESM-1v, position-disjoint (`contiguous`/`modulo`) splits, against
published ESCOTT/GEMME/S3F_MSA **plus** a supervised one-hot+biochemical control. See HANDOFF §4 P4.
