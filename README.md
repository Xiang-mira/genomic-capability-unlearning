# Genomic foundation model capability — benchmark audit + viral negative result

Branch: `viral-benchmark-continuation`.

Two overlapping programmes sharing one harness. **Read
[reports/RESEARCH_PLAN.md](reports/RESEARCH_PLAN.md) first.**

| | **Track A — benchmark & method audit** | **Track B — viral capability** |
|:--|:--|:--|
| question | across existing benchmarks, what does each *method class* achieve? | is there model-specific viral capability over the strongest comparator on a defensible split? |
| benchmarks | GENEB, GUE, NT, EPI, DART-Eval | HVUE, ViroBench, GUE viral, ProteinGym viral, escape |
| comparators | k-mer, CNN ladder | + alignment nearest-hit, Kraken2, MSA |
| contribution | published gaps are largely baseline artifacts | scoped negative result, positively controlled by Track A |

Track B's negative result is only credible because Track A shows the same harness detects real
capability elsewhere. Track A's headline was discovered *because* Track B forced a proper comparator.

## Documents

| doc | what it is |
|:--|:--|
| **[reports/RESEARCH_PLAN.md](reports/RESEARCH_PLAN.md)** | the two tracks, status, and what each still needs |
| **[reports/PROTOCOL.md](reports/PROTOCOL.md)** | 27 binding rules; each one exists because violating it changed a conclusion here |
| **[reports/TESTED_MATRIX.md](reports/TESTED_MATRIX.md)** | benchmark × method × regime, what is done / partial / invalid, and claims withdrawn |
| [reports/BASELINE_CAPACITY_CEILING.md](reports/BASELINE_CAPACITY_CEILING.md) | the receptive-field finding |
| [reports/CROSS_CLUSTER_SYNTHESIS.md](reports/CROSS_CLUSTER_SYNTHESIS.md) | both clusters reconciled |
| [reports/PAPER_OUTLINE.md](reports/PAPER_OUTLINE.md) | thesis, sections, figures, defensible-vs-gated claims |
| [ANALYTICAL_RESULTS.md](ANALYTICAL_RESULTS.md) | full per-task tables |

## The four findings that shape everything else

1. **Baseline receptive field, not capacity, binds on positionally-structured tasks.** On 600bp
   splice, ResNet at 9.44M params (RF 89bp) scores 0.336 MCC; a U-Net at 0.26M (global RF) scores
   0.951 — 36× fewer parameters, +0.62 MCC. Published FM-vs-CNN splice gaps of +0.31–0.60 shrink to
   **+0.02–0.03**. Always run `capacity_sweep.py` before quoting a baseline.
2. **Homology audits need `easy-search`, not `easy-cluster`.** `-c 0.9` requires 90% *bidirectional*
   coverage and is blind to a test sequence sharing half its length at high identity. HVUE
   Pathogenecity is **80.5%** leaked by that measure and retains **96 of 5,194** homology-clean test
   rows; Transmissibility 60 of 4,956. Those two tasks cannot support a clean evaluation for anyone.
3. **On taxonomy, an alignment baseline is mandatory.** Alignment nearest-hit gets macro-F1 0.7383
   on ViroBench family vs 0.6148 for the best gLM — and 0.9915 accuracy on the 85% it can align,
   over only 10–13% of query length. Viral family taxonomy is determined by short conserved regions.
4. **Match the effective context in bp, per method, per cell.** The ViroBench frozen probe saw
   38.8% of test bp (`max_win 16 × 2048`) against an unbounded k-mer. Fixing that reversed an
   apparent +0.037 FM win to −0.012.

## Layout

```
scripts/common/              shared harness — use these for BOTH tracks
  paths.py                   env-var path resolution (VB_*); all other dirs shim to this
  capacity_sweep.py          architecture x capacity ladder (dilated/U-Net/ResNet, 0.04M-9.4M)
  partial_overlap_audit.py   easy-search partial-overlap leakage audit
  build_strict_splits.py     drop test rows with a train hit above threshold
  paired_bootstrap.py        paired bootstrap + pre-declared equivalence margins
  audit_splits.py            split-integrity + dedup checks
  download_data.py           GUE viral + ViroBench fetch
scripts/track_a_benchmarks/
  splice_positive_control.py frozen-probe arm
  splice_finetune.py         full-FT arm; LR sweep, warmup, fresh-weight guard
  gue_baselines.py           k-mer + CNN on GUE  (--test_csv to override the test set)
  gue_glm.py                 gLMs on GUE
  aggregate_splice.py        three-regime table, regimes kept separate
scripts/track_b_viral/
  hvue_cnn.py hvue_glm.py hvue_evo_lora.py
  virobench_baselines.py     k-mer + CNN  (--kmer_cap / --cnn_len -> DECLARE CONTEXT)
  virobench_frozen_probe.py  ViroBench protocol  (--layer -> SWEEP ON DEV)
  virobench_alignment_baseline.py   mmseqs nearest-hit label transfer
  virobench_glm.py build_identity_splits.py
```

Data, weights, logs and run outputs are gitignored.

## Quickstart

```bash
pip install -r scripts/common/requirements.txt
export VB_ROOT=/scratch/$USER/viral-bench VB_OUT=$VB_ROOT/results
export VB_HVUE_DIR=$VB_ROOT/data/hvue VB_MMSEQS=$(which mmseqs) HF_HOME=$VB_ROOT/hf_cache

cd scripts/common && python download_data.py --what both
python capacity_sweep.py --dataset splice --task splice_sites_all --seeds 42 43   # baseline ceiling
python partial_overlap_audit.py                                                  # leakage, before any claim
```

`paths.py` resolves everything from `VB_*` env vars; no absolute paths are hard-coded.
`VB_SPLIT_DIR` / `VB_SPLIT_SUFFIX` point `capacity_sweep.py` at an alternative HVUE split.

## Supported models

`nt_v2_500m`, `gena_lm`, `hyenadna`, `lucavirus` (ViroBench only), `evo` (LoRA). DNABERT-2 is
excluded — `config_class` conflict with transformers 4.48.

**Two loading traps:** `AutoModelForSequenceClassification` on GENA-LM silently discards all 48
pretrained LayerNorms (pre-LN checkpoint vs post-LN HF class) and the model collapses to the
majority class at every LR — use `AutoModel` + your own head, and keep
`assert_no_fresh_encoder_weights()`. And LucaVirus's tokenizer returns `token_type_ids` of the
wrong length under padding.

## The single most important open gap

**No CNN baseline exists for any GENEB task.** The entire non-viral positive control is
k-mer-anchored, and the k-mer is demonstrably the weaker of the two baselines on positional tasks.
On the one overlapping task, our CNN ladder scores **0.9527** where GENEB's fair k-mer scores 0.387
and its best-of-40 published submission scores 0.685. Until `capacity_sweep.py` is run on the
GENEB sentinel, we do not know whether those wins survive — and Track B's credibility depends on
them. See [reports/RESEARCH_PLAN.md](reports/RESEARCH_PLAN.md) §Track A.
