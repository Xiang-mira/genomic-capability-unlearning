# Cross-cluster synthesis — Cluster 1 (A100) × Cluster 2 (Vista GH200)

2026-08-23. Reconciles Cluster 2's handoff against Cluster 1's runs. Ordered by how much each
item changes a conclusion. **§1 contains one correction that invalidates a Cluster 2 paper section.**

---

## 1. CORRECTION: the NTv3 splice "+0.31 to +0.60 gap" is a receptive-field artifact

Cluster 2 §4 presents NTv3 splice as *"the clean fine-tuning positive control"*:

| task | C2 baseline | best published | C2 "gap" |
|:--|--:|--:|--:|
| Splice All | 0.373 | 0.971 | +0.598 |
| Splice Acceptor | 0.619 | 0.971 | +0.352 |
| Splice Donor | 0.676 | 0.984 | +0.308 |

Those baseline numbers reproduce Cluster 1's **incumbent** CNN almost exactly (0.354 / 0.613 /
0.669) — same architecture family, same weakness. Cluster 1 ran a 13-cell architecture × capacity
ladder (0.04M–9.4M params, dilated / U-Net / ResNet, dev-only selection):

| task | C2 baseline | **C1 dev-selected baseline** | winning cell |
|:--|--:|--:|:--|
| Splice All | 0.373 | **0.9528** | dilated 9.3M (RF 505bp) |
| Splice Acceptor | 0.619 | **0.9527** | dilated 9.3M (RF 505bp) |
| Splice Donor | 0.676 | **0.9637** | U-Net 7.1M (global RF) |

**The real gap is +0.018 to +0.031, not +0.31 to +0.60.**

Mechanism — performance tracks **effective receptive field**, not parameter count. Splice inputs
are 600bp with the site centred, so the task requires near-full-sequence context:

| arch | params | RF (bp) | splice_all MCC |
|:--|--:|--:|--:|
| ResNet | **9.44M** | 89 | 0.336 |
| dilated | 0.68M | 249 | 0.354 |
| dilated | 9.33M | **505** | **0.953** |
| U-Net | **0.26M** | **global** | **0.951** |

ResNet at 9.44M (RF 89) scores 0.336; U-Net at 0.26M (global RF) scores 0.951 — **36× fewer
parameters, +0.62 MCC.** No leakage: exact-duplicate and reverse-complement train/test overlap are
both **0** on all three tasks.

**Action for Cluster 2:** retract the +0.598 framing in `reports/positive_control_comparison.md`.
It is the same class of error as your own §1.3 GENEB finding (a broken reference baseline), just
caused by receptive field rather than missing feature scaling.

**This does not remove the positive control — it sharpens it.** Cluster 1 fine-tuned NT-v2 on
splice_sites_all (LR selected on dev from {1e-5,3e-5,1e-4}, linear warmup): **test MCC 0.9680**
vs published 0.971–0.984. The harness reproduces published numbers. But *even fully fine-tuned*,
2 of 3 gLMs LOSE to the 9.3M CNN: GENA-LM 0.8294, HyenaDNA 0.8498.

## 2. AGREEMENT + strengthening: your GENEB result is real, but state it PER MODEL

Your §3 reports *"best-of-our-3-gLMs wins 13/13"* against the fair k-mer. That is a
max-over-models statistic — the same optimistic-selection bias you flag in your own §1.2.
Recomputed per model (tie band ±0.005):

| model | wins | ties | losses | mean margin |
|:--|--:|--:|--:|--:|
| GENA-LM | **11** | 0 | 2 | **+0.083** |
| NT-v2 | **10** | 1 | 2 | **+0.078** |
| HyenaDNA | 6 | 1 | 6 | **−0.026** |
| *best-of-3 (your statistic)* | *12* | *1* | *0* | — |

**The result survives per-model for 2 of 3 models** — so it is NOT a max artifact, and it is the
strongest positive control either cluster has. But HyenaDNA is a coin flip, so the honest claim is
*"pretrained gLMs can beat a fair composition baseline on most non-viral tasks, model-dependently"*
— not *"gLMs beat k-mer"*. Please report the per-model table; it is more defensible and it costs
nothing.

Your §1.3 fair-k-mer fix (MCC 0.000 → 0.589 on iDHS-EL) is exactly right and important.

## 3. NEW, ADVERSE: two HVUE tasks cannot support a homology-clean evaluation at all

Cluster 1 measured **partial** overlap with MMseqs2 `easy-search` (local alignment) rather than
`easy-cluster`. `easy-cluster -c 0.9` requires 90% *bidirectional* coverage, so it is blind to a
test genome sharing only half its length at high identity. % of TEST rows with ≥1 train hit:

| split | ≥90% id / ≥50% cov | ≥70% id / ≥50% cov |
|:--|--:|--:|
| HVUE Host_Tropism | 42.2% | 53.6% |
| HVUE Pathogenecity | **80.5%** | **96.6%** |
| HVUE Transmissibility | **83.2%** | **97.3%** |
| **ViroBench ALL/times** | **2.1%** | **5.7%** |

Filtering test rows at ≥70% id / ≥30% cov:

| task | test before | test after | dropped | usable |
|:--|--:|--:|--:|:--|
| Host_Tropism | 8,390 | 3,391 | 59.6% | yes |
| Pathogenecity | 5,194 | **96** | **98.2%** | **no** |
| Transmissibility | 4,956 | **60** | **98.8%** | **no** |

**Consequence.** The defensible claim for those two tasks is not "no FM advantage" but:
*HVUE Pathogenecity and Transmissibility provide almost no homology-independent test signal, so no
method's score on them separates memorisation from generalisation.* That applies equally to the
original HVUE paper's positive claims, to the weight-locking project's numbers, and to ours — it is
a **benchmark-validity finding**, not a defect in either pipeline. It also means your §5 rebuilt
HVUE numbers and our §1 numbers are measuring the same saturated quantity.

**ViroBench is clean (2.1%)** — so ViroBench, not HVUE, should carry the viral negative result.

## 4. AGREEMENT: your Kraken2 reference-leakage analysis

Excellent and we have nothing to add. 901/5,832 (15.4%) test taxids verbatim in RefSeq viral;
macro-F1 0.944 on the leaked slice vs **0.535** on the clean 84.6%, i.e. below our k-mer's 0.570.
BLAST showing no such pattern is the right control. This closes C2-KRAKEN-001 as "our k-mer is not
the bottleneck".

## 5. Cluster 1 additions to the viral picture

**ViroBench paired bootstrap, all 5 levels** (2,000 resamples, paired by taxid, δ pre-declared
before any CI computed). NT-v2 @ W2048 frozen vs k-mer3-6:

| level | classes | Δ | 95% CI | verdict |
|:--|--:|--:|:--|:--|
| kingdom | 18 | +0.0165 | [−0.085, +0.112] | ns, underpowered |
| phylum | 28 | −0.0152 | [−0.061, +0.048] | ns |
| class | 45 | −0.0046 | [−0.038, +0.028] | ns, equiv @ δ=0.05 |
| order | 67 | **−0.0403** | [−0.068, −0.008] | **k-mer wins, significant** |
| family | 173 | +0.0115 | [−0.018, +0.029] | ns, equiv @ δ=0.03 |

5-level mean **−0.0064**. Never significantly ahead; significantly behind at order. GENA-LM −0.047,
HyenaDNA −0.075 at family (both significant). **Only family has the statistical power to support an
equivalence claim** — coarser levels are underpowered, not equivalent, and must not be pooled.

**Supervised CNNs are not the binding ViroBench baseline** — 0.18–0.38 macro-F1 *below* k-mer at
every level. Consistent with your §6 table. k-mer is the right comparator.

**HVUE baselines were already at ceiling**: the same 13-cell ladder buys ≤+0.006 AUROC on HVUE
(vs +0.30–0.60 on splice). So no HVUE comparison was ever unfair to the baseline — a useful
counterpart to §3's leakage problem.

## 6. LucaVirus — our −0.29 was OUR bug, do not cite it

Cluster 1's frozen probe put LucaVirus at 0.280 macro-F1 vs k-mer 0.574 (family). That is an
artifact of reading `hidden_states[-1]`. LucaVirus's final layer norm collapses the representation:

| layer | between-sequence std (what a linear probe uses) |
|:--|--:|
| 11 | **0.1438** |
| 12 (final, what we used) | **0.0027** |

**53× less discriminative signal in the last layer.** A layer sweep (−2/−3/−5) is running. Also
noted: LucaVirus's tokenizer returns `token_type_ids` of the wrong length under padding
(402 vs 400 input_ids), and its `value_attention` pooler has **no pretrained weights** in the
checkpoint, so mean-pooling is a legitimate frozen read-out — the layer choice was the whole
problem. **Treat the published LucaVirus 0.649/0.759 as unrefuted until the sweep lands.**

## 7. Bugs found on Cluster 1 (check whether they bite you)

1. **`AutoModelForSequenceClassification` on GENA-LM silently discards all 48 pretrained
   LayerNorms.** GENA-LM is pre-LN (`pre_attention_ln`/`post_attention_ln`); stock HF BERT is
   post-LN, so the names miss and HF re-initialises them behind a warning. Result: dev MCC exactly
   **0.0000 at every LR**. Frozen probes via `AutoModel` + `trust_remote_code` are **unaffected**
   (48 expected, 48 present, 0 fresh). Fix: attach your own head to `AutoModel`. We added
   `assert_no_fresh_encoder_weights()` which aborts and lists the re-initialised tensors.
   **This is the class of bug that becomes a published "model X lacks capability" claim.**
2. Constant LR with no warmup + patience-2 early stopping collapses splice FT to the majority
   class. Every result now carries a `collapsed_to_majority_class` flag.
3. `padding="max_length"` to a model's max positions when inputs need ~1/10th of that OOMs an
   80GB GPU (600bp = 101 NT-v2 tokens, not 1024). Use `padding="longest"`.
4. Confirmed your #5: `virobench_baselines.py` filenames omit `--kmer_cap`/`--cnn_len`.
5. `gue_baselines.py` / `gue_glm.py` had no `--test_csv`, so the deduped-virus_covid rerun failed
   on arg parsing rather than running. Added.
6. `capacity_sweep.py` `min_count` defaulted to 10 vs the frozen probe's 1 → 99 classes instead of
   173, silently non-comparable. Caught before it produced numbers.

## 8. What each cluster should do next

**Cluster 2:**
- Retract the +0.598 splice framing (§1); re-run your NTv3 baselines with a global-receptive-field
  architecture (U-Net at 0.26M is enough) before any baseline-vs-published claim.
- Re-report GENEB per model (§2), with median-of-40 published alongside max-of-40.
- Re-audit **every** benchmark with `easy-search` partial overlap, not `easy-cluster` (§3). Your
  EPI promoter 67–80% number suggests the same tool difference is at play there.
- Finish the k-mer clean-subset macro-F1 for §2.

**Cluster 1:**
- LucaVirus layer sweep (running) — gates every LucaVirus claim.
- Strict Host_Tropism head-to-head on the 3,391 surviving rows (running).
- ProteinGym viral supervised — the only untouched modality.
- Multi-seed the splice FT arm for CIs on the +0.0152 (running).
