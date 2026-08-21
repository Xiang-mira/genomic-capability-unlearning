# Analytical results — both clusters, all viral benchmarks

2026-08-21. Merges this cluster's runs with Vista's (`CLUSTER_HANDOFF_FROM_VISTA.md`,
`reports/hvue_real_data_verification.md`, `reports/virobench_full_spec_baselines.md`).
Baseline convention throughout: **`best(k-mer3-5, k-mer3-6, CNN)` evaluated at the model's own
effective context**, mean over seeds (never best-of-N).

---

## 1. HVUE core — two clusters, independently rebuilt splits, converge on zero wins

MMseqs2 90%-identity-disjoint hsd0, no baseline gate, 3 seeds per cell on both sides.

**Baselines reproduce across clusters** (independent rebuilds of the split):

| task | k-mer here | k-mer Vista | CNN here | CNN Vista |
|:--|--:|--:|--:|--:|
| Host_Tropism | 0.9194 | 0.9171 | 0.9482 | 0.9491 |
| Pathogenecity | 0.9479 | 0.9558 | 0.9667 | 0.9715 |
| Transmissibility | 0.9085 | 0.9224 | 0.9202 | 0.9340 |

Max disagreement 0.014 — the split construction is reproducible.

**gLM excess over `best(k-mer, CNN)`, 3 seeds:**

| task | model | here | Vista |
|:--|:--|--:|--:|
| Host_Tropism | NT-v2-500M | **+0.0012** | **+0.0003** |
| Host_Tropism | HyenaDNA | −0.0044 | −0.0119 |
| Host_Tropism | GENA-LM | −0.0111 | −0.0106 |
| Pathogenecity | NT-v2-500M | −0.0257 | −0.0283 |
| Pathogenecity | HyenaDNA | −0.0432 | −0.0164 |
| Pathogenecity | GENA-LM | −0.0173 | −0.0074 |
| Transmissibility | NT-v2-500M | −0.0508 | −0.0325 |
| Transmissibility | HyenaDNA | −0.0727 | −0.0373 |
| Transmissibility | GENA-LM | −0.0164 | −0.0177 |

**18 cells across two clusters: 0 wins, 2 ties (+0.0012, +0.0003), 16 losses.** Both "wins" are
NT-v2 on Host_Tropism and both are inside seed noise. Vista's framing is right — this is a tie,
not a win, and my earlier "8 losses, 1 win" should read **"8 losses, 1 tie."**

The CNN is the binding baseline in **9 of 9** cells (it beats the k-mer by +0.012 to +0.029
everywhere). Reporting k-mer alone would have shown 3 spurious gLM wins.

---

## 2. ViroBench context ladder — the k-mer's advantage is NOT a context artifact

DNA / `times` (temporal) / family, 44 classes. Baselines capped at each gLM's effective context.

| effective context | best k-mer | CNN | best gLM at that context | winner |
|--:|--:|--:|:--|:--|
| 3.1 kb | **0.8469** | 0.5623 | GENA-LM 0.6378 | **k-mer +0.209** |
| 6.1 kb | 0.8866 | 0.6451 | NT-v2 **0.8867** | **tie (+0.0001)** |
| 20 kb | **0.9297** | 0.6774 | HyenaDNA 0.8784 | **k-mer +0.051** |
| whole genome (43.5 kb) | **0.9527** | 0.6563–0.6858 | — | — |

**Correction to my own earlier claim:** I reported "NT-v2 +0.021 at 6.1 kb" using only k-mer3-5
(0.8662). k-mer3-6 at that cap reaches 0.8866, making it a **dead tie**, not a win. So the
matched-context ladder now shows **zero gLM wins at any context** — k-mer wins at 3.1 kb and
20 kb and ties at 6.1 kb.

This also settles the disagreement Vista raised and then retracted: their reading that gLMs show
capability the CNN lacks is correct *against the CNN* (HyenaDNA +0.236 at 20 kb) but the CNN is
not the binding baseline on taxonomy — the k-mer is, and it beats HyenaDNA at the same 20 kb.

Whole-genome run-to-run spread: k-mer3-6 0.9370–0.9527, CNN 0.6563–0.6858. Treat ±0.02 as the
resolution on this task.

---

## 3. ViroBench at full spec — and the one genuine open positive in the programme

Vista ran `--mod ALL` (46,651), `--min_count 1`, all 5 levels, `times` split — removing the two
caveats that made earlier numbers non-comparable to ViroBench's published figures.

| level | classes | k-mer3-6 (whole genome) | CNN (20 kb) |
|:--|--:|--:|--:|
| kingdom | 18 | 0.560 | 0.294 |
| phylum | 28 | 0.555 | 0.325 |
| class | 45 | 0.520 | 0.327 |
| order | 67 | 0.599 | 0.238 |
| family | 173 | **0.570** | 0.208 |

k-mer wins every level by 0.19–0.36. But placed against ViroBench's published numbers **on the
matching split** (their T-split, from the ViroBench paper's own table):

| method | T-split taxonomy macro-F1 | source |
|:--|--:|:--|
| BLAST | 0.412 | published |
| **our k-mer3-6** | **0.570** | this work |
| **LucaVirus** | **0.649** | published |

Two readings, both important:

1. **Our k-mer beats their alignment baseline by +0.158.** ViroBench's reported model advantage
   over BLAST (+0.237 on T-split) shrinks to **+0.079** once the baseline is a properly-fitted
   whole-genome k-mer rather than BLAST. Most of their headline gap is baseline weakness.
2. **LucaVirus still leads by +0.079.** This is the **only** credible case anywhere in this
   programme of a viral foundation model beating a well-fitted classical baseline on a genuinely
   disjoint split. It is the one live candidate for a qualified unlearning target.

**Three checks before treating it as established:**
- which taxonomic level their 64.91 refers to (ours is family; if theirs is kingdom, the
  comparison is invalid — kingdom is a far easier 18-class problem)
- whether their `ALL` set matches ours
- LucaVirus's effective context — if it reads whole genomes the comparison is fair; if it reads
  less, it is winning *despite* less context, which strengthens their claim

---

## 4. GUE viral — both tasks, and a leakage finding that undercuts one

| task | classes | best baseline | GENA-LM | HyenaDNA | NT-v2 | best gLM vs baseline |
|:--|--:|--:|--:|--:|--:|--:|
| `virus_covid` | 9 | **k-mer 0.7282** | 0.6662 | 0.4499 | 0.6850 | **−0.043** |
| `virus_species_40` | 25 | **k-mer 0.4407** | 0.3643 | 0.4329 | 0.4443 | **+0.004 (tie)** |

Both losses or ties. But Vista found that **`virus_covid`'s official split has 11.9% exact-duplicate
test sequences present verbatim in train** — not previously flagged. That inflates every method's
score on that task and makes it unusable as an OOD measurement without a dedup pass. It does not
change the ranking (all methods benefit), but the absolute numbers should not be quoted.

---

## 5. Harness validity — three independent confirmations

The negatives above are not a broken-pipeline artifact:

| check | result |
|:--|:--|
| **NTv3 splice sites**, chromosome-disjoint (0% overlap verified) | gLMs beat best classical baseline by **+0.30 to +0.60 MCC** — a large, clean, disjoint-split positive control |
| **DART-Eval Task 1** reproduction | ab-initio CNN within **0.4%** of the paper's published numbers (0.8423 vs 0.8460 acc; 0.9264 vs 0.927 AUROC) |
| **Cross-cluster baseline agreement** | identity-split k-mer and CNN reproduce to ≤0.014 across two independent rebuilds |

So when a gLM loses on a viral task here, the pipeline is capable of detecting a win — it detects
one on splice sites at +0.30–0.60 MCC.

---

## 6. Split-integrity findings (cumulative)

| split | problem | magnitude |
|:--|:--|:--|
| HVUE composition-cluster | built in k-mer5-PCA space **and** accepted only if k-mer loses ≥0.03 (`GATE=0.03`) | moves k-mer by **0.12–0.18** vs identity holdout |
| HVUE identity @90% after 99% dedup | too permissive — barely OOD | k-mer drops only 0.003–0.014 |
| ViroBench `genus` ("G-split") | **82–84% genus overlap**; only taxid is held out | it is a record-level holdout, not genus-disjoint |
| GUE `virus_covid` official | **11.9% exact test duplicates in train** | inflates all methods |
| GUE EPI (positive-control sweep) | **67–80% promoter leakage** train→test, MMseqs2-verified | corroborates BENGI/LOCO-EPI critique |
| ViroBench `times` | clean: zero date overlap, 1–2% species overlap | **the split to use** |

Identity-threshold sweep (90/70/50/30%) shows the k-mer barely degrades at any threshold
(0.9066–0.9209 on Host_Tropism), so composition-clustering is not a stricter OOD test — it is a
differently-biased one.

---

## 7. Bottom line

**Across 18 HVUE cells, 4 ViroBench contexts, 2 GUE tasks, ViroBench LOFO host prediction, and
the escape and ProteinGym arms — measured on two clusters with independently rebuilt splits —
biological foundation models do not beat `best(k-mer, CNN)` at matched context on any viral task
we control end-to-end.** Best case is a tie (+0.0003 to +0.0012).

**One exception is live and unresolved:** LucaVirus on ViroBench taxonomy, T-split, appears to
beat our properly-fitted k-mer by +0.079. Everything else in the programme is a loss or a tie.
If that survives the three checks in §3, it is the qualified target the unlearning project has
been looking for — and it would be the *only* one. If it does not survive, the viral modality is
closed.

**For the unlearning question specifically**, two facts are independent of the LucaVirus check:
frozen viral representations sit 0.02–0.25 AUROC *below* baseline (nothing localisable to excise),
and a 0.64M CNN trained from scratch matches or beats every gLM on 8 of 9 HVUE cells (the
capability is not model-gated, so removing it from open weights denies it to nobody).

---

## 8. Immediate next actions

| # | action | why | where |
|:--|:--|:--|:--|
| 1 | **Confirm ViroBench's taxonomic level** for the 47.67/75.88 and 41.22/64.91 figures | decides whether the one live positive is real | paper check, no compute |
| 2 | Run **LucaVirus** on our full-spec `times` split at family level | direct head-to-head, same data, same level | either cluster; weights on HF |
| 3 | **BLAST + Kraken2** baselines at full spec | Kraken2 is a k-mer classifier by design — the honest SOTA taxonomy comparator | Vista offered to take this |
| 4 | Dedup `virus_covid` and re-measure | 11.9% leakage makes current numbers unquotable | here |
| 5 | gLMs at ViroBench full spec (46,651 rows, 5 levels) | only the 6,042-row DNA/family subset has gLM numbers | ~7.7× cost, either cluster |
| 6 | **Supervised single-variant** (22 viral ProteinGym assays, position-disjoint) | the only untouched modality; leaderboard is all zero-shot | here — data is local |
| 7 | Escape completion (LASV/SARS2/H5 LoRA + full, matched *n*) | 3 of 4 antigens incomplete | here — data is local |
