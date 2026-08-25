# What has actually been tested — benchmark × method × regime

Legend: **D** done and trusted · **P** partial · **X** not run · **!** run but invalid/superseded

## Track A — benchmark & method audit

| benchmark | tasks | k-mer | CNN ladder | frozen probe | LoRA | full FT |
|:--|--:|:--|:--|:--|:--|:--|
| NT splice | 3 | **D** | **D** (13-cell) | **D** (3 models) | X | **D** (3 models × 3 LR × 3 seeds) |
| GENEB sentinel | 13 | **D** (fair refit) | **X ← critical gap** | **D** (3 models, layer NOT swept) | X | X |
| GENEB remainder | 87 | X | X | X | X | X |
| GUE non-viral | 12 | **D** | **P** (1 arch) | X | X | vs published only |
| NT benchmark | 18 | **D** | **P** (1 arch) | X | X | vs published only |
| EPI | 6 | **D** | **D** | X | X | X |
| DART-Eval task 1 | 1 | **D** | **D** | X | X | X |

## Track B — viral capability

| benchmark | tasks | k-mer | CNN ladder | alignment | frozen probe | LoRA | full FT |
|:--|--:|:--|:--|:--|:--|:--|:--|
| ViroBench ALL/times | 5 levels | **D** (matched ctx at family; **P** elsewhere) | **D** | **D** family only | **D** (layer swept at family) | X | X |
| ViroBench ALL/genus | 5 levels | X | X | X | **X ← harder split, no gLM data** | X | X |
| ViroBench DNA/RNA subsets | — | **P** | X | X | **P** | X | X |
| HVUE Host_Tropism (strict) | 1 | **D** | **D** | X | **D** (3 models × 3 seeds) | X | X |
| HVUE Path./Trans. | 2 | ! | ! | X | ! | ! | ! |
| HVUE composition-gated splits | 3 | ! | ! | X | ! | ! | ! |
| GUE viral (deduped) | 2 | **D** | **D** | X | X | X | **D** (3 models × 3 seeds) |
| ProteinGym viral | 22 | X | X | X (MSA needed) | X | X | X |
| Antibody escape | 3 | **D** (MSA) | X | X | X | **P** | **P** |

**!** on HVUE Path./Trans. = the task retains 96/5,194 and 60/4,956 homology-clean test rows;
no method's score there separates memorisation from generalisation. **!** on composition-gated
splits = built in the k-mer's feature space and gated on k-mer degradation; invalid by construction.

## Models covered

| model | params | HVUE | ViroBench | GUE viral | splice | GENEB |
|:--|--:|:--|:--|:--|:--|:--|
| NT-v2-500M | 500M | D | D | D | D | D |
| GENA-LM-bert-base-t2t | 111M | D | D | D | D | D |
| HyenaDNA-medium-160k | 7M | D | D | D | D | D |
| LucaVirus | 944M | X | D (4 layers) | X | X | X |
| Evo-1-8k | 6.5B | P (LoRA) | X | X | X | X |
| ESM-2 / ESM-1v / pLMs | — | — | — | — | — | X (ProteinGym pending) |

## Splits used, and what each licenses

| split | construction | licenses | status |
|:--|:--|:--|:--|
| composition-cluster (gated) | kmer5→PCA→KMeans, accepted iff k-mer drops ≥0.03 | **nothing** | invalid |
| identity-disjoint | `easy-cluster -c 0.9` (90% bidirectional) | duplicate removal | insufficient |
| **strict** | `easy-search`, drop test rows ≥70% id over ≥30% cov | homology-clean claims | **only valid HVUE split** |
| ViroBench `times` | train ≤2017-10-21 / test ≥2020-02-03 | temporal generalisation | clean (2.2% dup) |
| ViroBench `genus` | whole genera withheld | **harder** generalisation | **unused** |
| NT splice | chromosome-disjoint, 0% exact/revcomp | FT positive control | clean, verified |
| GUE official | unverified | in-distribution only | must not be called OOD |
| GUE virus_covid deduped | 11.9% exact dups removed | usable | clean |

## Headline numbers we stand behind

| claim | evidence |
|:--|:--|
| HVUE Path./Trans. cannot support homology-clean evaluation | 96/5,194 and 60/4,956 surviving rows |
| Baseline receptive field, not capacity, binds on positional tasks | ResNet 9.44M/RF89 = 0.336 vs U-Net 0.26M/global = 0.951 |
| Harness reproduces published splice numbers when regime matches | NT-v2 FT 0.9674 ± 0.0025 vs published 0.971–0.984 |
| Viral family taxonomy is an alignment task | alignment 0.7383 vs best gLM 0.6148; 0.9915 acc on aligned subset at 10–13% coverage |
| No viral advantage at matched context | ViroBench family: k-mer3-5 0.6231 > NT-v2 0.6111 (dev-selected) |
| Deduplication does not change the GUE viral ranking | k-mer 0.7171 > NT-v2 0.6764 on n=8,050 |

## Claims withdrawn

- ~~NT-v2 significantly beats k-mer on ViroBench family~~ — context asymmetry + incomplete k-mer sweep.
- ~~Our k-mer is the ViroBench ceiling~~ — alignment is +0.16 above it.
- ~~gLMs beat the baseline on GENEB 13/13~~ — k-mer-anchored; no CNN was run.
- ~~Published splice gap is +0.31 to +0.60~~ — receptive-field artifact; real gap +0.02–0.03.
- ~~LucaVirus's deficit is a read-out artifact~~ — persists across 4 layers (−0.28).
