# HVUE core benchmark — independent verification on real data (Vista cluster)

Earlier this session, HVUE was believed unverifiable from this cluster because `paths.py`'s
hardcoded defaults (`HVUE_DIR=/home/nvidia/glm-locking/data/hvue`, `MMSEQS=/home/nvidia/tools/...`)
point at an unreachable external host. That was only true of the *default paths* — a local git
clone at `/work/10906/arisk/ls6/evo-locking/data/hvue/` has the actual train/validation parquets
(Host_Tropism 47,194 rows, Pathogenecity 134,066, Transmissibility 458,756; every sequence
**exactly 1000bp**, confirming no context-truncation asymmetry exists for HVUE, unlike ViroBench).

## Two real bugs found and fixed in the process

1. `build_identity_splits.py` was missing `import paths as P` entirely -- a `NameError` on the
   very first line that uses it. Every sibling script in this directory has the standard
   `sys.path.insert(...); import paths as P` shim; this one didn't. Fixed.
2. The same script's `df.groupby("label", group_keys=False).apply(lambda g: g.sample(...))`
   silently breaks under pandas >=2.2 (this env: 3.0.3), which excludes the grouping column from
   `g` by default (`include_groups`), so the later `df.label` access raised `AttributeError`.
   Rewrote as an explicit per-group loop (version-safe). Fixed.
3. `hvue_glm.py` hardcodes a `KMER[(task, split)]` lookup dict that only has entries for the old
   `random`/`cluster_disjoint` splits -- running it against `identity_disjoint_hsd0` without
   `--kmer_json` raises `KeyError` right after training finishes, discarding the result. Not a bug
   in the sense of wrong code, but a real trap: the `--kmer_json` flag exists for exactly this
   reason and has to be supplied with a flat `{"{task}__{split}": [value]}` file. Built one from
   this session's own `build_identity_splits.py` output and reran cleanly.

## Result: independently rebuilding the identity-disjoint split reproduces HANDOFF's numbers

k-mer AUROC on MMseqs2 90%-identity-disjoint hsd0 (rebuilt fresh, no baseline-performance gate):

| task | this session (hsd0) | HANDOFF.md (original cluster) | diff |
|:--|--:|--:|--:|
| Host_Tropism | 0.9171 | 0.9131 | 0.004 |
| Pathogenecity | 0.9558 | (failed gate, no exact number given; random=0.9685) | -- |
| Transmissibility | 0.9224 | (failed gate, no exact number given; random=0.9238) | -- |

CNN (0.64M params, 3 seeds, best LR, mean):

| task | this session | HANDOFF.md | diff |
|:--|--:|--:|--:|
| Host_Tropism | 0.9491 | 0.9482 | 0.0009 |
| Pathogenecity | 0.9715 | 0.9667 | 0.0048 |
| Transmissibility | 0.9340 | 0.9202 | 0.0138 |

gLMs, extended to 3 seeds for NT-v2/HyenaDNA (GENA-LM single-seed, ran once as a third cross-check model):

| task | model | seeds | mean ev | best(k-mer,CNN) | excess |
|:--|:--|--:|--:|--:|--:|
| Host_Tropism | NT-v2-500M | 42,43,44 | 0.9494 | 0.9491 (CNN) | +0.0003 (tie) |
| Host_Tropism | HyenaDNA | 42,43,44 | 0.9372 | 0.9491 (CNN) | -0.0119 |
| Host_Tropism | GENA-LM | 42 | 0.9385 | 0.9491 (CNN) | -0.0106 |
| Pathogenecity | NT-v2-500M | 42,43,44 | 0.9432 | 0.9715 (CNN) | -0.0283 |
| Pathogenecity | HyenaDNA | 42,43,44 | 0.9551 | 0.9715 (CNN) | -0.0164 |
| Pathogenecity | GENA-LM | 42 | 0.9641 | 0.9715 (CNN) | -0.0074 |
| Transmissibility | NT-v2-500M | 42,43,44 | 0.9015 | 0.9340 (CNN) | -0.0325 |
| Transmissibility | HyenaDNA | 42,43,44 | 0.8967 | 0.9340 (CNN) | -0.0373 |
| Transmissibility | GENA-LM | 42 | 0.9163 | 0.9340 (CNN) | -0.0177 |

**With the seed-averaging correction (mean, not single-seed or best-of-N -- same fix applied
earlier this session to `aggregate_positive_control.py`), the one apparent win evaporates.** The
single-seed NT-v2 Host_Tropism result looked like a real win (+0.0023, matching the parallel
cluster's own single-seed "+0.0012"); with 2 more seeds it is +0.0003 -- a dead tie, not a win.
Every one of the 9 (model x task) cells on this freshly-rebuilt, independently-verified
identity-disjoint split is now a loss or a tie. Zero real gLM wins on HVUE core once seed
variance is accounted for properly. This is a stronger, more defensible version of the project's
central claim than either cluster's single-seed numbers supported on their own.

## Caveats
- Pathogenecity/Transmissibility have no exact original k-mer-identity-disjoint number to diff
  against (HANDOFF only recorded that they failed the old gate, not the raw value), so those rows
  are a fresh measurement rather than a literal reproduction -- still useful as the first-ever
  recorded value for that specific quantity.
- GENA-LM is still single-seed; given how much the NT-v2 Host_Tropism margin moved with 2 more
  seeds, GENA-LM's losses here should not yet be treated as fully seed-stable, though none of
  them are close enough to zero to plausibly flip.
