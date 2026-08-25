# Group 0 Repo Audit

## Sync Audit

Commands completed:

- `git status --short`
- `git remote -v`
- `git fetch origin --prune`
- `git branch -a | grep viral-benchmark-continuation`
- `git switch -c viral-benchmark-continuation --track origin/viral-benchmark-continuation`
- `git pull --ff-only origin viral-benchmark-continuation`
- `git status`
- `git branch --show-current`
- `git rev-parse HEAD`
- `git rev-parse origin/viral-benchmark-continuation`
- `git log --oneline --decorate -n 20`

Verified:

```text
HEAD == origin/viral-benchmark-continuation == 2cf5c967ce6739026e7fabd2381b394b5add4b64
```

Development branch:

```text
xiang/viral-benchmark-continuation-a1
```

## Required Documents

Read:

- `reports/PAPER_DESIGN.md`
- `reports/PROTOCOL.md`
- `reports/TESTED_MATRIX.md`
- `reports/RESEARCH_PLAN.md`
- `README.md`

## Required Harness

Present:

- `scripts/common/capacity_sweep.py`
- `scripts/common/partial_overlap_audit.py`
- `scripts/common/build_strict_splits.py`
- `scripts/common/paired_bootstrap.py`
- `scripts/track_a_benchmarks/`
- `scripts/track_b_viral/`

Finding:

- `capacity_sweep.py` supported `splice`, `hvue`, and `virobench`; GENEB loader was absent.
- GENEB data was not present in the checked paths.

## Scientific State From Source Docs

- Splice is still a positive control, but only a small-margin one.
- LucaVirus is withdrawn as a viral positive anchor.
- HVUE Pathogenecity and Transmissibility are `UNEVALUABLE`, not clean negatives.
- HVUE Host_Tropism is evaluable and has a tiny NT-v2 positive result of about `+0.0059`.
- GENEB sentinel remains the critical open A1 gap because no CNN baseline was run.
- ProteinGym viral supervised and ViroBench genus remain future Group 3 tasks.
- A2 Borzoi-style long-range anchor is defined but not to be launched before manual approval.

## Deliverables

- `reports/xiang_execution/current_experiment_state.csv`
- `reports/xiang_execution/current_experiment_state.json`
- `reports/xiang_execution/CURRENT_STATE.md`
- `reports/xiang_execution/GROUP0_REPO_AUDIT.md`
