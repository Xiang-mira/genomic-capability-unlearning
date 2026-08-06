# Downstream-First Reaudit Runbook

This runbook implements the new rule: primary conclusions come from target and
retain downstream behavior, not from fixed/fresh probes or perplexity.

## 1. Audit Artifacts

```bash
python phase2/downstream_reaudit.py audit
```

Outputs:

- `data/phase2/downstream_reaudit/checkpoint_inventory.csv`
- `data/phase2/downstream_reaudit/task_inventory.csv`
- `data/phase2/downstream_reaudit/shortcut_baseline_inventory.csv`
- `data/phase2/downstream_reaudit/split_integrity_report.md`
- `data/phase2/downstream_reaudit/artifact_inventory.json`
- `data/phase2/downstream_reaudit/downstream_reaudit_eval_manifest.csv`

Use `--hash-files` for the final locked run. It hashes the large benchmark
manifest and checkpoint weights, so it is intentionally optional.

Before using any new random/disjoint split pair for formal claims, run:

```bash
python phase2/check_split_validity.py \
  --manifest data/phase2/downstream_reaudit/downstream_reaudit_eval_manifest.csv \
  --baseline-csv data/phase2/kmer_baselines/kmer_metrics.csv \
  --out-csv data/phase2/downstream_reaudit/split_validity.csv
```

Formal route decisions should only use `primary_forget` rows. The
`negative_control` rows are diagnostic-only and must not affect ranking.

## 2. Generate Downstream Commands

```bash
python phase2/downstream_reaudit.py write-commands \
  --python-bin /home/teacher1/miniconda3/envs/UT-p1/bin/python \
  --device cuda:0
```

This writes:

- `data/phase2/downstream_reaudit/run_downstream_reaudit.sh`
- `data/phase2/downstream_reaudit/run_downstream_reaudit_commands.json`

The generated script runs supervised LoRA downstream evaluation for every
runnable checkpoint and seed. It defaults to the filtered manifest produced by
the audit step, where primary/secondary target groups are separated and the
known Calici-confounded HVUE tasks are excluded. It skips audit-only checkpoints
with missing weights.

For a smoke test:

```bash
python phase2/downstream_reaudit.py write-commands \
  --benchmark-manifest data/benchmarks/hvue_gue_pilot_slim_manifest.csv \
  --cohort global_host_tropism \
  --checkpoint projection_rank32 \
  --seeds 42
```

Then run the generated shell script.

## 3. Full Reaudit

After reviewing the commands, run:

```bash
bash data/phase2/downstream_reaudit/run_downstream_reaudit.sh
```

Each checkpoint/seed writes to:

```text
data/phase2/downstream_reaudit/<cohort>/<checkpoint>/seed_<seed>/
```

The expected downstream result file is `eval_benchmarks.csv`.

## 4. Aggregate Selection Decisions

```bash
python phase2/downstream_reaudit.py aggregate
```

Outputs:

- `data/phase2/downstream_reaudit/downstream_group_scores.csv`
- `data/phase2/downstream_reaudit/downstream_selection_summary.csv`
- `data/phase2/downstream_reaudit/downstream_reaudit_report.md`

Only `selective_unlearning_candidate` is a pass state for recovery experiments.
Probe/PPL evidence remains diagnostic and should be read from the existing
Task5A audit directory:

```text
data/phase2/audits/task5a_identity_reaudit_20260713/
```

## 5. Stage 1 TAR Smoke

For repeated Stage 1 work, it is tempting to first materialize a filtered
manifest containing only the two formal target tasks:

```bash
python - <<'PY'
import csv, sys
from pathlib import Path
csv.field_size_limit(sys.maxsize)
src = Path('data/phase2/downstream_reaudit/downstream_reaudit_eval_manifest.csv')
out = Path('data/phase2/downstream_reaudit/downstream_reaudit_eval_manifest_stage1_formal_targets.csv')
tasks = {'hvue_human_host_tropism', 'hvue_human_virus_pathogenicity_cini'}
with src.open(newline='') as fin, out.open('w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row.get('task') in tasks:
            writer.writerow(row)
print(out)
PY
```

That task-only filtered manifest is still useful for quick random-split probes,
but it does not carry explicit `split_type` metadata and must not be used as
evidence for formal `cluster_disjoint` TAR smoke.

For true Stage 1 formal-target smoke, first build task-specific manifests with
explicit split semantics:

```bash
python phase2/audit_stage1_target_sources.py
python phase2/build_stage1_formal_target_manifests.py
```

If you want one orchestration entrypoint for the current host-only Stage 1
workflow, use:

```bash
/home/teacher1/miniconda3/envs/UT-p1/bin/python phase2/run_stage1_hostonly_formal.py \
  --build-option-b \
  --option-b-best-candidate-json data/phase2/stage1_option_b_initializer/best_candidate.json \
  --execute-smoke \
  --execute
```

This writes a preview plan to
`data/phase2/stage1_hostonly_formal_plan.json` before executing the steps.

This currently emits:

- `data/phase2/stage1_formal_target_manifests/hvue_human_host_tropism_cluster_disjoint.csv`
- `data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv`
- `data/phase2/stage1_formal_target_manifests/stage1_formal_target_manifest_report.json`
- `data/phase2/stage1_formal_target_manifests/stage1_target_source_audit.json`

At the moment, the generated formal-target manifest only contains
`hvue_human_host_tropism` on a validated `cluster_disjoint` split. The report
explicitly records `hvue_human_virus_pathogenicity_cini` as missing until a
task-specific disjoint source is provided.

The source audit also records why CINI is still blocked in-repo today: the raw
HVUE CINI CSVs expose only `sequence,label`, and the unified manifest retains
only `family=mixed` plus coarse grouping, which is not enough to reconstruct a
validated `cluster_disjoint` split.

Generate Stage 1 TAR smoke commands against the explicit formal-target
manifest:

```bash
python phase2/tar_feasibility_smoke.py \
  --project-root . \
  --benchmark-manifest data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv \
  --tasks hvue_human_host_tropism \
  --recipes k0_no_attack,lora_r8_lr1e5_l5l9 \
  --test-max-rows 256 \
  --out-dir data/phase2/tar_feasibility_smoke_formal_targets
```

The runner now validates that the requested `--split-type` is actually present
in the manifest, so a random-only manifest cannot silently masquerade as
`cluster_disjoint`.

This writes `commands.json` with one entry per `variant_id × attack_recipe_id`. The
script now bootstraps its own import path, so it can be called directly without
manually exporting `PYTHONPATH`.

After an `--execute` run completes, the runner also writes a flat summary table
at `stage1_smoke_summary.csv` under the chosen `--out-dir`.

If `--kmer-baseline-csv` is provided, the runner backfills both
`kmer_baseline_score` and `metric_excess_over_kmer` into each downstream result
row and into the flat smoke summary.

If the smoke has already been run and you only want to attach a newly finished
k-mer baseline, rerun the same command with `--backfill-only` and without
`--execute`.

To generate a k-mer baseline for the validated host-only formal-target
manifest:

```bash
python phase2/eval_kmer_baseline.py \
  --benchmark-manifest data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv \
  --out-csv data/phase2/kmer_baselines/stage1_formal_targets_available_kmer.csv \
  --max-length 512 \
  --kmer-min 3 \
  --kmer-max 6 \
  --c-grid 0.001,0.01,0.1,1,10
```

To compare initializer / attacked-checkpoint variants, pass a JSON list via
`--variant-spec-json`. Each entry may include:

- `variant_id`
- `initializer_label`
- `k0_ckpt`
- `attacked_ckpt`
- `attacked_ckpt_by_recipe`
- `readout_disruption_flag`

To build a reusable host-only smoke spec from the currently checked-in control
checkpoints:

```bash
python phase2/build_stage1_smoke_variants.py
```

This writes:

- `data/phase2/stage1_variant_specs/stage1_hostonly_smoke_variants.json`
- `data/phase2/stage1_variant_specs/stage1_hostonly_smoke_variants_report.json`

The generated spec intentionally includes only variants that are actually
available in the repo today:

- `option_a_base`
- `legacy_projinit_control`

The companion report explicitly records that the planned `Option B/C/D`
initializer families still do not have checked-in artifacts.

Once a generated Option B initializer checkpoint exists at
`data/phase2/stage1_option_b_initializer/hostonly/weights.safetensors`, the
same builder upgrades `option_b_classification_ce` from missing to available
and wires it to `recipe_ids=["k0_no_attack"]` until attacked checkpoints are
materialized. If that default path is absent, the builder now falls back to
`data/phase2/stage1_option_b_initializer/best_candidate.json` and auto-wires
the selected `weights_path`.

To build that minimal host-only Option B artifact:

```bash
/home/teacher1/miniconda3/envs/UT-p1/bin/python phase2/build_stage1_option_b_initializer.py \
  --benchmark-manifest data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv \
  --target-task hvue_human_host_tropism \
  --split-type cluster_disjoint \
  --retain-csv data/phase2/splits/retain.csv \
  --target-train-max-rows 256 \
  --target-val-max-rows 128 \
  --target-test-max-rows 128 \
  --retain-max-rows 256 \
  --elicitation-steps 20 \
  --ascent-steps 20 \
  --out-dir data/phase2/stage1_option_b_initializer/hostonly
```

To compare a few named Option B candidate settings without hand-editing each
command, use the sweep helper:

```bash
/home/teacher1/miniconda3/envs/UT-p1/bin/python phase2/run_stage1_option_b_sweep.py
```

This writes a plan preview to
`data/phase2/stage1_option_b_initializer/sweep_plan.json` and a summary table
to `data/phase2/stage1_option_b_initializer/sweep_summary.csv`. Add
`--execute` to actually run the listed configs.

To materialize the current best Option B checkpoint choice across the known
summary tables:

```bash
python phase2/select_stage1_option_b_candidate.py
```

This writes `data/phase2/stage1_option_b_initializer/best_candidate.json`.

As of July 25, 2026, the current best checked-in candidate is
`retain_heavy_40x40`, which points to
`data/phase2/stage1_option_b_initializer/sweep_runs/retain_heavy_40x40/weights.safetensors`.

To turn the current host-only Stage 1 artifacts into a strict Stage 2
initializer-ablation plan, use:

```bash
python phase2/plan_stage2_initializer_ablation.py \
  --variant-spec-json data/phase2/stage1_variant_specs_best_optionb/stage1_hostonly_smoke_variants.json \
  --existing-summary-csv data/phase2/tar_feasibility_smoke_best_optionb_k0_compare/stage1_smoke_summary.csv
```

This writes a reduced two-variant spec plus a readiness report under
`data/phase2/stage2_initializer_ablation/`. At the moment it should report that
`Option A` and the current best `Option B` are only jointly runnable on
`k0_no_attack`; attacked-recipe comparison stays blocked until attacked
checkpoints exist for the selected initializer arm(s).

Once attacked checkpoints have been materialized for more than one recipe, merge
the per-recipe variant specs into one reusable compare spec:

```bash
python phase2/merge_stage2_attacked_variants.py \
  --variant-spec-json data/phase2/stage2_attacked_compare_variants.json \
  --variant-spec-json data/phase2/stage2_attacked_compare_r16_variants.json \
  --out-json data/phase2/stage2_attacked_compare_r8_r16_variants.json
```

As of July 25, 2026, the merged host-only attacked compare can already run on:

- `lora_r8_lr1e5_l5l9`
- `lora_r16_lr5e5_l5l9`

using:

```bash
/home/teacher1/miniconda3/envs/UT-p1/bin/python phase2/tar_feasibility_smoke.py \
  --project-root . \
  --python-bin /home/teacher1/miniconda3/envs/UT-p1/bin/python \
  --benchmark-manifest data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv \
  --tasks hvue_human_host_tropism \
  --variant-spec-json data/phase2/stage2_attacked_compare_r8_r16_variants.json \
  --recipes lora_r8_lr1e5_l5l9,lora_r16_lr5e5_l5l9 \
  --validation-max-rows 32 \
  --test-max-rows 32 \
  --out-dir data/phase2/tar_feasibility_smoke_stage2_attacked_compare_r8_r16 \
  --execute
```

Example:

```json
[
  {
    "variant_id": "option_a_base",
    "initializer_label": "none"
  },
  {
    "variant_id": "option_b_ce_init",
    "initializer_label": "classification_ce",
    "k0_ckpt": "data/phase2/checkpoints_tuned/refseq_gd_projinit_random_ar5_s1000/weights.safetensors",
    "attacked_ckpt_by_recipe": {
      "lora_r8_lr1e5_l5l9": "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s500/weights.safetensors",
      "full_lr1e5_all": "data/phase2/checkpoints_tuned/refseq_gd_projinit_full_ar5_s200/weights.safetensors"
    },
    "readout_disruption_flag": "readout_disruption"
  }
]
```

Example host-only formal smoke using the generated variant spec:

```bash
/home/teacher1/miniconda3/envs/UT-p1/bin/python phase2/tar_feasibility_smoke.py \
  --project-root . \
  --python-bin /home/teacher1/miniconda3/envs/UT-p1/bin/python \
  --benchmark-manifest data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv \
  --tasks hvue_human_host_tropism \
  --variant-spec-json data/phase2/stage1_variant_specs/stage1_hostonly_smoke_variants.json \
  --recipes k0_no_attack,lora_r8_lr1e5_l5l9,full_lr1e5_all \
  --validation-max-rows 128 \
  --test-max-rows 256 \
  --out-dir data/phase2/tar_feasibility_smoke_formal_targets_hostonly_variants \
  --execute
```

## Guardrails

- Do not rank `global_host_tropism` checkpoints against `coronaviridae`
  checkpoints.
- Do not treat fresh separability as a failure criterion by itself.
- Do not treat fixed probe collapse as success by itself.
- Do not replace viral/GUE retain downstream results with retain PPL.
