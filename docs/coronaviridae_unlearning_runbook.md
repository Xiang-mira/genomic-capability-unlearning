# Coronaviridae Unlearning Runbook

This runbook implements the corrected plan for Coronaviridae erasure. Do not
launch GPU experiments while unrelated GPU jobs are active; the commands below
are grouped into CPU-only checks and later GPU runs.

## CPU-Only Checks

Use these after code changes. They do not run Evo forward passes.

```bash
python -m py_compile \
  phase2/eval_unlearn.py \
  phase2/project_probe_nullspace.py \
  phase2/unlearn_probe.py \
  phase2/build_adaptive_probe_basis.py \
  phase2/aggregate_task2_results.py \
  phase2/build_gue_augmented_retain.py
```

Build adaptive bases only from cached feature chunks:

```bash
python phase2/build_adaptive_probe_basis.py \
  --feature-dir data/family_targets/coronaviridae/features \
  --out-dir data/phase2/adaptive_probe_bases \
  --target-name coronaviridae \
  --layers 0-10 \
  --max-rank 8 \
  --stop-separability 0.55
```

Current CPU status:

- Static checks pass for the corrected evaluation, projection, probe-guided
  training, aggregation, GUE augmentation, and adaptive-basis scripts.
- Adaptive Coronaviridae bases have been built for layers 0-10 under
  `data/phase2/adaptive_probe_bases/coronaviridae/`.
- With `max_rank=8`, every layer hit the rank cap before the 0.55 stop gate;
  final validation separability remains high, from 0.9024 at layer 6 to 0.9899
  at layer 0. Treat this as evidence that rank 8 is not enough on cached base
  features, not as a passed erasure gate.

## GPU Runs After The GPU Is Free

First re-audit the current best checkpoint with corrected metrics:

```bash
python phase2/eval_unlearn.py \
  --ckpt data/phase2/checkpoints_projection_opt/projopt_host5_9_coro4_10_coro125/weights.safetensors \
  --internal-target-config phase2/internal_eval_targets.json \
  --layers 0-15 \
  --fresh-probe \
  --device cuda:0
```

Run the early-layer projection sweep:

```bash
python phase2/run_task2_sweeps.py \
  --config phase2/sweep_configs/projection_coro_early.json \
  --out-dir data/phase2/checkpoints_projection_coro_early \
  --internal-layers 0-15 \
  --device cuda:0
```

After adaptive bases are built, run adaptive-rank projection:

```bash
python phase2/run_task2_sweeps.py \
  --config phase2/sweep_configs/projection_adaptive_basis.json \
  --out-dir data/phase2/checkpoints_projection_adaptive \
  --internal-layers 0-15 \
  --device cuda:0
```

For projection-initialized gradient unlearning, use the corrected objective:

```bash
python phase2/unlearn_probe.py \
  --init-from-run <best_projection_run> \
  --forget-objective logit_zero \
  --alpha-retain 10 \
  --steps 500 \
  --lr 5e-6 \
  --save-steps 100,200,300,400,500 \
  --device cuda:0
```

## Acceptance Gates

- Report fixed probe AUROC and `separability=max(AUC,1-AUC)`.
- Report fresh probe AUROC/separability on held-out splits.
- Initial gate: max fresh separability below 0.60 on target layers.
- Formal linear gate: max fresh separability at or below 0.55 on layers 0-12, then full-layer confirmation.
- Retain gate: GUE mean AUROC absolute delta must stay at or above -0.05, with per-task and worst-task deltas reported.
- Recovery gate: fixed-budget LoRA recovery should remain below HVUE Coronaviridae AUROC 0.70.
