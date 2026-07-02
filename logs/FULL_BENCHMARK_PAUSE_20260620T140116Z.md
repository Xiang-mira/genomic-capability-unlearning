# Full Benchmark Pause Record

- Paused at: `2026-06-20T14:01:16Z`
- Stop method: one `SIGINT` sent to the active evaluator (`PID 361513`); the evaluator recorded `status: interrupted`, and its parent workflow then exited with status 130.
- GPU state after pause: no experiment compute process remains.
- Main log: `logs/hvue_complete_selection_and_full_20260620_001843.log`
- Watchdog note: the existing watchdog (`PID 330186`) restarted the top-level pilot wrapper once at `2026-06-20T14:01:50Z`. The restarted process group and watchdog were then stopped together with `SIGINT`. A final delayed check found no benchmark, evaluator, wrapper, or watchdog process and no GPU compute process. The brief restart did not alter the completed pilot Base CSV or its `complete 6/6` progress record.

## Completed work

- Pilot selection: complete for Base plus all 12 candidates (`13/13` result directories complete).
- Rankings: complete.
- Selected full runs, in execution order:
  1. `lora_rmu_full_sc200_s200`
  2. `lora_gd_full_ar5_s500`
  3. `lora_gd_full_ar3_s200`
  4. `lora_rmu_full_sc50_s200`
- Full Benchmark Base: `1/44` tasks complete.
- Completed Full Benchmark task: `gue_emp_h3`.
- Interrupted task: `gue_emp_h3k14ac` at approximately step 300. This task has no committed result row and will restart from its beginning.

## Preserved artifacts

- Pilot rankings CSV: `data/phase2/benchmark_pilot_lora/pilot_rankings.csv`
  - SHA-256: `88da0a601f6f0bfa7bd59754a9a805f25f171fc74c6ea27f4cf59d41e3ba7833`
- Pilot rankings JSON: `data/phase2/benchmark_pilot_lora/pilot_rankings.json`
  - SHA-256: `88c34f1d6f767925f3646e9104642dea4646fd5f6309b3eba3e2086a2a54219d`
- Full Base result CSV: `data/phase2/full_benchmarks_lora_selected/base/eval_benchmarks.csv`
  - SHA-256: `2571ecd2755084640f9d8264d5cf63701f2ec635d9c90b39bcc53f5ce512f503`
- Full Base progress: `data/phase2/full_benchmarks_lora_selected/base/eval_benchmarks_progress.json`
  - Recorded state: `interrupted`, `completed_tasks: 1`, `expected_tasks: 44`, `exit_reason: received SIGINT`.

## Resume procedure

From `/home/teacher1/UT-project1/project1`, run:

```bash
PHASE2_PYTHON=/home/teacher1/miniconda3/envs/UT-p1/bin/python \
  bash phase2/run_selected_full_benchmark.sh \
  >> logs/hvue_complete_selection_and_full_20260620_001843.log 2>&1
```

The workflow passes `--resume`. It should load the fixed pilot rankings, skip the completed `gue_emp_h3` Base row, restart `gue_emp_h3k14ac`, finish the remaining Base tasks, then evaluate the four selected checkpoints and write final rankings.

Do not restart with `phase2/run_hvue_complete_selection_and_full.sh`; that wrapper begins with the already-completed pilot stage. Resume directly with `phase2/run_selected_full_benchmark.sh` as shown above.
