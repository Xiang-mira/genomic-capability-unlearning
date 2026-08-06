"""Plan and run confirmatory Host Tropism LoRA attackers for Stage 1."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUT_ROOT = Path("data/phase2/lora_subspace_targeting_20260729")
DEFAULT_MANIFEST = Path("data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv")
DEFAULT_PYTHON = Path("/home/teacher1/miniconda3/envs/UT-p1/bin/python")
DEFAULT_TASK = "hvue_human_host_tropism"
KMER_AUROC_TEST = 0.8554553475149496
FROZEN_STAGE1_RULES = {
    "frozen_before_confirmatory_completion_utc": "2026-07-30T03:40:00+00:00",
    "decision_inputs": [
        "validation AUROC/MCC grouping",
        "weight-space stability",
        "matched control comparisons",
    ],
    "non_decision_inputs": [
        "test AUROC",
        "test MCC",
        "test prediction labels",
        "test excess metrics",
    ],
    "dual_metric_strong_definition": {
        "validation_auroc": "> 0.8339622641509434",
        "validation_mcc": "> 0.5260285646629745",
    },
    "near_parity_tolerance": {
        "validation_auroc_within": 0.02,
        "validation_mcc_within": 0.05,
    },
    "minimum_independent_adapters_for_consensus": 3,
    "minimum_rank_specific_strong_adapters": 3,
    "strong_vs_control_rule": "real validation-strong top-k overlap must exceed singular-value-matched random orientation controls by both fold and absolute-margin thresholds",
    "cross_rank_stability": "hard threshold only for go_to_stage2; not required for rank_specific_go_to_stage2",
    "allowed_outcomes": [
        "go_to_stage2",
        "rank_specific_go_to_stage2",
        "conditional_go_requires_more_evidence",
        "heterogeneous_recovery_paths",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_for_run(out_root: Path, run_id: str, rank: int, lr: str, seed: int) -> list[str]:
    out_dir = out_root / "confirmatory_adapter_reruns" / run_id
    pred_dir = out_root / "confirmatory_adapter_predictions" / run_id
    return [
        str(DEFAULT_PYTHON),
        "-u",
        "phase2/eval_benchmarks.py",
        "--benchmark-manifest",
        str(DEFAULT_MANIFEST),
        "--benchmark-scope",
        "task",
        "--task-filter",
        DEFAULT_TASK,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--epochs",
        "3",
        "--max-steps",
        "0",
        "--eval-every",
        "200",
        "--validation-max-rows",
        "0",
        "--test-max-rows",
        "0",
        "--lr",
        lr,
        "--lora-rank",
        str(rank),
        "--lora-alpha",
        str(rank * 2),
        "--lora-dropout",
        "0.0",
        "--train-batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--max-length",
        "512",
        "--device",
        "cuda:0",
        "--cpu-threads",
        "16",
        "--metric-for-best",
        "auroc",
        "--split-type",
        "cluster_disjoint",
        "--kmer-baseline-score",
        str(KMER_AUROC_TEST),
        "--export-predictions",
        "--prediction-dir",
        str(pred_dir),
    ]


def run_status(out_root: Path, run_id: str) -> dict[str, str]:
    run_dir = out_root / "confirmatory_adapter_reruns" / run_id
    pred_dir = out_root / "confirmatory_adapter_predictions" / run_id
    results = run_dir / "eval_benchmarks.csv"
    ckpt = run_dir / "checkpoints" / DEFAULT_TASK / "best.pt"
    val = pred_dir / f"{DEFAULT_TASK}_val_predictions.csv"
    test = pred_dir / f"{DEFAULT_TASK}_test_predictions.csv"
    complete = all(path.exists() for path in (results, ckpt, val, test))
    return {
        "status": "complete" if complete else "pending",
        "results_path": str(results) if results.exists() else "",
        "adapter_path": str(ckpt) if ckpt.exists() else "",
        "validation_prediction_path": str(val) if val.exists() else "",
        "test_prediction_path": str(test) if test.exists() else "",
    }


def build_plan(out_root: Path) -> dict[str, object]:
    batches = [
        {
            "batch_id": "confirmatory_batch_a_rank16_lr5e-5",
            "purpose": "replicate the strongest corrected validation dual-metric configuration family; rank16/lr5e-5 had 2 of 3 selected reruns pass validation AUROC and MCC",
            "rank": 16,
            "learning_rate": "5e-5",
            "seeds": [45, 46, 47, 48, 49],
        },
        {
            "batch_id": "confirmatory_batch_b_rank32_lr5e-5",
            "purpose": "different-rank promising configuration; rank32/lr5e-5 includes a validation dual-metric strong seed and is the rank32 counterpart to Batch A",
            "rank": 32,
            "learning_rate": "5e-5",
            "seeds": [45, 46, 47, 48, 49],
        },
    ]
    runs = []
    for batch in batches:
        for seed in batch["seeds"]:
            run_id = f"{batch['batch_id']}_seed{seed}"
            runs.append(
                {
                    "run_id": run_id,
                    "batch_id": batch["batch_id"],
                    "rank": batch["rank"],
                    "learning_rate": batch["learning_rate"],
                    "seed": seed,
                    "frozen_settings": {
                        "split": "cluster_disjoint",
                        "input_policy": "first 512 raw characters as used by eval_benchmarks.py",
                        "target_modules": "all Linear modules under every Evo block",
                        "training_budget": "epochs=3,max_steps=0,eval_every=200,patience=3",
                        "mcc_threshold_rule": "select on validation only, freeze for test",
                        "metric_for_best": "auroc",
                    },
                    "command": command_for_run(out_root, run_id, int(batch["rank"]), str(batch["learning_rate"]), int(seed)),
                    **run_status(out_root, run_id),
                }
            )
    return {
        "status": "planned",
        "generated_at_utc": utc_now(),
        "stage2_formal_allowed": False,
        "stage2_exploratory_single_run_allowed": True,
        "frozen_stage1_rules": FROZEN_STAGE1_RULES,
        "validation_baselines": {
            "auroc": 0.8339622641509434,
            "mcc": 0.5260285646629745,
        },
        "test_baselines": {
            "auroc": KMER_AUROC_TEST,
            "mcc": 0.5991934875548052,
        },
        "batches": batches,
        "runs": runs,
    }


def run_queue(args: argparse.Namespace) -> None:
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    for dirname in ("confirmatory_adapter_reruns", "confirmatory_adapter_predictions", "confirmatory_effective_updates"):
        (out_root / dirname).mkdir(exist_ok=True)
    plan = build_plan(out_root)
    write_json(out_root / "confirmatory_attack_plan.json", plan)
    log_path = args.log_path or out_root / "confirmatory_attack_queue.log"
    registry_path = out_root / "confirmatory_attack_registry.json"
    pending = [row for row in plan["runs"] if row["status"] != "complete"]
    if args.max_runs > 0:
        pending = pending[: args.max_runs]
    registry = {
        "status": "running",
        "started_at_utc": utc_now(),
        "planned_run_count": len(plan["runs"]),
        "pending_this_invocation": len(pending),
        "completed": [],
        "failed": [],
        "log_path": str(log_path),
    }
    write_json(registry_path, registry)
    with log_path.open("a") as log:
        log.write(f"[{utc_now()}] queue_start pending={len(pending)} total={len(plan['runs'])}\n")
        for row in pending:
            run_id = str(row["run_id"])
            command = [str(part) for part in row["command"]]
            registry["current_run_id"] = run_id
            registry["current_command"] = command
            registry["updated_at_utc"] = utc_now()
            write_json(registry_path, registry)
            log.write(f"[{utc_now()}] run_start {run_id}\n")
            log.write("COMMAND " + " ".join(command) + "\n")
            log.flush()
            started = time.time()
            result = subprocess.run(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - started
            if result.returncode == 0:
                registry["completed"].append({"run_id": run_id, "elapsed_sec": elapsed})
                log.write(f"[{utc_now()}] run_complete {run_id} elapsed_sec={elapsed:.3f}\n")
            else:
                registry["failed"].append({"run_id": run_id, "elapsed_sec": elapsed, "returncode": result.returncode})
                registry["status"] = "failed"
                registry["updated_at_utc"] = utc_now()
                write_json(registry_path, registry)
                log.write(f"[{utc_now()}] run_failed {run_id} returncode={result.returncode} elapsed_sec={elapsed:.3f}\n")
                log.flush()
                if not args.keep_going:
                    return
            registry["updated_at_utc"] = utc_now()
            write_json(registry_path, registry)
            log.flush()
    final_plan = build_plan(out_root)
    write_json(out_root / "confirmatory_attack_plan.json", final_plan)
    registry["status"] = "complete" if not registry["failed"] else "complete_with_failures"
    registry["completed_at_utc"] = utc_now()
    registry.pop("current_run_id", None)
    registry.pop("current_command", None)
    write_json(registry_path, registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main() -> None:
    run_queue(build_parser().parse_args())


if __name__ == "__main__":
    main()
