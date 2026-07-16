"""Sequential route-decision pipeline with preflight, benchmarks, and reports."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = Path("/home/teacher1/UT-project1/project1")
DEFAULT_PYTHON = "/home/teacher1/miniconda3/envs/UT-p1/bin/python"

CHECKPOINTS = {
    "best_gd_from_task5a": "data/phase2/checkpoints_tuned/refseq_gd_projinit_full_ar5_s200/weights.safetensors",
    "gd_random_control": "data/phase2/checkpoints_tuned/refseq_gd_projinit_random_ar5_s1000/weights.safetensors",
    "projection_rank32": "data/phase2/checkpoints_projection_adaptive_rank32/projopt_host5_9_coro0_10_adaptive_basis_rank32/weights.safetensors",
}


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def append_step(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


class PipelineError(RuntimeError):
    pass


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)


def set_status(path: Path, status: str, **extra: Any) -> None:
    write_json(path, {"updated_at": now(), "status": status, **extra})


def assert_output(path: Path, description: str) -> None:
    if not path.exists():
        raise PipelineError(f"missing {description}: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise PipelineError(f"empty {description}: {path}")


def scientific_state_md(project_root: Path) -> str:
    return "\n".join(
        [
            "# Current Scientific State Freeze",
            "",
            "- Full benchmark: GD shows forget signal but insufficient selectivity; RMU is safer on retain but forget remains weak.",
            "- Task 0-3 / Task 7 / Task 7-R / Task 7-S: current capability gate remains confounded and cannot serve as a formal success gate.",
            "- Task 5A / 5B: projection is mainly a readout-displacement / historical mechanism anchor; strongest GD may still reflect general damage; localized GD remains the highest-priority old-route candidate; RMU is a retain-stable secondary reference.",
            "- Success criterion for this round: decide whether the old route is still worth continued investment, not whether success has already been proven.",
            "",
            "## Fixed Sources",
            "",
            "- `data/phase2/audits/task5a_identity_reaudit_20260713`",
            "- `data/phase2/audits/task7r_capability_probe_20260714`",
            "- `data/phase2/audits/task7s_clean_gate_20260715`",
            "- `docs/full_benchmark_results.md`",
            "- `docs/causal_chain_validation_plan.md`",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--out-dir", default="data/phase2/route_decision_20260715")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-manifest", default="data/benchmarks/hvue_gue_pilot_slim_manifest.csv")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--validation-max-rows", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--metric-for-best", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    reports_dir = out_dir / "reports"
    benchmarks_dir = out_dir / "benchmarks"
    status_path = out_dir / "pipeline_status.json"
    steps_path = out_dir / "pipeline_steps.jsonl"
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(name: str, command: list[str], expected_outputs: list[Path] | None = None) -> None:
        append_step(steps_path, {"event": "start", "step": name, "command": command, "time": now()})
        set_status(status_path, "running", step=name, command=command)
        started = time.time()
        result = run_command(command, project_root)
        elapsed = time.time() - started
        append_step(
            steps_path,
            {
                "event": "finish",
                "step": name,
                "returncode": result.returncode,
                "elapsed_sec": elapsed,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
                "time": now(),
            },
        )
        if result.returncode != 0:
            set_status(status_path, "failed", step=name, returncode=result.returncode)
            raise PipelineError(f"stage failed {name}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        for output in expected_outputs or []:
            assert_output(output, f"{name} output")

    set_status(status_path, "started", out_dir=str(out_dir), project_root=str(project_root))

    if not args.skip_preflight:
        preflight_cmd = [
            args.python_bin,
            "-u",
            "phase2/preflight_route_decision.py",
            "--project-root",
            str(project_root),
            "--python-bin",
            args.python_bin,
            "--out-dir",
            str(out_dir.relative_to(project_root)),
            "--device",
            args.device,
            "--benchmark-manifest",
            args.benchmark_manifest,
            "--cpu-threads",
            str(args.cpu_threads),
            "--train-batch-size",
            str(args.train_batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--validation-max-rows",
            str(args.validation_max_rows),
            "--max-length",
            str(args.max_length),
            "--epochs",
            str(args.epochs),
            "--max-steps",
            str(args.max_steps),
            "--eval-every",
            str(args.eval_every),
            "--patience",
            str(args.patience),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--lora-rank",
            str(args.lora_rank),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--metric-for-best",
            args.metric_for_best,
            "--seed",
            str(args.seed),
        ]
        run_stage(
            "preflight",
            preflight_cmd,
            expected_outputs=[
                out_dir / "preflight" / "preflight_env.json",
                out_dir / "preflight" / "preflight_paths.json",
                out_dir / "preflight" / "preflight_inputs.json",
                out_dir / "preflight" / "retain_audit.json",
                out_dir / "preflight" / "run_manifest_lock.json",
            ],
        )

    (reports_dir / "current_scientific_state.md").write_text(scientific_state_md(project_root) + "\n")
    append_step(steps_path, {"event": "write", "step": "freeze_current_state", "time": now()})

    common_eval = [
        "--benchmark-manifest",
        args.benchmark_manifest,
        "--benchmark-scope",
        "all",
        "--resume",
        "--device",
        args.device,
        "--batch-size",
        "1",
        "--cpu-threads",
        str(args.cpu_threads),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--patience",
        str(args.patience),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--metric-for-best",
        args.metric_for_best,
        "--max-length",
        str(args.max_length),
        "--seed",
        str(args.seed),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--validation-max-rows",
        str(args.validation_max_rows),
        "--discard-task-checkpoint",
    ]

    for name in ["gd_random_control", "best_gd_from_task5a", "projection_rank32"]:
        out_path = benchmarks_dir / name
        command = [args.python_bin, "-u", "phase2/eval_benchmarks.py", "--out-dir", str(out_path), "--ckpt", CHECKPOINTS[name], *common_eval]
        run_stage(
            f"benchmark:{name}",
            command,
            expected_outputs=[out_path / "eval_benchmarks_summary.json", out_path / "eval_benchmarks.csv", out_path / "eval_benchmarks_progress.json"],
        )

    for name, rel in {
        "base": "data/phase2/base_benchmarks_slim/eval_benchmarks_summary.json",
        "gd_loc_s1000": "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/eval_benchmarks_summary.json",
    }.items():
        assert_output(project_root / rel, f"reused summary {name}")
        append_step(steps_path, {"event": "reuse", "step": f"reuse:{name}", "path": rel, "time": now()})

    summary_cmd = [
        args.python_bin,
        "-u",
        "phase2/summarize_route_decision.py",
        "--project-root",
        str(project_root),
        "--route-root",
        str(out_dir.relative_to(project_root)),
    ]
    run_stage(
        "summary",
        summary_cmd,
        expected_outputs=[
            reports_dir / "route_decision_main_comparison.csv",
            reports_dir / "route_decision_main_comparison.md",
            reports_dir / "route_decision_summary.json",
            reports_dir / "route_decision_one_pager.md",
        ],
    )

    summary = read_json(reports_dir / "route_decision_summary.json")
    go_no_go = summary.get("go_no_go", "no_go")
    final_decision = summary.get("final_decision", "unknown")
    winner = summary.get("winner")
    append_step(steps_path, {"event": "decision", "step": "go_no_go", "go_no_go": go_no_go, "winner": winner, "final_decision": final_decision, "time": now()})

    if go_no_go == "go" and winner:
        diag_root = out_dir / "controlled_diagnostic"
        diag_root.mkdir(parents=True, exist_ok=True)
        for name in [winner, "gd_random_control"]:
            if name == "gd_loc_s1000":
                ckpt = "data/phase2/checkpoints_tuned/refseq_gd_projinit_loc_ar5_s1000/weights.safetensors"
            elif name == "best_gd_from_task5a":
                ckpt = CHECKPOINTS["best_gd_from_task5a"]
            else:
                ckpt = CHECKPOINTS[name]
            out_path = diag_root / name
            command = [args.python_bin, "-u", "phase2/eval_benchmarks.py", "--out-dir", str(out_path), "--ckpt", ckpt, *common_eval]
            run_stage(
                f"controlled_diagnostic:{name}",
                command,
                expected_outputs=[out_path / "eval_benchmarks_summary.json", out_path / "eval_benchmarks.csv"],
            )
    else:
        append_step(steps_path, {"event": "skip", "step": "controlled_diagnostic_if_go", "reason": "go_no_go=false", "time": now()})

    final_report_lines = [
        "# Route Decision Final Report",
        "",
        f"- final_decision: `{final_decision}`",
        f"- go_no_go: `{go_no_go}`",
        f"- winner: `{winner or 'none'}`",
        "",
        "## Deliverables",
        "",
        "- `reports/current_scientific_state.md`",
        "- `reports/route_decision_main_comparison.csv`",
        "- `reports/route_decision_main_comparison.md`",
        "- `reports/route_decision_one_pager.md`",
        "- `reports/route_decision_summary.json`",
        "- `preflight/preflight_env.json`",
        "- `preflight/preflight_paths.json`",
        "- `preflight/preflight_inputs.json`",
        "- `preflight/run_manifest_lock.json`",
        "- `pipeline_status.json`",
        "- `pipeline_steps.jsonl`",
        "",
        "All outputs remain diagnostic and non-formal.",
    ]
    (reports_dir / "final_route_decision_report.md").write_text("\n".join(final_report_lines) + "\n")
    set_status(status_path, "complete" if go_no_go == "go" else "stopped_by_go_no_go", final_decision=final_decision, winner=winner)


if __name__ == "__main__":
    main()
