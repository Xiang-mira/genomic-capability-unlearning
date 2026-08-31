"""Plan and optionally execute the standalone Experiment 3 prefilter stage.

The prefilter stage separates three concerns:
1. checkpoint/load integrity smoke;
2. retain-set perplexity and internal diagnostics via eval_unlearn.py;
3. lightweight downstream triage on host-tropism + GUE + viral-retain tasks.

This script reads the standalone candidate registry produced by
phase2/standalone_single_lora_intervention.py and writes executable command
artifacts for every arm.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata


DEFAULT_OUT_DIR = PROJECT_ROOT / "data/phase2/standalone_single_lora_intervention_20260730"
DEFAULT_PREFILTER_DIRNAME = "standalone_prefilter_runs"
DEFAULT_PROJECT_PYTHON = project_python()
DEFAULT_BENCHMARK_MANIFEST = PROJECT_ROOT / "data/benchmarks/hvue_gue_manifest.csv"
# Freeze prefilter retain triage to the established slim GUE retain panel plus
# the full viral non-target panel already present in the manifest.
DEFAULT_TRIAGE_TASKS = (
    "hvue_human_host_tropism",
    "gue_emp_h3",
    "gue_human_tf_1",
    "gue_mouse_1",
    "gue_prom_300_notata",
    "gue_splice_reconstructed",
    "virobench_all_taxon_genus",
    "virobench_all_taxon_times",
    "virobench_dna_taxon_genus",
    "virobench_dna_taxon_times",
    "virobench_rna_taxon_genus",
    "virobench_rna_taxon_times",
)
DEFAULT_GUE_PROTOCOL_STATUS = "partial_GUE_retain_evaluation"
DEFAULT_GUE_PROTOCOL_NOTE = (
    "current prefilter plan uses a validated 5-task GUE retain subset; "
    "an unambiguous frozen 7-task GUE protocol was not recoverable from repo artifacts during audit"
)
DEFAULT_STARTED_AT = "2026-07-30T11:30:34Z"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def serializable_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return argparse.Namespace(**payload)


def parse_tasks(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def arm_run_dir(out_dir: Path, arm_id: str) -> Path:
    return out_dir / DEFAULT_PREFILTER_DIRNAME / arm_id


def prefilter_arm_status(out_dir: Path, arm_id: str) -> dict[str, bool]:
    run_dir = arm_run_dir(out_dir, arm_id)
    return {
        "integrity": (run_dir / "integrity_smoke.json").exists(),
        "eval_unlearn": (run_dir / "eval_unlearn" / "eval_ppl.json").exists(),
        "downstream": (run_dir / "downstream" / "eval_benchmarks_summary.json").exists(),
    }


def summarize_prefilter_state(out_dir: Path, command_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_arm: list[dict[str, Any]] = []
    completed_arms = 0
    active_arm = ""
    for row in command_rows:
        status = prefilter_arm_status(out_dir, str(row["arm_id"]))
        done = all(status.values())
        if done:
            completed_arms += 1
        elif any(status.values()) and not active_arm:
            active_arm = str(row["arm_id"])
        per_arm.append({"arm_id": row["arm_id"], **status, "completed": done})
    if not active_arm:
        for row in per_arm:
            if not row["completed"]:
                active_arm = str(row["arm_id"])
                break
    overall_status = "planned_prefilter_not_run"
    if completed_arms == len(command_rows):
        overall_status = "complete"
    elif completed_arms > 0 or active_arm or any(any(v for k, v in row.items() if k in {"integrity", "eval_unlearn", "downstream"}) for row in per_arm):
        overall_status = "in_progress"
    return {
        "status": overall_status,
        "started_at": DEFAULT_STARTED_AT,
        "completed_arms": completed_arms,
        "active_arm": active_arm,
        "total_arms": len(command_rows),
        "arm_progress": per_arm,
    }


def build_eval_unlearn_cmd(args: argparse.Namespace, arm: dict[str, Any], run_dir: Path) -> list[str]:
    ckpt = "base" if arm["arm_type"] == "base" else str(arm["checkpoint_path"])
    cmd = [
        args.python_bin,
        "-u",
        "phase2/eval_unlearn.py",
        "--ckpt",
        ckpt,
        "--out-dir",
        str(run_dir / "eval_unlearn"),
        "--model-dir",
        str(args.model_dir),
        "--config-path",
        args.config_path,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--max-eval",
        str(args.max_eval),
        "--checkpoint-name",
        str(arm["arm_id"]),
        "--method-family",
        str(arm["arm_type"]),
        "--seed",
        str(args.seed),
    ]
    if arm["arm_type"] == "base":
        cmd.append("--base-checkpoint")
    return cmd


def build_downstream_cmd(args: argparse.Namespace, arm: dict[str, Any], run_dir: Path) -> list[str]:
    cmd = [
        args.python_bin,
        "-u",
        "phase2/eval_benchmarks.py",
        "--benchmark-manifest",
        str(args.benchmark_manifest),
        "--benchmark-scope",
        "all",
        "--task-filter",
        ",".join(args.triage_tasks),
        "--out-dir",
        str(run_dir / "downstream"),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--validation-max-rows",
        str(args.validation_max_rows),
        "--test-max-rows",
        str(args.test_max_rows),
        "--lr",
        str(args.lr),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-length",
        str(args.max_length),
        "--device",
        args.device,
        "--cpu-threads",
        str(args.cpu_threads),
        "--discard-task-checkpoint",
        "--resume",
        "--export-predictions",
        "--prediction-dir",
        str(run_dir / "predictions"),
    ]
    if arm["arm_type"] != "base":
        cmd[3:3] = ["--ckpt", str(arm["checkpoint_path"])]
    return cmd


def build_integrity_smoke_cmd(args: argparse.Namespace, arm: dict[str, Any], run_dir: Path) -> list[str]:
    ckpt_expr = "''" if arm["arm_type"] == "base" else repr(str(arm["checkpoint_path"]))
    base_flag = "True" if arm["arm_type"] == "base" else "False"
    script = f"""
from pathlib import Path
import json
import torch
from phase1.utils import load_local_checkpoint
from phase2.checkpoint_io import apply_checkpoint
from phase2.project_python import project_python
from evo.tokenizer import CharLevelTokenizer

model = load_local_checkpoint({repr(str(args.model_dir))}, {repr(args.config_path)}, device={repr(args.device)})
if not {base_flag}:
    apply_checkpoint(model, {ckpt_expr}, log_prefix='standalone-prefilter-smoke')
model.eval()
tok = CharLevelTokenizer(512)
ids = torch.tensor([tok.tokenize('ACGT' * 64)], dtype=torch.long, device={repr(args.device)})
mask = torch.ones_like(ids)
with torch.no_grad():
    logits, _ = model(ids, padding_mask=mask)
payload = {{
  'arm_id': {repr(arm['arm_id'])},
  'arm_type': {repr(arm['arm_type'])},
  'nan_count': int(sum(torch.isnan(v).sum().item() for v in model.state_dict().values() if torch.is_floating_point(v))),
  'inf_count': int(sum(torch.isinf(v).sum().item() for v in model.state_dict().values() if torch.is_floating_point(v))),
  'forward_ok': True,
  'logit_isfinite': bool(torch.isfinite(logits).all().item()),
}}
Path({repr(str(run_dir / 'integrity_smoke.json'))}).write_text(json.dumps(payload, indent=2) + '\\n')
print(json.dumps(payload))
"""
    return [args.python_bin, "-c", script]


def load_registry_arms(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return list(payload.get("arms") or [])


def write_command_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = out_dir / DEFAULT_PREFILTER_DIRNAME
    runs_root.mkdir(exist_ok=True)

    arms = load_registry_arms(args.candidate_registry_json)
    command_rows: list[dict[str, Any]] = []
    for arm in arms:
        run_dir = arm_run_dir(out_dir, str(arm["arm_id"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        integrity_cmd = build_integrity_smoke_cmd(args, arm, run_dir)
        ppl_cmd = build_eval_unlearn_cmd(args, arm, run_dir)
        downstream_cmd = build_downstream_cmd(args, arm, run_dir)
        command_rows.append(
            {
                "arm_id": arm["arm_id"],
                "arm_type": arm["arm_type"],
                "status": arm["status"],
                "checkpoint_path": arm.get("checkpoint_path", ""),
                "integrity_smoke_cmd": integrity_cmd,
                "retain_ppl_cmd": ppl_cmd,
                "downstream_triage_cmd": downstream_cmd,
            }
        )

    script_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in command_rows:
        script_lines.append(f"# {row['arm_id']}")
        script_lines.append(shell_join(row["integrity_smoke_cmd"]))
        script_lines.append(shell_join(row["retain_ppl_cmd"]))
        script_lines.append(shell_join(row["downstream_triage_cmd"]))
        script_lines.append("")
    script_path = runs_root / "run_standalone_prefilter.sh"
    script_path.write_text("\n".join(script_lines) + "\n")
    script_path.chmod(0o755)

    write_json(runs_root / "standalone_prefilter_commands.json", {"commands": command_rows})
    write_csv(
        out_dir / "standalone_prefilter_metrics.csv",
        [],
        [
            "arm_id",
            "arm_type",
            "integrity_status",
            "nan_count",
            "inf_count",
            "forward_ok",
            "logit_isfinite",
            "retain_ppl",
            "base_retain_ppl",
            "retain_ppl_delta_fraction",
            "host_tropism_metric",
            "gue_retain_mean",
            "viral_retain_mean",
            "prefilter_decision",
            "notes",
        ],
    )
    live_state = summarize_prefilter_state(out_dir, command_rows)
    write_json(
        out_dir / "standalone_prefilter_report.json",
        {
            "generated_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            "status": live_state["status"],
            "started_at": live_state["started_at"],
            "completed_arms": live_state["completed_arms"],
            "active_arm": live_state["active_arm"],
            "total_arms": live_state["total_arms"],
            "arm_progress": live_state["arm_progress"],
            "candidate_registry_json": str(args.candidate_registry_json),
            "benchmark_manifest": str(args.benchmark_manifest),
            "triage_tasks": list(args.triage_tasks),
            "gue_protocol_status": DEFAULT_GUE_PROTOCOL_STATUS,
            "gue_protocol_note": DEFAULT_GUE_PROTOCOL_NOTE,
            "command_script": str(script_path),
            "arm_count": len(command_rows),
        },
    )
    (out_dir / "standalone_prefilter_report.md").write_text(
        "\n".join(
            [
                "# Standalone Prefilter Report",
                "",
                "The prefilter command plan is active.",
                "",
                f"- Candidate arms: `{len(command_rows)}`",
                f"- Status: `{live_state['status']}`",
                f"- Started at: `{live_state['started_at']}`",
                f"- Completed arms: `{live_state['completed_arms']}`",
                f"- Active arm: `{live_state['active_arm']}`",
                f"- Benchmark manifest: `{args.benchmark_manifest}`",
                f"- Triage tasks: `{', '.join(args.triage_tasks)}`",
                f"- GUE protocol status: `{DEFAULT_GUE_PROTOCOL_STATUS}`",
                f"- Script: `{script_path}`",
                "",
            ]
        )
        + "\n"
    )
    write_metadata(
        runs_root / "standalone_prefilter_commands_metadata.json",
        build_run_metadata(
            args=serializable_args(args),
            data_paths=[str(args.candidate_registry_json), str(args.benchmark_manifest)],
            extra={
                "phase": "standalone_prefilter_plan",
                "arm_count": len(command_rows),
                "prefilter_status": live_state["status"],
                "started_at": live_state["started_at"],
                "completed_arms": live_state["completed_arms"],
                "active_arm": live_state["active_arm"],
                "triage_tasks": list(args.triage_tasks),
                "gue_protocol_status": DEFAULT_GUE_PROTOCOL_STATUS,
                "gue_protocol_note": DEFAULT_GUE_PROTOCOL_NOTE,
                "script_path": str(script_path),
            },
        ),
    )
    return {"script_path": script_path, "commands": command_rows}


def execute_commands(commands: list[dict[str, Any]], *, only_stage: str) -> None:
    stage_key = {
        "integrity": "integrity_smoke_cmd",
        "ppl": "retain_ppl_cmd",
        "downstream": "downstream_triage_cmd",
        "all": None,
    }[only_stage]
    for row in commands:
        items = [stage_key] if stage_key else ["integrity_smoke_cmd", "retain_ppl_cmd", "downstream_triage_cmd"]
        for key in items:
            cmd = row[key]
            print(f"[standalone-prefilter] execute {row['arm_id']} {key}: {shell_join(cmd)}", flush=True)
            code = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
            if code != 0:
                raise SystemExit(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-registry-json", type=Path, default=DEFAULT_OUT_DIR / "standalone_candidate_registry.json")
    parser.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST)
    parser.add_argument("--triage-tasks", type=parse_tasks, default=list(DEFAULT_TRIAGE_TASKS))
    parser.add_argument("--python-bin", default=DEFAULT_PROJECT_PYTHON)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-eval", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-max-rows", type=int, default=1000)
    parser.add_argument("--test-max-rows", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-stage", choices=["integrity", "ppl", "downstream", "all"], default="all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = write_command_artifacts(args)
    print(f"[standalone-prefilter] wrote command plan to {payload['script_path']}")
    if args.execute:
        execute_commands(payload["commands"], only_stage=args.execute_stage)


if __name__ == "__main__":
    main()
