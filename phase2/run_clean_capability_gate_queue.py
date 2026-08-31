"""Screen-friendly queue for clean capability gate construction and mini re-audit.

This queue follows the "clean capability gate first" recovery plan:
freeze the failed Task 7-R evidence, build matched hard-negative candidates,
run validity + base-only smoke, and only then run a small checkpoint re-audit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.build_clean_capability_candidates import build_signature as candidate_build_signature
from phase2.eval_capability_probe import probe_signature as capability_probe_signature
from phase2.probe_validity_audit import HARD_STOP_ACTIONS, probe_validity_signature
from phase2.run_metadata import build_run_metadata, file_sha256, stable_hash, write_metadata
from phase2.run_task5a_identity_reaudit import TASK3_CONTEXT
from phase2.summarize_clean_capability_gate_smoke import smoke_summary_signature
from phase2.summarize_identity_capability_calibration import summary_signature as calibration_summary_signature
from phase2.project_python import project_python


class QueueStopped(RuntimeError):
    pass


DEFAULT_PROJECT_PYTHON = project_python()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def queue_metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "clean_gate_queue_metadata.json"


def base_manifest_metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "manifests" / "base_only_checkpoint_manifest_metadata.json"


def mini_manifest_metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "manifests" / "mini_task5b_checkpoint_manifest_metadata.json"


def write_queue_metadata(args: argparse.Namespace) -> None:
    write_metadata(
        queue_metadata_path(args),
        build_run_metadata(
            args=args,
            source_checkpoint=args.model_dir,
            data_paths=[
                args.source_queue_status,
                args.config_path,
                args.task7_calibration,
                Path(args.source_task7r_dir) / "capability_dataset_manifest.csv",
                Path(args.source_task7r_dir) / "identity_capability_calibration.json",
                Path(args.task5a_out_dir) / "task5a_for_task7_checkpoint_manifest.json",
                Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json",
            ],
            extra={
                "phase": "clean_capability_gate_queue",
                "task": "clean_capability_gate_queue",
                "task3_context": TASK3_CONTEXT,
                "out_root": args.out_root,
                "source_task7r_dir": args.source_task7r_dir,
                "source_queue_status": args.source_queue_status,
                "task5a_out_dir": args.task5a_out_dir,
                "task7_calibration": args.task7_calibration,
                "stop_on_low_disk_gb": args.stop_on_low_disk_gb,
                "match_quantiles": args.match_quantiles,
                "smoke_layers": args.smoke_layers,
                "mini_layers": args.mini_layers,
                "probe_seeds": args.probe_seeds,
                "c_grid": args.c_grid,
                "validity_c_grid": args.validity_c_grid,
                "n_bootstrap": args.n_bootstrap,
                "batch_size": args.batch_size,
                "device": args.device,
                "cuda_visible_devices": args.cuda_visible_devices,
                "checkpoint_format": args.checkpoint_format,
                "python": args.python,
            },
        ),
    )


def write_manifest_metadata(
    *,
    args: argparse.Namespace,
    path: Path,
    phase: str,
    task: str,
    entries: list[dict[str, Any]],
    extra_data_paths: list[str | Path],
) -> None:
    checkpoint_names = [str(entry.get("checkpoint_name", "")) for entry in entries]
    source_names = [str(entry.get("source_checkpoint_name", entry.get("checkpoint_name", ""))) for entry in entries]
    metadata_path = base_manifest_metadata_path(args) if task == "clean_gate_base_only_manifest" else mini_manifest_metadata_path(args)
    write_metadata(
        metadata_path,
        build_run_metadata(
            args=args,
            source_checkpoint=f"{task}_builder",
            data_paths=[*extra_data_paths, path],
            extra={
                "phase": phase,
                "task": task,
                "manifest_path": str(path),
                "checkpoint_count": len(entries),
                "checkpoint_names": checkpoint_names,
                "source_checkpoint_names": source_names,
                "checkpoint_name_hash": stable_hash(checkpoint_names),
                "source_checkpoint_name_hash": stable_hash(source_names),
            },
        ),
    )


def same_signature(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    actual = read_json(path)
    return actual == expected


def project_python(args: argparse.Namespace) -> str:
    candidate = args.python or os.environ.get("PYTHON") or DEFAULT_PROJECT_PYTHON
    if not Path(candidate).exists():
        raise FileNotFoundError(
            f"Configured project python does not exist: {candidate}. "
            "Expected an environment satisfying environment.yml; see README.md, Quickstart."
        )
    return candidate


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024**3)


def write_status(args: argparse.Namespace, status: str, **extra: Any) -> None:
    payload = {
        "updated_at": now(),
        "status": status,
        "task3_context": TASK3_CONTEXT,
        "out_root": args.out_root,
        "source_task7r_dir": args.source_task7r_dir,
        "task5a_out_dir": args.task5a_out_dir,
        "queue_metadata_path": str(queue_metadata_path(args)),
        **extra,
    }
    write_json(Path(args.out_root) / "clean_gate_queue_status.json", payload)


def require_disk(args: argparse.Namespace, stage: str) -> None:
    free = free_disk_gb(Path(args.out_root))
    if free < args.stop_on_low_disk_gb:
        write_status(args, "stopped_low_disk", stage=stage, free_disk_gb=free)
        raise QueueStopped(f"low disk before {stage}: free={free:.2f}G threshold={args.stop_on_low_disk_gb:.2f}G")
    print(f"[clean-queue] disk ok before {stage}: free={free:.2f}G", flush=True)


def run_command(args: argparse.Namespace, env: dict[str, str], stage: str, command: list[str]) -> None:
    print(f"[clean-queue] start {stage}: {' '.join(command)}", flush=True)
    started = time.time()
    result = subprocess.run(command, env=env)
    elapsed = time.time() - started
    print(f"[clean-queue] finish {stage}: returncode={result.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    if result.returncode != 0:
        write_status(args, "failed", stage=stage, returncode=result.returncode, elapsed_sec=elapsed)
        raise SystemExit(result.returncode)


def write_failure_diagnosis(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_task7r_dir)
    queue_status = read_json(Path(args.source_queue_status))
    calibration = read_json(source_dir / "identity_capability_calibration.json")
    decision = calibration.get("decision", {})
    out_path = Path(args.out_root) / "task7r_failure_diagnosis.md"
    lines = [
        "# Task 7-R Failure Diagnosis",
        "",
        f"- generated_at: {now()}",
        f"- task7r_status: {queue_status.get('status', 'unknown')}",
        f"- formal_success_allowed: {decision.get('formal_success_allowed')}",
        f"- capability_probe_status: {decision.get('capability_probe_status')}",
        f"- recommended_action: {decision.get('recommended_action')}",
        f"- hidden_mean_separability: {decision.get('hidden_mean_separability')}",
        f"- shortcut_best_mean_separability: {decision.get('shortcut_best_mean_separability')}",
        f"- hidden_incremental_auroc_mean: {decision.get('hidden_incremental_auroc_mean')}",
        f"- reason: {decision.get('reason')}",
        "",
        "## Blocked Downstream Actions",
        "",
        "- Task 8 is blocked until a clean capability gate exists.",
        "- Task 5B-v2 is blocked until a clean capability gate exists.",
        "- No large model-modification sweep should run before a clean gate exists.",
        "- No final checkpoint selection or formal success claim is allowed from Task 7-R.",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def base_only_manifest(args: argparse.Namespace) -> Path:
    path = Path(args.out_root) / "manifests" / "base_only_checkpoint_manifest.json"
    if path.exists() and args.resume:
        return path
    entries = [
        {
            "checkpoint_name": "base",
            "source_checkpoint_name": "base",
            "method_family": "base",
            "checkpoint_path": "",
            "checkpoint_exists": True,
            "source_selection_role": "base_reference",
        }
    ]
    write_json(
        path,
        {
            "created_at": now(),
            "task": "clean_gate_base_only_manifest",
            "checkpoints": entries,
        },
    )
    write_manifest_metadata(
        args=args,
        path=path,
        phase="clean_gate_base_only_manifest",
        task="clean_gate_base_only_manifest",
        entries=entries,
        extra_data_paths=[],
    )
    return path


def mini_checkpoint_manifest(args: argparse.Namespace) -> Path:
    path = Path(args.out_root) / "manifests" / "mini_task5b_checkpoint_manifest.json"
    if path.exists() and args.resume:
        return path
    task7_manifest = read_json(Path(args.task5a_out_dir) / "task5a_for_task7_checkpoint_manifest.json")
    task5a_summary = read_json(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json")
    summary_rows = {row.get("checkpoint_name"): row for row in task5a_summary.get("rows", [])}
    by_name = {row.get("checkpoint_name"): row for row in task7_manifest.get("checkpoints", [])}

    names = [
        "base",
        "projection_rank32",
        "best_rmu_from_task5a",
        "best_gd_from_task5a",
        "gd_random_control",
        "gd_loc_s1000",
    ]
    entries = []
    for name in names:
        if name in by_name:
            entries.append(by_name[name])
            continue
        row = summary_rows.get(name)
        if not row:
            continue
        entries.append(
            {
                "checkpoint_name": name,
                "source_checkpoint_name": name,
                "method_family": row.get("method_family", "unknown"),
                "checkpoint_path": row.get("checkpoint_path", ""),
                "checkpoint_exists": row.get("checkpoint_exists", True),
                "source_selection_role": "mini_task5b_selected_control_or_candidate",
                "retain_safety_flag": row.get("retain_safety_flag"),
            }
        )
    write_json(
        path,
        {
            "created_at": now(),
            "task": "mini_task5b_checkpoint_manifest",
            "checkpoints": entries,
        },
    )
    write_manifest_metadata(
        args=args,
        path=path,
        phase="mini_task5b_checkpoint_manifest",
        task="mini_task5b_checkpoint_manifest",
        entries=entries,
        extra_data_paths=[
            Path(args.task5a_out_dir) / "task5a_for_task7_checkpoint_manifest.json",
            Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json",
        ],
    )
    return path


def candidate_index_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "candidates" / "candidate_index.json"


def candidate_build_ready(args: argparse.Namespace) -> bool:
    signature_path = Path(args.out_root) / "candidates" / "candidate_build_signature.json"
    expected_signature = candidate_build_signature(
        Path(args.source_task7r_dir) / "capability_dataset_manifest.csv",
        argparse.Namespace(
            quantiles=args.match_quantiles,
            seeds=args.probe_seeds,
            c_grid=args.c_grid,
            n_bootstrap=args.n_bootstrap,
        ),
    )
    if not same_signature(signature_path, expected_signature):
        return False
    index = read_json(candidate_index_path(args))
    candidates = index.get("candidates", [])
    if not candidates:
        return False
    return all(
        candidate.get("candidate_dir")
        and Path(candidate["candidate_dir"]).exists()
        and Path(candidate["candidate_dir"], "capability_dataset_manifest.csv").exists()
        and Path(candidate["candidate_dir"], "capability_dataset_audit.json").exists()
        for candidate in candidates
    )


def validity_ready(args: argparse.Namespace, candidate_dir: Path) -> bool:
    signature_path = candidate_dir / "probe_validity" / "probe_validity_signature.json"
    expected_signature = probe_validity_signature(
        config_path=candidate_dir / "task7r_internal_target_config.json",
        target_hashes={
            str(path): file_sha256(path)
            for path in sorted((candidate_dir / "formal_task_manifests").glob("*.csv"))
        },
        args=argparse.Namespace(
            seeds=args.probe_seeds,
            c_grid=args.validity_c_grid,
            kmer_min=3,
            kmer_max=6,
            n_bootstrap=args.n_bootstrap,
        ),
    )
    return (candidate_dir / "probe_validity" / "probe_validity_audit.json").exists() and same_signature(
        signature_path,
        expected_signature,
    )


def smoke_ready(args: argparse.Namespace, candidate_dir: Path, smoke_dir: Path, checkpoint_manifest: Path) -> bool:
    signature_path = smoke_dir / "capability_probe_signature.json"
    expected_signature = capability_probe_signature(
        dataset_path=candidate_dir / "capability_dataset_manifest.csv",
        dataset_audit_path=candidate_dir / "capability_dataset_audit.json",
        checkpoint_manifest=checkpoint_manifest,
        args=argparse.Namespace(
            split_column="split",
            layers=args.smoke_layers,
            batch_size=args.batch_size,
            max_length=512,
            seeds=args.probe_seeds,
            c_grid=args.c_grid,
            device=args.device,
            model_dir=args.model_dir,
            config_path=args.config_path,
            checkpoint_format=args.checkpoint_format,
            save_feature_cache=True,
        ),
    )
    calibration_signature = calibration_summary_signature(
        argparse.Namespace(
            mode="task7",
            metrics=str(smoke_dir / "capability_probe_metrics.csv"),
            dataset_audit=str(candidate_dir / "capability_dataset_audit.json"),
            task5a_summary=str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
            task7_calibration=args.task7_calibration,
        )
    )
    return (
        (smoke_dir / "capability_probe_metrics.csv").exists()
        and (smoke_dir / "identity_capability_calibration.json").exists()
        and same_signature(signature_path, expected_signature)
        and same_signature(smoke_dir / "identity_capability_calibration_signature.json", calibration_signature)
    )


def smoke_summary_ready(args: argparse.Namespace) -> bool:
    summary_dir = Path(args.out_root) / "smoke_summary"
    return (summary_dir / "clean_gate_smoke_summary.json").exists() and same_signature(
        summary_dir / "clean_gate_smoke_summary_signature.json",
        smoke_summary_signature(candidate_index_path(args), Path(args.out_root) / "smoke"),
    )


def mini_task5b_ready(args: argparse.Namespace, candidate_dir: Path, mini_dir: Path, checkpoint_manifest: Path) -> bool:
    expected_probe_signature = capability_probe_signature(
        dataset_path=candidate_dir / "capability_dataset_manifest.csv",
        dataset_audit_path=candidate_dir / "capability_dataset_audit.json",
        checkpoint_manifest=checkpoint_manifest,
        args=argparse.Namespace(
            split_column="split",
            layers=args.mini_layers,
            batch_size=args.batch_size,
            max_length=512,
            seeds=args.probe_seeds,
            c_grid=args.c_grid,
            device=args.device,
            model_dir=args.model_dir,
            config_path=args.config_path,
            checkpoint_format=args.checkpoint_format,
            save_feature_cache=True,
        ),
    )
    expected_summary_signature = calibration_summary_signature(
        argparse.Namespace(
            mode="task5b",
            metrics=str(mini_dir / "capability_probe_metrics.csv"),
            dataset_audit=str(candidate_dir / "capability_dataset_audit.json"),
            task5a_summary=str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
            task7_calibration=str(Path(mini_dir).parent.parent / "smoke" / candidate_dir.name / "identity_capability_calibration.json"),
        )
    )
    return (
        (mini_dir / "capability_probe_metrics.csv").exists()
        and (mini_dir / "task5b_capability_reaudit_summary.json").exists()
        and same_signature(mini_dir / "capability_probe_signature.json", expected_probe_signature)
        and same_signature(mini_dir / "task5b_capability_reaudit_signature.json", expected_summary_signature)
    )


def run_candidate_builder(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.resume and candidate_build_ready(args):
        print("[clean-queue] skip build-clean-candidates (resume)", flush=True)
        return
    require_disk(args, "build-clean-candidates")
    command = [
        project_python(args),
        "phase2/build_clean_capability_candidates.py",
        "--source-manifest",
        str(Path(args.source_task7r_dir) / "capability_dataset_manifest.csv"),
        "--out-root",
        str(Path(args.out_root) / "candidates"),
        "--quantiles",
        args.match_quantiles,
        "--seeds",
        args.probe_seeds,
        "--c-grid",
        args.c_grid,
        "--n-bootstrap",
        str(args.n_bootstrap),
    ]
    run_command(args, env, "build-clean-candidates", command)


def run_validity_and_smoke(args: argparse.Namespace, env: dict[str, str]) -> None:
    index = read_json(candidate_index_path(args))
    base_manifest = base_only_manifest(args)
    for candidate in index.get("candidates", []):
        candidate_dir = Path(candidate["candidate_dir"])
        smoke_dir = Path(args.out_root) / "smoke" / candidate["candidate_name"]
        if not (args.resume and validity_ready(args, candidate_dir)):
            require_disk(args, f"validity:{candidate['candidate_name']}")
            run_command(
                args,
                env,
                f"validity:{candidate['candidate_name']}",
                [
                    project_python(args),
                    "phase2/probe_validity_audit.py",
                    "--internal-target-config",
                    str(candidate_dir / "task7r_internal_target_config.json"),
                    "--out-dir",
                    str(candidate_dir / "probe_validity"),
                    "--seeds",
                    args.probe_seeds,
                    "--c-grid",
                    args.validity_c_grid,
                    "--n-bootstrap",
                    str(args.n_bootstrap),
                ],
            )
        else:
            print(f"[clean-queue] skip validity:{candidate['candidate_name']} (resume)", flush=True)
        validity = read_json(candidate_dir / "probe_validity" / "probe_validity_audit.json")
        validity_action = validity.get("decision", {}).get("action", "")
        if validity.get("decision", {}).get("hard_stop") or validity_action in HARD_STOP_ACTIONS:
            write_status(
                args,
                "stopped_validity_hard_stop",
                candidate_name=candidate["candidate_name"],
                validity_action=validity_action,
                hard_stop_reasons=validity.get("decision", {}).get("hard_stop_reasons", []),
            )
            raise QueueStopped(
                f"validity hard stop for {candidate['candidate_name']}: "
                f"{','.join(validity.get('decision', {}).get('hard_stop_reasons', [])) or validity_action}"
            )
        if not (args.resume and smoke_ready(args, candidate_dir, smoke_dir, base_manifest)):
            require_disk(args, f"smoke-probe:{candidate['candidate_name']}")
            run_command(
                args,
                env,
                f"smoke-probe:{candidate['candidate_name']}",
                [
                    project_python(args),
                    "phase2/eval_capability_probe.py",
                    "--dataset-manifest",
                    str(candidate_dir / "capability_dataset_manifest.csv"),
                    "--dataset-audit",
                    str(candidate_dir / "capability_dataset_audit.json"),
                    "--checkpoint-manifest",
                    str(base_manifest),
                    "--out-dir",
                    str(smoke_dir),
                    "--layers",
                    args.smoke_layers,
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--model-dir",
                    args.model_dir,
                    "--config-path",
                    args.config_path,
                    "--checkpoint-format",
                    args.checkpoint_format,
                    "--seeds",
                    args.probe_seeds,
                    "--c-grid",
                    args.c_grid,
                ],
            )
            run_command(
                args,
                env,
                f"smoke-summary:{candidate['candidate_name']}",
                [
                    project_python(args),
                    "phase2/summarize_identity_capability_calibration.py",
                    "--mode",
                    "task7",
                    "--out-dir",
                    str(smoke_dir),
                    "--metrics",
                    str(smoke_dir / "capability_probe_metrics.csv"),
                    "--dataset-audit",
                    str(candidate_dir / "capability_dataset_audit.json"),
                    "--task5a-summary",
                    str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
                    "--task7-calibration",
                    str(args.task7_calibration),
                ],
            )
        else:
            print(f"[clean-queue] skip smoke-probe:{candidate['candidate_name']} (resume)", flush=True)


def summarize_smoke(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    summary_dir = Path(args.out_root) / "smoke_summary"
    if args.resume and smoke_summary_ready(args):
        print("[clean-queue] skip summarize-clean-smoke (resume)", flush=True)
        return read_json(summary_dir / "clean_gate_smoke_summary.json")
    run_command(
        args,
        env,
        "summarize-clean-smoke",
        [
            project_python(args),
            "phase2/summarize_clean_capability_gate_smoke.py",
            "--candidate-index",
            str(candidate_index_path(args)),
            "--smoke-root",
            str(Path(args.out_root) / "smoke"),
            "--out-dir",
            str(summary_dir),
        ],
    )
    return read_json(summary_dir / "clean_gate_smoke_summary.json")


def run_mini_task5b(args: argparse.Namespace, env: dict[str, str], selected_candidate: dict[str, Any]) -> None:
    candidate_name = selected_candidate["candidate_name"]
    candidate_dir = Path(selected_candidate["candidate_dir"])
    smoke_dir = Path(selected_candidate["smoke_dir"])
    mini_dir = Path(args.out_root) / "mini_task5b" / candidate_name
    manifest = mini_checkpoint_manifest(args)
    if args.resume and mini_task5b_ready(args, candidate_dir, mini_dir, manifest):
        print(f"[clean-queue] skip mini-task5b:{candidate_name} (resume)", flush=True)
        return
    require_disk(args, f"mini-task5b:{candidate_name}")
    run_command(
        args,
        env,
        f"mini-task5b-probe:{candidate_name}",
        [
            project_python(args),
            "phase2/eval_capability_probe.py",
            "--dataset-manifest",
            str(candidate_dir / "capability_dataset_manifest.csv"),
            "--dataset-audit",
            str(candidate_dir / "capability_dataset_audit.json"),
            "--checkpoint-manifest",
            str(manifest),
            "--out-dir",
            str(mini_dir),
            "--layers",
            args.mini_layers,
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--model-dir",
            args.model_dir,
            "--config-path",
            args.config_path,
            "--checkpoint-format",
            args.checkpoint_format,
            "--seeds",
            args.probe_seeds,
            "--c-grid",
            args.c_grid,
        ],
    )
    run_command(
        args,
        env,
        f"mini-task5b-summary:{candidate_name}",
        [
            project_python(args),
            "phase2/summarize_identity_capability_calibration.py",
            "--mode",
            "task5b",
            "--out-dir",
            str(mini_dir),
            "--metrics",
            str(mini_dir / "capability_probe_metrics.csv"),
            "--dataset-audit",
            str(candidate_dir / "capability_dataset_audit.json"),
            "--task5a-summary",
            str(Path(args.task5a_out_dir) / "task5a_identity_reaudit_summary.json"),
            "--task7-calibration",
            str(smoke_dir / "identity_capability_calibration.json"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PROJECT_PYTHON)
    parser.add_argument("--out-root", default="data/phase2/audits/task7s_clean_gate_20260715")
    parser.add_argument("--source-task7r-dir", default="data/phase2/audits/task7r_capability_probe_20260714")
    parser.add_argument(
        "--source-queue-status",
        default="data/phase2/audits/task5b_v2_clean_probe_20260714/task7r8_5bv2_queue_status.json",
    )
    parser.add_argument("--task5a-out-dir", default="data/phase2/audits/task5a_identity_reaudit_20260713")
    parser.add_argument(
        "--task7-calibration",
        default="data/phase2/audits/task7_capability_probe_20260713/identity_capability_calibration.json",
    )
    parser.add_argument("--stop-on-low-disk-gb", type=float, default=60.0)
    parser.add_argument("--match-quantiles", default="1.0,0.75,0.50")
    parser.add_argument("--smoke-layers", default="0,4,8,12,15")
    parser.add_argument("--mini-layers", default="0,4,8,12,15")
    parser.add_argument("--probe-seeds", default="42,43,44")
    parser.add_argument("--c-grid", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--validity-c-grid", default="0.001,0.01,0.1,1.0,10.0")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--model-dir", default="./evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--checkpoint-format", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    write_queue_metadata(args)
    write_status(args, "running", started_at=now())

    diagnosis = write_failure_diagnosis(args)
    print(f"[clean-queue] wrote failure diagnosis to {diagnosis}", flush=True)

    run_candidate_builder(args, env)
    run_validity_and_smoke(args, env)
    smoke_summary = summarize_smoke(args, env)
    selected = smoke_summary.get("selected_candidate")
    if not selected:
        write_status(
            args,
            "stopped_no_clean_gate",
            smoke_summary=str(Path(args.out_root) / "smoke_summary" / "clean_gate_smoke_summary.json"),
        )
        raise QueueStopped("no candidate met clean-gate smoke criteria")
    run_mini_task5b(args, env, selected)
    write_status(
        args,
        "completed",
        completed_at=now(),
        selected_candidate=selected.get("candidate_name"),
        smoke_summary=str(Path(args.out_root) / "smoke_summary" / "clean_gate_smoke_summary.json"),
    )
    print("[clean-queue] completed clean gate queue", flush=True)


if __name__ == "__main__":
    try:
        main()
    except QueueStopped as exc:
        print(f"[clean-queue] stopped: {exc}", flush=True)
        raise SystemExit(3)
