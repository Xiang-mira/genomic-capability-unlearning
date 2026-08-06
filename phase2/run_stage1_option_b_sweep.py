"""Run and summarize a small host-only Stage 1 Option B initializer sweep."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.run_metadata import build_run_metadata, write_metadata


DEFAULT_CONFIGS = [
    {
        "config_id": "smoke_2x2",
        "elicitation_steps": 2,
        "ascent_steps": 2,
        "alpha_target": 1.0,
        "alpha_retain": 1.0,
        "target_train_max_rows": 64,
        "target_val_max_rows": 64,
        "target_test_max_rows": 64,
        "retain_max_rows": 64,
    },
    {
        "config_id": "formal_20x20",
        "elicitation_steps": 20,
        "ascent_steps": 20,
        "alpha_target": 1.0,
        "alpha_retain": 1.0,
        "target_train_max_rows": 256,
        "target_val_max_rows": 128,
        "target_test_max_rows": 128,
        "retain_max_rows": 256,
    },
    {
        "config_id": "retain_heavy_20x20",
        "elicitation_steps": 20,
        "ascent_steps": 20,
        "alpha_target": 1.0,
        "alpha_retain": 2.0,
        "target_train_max_rows": 256,
        "target_val_max_rows": 128,
        "target_test_max_rows": 128,
        "retain_max_rows": 256,
    },
]


def load_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text())
        if not isinstance(payload, list) or not payload:
            raise ValueError("--config-json must contain a non-empty JSON list")
        return payload
    return list(DEFAULT_CONFIGS)


def build_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    out_dir = Path(args.out_root) / str(config["config_id"])
    cmd = [
        args.python_bin,
        "phase2/build_stage1_option_b_initializer.py",
        "--benchmark-manifest",
        args.benchmark_manifest,
        "--target-task",
        args.target_task,
        "--split-type",
        args.split_type,
        "--retain-csv",
        args.retain_csv,
        "--target-train-max-rows",
        str(config.get("target_train_max_rows", args.target_train_max_rows)),
        "--target-val-max-rows",
        str(config.get("target_val_max_rows", args.target_val_max_rows)),
        "--target-test-max-rows",
        str(config.get("target_test_max_rows", args.target_test_max_rows)),
        "--retain-max-rows",
        str(config.get("retain_max_rows", args.retain_max_rows)),
        "--elicitation-steps",
        str(config.get("elicitation_steps", args.elicitation_steps)),
        "--ascent-steps",
        str(config.get("ascent_steps", args.ascent_steps)),
        "--eval-every",
        str(config.get("eval_every", args.eval_every)),
        "--train-batch-size",
        str(config.get("train_batch_size", args.train_batch_size)),
        "--eval-batch-size",
        str(config.get("eval_batch_size", args.eval_batch_size)),
        "--alpha-target",
        str(config.get("alpha_target", args.alpha_target)),
        "--alpha-retain",
        str(config.get("alpha_retain", args.alpha_retain)),
        "--out-dir",
        str(out_dir),
    ]
    init_ckpt = str(config.get("init_ckpt", "")).strip()
    if init_ckpt:
        cmd.extend(["--init-ckpt", init_ckpt])
    return cmd


def summarize_runs(out_root: Path, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        config_id = str(config["config_id"])
        meta_path = out_root / config_id / "meta.json"
        if not meta_path.exists():
            rows.append({"config_id": config_id, "status": "missing_meta"})
            continue
        meta = json.loads(meta_path.read_text())
        val_metrics = meta.get("val_metrics_after_ascent", {})
        test_metrics = meta.get("test_metrics_after_ascent", {})
        rows.append(
            {
                "config_id": config_id,
                "status": "completed",
                "target_task": meta.get("target_task", ""),
                "split_type": meta.get("split_type", ""),
                "elicitation_steps": meta.get("elicitation_steps", ""),
                "ascent_steps": meta.get("ascent_steps", ""),
                "alpha_target": meta.get("alpha_target", ""),
                "alpha_retain": meta.get("alpha_retain", ""),
                "retain_train_rows": meta.get("retain_train_rows", ""),
                "selected_tensor_count": meta.get("selected_tensor_count", ""),
                "val_auroc_after_ascent": val_metrics.get("auroc", ""),
                "test_auroc_after_ascent": test_metrics.get("auroc", ""),
                "readout_disruption_flag": meta.get("readout_disruption_flag", ""),
                "weights_path": meta.get("weights_path", ""),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_metadata(
    args: argparse.Namespace,
    *,
    preview_path: Path,
    summary_path: Path,
    configs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> Path:
    metadata_path = summary_path.with_name(f"{summary_path.stem}_metadata.json")
    write_metadata(
        metadata_path,
        build_run_metadata(
            args=args,
            data_paths=[args.benchmark_manifest, args.retain_csv, args.config_json],
            extra={
                "phase": "run_stage1_option_b_sweep",
                "preview_json": str(preview_path),
                "summary_csv": str(summary_path),
                "config_ids": [str(config["config_id"]) for config in configs],
                "config_count": len(configs),
                "completed_count": sum(1 for row in rows if row.get("status") == "completed"),
                "execute": bool(args.execute),
            },
        ),
    )
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config-json", default="")
    parser.add_argument("--preview-json", default="data/phase2/stage1_option_b_initializer/sweep_plan.json")
    parser.add_argument("--summary-csv", default="data/phase2/stage1_option_b_initializer/sweep_summary.csv")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--benchmark-manifest", default="data/phase2/stage1_formal_target_manifests/stage1_formal_targets_available_manifest.csv")
    parser.add_argument("--target-task", default="hvue_human_host_tropism")
    parser.add_argument("--split-type", default="cluster_disjoint")
    parser.add_argument("--retain-csv", default="data/phase2/splits/retain.csv")
    parser.add_argument("--out-root", default="data/phase2/stage1_option_b_initializer/sweep_runs")
    parser.add_argument("--target-train-max-rows", type=int, default=256)
    parser.add_argument("--target-val-max-rows", type=int, default=128)
    parser.add_argument("--target-test-max-rows", type=int, default=128)
    parser.add_argument("--retain-max-rows", type=int, default=256)
    parser.add_argument("--elicitation-steps", type=int, default=20)
    parser.add_argument("--ascent-steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--alpha-target", type=float, default=1.0)
    parser.add_argument("--alpha-retain", type=float, default=1.0)
    args = parser.parse_args()

    configs = load_configs(args)
    plan = []
    for config in configs:
        plan.append(
            {
                "config_id": str(config["config_id"]),
                "cmd": build_command(args, config),
            }
        )

    preview_path = PROJECT_ROOT / args.preview_json
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[option-b-sweep] wrote {preview_path}")

    if args.execute:
        for item in plan:
            print(f"[option-b-sweep] run config={item['config_id']}", flush=True)
            result = subprocess.run(item["cmd"], cwd=str(PROJECT_ROOT), check=False)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

    rows = summarize_runs(PROJECT_ROOT / args.out_root, configs)
    summary_path = PROJECT_ROOT / args.summary_csv
    write_summary(summary_path, rows)
    metadata_path = write_sweep_metadata(
        args,
        preview_path=preview_path,
        summary_path=summary_path,
        configs=configs,
        rows=rows,
    )
    print(f"[option-b-sweep] wrote {summary_path}")
    print(f"[option-b-sweep] wrote {metadata_path}")


if __name__ == "__main__":
    main()
