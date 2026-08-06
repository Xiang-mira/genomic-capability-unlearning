"""Artifact validation and temporary materialization for Experiment 3.

The formal relearning protocol trains target readouts, but artifact retention
must avoid keeping duplicate full-model exports for fresh-LoRA arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo.tokenizer import CharLevelTokenizer
from phase1.utils import load_local_checkpoint
from phase2.checkpoint_io import save_checkpoint
from phase2.eval_benchmarks import (
    apply_checkpoint,
    deterministic_stratified_subset,
    evaluate_model,
    load_trainable_state_dict,
    read_benchmark_manifest,
    select_mcc_threshold_from_validation,
    split_records,
)
from phase2.lora_utils import PooledEvoClassifier, classification_metrics, inject_lora_all_blocks, merge_lora_adapters, remove_lora_adapters
from phase2.run_metadata import file_sha256


TASK = "hvue_human_host_tropism"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_expected_metrics(results_csv: Path) -> dict[str, float]:
    with results_csv.open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row.get("task") == TASK)
    return {
        "auroc": float(row["auroc"]),
        "mcc": float(row["mcc"]),
        "validation_selected_mcc_threshold": float(row["validation_selected_mcc_threshold"]),
    }


def load_task_records(args: argparse.Namespace):
    records = read_benchmark_manifest(
        args.benchmark_manifest,
        benchmark_scope="task",
        task_filter={TASK},
        requested_split_type="cluster_disjoint",
    )
    by_task = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)
    task_records = by_task[TASK]
    splits = split_records(task_records)
    val_records = deterministic_stratified_subset(splits["val"], 0, args.seed, TASK)
    test_records = deterministic_stratified_subset(splits["test"], 0, args.seed, f"{TASK}:test")
    return task_records, val_records, test_records


def build_task_model(args: argparse.Namespace, mode: str):
    model = load_local_checkpoint(args.model_dir, args.config_path, device=args.device)
    if args.starting_ckpt:
        apply_checkpoint(model, args.starting_ckpt)
    if mode == "lora":
        inject_lora_all_blocks(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    hidden_dim = int(model.blocks[0].pre_norm.scale.shape[0])
    task_model = PooledEvoClassifier(model, hidden_dim, 2, "classification").to(args.device)
    return model, task_model


def load_binary_labels(records, label_to_id: dict[str, int]) -> np.ndarray:
    return np.array([label_to_id[record.label] for record in records], dtype=np.int64)


def validate_predictions(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    expected = read_expected_metrics(Path(args.results_csv))
    task_records, val_records, test_records = load_task_records(args)
    labels = sorted({record.label for record in task_records})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    val_labels = load_binary_labels(val_records, label_to_id)
    test_labels = load_binary_labels(test_records, label_to_id)

    model, task_model = build_task_model(args, mode)
    try:
        checkpoint = torch.load(args.adapter_head_ckpt, map_location="cpu")
        state = checkpoint["state_dict"]
        if mode == "full_ft":
            head_state = {key: value for key, value in state.items() if key.startswith("head.")}
            if not head_state:
                raise RuntimeError("full-FT best.pt contains no head.* tensors")
            if args.canonical_ckpt:
                apply_checkpoint(model, args.canonical_ckpt)
            load_trainable_state_dict(task_model, head_state, args.device)
        else:
            load_trainable_state_dict(task_model, state, args.device)

        tokenizer = CharLevelTokenizer(args.max_length)
        _val_loss, _val_metrics, val_predictions = evaluate_model(
            task_model,
            val_records,
            val_labels,
            tokenizer,
            args.eval_batch_size,
            args.max_length,
            args.device,
            "classification",
            2,
            return_predictions=True,
        )
        _test_loss, test_metrics, test_predictions = evaluate_model(
            task_model,
            test_records,
            test_labels,
            tokenizer,
            args.eval_batch_size,
            args.max_length,
            args.device,
            "classification",
            2,
            return_predictions=True,
        )
        threshold, _mcc = select_mcc_threshold_from_validation(
            val_predictions["y_true"],
            val_predictions["y_score"][:, 1],
        )
        thresholded = (test_predictions["y_score"][:, 1] >= threshold).astype(np.int64)
        thresholded_metrics = classification_metrics(
            test_predictions["y_true"],
            thresholded,
            test_predictions["y_score"],
            2,
        )
        test_metrics.update({"mcc": thresholded_metrics["mcc"]})

        auroc_delta = abs(float(test_metrics["auroc"]) - expected["auroc"])
        mcc_delta = abs(float(test_metrics["mcc"]) - expected["mcc"])
        threshold_delta = abs(float(threshold) - expected["validation_selected_mcc_threshold"])
        auroc_tolerance = float(args.auroc_tolerance)
        mcc_tolerance = float(args.mcc_tolerance)
        threshold_tolerance = float(args.threshold_tolerance)
        validation_pass = auroc_delta <= auroc_tolerance and mcc_delta <= mcc_tolerance
        warnings = []
        if threshold_delta > threshold_tolerance:
            warnings.append(
                "validation threshold drift exceeded advisory tolerance but is not blocking"
            )
        validation = {
            "status": "pass" if validation_pass else "fail",
            "mode": mode,
            "validation_method": "reload_recompute_with_bfloat16_tolerance",
            "expected": expected,
            "observed": {
                "auroc": float(test_metrics["auroc"]),
                "mcc": float(test_metrics["mcc"]),
                "validation_selected_mcc_threshold": float(threshold),
            },
            "metric_deltas": {
                "auroc": auroc_delta,
                "mcc": mcc_delta,
                "validation_selected_mcc_threshold": threshold_delta,
            },
            "metric_tolerances": {
                "auroc": auroc_tolerance,
                "mcc": mcc_tolerance,
                "validation_selected_mcc_threshold": threshold_tolerance,
            },
            "warnings": warnings,
            "adapter_head_ckpt": args.adapter_head_ckpt,
            "adapter_head_sha256": file_sha256(Path(args.adapter_head_ckpt)),
            "canonical_ckpt": args.canonical_ckpt,
            "canonical_sha256": file_sha256(Path(args.canonical_ckpt)) if args.canonical_ckpt else "",
        }
        if args.output_json:
            write_json(Path(args.output_json), validation)
        if validation["status"] != "pass":
            raise RuntimeError(f"artifact validation failed: {validation}")
        return validation
    finally:
        task_model.close()
        if mode == "lora":
            remove_lora_adapters(model)
        del task_model, model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def materialize_lora(args: argparse.Namespace) -> None:
    model, task_model = build_task_model(args, "lora")
    try:
        checkpoint = torch.load(args.adapter_head_ckpt, map_location="cpu")
        load_trainable_state_dict(task_model, checkpoint["state_dict"], args.device)
        merged = merge_lora_adapters(model)
        result = save_checkpoint(
            model,
            args.output_ckpt,
            policy="full",
            metadata={
                "checkpoint_policy": "full",
                "artifact_role": "temporary_merged_lora_for_retain",
                "source_adapter_head_ckpt": args.adapter_head_ckpt,
                "merged_lora_modules": ",".join(merged),
            },
        )
        if not result.saved:
            raise RuntimeError(f"temporary materialization skipped: {result.skipped_reason}")
        write_json(
            Path(args.output_json),
            {
                "status": "pass",
                "output_ckpt": args.output_ckpt,
                "output_sha256": file_sha256(Path(args.output_ckpt)),
                "source_adapter_head_ckpt": args.adapter_head_ckpt,
                "source_adapter_head_sha256": file_sha256(Path(args.adapter_head_ckpt)),
                "merged_lora_module_count": len(merged),
            },
        )
    finally:
        task_model.close()
        remove_lora_adapters(model)
        del task_model, model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def extract_full_ft_head(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.adapter_head_ckpt, map_location="cpu")
    state = checkpoint["state_dict"]
    head_state = {key: value.cpu() for key, value in state.items() if key.startswith("head.")}
    if not head_state:
        raise RuntimeError("full-FT best.pt contains no head.* tensors")
    Path(args.output_head).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head_state, "source_best_pt": args.adapter_head_ckpt}, args.output_head)
    write_json(
        Path(args.output_json),
        {
            "status": "pass",
            "output_head": args.output_head,
            "output_head_sha256": file_sha256(Path(args.output_head)),
            "source_best_pt": args.adapter_head_ckpt,
            "source_best_pt_sha256": file_sha256(Path(args.adapter_head_ckpt)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate_lora", "materialize_lora", "validate_full_ft", "extract_full_ft_head"], required=True)
    parser.add_argument("--adapter-head-ckpt", required=True)
    parser.add_argument("--starting-ckpt", default="")
    parser.add_argument("--canonical-ckpt", default="")
    parser.add_argument("--output-ckpt", default="")
    parser.add_argument("--output-head", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--results-csv", default="")
    parser.add_argument("--benchmark-manifest", default="data/phase2/stage1_formal_target_manifests/hvue_human_host_tropism_cluster_disjoint.csv")
    parser.add_argument("--model-dir", default="evo-1-8k-base")
    parser.add_argument("--config-path", default="configs/evo-1-8k-base_inference.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1049)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-2,
        help="Deprecated alias retained for compatibility; use per-metric tolerances.",
    )
    parser.add_argument(
        "--auroc-tolerance",
        type=float,
        default=2e-2,
        help="Absolute AUROC tolerance for reload reproduction; GPU bf16 evaluation is not bit-exact.",
    )
    parser.add_argument(
        "--mcc-tolerance",
        type=float,
        default=5e-2,
        help="Absolute MCC tolerance for reload reproduction; MCC is threshold-sensitive.",
    )
    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=2e-2,
        help="Absolute validation-threshold tolerance for reload reproduction.",
    )
    args = parser.parse_args()

    if args.mode == "validate_lora":
        validate_predictions(args, "lora")
    elif args.mode == "materialize_lora":
        if not args.output_ckpt:
            raise ValueError("--output-ckpt is required for materialize_lora")
        materialize_lora(args)
    elif args.mode == "validate_full_ft":
        if not args.canonical_ckpt:
            raise ValueError("--canonical-ckpt is required for validate_full_ft")
        validate_predictions(args, "full_ft")
    elif args.mode == "extract_full_ft_head":
        if not args.output_head:
            raise ValueError("--output-head is required for extract_full_ft_head")
        extract_full_ft_head(args)


if __name__ == "__main__":
    main()
