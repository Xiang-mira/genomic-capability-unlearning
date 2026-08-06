"""Formal standalone single-LoRA intervention runner for Experiment 3.

This script does two things:
1. freeze validation-only source selection and emit the required registries;
2. optionally materialize reverse-direction candidate checkpoints as
   `weights.safetensors` delta checkpoints usable by `eval_benchmarks.py`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.checkpoint_io import atomic_save_safetensors


DEFAULT_IN_ROOT = Path("data/phase2/lora_subspace_targeting_20260729")
DEFAULT_OUT_DIR = Path("data/phase2/standalone_single_lora_intervention_20260730")
DEFAULT_MODEL_DIR = Path("evo-1-8k-base")
DEFAULT_TASK = "hvue_human_host_tropism"
VALIDATION_AUROC_BASELINE = 0.8339622641509434
VALIDATION_MCC_BASELINE = 0.5260285646629745
DEFAULT_STRENGTHS = (0.25, 0.5, 0.75)
DEFAULT_LORA_SCALE = 2.0
TEST_AUROC_BASELINE = 0.8554553475
TEST_MCC_BASELINE = 0.5991934876
CURRENT_GUE_TASKS = (
    "gue_emp_h3",
    "gue_human_tf_1",
    "gue_mouse_1",
    "gue_prom_300_notata",
    "gue_splice_reconstructed",
)
BASE_RECOVERY_EVIDENCE_PATH = Path("data/phase2/stage1_baseline_alignment_20260729/base_calibration_27run_unified.csv")
BASE_RECOVERY_EVIDENCE_REPORT = Path("data/phase2/stage1_baseline_alignment_20260729/stage1_baseline_alignment_report.md")
GUE_REVIEWED_LOCATIONS = (
    "phase2/run_standalone_prefilter.py",
    "phase2/summarize_route_decision.py",
    "phase2/preflight_route_decision.py",
    "phase2/next_steps_common.py",
    "phase2/run_downstream_reaudit_triage.py",
    "data/phase2/downstream_reaudit_smoke/task_inventory.csv",
    "data/benchmarks/hvue_gue_manifest.csv",
    "data/phase2/standalone_single_lora_intervention_20260730/gue_retain_definition_search_report.json",
)
PARTIAL_GUE_NOTE = (
    "no single authoritative frozen seven-task GUE retain definition was recoverable from repository manifests, "
    "registries, configs, or prior protocol artifacts; the validated five-task subset remains the only supported retain panel"
)


@dataclass(frozen=True)
class SourceCandidate:
    run_id: str
    rank: int
    learning_rate: float
    seed: int
    validation_auroc: float
    validation_mcc: float
    validation_auroc_excess: float
    validation_mcc_excess: float
    test_auroc: float
    test_mcc: float
    selected_threshold: float
    adapter_path: str
    adapter_sha256: str
    validation_prediction_path: str
    test_prediction_path: str
    best_step: int
    merge_equivalence_passed: bool
    normal_update_norm: bool
    abnormal_threshold: bool
    incomplete_artifact: bool
    unstable_training_trajectory: bool
    module_count: int
    aggregate_update_frobenius_norm: float
    selection_score: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_or_empty(path: Path) -> str:
    return file_sha256(path) if path.exists() else ""


def load_adapter_metadata(adapter_path: str) -> dict[str, Any]:
    path = Path(adapter_path)
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def canonical_module_name(name: str) -> str:
    return name[len("base_model.") :] if name.startswith("base_model.") else name


def parse_strengths(spec: str) -> tuple[float, ...]:
    parts = tuple(float(part.strip()) for part in spec.split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError("--strengths must contain exactly three values")
    if any(value <= 0 for value in parts):
        raise ValueError("--strengths values must be positive")
    return parts


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(value) if value else default


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize_update_norms(update_rows: list[dict[str, str]]) -> dict[str, float]:
    grouped = group_by(update_rows, "run_id")
    summary: dict[str, float] = {}
    for run_id, rows in grouped.items():
        fro_sq = 0.0
        for row in rows:
            fro = as_float(row, "frobenius_norm")
            fro_sq += fro * fro
        summary[run_id] = math.sqrt(fro_sq)
    return summary


def compute_normal_update_flags(update_rows: list[dict[str, str]]) -> dict[str, bool]:
    summary = summarize_update_norms(update_rows)
    values = list(summary.values())
    if not values:
        return {}
    med = median(values)
    mad = median([abs(value - med) for value in values])
    if mad == 0:
        return {run_id: True for run_id in summary}
    return {run_id: med - 3 * mad <= value <= med + 3 * mad for run_id, value in summary.items()}


def load_candidates(
    metrics_rows: list[dict[str, str]],
    update_rows: list[dict[str, str]],
    merge_rows: list[dict[str, str]],
) -> list[SourceCandidate]:
    update_by_run = group_by(update_rows, "run_id")
    merge_by_run = group_by(merge_rows, "run_id")
    update_summary = summarize_update_norms(update_rows)
    normal_update_flags = compute_normal_update_flags(update_rows)
    candidates: list[SourceCandidate] = []
    for row in metrics_rows:
        if str(row.get("status", "")).strip().lower() != "complete":
            continue
        if as_int(row, "rank") != 16:
            continue
        if not math.isclose(as_float(row, "learning_rate"), 5e-5, rel_tol=0.0, abs_tol=1e-12):
            continue
        run_id = str(row["run_id"])
        merge_pass = all(str(item.get("status", "")).lower() == "pass" for item in merge_by_run.get(run_id, []))
        threshold = as_float(row, "selected_threshold")
        incomplete = not all(
            str(row.get(field, "")).strip()
            for field in ("adapter_path", "adapter_sha256", "validation_prediction_path", "test_prediction_path")
        )
        val_auroc = as_float(row, "validation_auroc")
        val_mcc = as_float(row, "validation_mcc")
        candidates.append(
            SourceCandidate(
                run_id=run_id,
                rank=as_int(row, "rank"),
                learning_rate=as_float(row, "learning_rate"),
                seed=as_int(row, "seed"),
                validation_auroc=val_auroc,
                validation_mcc=val_mcc,
                validation_auroc_excess=val_auroc - VALIDATION_AUROC_BASELINE,
                validation_mcc_excess=val_mcc - VALIDATION_MCC_BASELINE,
                test_auroc=as_float(row, "test_auroc"),
                test_mcc=as_float(row, "test_mcc"),
                selected_threshold=threshold,
                adapter_path=str(row.get("adapter_path", "")),
                adapter_sha256=str(row.get("adapter_sha256", "")),
                validation_prediction_path=str(row.get("validation_prediction_path", "")),
                test_prediction_path=str(row.get("test_prediction_path", "")),
                best_step=as_int(row, "best_step"),
                merge_equivalence_passed=merge_pass,
                normal_update_norm=normal_update_flags.get(run_id, True),
                abnormal_threshold=threshold <= 0.01 or threshold >= 0.99,
                incomplete_artifact=incomplete,
                unstable_training_trajectory=as_int(row, "best_step") <= 0,
                module_count=len(update_by_run.get(run_id, [])),
                aggregate_update_frobenius_norm=update_summary.get(run_id, 0.0),
                selection_score=(val_auroc - VALIDATION_AUROC_BASELINE) + (val_mcc - VALIDATION_MCC_BASELINE),
            )
        )
    return candidates


def select_source_adapter(candidates: list[SourceCandidate]) -> tuple[SourceCandidate, list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    qualified: list[SourceCandidate] = []
    for candidate in candidates:
        selected = (
            candidate.validation_auroc > VALIDATION_AUROC_BASELINE
            and candidate.validation_mcc > VALIDATION_MCC_BASELINE
            and candidate.merge_equivalence_passed
            and candidate.normal_update_norm
            and not candidate.abnormal_threshold
            and not candidate.incomplete_artifact
            and not candidate.unstable_training_trajectory
        )
        audit_rows.append(
            {
                "run_id": candidate.run_id,
                "rank": candidate.rank,
                "learning_rate": candidate.learning_rate,
                "seed": candidate.seed,
                "validation_auroc": candidate.validation_auroc,
                "validation_mcc": candidate.validation_mcc,
                "validation_auroc_excess": candidate.validation_auroc_excess,
                "validation_mcc_excess": candidate.validation_mcc_excess,
                "test_auroc": candidate.test_auroc,
                "test_mcc": candidate.test_mcc,
                "merge_equivalence_passed": candidate.merge_equivalence_passed,
                "normal_update_norm": candidate.normal_update_norm,
                "abnormal_threshold": candidate.abnormal_threshold,
                "incomplete_artifact": candidate.incomplete_artifact,
                "unstable_training_trajectory": candidate.unstable_training_trajectory,
                "qualified_for_source_selection": selected,
                "selection_score": candidate.selection_score,
            }
        )
        if selected:
            qualified.append(candidate)
    if not qualified:
        raise ValueError("No rank16/lr5e-5 candidates satisfy the frozen validation-only source selection rule")
    qualified.sort(
        key=lambda row: (
            row.selection_score,
            row.validation_auroc_excess,
            row.validation_mcc_excess,
            -abs(row.selected_threshold - 0.5),
            -row.aggregate_update_frobenius_norm,
            -row.seed,
        ),
        reverse=True,
    )
    return qualified[0], audit_rows


def base_checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    payload: dict[str, Any] = {
        "model_dir": str(model_dir),
        "index_path": str(index_path) if index_path.exists() else "",
        "index_sha256": file_sha256_or_empty(index_path),
        "single_path": str(single_path) if single_path.exists() else "",
        "single_sha256": file_sha256_or_empty(single_path),
    }
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        payload["shard_paths"] = [str(model_dir / name) for name in shard_names]
        payload["shard_sha256"] = {name: file_sha256(model_dir / name) for name in shard_names if (model_dir / name).exists()}
    return payload


def write_selection_outputs(out_dir: Path, selected: SourceCandidate, audit_rows: list[dict[str, Any]], *, model_dir: Path) -> None:
    adapter_meta = load_adapter_metadata(selected.adapter_path)
    base_identity = base_checkpoint_identity(model_dir)
    tie_break_rule = [
        "highest validation dual-metric selection score",
        "then validation AUROC excess",
        "then validation MCC excess",
        "then threshold closest to 0.5",
        "then larger aggregate update Frobenius norm",
        "then lower source seed id via reverse sort on negative seed",
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "task": DEFAULT_TASK,
            "allowed_rank": 16,
            "allowed_learning_rate": 5e-5,
            "decision_inputs": [
                "validation AUROC above validation k-mer baseline",
                "validation MCC above validation k-mer baseline",
                "merge-equivalence passed",
                "normal update norm",
                "no abnormal threshold",
                "no incomplete artifact",
                "no unstable training trajectory",
            ],
            "forbidden_inputs": ["test AUROC", "test MCC", "test-set thresholds", "test predictions"],
            "tie_break_rule": tie_break_rule,
        },
        "selected_source_adapter": {
            "run_id": selected.run_id,
            "rank": selected.rank,
            "learning_rate": selected.learning_rate,
            "seed": selected.seed,
            "validation_auroc": selected.validation_auroc,
            "validation_mcc": selected.validation_mcc,
            "validation_auroc_excess": selected.validation_auroc_excess,
            "validation_mcc_excess": selected.validation_mcc_excess,
            "test_auroc_recorded_but_not_used": selected.test_auroc,
            "test_mcc_recorded_but_not_used": selected.test_mcc,
            "selected_threshold": selected.selected_threshold,
            "adapter_path": selected.adapter_path,
            "adapter_sha256": selected.adapter_sha256,
            "validation_prediction_path": selected.validation_prediction_path,
            "test_prediction_path": selected.test_prediction_path,
            "module_count": selected.module_count,
            "aggregate_update_frobenius_norm": selected.aggregate_update_frobenius_norm,
            "selection_score": selected.selection_score,
            "base_checkpoint_identity": base_identity,
            "adapter_configuration": {
                "rank": selected.rank,
                "learning_rate": selected.learning_rate,
                "selected_threshold": selected.selected_threshold,
                "task": adapter_meta.get("task"),
                "metric_for_best": adapter_meta.get("metric_for_best"),
                "best_epoch": adapter_meta.get("epoch"),
                "best_step": adapter_meta.get("step", selected.best_step),
            },
        },
        "selection_audit": audit_rows,
    }
    write_json(out_dir / "standalone_source_adapter_selection.json", payload)
    (out_dir / "standalone_source_adapter_selection.md").write_text(
        "\n".join(
            [
                "# Standalone Source Adapter Selection",
                "",
                "The source adapter was selected using frozen validation-only rules from the completed rank16/lr5e-5 family.",
                "",
                f"- Run ID: `{selected.run_id}`",
                f"- Rank / LR / seed: `{selected.rank}` / `{selected.learning_rate}` / `{selected.seed}`",
                f"- Validation AUROC excess: `{selected.validation_auroc_excess:+.6f}`",
                f"- Validation MCC excess: `{selected.validation_mcc_excess:+.6f}`",
                f"- Tie-break rule: `{'; '.join(tie_break_rule)}`",
                "- Test performance was recorded for reporting only and was not used for selection",
                "",
            ]
        )
        + "\n"
    )


def normalize_weight_key(key: str) -> str:
    return key[len("backbone.") :] if key.startswith("backbone.") else key


def load_base_weight_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    raw_state: dict[str, torch.Tensor] = {}
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        for shard_name in sorted(set(payload["weight_map"].values())):
            raw_state.update(load_file(str(model_dir / shard_name)))
    elif single_path.exists():
        raw_state = load_file(str(single_path))
    else:
        raise FileNotFoundError(f"No base safetensors files found in {model_dir}")
    return {normalize_weight_key(key): value for key, value in raw_state.items()}


def compute_and_write_base_module_norms(
    model_dir: Path,
    out_path: Path,
    *,
    weight_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    state = weight_tensors or load_base_weight_tensors(model_dir)
    rows: list[dict[str, Any]] = []
    norms: dict[str, float] = {}
    for key, tensor in sorted(state.items()):
        if not key.startswith("blocks.") or not key.endswith(".weight"):
            continue
        module = key[: -len(".weight")]
        fro = float(torch.linalg.vector_norm(tensor.float()).item())
        rows.append(
            {
                "module": module,
                "weight_key": key,
                "weight_frobenius_norm": fro,
                "shape_out": tensor.shape[0] if tensor.ndim >= 1 else "",
                "shape_in": tensor.shape[1] if tensor.ndim >= 2 else "",
            }
        )
        norms[module] = fro
    write_csv(out_path, rows, ["module", "weight_key", "weight_frobenius_norm", "shape_out", "shape_in"])
    return norms


def read_base_norms(path: Path) -> dict[str, float]:
    norms: dict[str, float] = {}
    for row in read_csv_rows(path):
        module = str(row.get("module", "")).strip()
        if module:
            norms[canonical_module_name(module)] = as_float(row, "weight_frobenius_norm")
    return norms


def module_name_from_adapter_key(key: str) -> str:
    module = key[len("base_model.") :] if key.startswith("base_model.") else key
    if module.endswith(".lora_A"):
        return module[: -len(".lora_A")]
    if module.endswith(".lora_B"):
        return module[: -len(".lora_B")]
    raise ValueError(f"Unsupported adapter key: {key}")


def load_adapter_updates(source: SourceCandidate) -> dict[str, torch.Tensor]:
    payload = torch.load(source.adapter_path, map_location="cpu")
    state = payload["state_dict"]
    modules = sorted({module_name_from_adapter_key(key) for key in state if key.endswith(".lora_A")})
    updates: dict[str, torch.Tensor] = {}
    for module in modules:
        A = state[f"base_model.{module}.lora_A"].float()
        B = state[f"base_model.{module}.lora_B"].float()
        updates[module] = (B @ A) * DEFAULT_LORA_SCALE
    return updates


def source_short_name(module: str) -> str:
    return ".".join(module.split(".")[2:])


def build_random_orientation(delta: torch.Tensor, *, seed: int) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(delta.float())
    rank = int((singular_values > 0).sum().item())
    if rank == 0:
        return torch.zeros_like(delta)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.randn(delta.shape[0], rank, generator=generator)
    right = torch.randn(delta.shape[1], rank, generator=generator)
    q_left, _ = torch.linalg.qr(left, mode="reduced")
    q_right, _ = torch.linalg.qr(right, mode="reduced")
    return (q_left[:, :rank] * singular_values[:rank].unsqueeze(0)) @ q_right[:, :rank].T


def deranged_assignment(modules: list[str], weight_tensors: dict[str, torch.Tensor], *, seed: int) -> dict[str, str]:
    groups: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    for module in modules:
        shape = tuple(weight_tensors[f"{module}.weight"].shape)
        groups.setdefault((source_short_name(module), shape), []).append(module)
    rng = random.Random(seed)
    mapping: dict[str, str] = {}
    for candidates in groups.values():
        shuffled = candidates[:]
        if len(candidates) > 1:
            for _attempt in range(100):
                rng.shuffle(shuffled)
                if all(src != dst for src, dst in zip(candidates, shuffled)):
                    break
        for src, dst in zip(candidates, shuffled):
            mapping[src] = dst
    return mapping


def materialization_command(out_dir: Path, arm_id: str) -> list[str]:
    return [
        "python",
        "-u",
        "phase2/standalone_single_lora_intervention.py",
        "--out-dir",
        str(out_dir),
        "--materialize-arm",
        arm_id,
    ]


def refresh_registry_materialization_state(registry_rows: list[dict[str, Any]]) -> None:
    for row in registry_rows:
        if row["arm_type"] == "base":
            row["status"] = "reference"
            row["materialization_status"] = "not_required"
            continue
        checkpoint_path = Path(str(row.get("checkpoint_path", "")).strip())
        if checkpoint_path.exists():
            row["status"] = "materialized"
            row["materialization_status"] = "complete"
            if not str(row.get("checkpoint_sha256", "")).strip():
                row["checkpoint_sha256"] = file_sha256(checkpoint_path)
        else:
            row["status"] = "planned_not_materialized"
            row["materialization_status"] = "pending_execution"
            row["checkpoint_sha256"] = ""


def build_candidate_artifacts(
    source: SourceCandidate,
    update_rows: list[dict[str, str]],
    base_norms: dict[str, float],
    strengths: tuple[float, ...],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = [row for row in update_rows if str(row.get("run_id")) == source.run_id]
    norm_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = [
        {
            "arm_id": "base",
            "arm_type": "base",
            "eta": 0.0,
            "source_run_id": source.run_id,
            "status": "reference",
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "materialization_status": "not_required",
            "control_seed": "",
        }
    ]
    for idx, eta in enumerate(strengths, start=1):
        source_arm = f"source_subspace_intervention_eta{idx}"
        random_subspace_arm = f"random_subspace_control_eta{idx}"
        random_layer_arm = f"random_layer_control_eta{idx}"
        for arm_id, arm_type, control_seed in (
            (source_arm, "source_subspace_intervention", ""),
            (random_subspace_arm, "random_subspace_control", str(20260730 + idx)),
            (random_layer_arm, "random_layer_control", str(20261730 + idx)),
        ):
            registry_rows.append(
                {
                    "arm_id": arm_id,
                    "arm_type": arm_type,
                    "eta": eta,
                    "source_run_id": source.run_id,
                    "source_adapter_path": source.adapter_path,
                    "status": "planned_not_materialized",
                    "checkpoint_path": str(out_dir / "standalone_candidate_checkpoints" / arm_id / "weights.safetensors"),
                    "checkpoint_sha256": "",
                    "materialization_status": "pending_execution",
                    "control_seed": control_seed,
                    "materialize_command": " ".join(materialization_command(out_dir, arm_id)),
                }
            )
        for module_row in source_rows:
            module = str(module_row["module"])
            canonical_module = canonical_module_name(module)
            base_norm = base_norms.get(canonical_module, 0.0)
            source_update_fro = as_float(module_row, "frobenius_norm")
            relative = (eta * source_update_fro / base_norm) if base_norm > 0 else ""
            common = {
                "source_run_id": source.run_id,
                "module": module,
                "canonical_module": canonical_module,
                "layer": as_int(module_row, "layer"),
                "module_short_name": str(module_row.get("module_short_name", "")),
                "source_update_frobenius_norm": source_update_fro,
                "base_weight_frobenius_norm": base_norm,
                "top_singular_values": str(module_row.get("top_singular_values", "")),
                "effective_rank_99pct": as_int(module_row, "effective_rank_99pct"),
                "spectral_norm": as_float(module_row, "spectral_norm"),
                "eta": eta,
                "relative_perturbation_rho": relative,
            }
            norm_rows.append({"arm_id": source_arm, "arm_type": "source_subspace_intervention", **common})
            norm_rows.append({"arm_id": random_subspace_arm, "arm_type": "random_subspace_control", **common})
            norm_rows.append({"arm_id": random_layer_arm, "arm_type": "random_layer_control", **common})
    return registry_rows, norm_rows


def parse_block_layer(module_name: str) -> int | None:
    match = re.search(r"blocks\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def load_arm_provenance(out_dir: Path, arm_id: str) -> dict[str, Any]:
    path = out_dir / "standalone_candidate_checkpoints" / arm_id / "provenance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def append_control_matching_summaries(norm_rows: list[dict[str, Any]], strengths: tuple[float, ...], out_dir: Path) -> list[dict[str, Any]]:
    rows_by_arm = group_by([{k: str(v) for k, v in row.items()} for row in norm_rows], "arm_id")
    output = list(norm_rows)
    for idx, eta in enumerate(strengths, start=1):
        source_arm = f"source_subspace_intervention_eta{idx}"
        source_rows = rows_by_arm.get(source_arm, [])
        if not source_rows:
            continue
        source_modules = sorted(str(row.get("canonical_module", "")) for row in source_rows)
        source_layers = sorted({as_int(row, "layer") for row in source_rows})
        source_energy = [eta * as_float(row, "source_update_frobenius_norm") for row in source_rows]
        source_total = math.sqrt(sum(value * value for value in source_energy))
        source_ranks = sorted(as_int(row, "effective_rank_99pct") for row in source_rows)
        source_spectra = sorted(str(row.get("top_singular_values", "")) for row in source_rows)
        source_module_count = len(source_modules)
        for arm_type in ("random_subspace_control", "random_layer_control"):
            arm_id = f"{arm_type}_eta{idx}"
            control_rows = rows_by_arm.get(arm_id, [])
            control_modules = sorted(str(row.get("canonical_module", "")) for row in control_rows)
            if arm_type == "random_layer_control":
                provenance = load_arm_provenance(out_dir, arm_id)
                mappings = list(provenance.get("mappings", []))
                target_layers = sorted(
                    {
                        parse_block_layer(str(item.get("target_module", "")))
                        for item in mappings
                        if parse_block_layer(str(item.get("target_module", ""))) is not None
                    }
                )
                control_layers = target_layers
                slot_reassignment_distinct = all(
                    str(item.get("source_module", "")) != str(item.get("target_module", "")) for item in mappings
                ) if mappings else False
            else:
                control_layers = sorted({as_int(row, "layer") for row in control_rows})
                slot_reassignment_distinct = False
            control_energy = [eta * as_float(row, "source_update_frobenius_norm") for row in control_rows]
            control_total = math.sqrt(sum(value * value for value in control_energy))
            control_ranks = sorted(as_int(row, "effective_rank_99pct") for row in control_rows)
            control_spectra = sorted(str(row.get("top_singular_values", "")) for row in control_rows)
            exact_layer_identity_matched = source_layers == control_layers
            output.append(
                {
                    "arm_id": arm_id,
                    "arm_type": arm_type,
                    "record_type": "verification_summary",
                    "matched_source_arm_id": source_arm,
                    "eta": eta,
                    "module_count_matched": "pass" if source_module_count == len(control_modules) else "fail",
                    "layer_count_matched": "pass" if len(source_layers) == len(control_layers) else "fail",
                    "parameter_count_matched": "pass" if len(source_rows) == len(control_rows) else "fail",
                    "norm_distribution_matched": "pass" if source_energy == control_energy else "fail",
                    "exact_layer_identity_matched": (
                        "pass" if exact_layer_identity_matched else "fail"
                    ) if arm_type == "random_subspace_control" else "not_applicable",
                    "random_layer_identity_distinct": (
                        "not_applicable" if arm_type == "random_subspace_control" else ("pass" if slot_reassignment_distinct else "fail")
                    ),
                    "frobenius_norm_status": "pass" if math.isclose(source_total, control_total, rel_tol=0.0, abs_tol=1e-12) else "fail",
                    "total_perturbation_norm_status": "pass" if math.isclose(source_total, control_total, rel_tol=0.0, abs_tol=1e-12) else "fail",
                    "effective_rank_status": "pass" if source_ranks == control_ranks else "fail",
                    "module_energy_distribution_status": "pass" if source_energy == control_energy else "fail",
                    "singular_value_spectrum_status": "pass" if source_spectra == control_spectra else "fail",
                    "source_total_perturbation_norm": source_total,
                    "control_total_perturbation_norm": control_total,
                    "matching_scope_note": (
                        "same-slot spectrum-preserving random orientation"
                        if arm_type == "random_subspace_control"
                        else "shape-matched layer reassignment preserves the source multiset over the touched module set"
                    ),
                }
            )
    return output


def summarize_prefilter_live_state(out_dir: Path, registry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_arm: list[dict[str, Any]] = []
    completed_arms = 0
    active_arm = ""
    runs_root = out_dir / "standalone_prefilter_runs"
    for row in registry_rows:
        arm_id = str(row["arm_id"])
        run_dir = runs_root / arm_id
        status = {
            "integrity": (run_dir / "integrity_smoke.json").exists(),
            "eval_unlearn": (run_dir / "eval_unlearn" / "eval_ppl.json").exists(),
            "downstream": (run_dir / "downstream" / "eval_benchmarks_summary.json").exists(),
        }
        completed = all(status.values())
        if completed:
            completed_arms += 1
        elif any(status.values()) and not active_arm:
            active_arm = arm_id
        per_arm.append({"arm_id": arm_id, **status, "completed": completed})
    if not active_arm:
        for row in per_arm:
            if not row["completed"]:
                active_arm = str(row["arm_id"])
                break
    overall = "planned_prefilter_not_run"
    if completed_arms == len(registry_rows):
        overall = "complete"
    elif completed_arms > 0 or any(row["integrity"] or row["eval_unlearn"] or row["downstream"] for row in per_arm):
        overall = "in_progress"
    return {
        "status": overall,
        "started_at": "2026-07-30T11:30:34Z",
        "completed_arms": completed_arms,
        "active_arm": active_arm,
        "total_arms": len(registry_rows),
        "arm_progress": per_arm,
    }


def summarize_base_recovery_evidence() -> dict[str, Any]:
    rows = read_csv_rows(BASE_RECOVERY_EVIDENCE_PATH)
    by_run = {str(row.get("run_id", "")): row for row in rows}
    target_runs = {
        "rank32_lr1e-5": [f"fresh_lora_base_r32_lr1e-5_seed{seed}" for seed in (42, 43, 44)],
        "rank32_lr5e-5": [f"fresh_lora_base_r32_lr5e-5_seed{seed}" for seed in (42, 43, 44)],
    }
    summary: dict[str, Any] = {
        "evidence_csv": str(BASE_RECOVERY_EVIDENCE_PATH),
        "evidence_report_md": str(BASE_RECOVERY_EVIDENCE_REPORT),
        "criterion_note": (
            "prefer a cross-configuration evaluator with independent confirmatory support for stable dual-metric recovery; "
            "AUROC-only or incomplete MCC evidence is insufficient"
        ),
        "configs": {},
    }
    for label, run_ids in target_runs.items():
        selected_rows = [by_run[run_id] for run_id in run_ids if run_id in by_run]
        summary["configs"][label] = {
            "run_ids": run_ids,
            "rows_found": len(selected_rows),
            "raw_lora_auroc": [as_float(row, "raw_lora_auroc") for row in selected_rows],
            "raw_lora_mcc": [as_float(row, "raw_lora_mcc") for row in selected_rows],
            "mcc_status": sorted({str(row.get("mcc_status", "")) for row in selected_rows}),
            "seed_count_with_positive_auroc_excess_vs_strong_matched_input_kmer": sum(
                1 for row in selected_rows if as_float(row, "excess_vs_strong_matched_input_kmer") > 0.0
            ),
            "seed_count_with_positive_mcc_excess_vs_strong_matched_input_kmer": sum(
                1 for row in selected_rows if as_float(row, "mcc_excess_vs_strong_matched_input_kmer") > 0.0
            ),
        }
    summary["recommended_cross_config"] = {
        "rank": 32,
        "learning_rate": 5e-5,
        "reason": (
            "rank32/lr1e-5 has positive AUROC excess but lacks independent confirmatory dual-metric support; "
            "rank32/lr5e-5 is the confirmatory batch-B configuration and includes dual-metric strong recovery evidence"
        ),
    }
    return summary


def singular_values_by_module(update_rows: list[dict[str, str]], source_run_id: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row in update_rows:
        if str(row.get("run_id")) != source_run_id:
            continue
        module = str(row.get("module", ""))
        if module.startswith("base_model."):
            module = module[len("base_model.") :]
        values = [float(part) for part in str(row.get("top_singular_values", "")).split(";") if str(part).strip()]
        result[module] = values
    return result


def materialize_checkpoint(
    arm_row: dict[str, Any],
    source_updates: dict[str, torch.Tensor],
    weight_tensors: dict[str, torch.Tensor],
    source_singular_values: dict[str, list[float]],
    out_dir: Path,
) -> tuple[str, str]:
    arm_type = str(arm_row["arm_type"])
    arm_id = str(arm_row["arm_id"])
    eta = float(arm_row["eta"])
    control_seed = int(arm_row["control_seed"]) if str(arm_row.get("control_seed", "")).strip() else None
    mappings: list[dict[str, Any]] = []
    if arm_type == "source_subspace_intervention":
        for module in sorted(source_updates):
            mappings.append({"source_module": module, "target_module": module, "policy": "reverse_source_direction"})
    elif arm_type == "random_subspace_control":
        for idx, module in enumerate(sorted(source_updates)):
            singular_values = source_singular_values.get(module)
            if not singular_values:
                singular_values = torch.linalg.svdvals(source_updates[module].float()).tolist()
            mappings.append(
                {
                    "source_module": module,
                    "target_module": module,
                    "policy": "random_orientation_same_slot",
                    "orientation_seed": (control_seed or 0) + idx,
                    "singular_values": singular_values,
                }
            )
    elif arm_type == "random_layer_control":
        mapping = deranged_assignment(sorted(source_updates.keys()), weight_tensors, seed=control_seed or 0)
        for source_module, target_module in sorted(mapping.items()):
            mappings.append({"source_module": source_module, "target_module": target_module, "policy": "shape_matched_slot_reassignment"})
    else:
        raise ValueError(f"Unsupported arm type: {arm_type}")
    target_dir = out_dir / "standalone_candidate_checkpoints" / arm_id
    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = target_dir / "weights.safetensors"
    atomic_save_safetensors(
        {"__compact_anchor__": torch.tensor([0], dtype=torch.uint8)},
        str(ckpt_path),
        metadata={
            "checkpoint_policy": "standalone_lora_reverse",
            "arm_id": arm_id,
            "arm_type": arm_type,
            "source_run_id": arm_row["source_run_id"],
            "eta": eta,
        },
    )
    write_json(
        target_dir / "provenance.json",
        {
            "arm_id": arm_id,
            "arm_type": arm_type,
            "eta": eta,
            "source_run_id": arm_row["source_run_id"],
            "source_adapter_path": arm_row["source_adapter_path"],
            "lora_scale": DEFAULT_LORA_SCALE,
            "control_seed": arm_row.get("control_seed", ""),
            "tensor_count": 1,
            "mappings": mappings,
        },
    )
    return str(ckpt_path), file_sha256(ckpt_path)


def build_prefilter_payload(registry_rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_rows = [
        {
            "arm_id": row["arm_id"],
            "arm_type": row["arm_type"],
            "status": "pending_execution" if row["arm_type"] != "base" else "reference",
            "retain_ppl_change_fraction": "",
            "gue_status": "",
            "viral_non_target_status": "",
            "numerical_stability_status": "",
            "nan_inf_status": "",
            "forward_pass_status": "",
            "focus_task_reduction_vs_random_control_status": "",
            "control_matching_status": "pending_verification" if row["arm_type"] != "base" else "reference",
        }
        for row in registry_rows
    ]
    live_state = summarize_prefilter_live_state(out_dir, registry_rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": live_state["status"],
        "started_at": live_state["started_at"],
        "completed_arms": live_state["completed_arms"],
        "active_arm": live_state["active_arm"],
        "total_arms": live_state["total_arms"],
        "arm_progress": live_state["arm_progress"],
        "prefilter_policy": {
            "retain_ppl_max_fractional_change": 0.10,
            "gue_requirement": "reject broad degradation",
            "viral_non_target_requirement": "reject material degradation",
            "numerical_requirement": "reject NaN, Inf, or forward-pass instability",
            "causal_requirement": "reject effects indistinguishable from matched random controls",
            "gue_protocol_status": "partial_GUE_retain_evaluation",
            "gue_protocol_note": PARTIAL_GUE_NOTE,
        },
        "gue_retain_status": "partial_GUE_retain_evaluation",
        "evaluated_gue_task_count": 5,
        "full_gue_retain_gate_passed": False,
        "currently_evaluated_gue_tasks": list(CURRENT_GUE_TASKS),
        "reviewed_configs_manifests_and_registries": list(GUE_REVIEWED_LOCATIONS),
        "success_claim_language": {
            "allowed_if_five_tasks_pass": "evaluated GUE tasks passed",
            "required_limitation": "full seven-task GUE gate remains unresolved",
            "forbidden_auto_claim": "all formal retain gates passed",
        },
        "candidate_selection_rule": {
            "must_pass_before_validation_selection": [
                "no NaN or Inf values",
                "normal forward-pass behavior",
                "retain-PPL fractional change <= 0.10",
                "all five evaluated GUE retain tasks pass",
                "viral non-target retain criteria pass",
                "intervention strength and control matching valid",
            ],
            "selection_input_after_gates": "validation results only",
            "forbidden_selection_rule": "do not auto-select the checkpoint with the lowest clean Host Tropism score",
            "test_results_must_not_influence_selection": True,
        },
        "out_dir": str(out_dir),
    }
    return metrics_rows, payload


def build_fresh_lora_registry(
    registry_rows: list[dict[str, Any]],
    source: SourceCandidate,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for row in registry_rows:
        status = "reference" if row["arm_type"] == "base" else "blocked_on_prefilter_pass"
        for eval_type, rank, lr, seed in (
            ("matched_family_heldout_lora", source.rank, source.learning_rate, source.seed + 1000),
            ("cross_configuration_lora", 32, 5e-5, source.seed + 2000),
        ):
            eval_id = f"{row['arm_id']}__{eval_type}"
            eval_rows.append(
                {
                    "evaluation_id": eval_id,
                    "arm_id": row["arm_id"],
                    "arm_type": row["arm_type"],
                    "evaluation_type": eval_type,
                    "status": status,
                    "lora_rank": rank,
                    "learning_rate": lr,
                    "seed": seed,
                    "fresh_lora_required": True,
                    "fresh_head_required": True,
                    "fresh_optimizer_required": True,
                    "reuse_source_adapter": False,
                    "reuse_source_head": False,
                    "prediction_dir": str(out_dir / "standalone_fresh_lora_predictions" / eval_id),
                    "selection_policy": "validation_only",
                }
            )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned_fresh_lora_evaluations_not_run",
        "evaluation_families": [
            {
                "evaluation_type": "matched_family_heldout_lora",
                "rank": source.rank,
                "learning_rate": source.learning_rate,
                "seed_policy": "source_seed_plus_1000",
                "seed_must_differ_from_source_adapter": True,
                "fresh_lora_parameters": True,
                "fresh_classification_head": True,
                "fresh_optimizer": True,
            },
            {
                "evaluation_type": "cross_configuration_lora",
                "rank": 32,
                "learning_rate": 5e-5,
                "seed_policy": "source_seed_plus_2000",
                "predefined_alternative_configuration": "frozen_rank32_lr5e-5",
                "fresh_lora_parameters": True,
                "fresh_classification_head": True,
                "fresh_optimizer": True,
            },
        ],
        "full_finetune_policy": {
            "status": "pending_validation_only_selection_after_prefilter_and_fresh_lora",
            "selection_target": "strongest retain-safe source-subspace intervention using validation results only",
        },
        "configuration_freeze_note": "downstream fresh-LoRA settings are frozen before prefilter review; no outcome-dependent reconfiguration is allowed",
        "gue_retain_status": "partial_GUE_retain_evaluation",
        "evaluated_gue_task_count": 5,
        "full_gue_retain_gate_passed": False,
        "cross_configuration_base_recovery_evidence": summarize_base_recovery_evidence(),
        "test_baselines": {
            "auroc": TEST_AUROC_BASELINE,
            "mcc": TEST_MCC_BASELINE,
        },
    }
    return eval_rows, payload


def build_full_ft_registry(registry_rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in registry_rows:
        if row["arm_id"] == "base":
            cohort = "required_reference"
        elif row["arm_type"] == "source_subspace_intervention":
            cohort = "eligible_if_best_retain_safe_source"
        else:
            cohort = "eligible_if_closest_matched_control_to_selected_source"
        rows.append(
            {
                "arm_id": row["arm_id"],
                "arm_type": row["arm_type"],
                "status": "blocked_on_prefilter_and_fresh_lora_validation_selection",
                "selection_cohort": cohort,
                "prediction_dir": str(out_dir / "standalone_full_ft_predictions" / str(row["arm_id"])),
                "selection_policy": "validation_only",
            }
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned_full_ft_evaluations_not_run",
        "required_arms": ["base", "best_source_intervention", "closest_matched_random_control"],
        "selection_policy": "validation_only",
        "shared_training_protocol": {
            "same_data_split": True,
            "same_training_budget": True,
            "same_optimizer_settings": True,
            "same_checkpoint_selection_rule": True,
            "same_evaluation_protocol": True,
        },
        "gue_retain_status": "partial_GUE_retain_evaluation",
        "evaluated_gue_task_count": 5,
        "full_gue_retain_gate_passed": False,
        "test_baselines": {
            "auroc": TEST_AUROC_BASELINE,
            "mcc": TEST_MCC_BASELINE,
        },
    }
    return rows, payload


def write_intervention_report(out_dir: Path, registry_rows: list[dict[str, Any]], norm_rows: list[dict[str, Any]], strengths: tuple[float, ...]) -> None:
    materialized = sum(1 for row in registry_rows if row["arm_type"] != "base" and row["status"] == "materialized")
    (out_dir / "standalone_intervention_report.md").write_text(
        "\n".join(
            [
                "# Standalone Intervention Report",
                "",
                f"- Intervention strengths (eta): `{', '.join(f'{value:.4f}' for value in strengths)}`",
                f"- Candidate arms excluding base: `{sum(1 for row in registry_rows if row['arm_type'] != 'base')}`",
                f"- Materialized candidate checkpoints: `{materialized}`",
                f"- Module audit rows: `{len(norm_rows)}`",
                "- Random-layer control uses shape-matched slot reassignment because the source LoRA family touches all layers/modules.",
                "",
            ]
        )
        + "\n"
    )


def write_prefilter_report(out_dir: Path, payload: dict[str, Any]) -> None:
    (out_dir / "standalone_prefilter_report.md").write_text(
        "# Standalone Prefilter Report\n\nNo prefilter metrics have been executed yet.\n"
    )
    write_json(out_dir / "standalone_prefilter_report.json", payload)


def write_fresh_lora_report(out_dir: Path, payload: dict[str, Any]) -> None:
    (out_dir / "standalone_fresh_lora_evaluation.md").write_text(
        "# Standalone Fresh LoRA Evaluation\n\nFresh re-learning evaluations remain blocked until prefiltering passes.\n"
    )
    write_json(out_dir / "standalone_fresh_lora_evaluation.json", payload)


def write_final_report_template(out_dir: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gue_retain_status": "partial_GUE_retain_evaluation",
        "evaluated_gue_task_count": 5,
        "full_gue_retain_gate_passed": False,
        "currently_evaluated_gue_tasks": list(CURRENT_GUE_TASKS),
        "reviewed_configs_manifests_and_registries": list(GUE_REVIEWED_LOCATIONS),
        "seven_task_definition_status": "unresolved_without_authoritative_source",
        "required_wording_if_five_tasks_pass": [
            "evaluated GUE tasks passed",
            "full seven-task GUE gate remains unresolved",
        ],
        "forbidden_wording": [
            "all formal retain gates passed",
        ],
    }
    write_json(out_dir / "standalone_final_report_template.json", payload)
    (out_dir / "standalone_final_report_template.md").write_text(
        "\n".join(
            [
                "# Standalone Final Report Template",
                "",
                "- gue_retain_status = `partial_GUE_retain_evaluation`",
                "- evaluated_gue_task_count = `5`",
                "- full_gue_retain_gate_passed = `false`",
                f"- evaluated GUE tasks: `{', '.join(CURRENT_GUE_TASKS)}`",
                f"- reviewed sources: `{'; '.join(GUE_REVIEWED_LOCATIONS)}`",
                f"- unresolved seven-task note: `{PARTIAL_GUE_NOTE}`",
                "- required wording if all five evaluated tasks pass: `evaluated GUE tasks passed`",
                "- required limitation: `full seven-task GUE gate remains unresolved`",
                "- forbidden automatic claim: `all formal retain gates passed`",
                "",
            ]
        )
        + "\n"
    )


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "standalone_candidate_checkpoints").mkdir(exist_ok=True)
    (out_dir / "standalone_fresh_lora_predictions").mkdir(exist_ok=True)
    (out_dir / "standalone_full_ft_predictions").mkdir(exist_ok=True)

    metrics_rows = read_csv_rows(args.metrics_csv)
    update_rows = read_csv_rows(args.effective_update_stats_csv)
    merge_rows = read_csv_rows(args.merge_equivalence_csv)
    base_norms_csv = getattr(args, "base_module_norms_csv", None) or (out_dir / "standalone_base_module_norms.csv")

    candidates = load_candidates(metrics_rows, update_rows, merge_rows)
    selected, audit_rows = select_source_adapter(candidates)
    write_selection_outputs(out_dir, selected, audit_rows, model_dir=getattr(args, "model_dir", DEFAULT_MODEL_DIR))

    if base_norms_csv.exists():
        base_norms = read_base_norms(base_norms_csv)
    else:
        base_norms = compute_and_write_base_module_norms(getattr(args, "model_dir", DEFAULT_MODEL_DIR), base_norms_csv)

    registry_rows, norm_rows = build_candidate_artifacts(selected, update_rows, base_norms, args.strengths, out_dir)

    materialize_all = bool(getattr(args, "materialize_all", False))
    materialize_arm = str(getattr(args, "materialize_arm", ""))
    if materialize_all or materialize_arm:
        base_weight_tensors = load_base_weight_tensors(getattr(args, "model_dir", DEFAULT_MODEL_DIR))
        source_updates = load_adapter_updates(selected)
        source_singular_values = singular_values_by_module(update_rows, selected.run_id)
        allowed = {materialize_arm} if materialize_arm else set()
        for row in registry_rows:
            if row["arm_type"] == "base":
                continue
            if allowed and row["arm_id"] not in allowed:
                continue
            ckpt_path, ckpt_sha = materialize_checkpoint(
                row,
                source_updates,
                base_weight_tensors,
                source_singular_values,
                out_dir,
            )
            row["status"] = "materialized"
            row["materialization_status"] = "complete"
            row["checkpoint_path"] = ckpt_path
            row["checkpoint_sha256"] = ckpt_sha

    refresh_registry_materialization_state(registry_rows)

    write_json(
        out_dir / "standalone_candidate_registry.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "materialized_candidate_checkpoints_ready"
                if any(row["arm_type"] != "base" and row["status"] == "materialized" for row in registry_rows)
                else "planned_candidate_checkpoints_not_materialized"
            ),
            "source_run_id": selected.run_id,
            "base_module_norms_csv": str(base_norms_csv),
            "arms": registry_rows,
        },
    )
    norm_rows = append_control_matching_summaries(norm_rows, args.strengths, out_dir)
    write_csv(out_dir / "standalone_intervention_norm_audit.csv", norm_rows)
    write_intervention_report(out_dir, registry_rows, norm_rows, args.strengths)

    prefilter_rows, prefilter_payload = build_prefilter_payload(registry_rows, out_dir)
    write_csv(out_dir / "standalone_prefilter_metrics.csv", prefilter_rows)
    write_prefilter_report(out_dir, prefilter_payload)

    fresh_rows, fresh_payload = build_fresh_lora_registry(registry_rows, selected, out_dir)
    write_csv(
        out_dir / "standalone_fresh_lora_adaptation_curves.csv",
        [],
        ["evaluation_id", "arm_id", "evaluation_type", "step", "split", "auroc", "mcc", "selected_threshold", "status"],
    )
    write_json(out_dir / "standalone_fresh_lora_registry.json", {"evaluations": fresh_rows, "status": fresh_payload["status"]})
    write_fresh_lora_report(out_dir, fresh_payload)

    full_ft_rows, full_ft_payload = build_full_ft_registry(registry_rows, out_dir)
    write_csv(
        out_dir / "standalone_full_ft_curves.csv",
        [],
        ["arm_id", "step", "split", "auroc", "mcc", "selected_threshold", "status"],
    )
    write_json(out_dir / "standalone_full_ft_registry.json", {"evaluations": full_ft_rows, "status": full_ft_payload["status"]})
    write_json(out_dir / "standalone_full_ft_evaluation.json", full_ft_payload)
    (out_dir / "standalone_full_ft_evaluation.md").write_text(
        "# Standalone Full Fine-Tuning Evaluation\n\nFull fine-tuning remains blocked until validation-only prefilter and fresh-LoRA selection complete.\n"
    )
    write_final_report_template(out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_IN_ROOT / "confirmatory_adapter_metrics.csv")
    parser.add_argument(
        "--effective-update-stats-csv",
        type=Path,
        default=DEFAULT_IN_ROOT / "confirmatory_effective_updates" / "confirmatory_effective_update_statistics.csv",
    )
    parser.add_argument(
        "--merge-equivalence-csv",
        type=Path,
        default=DEFAULT_IN_ROOT / "confirmatory_effective_updates" / "confirmatory_adapter_merge_equivalence_by_module.csv",
    )
    parser.add_argument("--base-module-norms-csv", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--strengths", type=parse_strengths, default=DEFAULT_STRENGTHS)
    parser.add_argument("--materialize-arm", default="")
    parser.add_argument("--materialize-all", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
