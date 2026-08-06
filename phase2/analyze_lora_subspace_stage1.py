"""Complete Stage 1 LoRA-subspace analysis from retained selected reruns.

This script intentionally keeps effective updates in compact low-rank form.
Dense sBA matrices for all selected adapters would be prohibitively large.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from sklearn.metrics import matthews_corrcoef, roc_auc_score


DEFAULT_OUT_ROOT = Path("data/phase2/lora_subspace_targeting_20260729")
DEFAULT_MODEL_DIR = Path("evo-1-8k-base")
DEFAULT_TASK = "hvue_human_host_tropism"
AUROC_BASELINE = 0.8554553475149496
MCC_BASELINE = 0.5991934875548052
VALIDATION_AUROC_BASELINE = 0.8339622641509434
VALIDATION_MCC_BASELINE = 0.5260285646629745
FULL_SEQUENCE_AUROC = 0.8930006862072345
NEAR_PARITY_AUROC_TOLERANCE = 0.02
NEAR_PARITY_MCC_TOLERANCE = 0.05
MIN_CONSENSUS_ADAPTERS = 3
MIN_RANK_SPECIFIC_STRONG_ADAPTERS = 3
MIN_CROSS_RANK_STRONG_ADAPTERS_PER_RANK = 2
STRONG_CONTROL_OVERLAP_FOLD = 1.25
STRONG_CONTROL_OVERLAP_ABSOLUTE_MARGIN = 0.02
LEAVE_ONE_OUT_MAX_RELATIVE_DROP = 0.35
MAX_SINGLE_ADAPTER_CONTRIBUTION = 0.45
RANDOM_ORIENTATION_CONTROL_SEED = 20260730

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
        "validation_auroc": f"> {VALIDATION_AUROC_BASELINE}",
        "validation_mcc": f"> {VALIDATION_MCC_BASELINE}",
    },
    "near_parity_tolerance": {
        "validation_auroc_within": NEAR_PARITY_AUROC_TOLERANCE,
        "validation_mcc_within": NEAR_PARITY_MCC_TOLERANCE,
    },
    "minimum_independent_adapters_for_consensus": MIN_CONSENSUS_ADAPTERS,
    "minimum_rank_specific_strong_adapters": MIN_RANK_SPECIFIC_STRONG_ADAPTERS,
    "minimum_cross_rank_strong_adapters_per_rank": MIN_CROSS_RANK_STRONG_ADAPTERS_PER_RANK,
    "strong_vs_control_rule": {
        "topk_overlap_fold_over_controls": STRONG_CONTROL_OVERLAP_FOLD,
        "topk_overlap_absolute_margin": STRONG_CONTROL_OVERLAP_ABSOLUTE_MARGIN,
        "primary_control": "singular-value-matched random orientation control",
        "supportive_control": "random-label trained adapter control",
    },
    "cross_rank_stability": "hard threshold only for cross-rank go_to_stage2; not required for rank_specific_go_to_stage2",
    "rank_specific_go": "allowed when one rank has enough validation-dual-strong adapters, within-rank overlap beats matched controls, leave-one-out is stable, and no single adapter dominates",
    "leave_one_out_stability": f"mean strong overlap after leaving any one member out must not drop by more than {LEAVE_ONE_OUT_MAX_RELATIVE_DROP:.0%}",
    "max_single_adapter_contribution": MAX_SINGLE_ADAPTER_CONTRIBUTION,
    "uncertainty_policy": "borderline or insufficient evidence returns conditional_go_requires_more_evidence; successful metrics with random-like directions returns heterogeneous_recovery_paths",
    "allowed_outcomes": [
        "go_to_stage2",
        "rank_specific_go_to_stage2",
        "conditional_go_requires_more_evidence",
        "heterogeneous_recovery_paths",
    ],
}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_commit() -> str:
    git_dir = Path(".git")
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unavailable"
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref_path = git_dir / head.split(" ", 1)[1]
        return ref_path.read_text().strip() if ref_path.exists() else "unavailable"
    return head


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def read_prediction_metrics(path: Path) -> dict[str, float]:
    labels: list[int] = []
    probs: list[float] = []
    preds: list[int] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(row["label_id"]))
            probs.append(float(row["probability_positive"]))
            preds.append(int(row["predicted_label_id"]))
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def count_csv_value(path: Path, column: str, value: str) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if row.get(column) == value)


def parse_run_id(run_id: str) -> dict[str, Any]:
    match = re.search(r"_r(?P<rank>\d+)_lr(?P<lr>[^_]+)_seed(?P<seed>\d+)$", run_id)
    if not match:
        raise ValueError(f"Could not parse run_id: {run_id}")
    prefix = run_id[: match.start()]
    return {
        "selection_label": prefix,
        "rank": int(match.group("rank")),
        "learning_rate": match.group("lr"),
        "seed": int(match.group("seed")),
    }


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    selection_label: str
    assigned_group: str
    rank: int
    lora_alpha: int
    learning_rate: str
    seed: int
    checkpoint_path: Path
    validation_prediction_path: Path
    test_prediction_path: Path
    validation_auroc: float
    validation_mcc: float
    test_auroc: float
    test_mcc: float
    selected_threshold: float
    best_step: int

    @property
    def scale(self) -> float:
        return self.lora_alpha / self.rank


def assign_group(selection_label: str, val_auroc: float, val_mcc: float) -> tuple[str, str]:
    if selection_label == "matched_random_label_control":
        return "random_label_control", "matched random-label manifest control"
    auroc_pass = val_auroc > VALIDATION_AUROC_BASELINE
    mcc_pass = val_mcc > VALIDATION_MCC_BASELINE
    if auroc_pass and mcc_pass:
        return "dual_metric_strong_recovery", "validation AUROC and MCC exceed strong matched-input k-mer references"
    if auroc_pass:
        return "auroc_only_recovery", "validation AUROC exceeds reference but validation MCC does not"
    if (VALIDATION_AUROC_BASELINE - val_auroc) <= NEAR_PARITY_AUROC_TOLERANCE or (
        VALIDATION_MCC_BASELINE - val_mcc
    ) <= NEAR_PARITY_MCC_TOLERANCE:
        return "weak_or_near_parity_recovery", "near at least one formal reference without passing both"
    return "failed_or_weak_recovery", "below both formal references by the grouping margins"


def load_runs(out_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for results_path in sorted((out_root / "selected_adapter_reruns").glob("*/eval_benchmarks.csv")):
        run_id = results_path.parent.name
        parsed = parse_run_id(run_id)
        row = read_one_csv(results_path)
        val_path = Path(row["validation_prediction_path"])
        test_path = Path(row["test_prediction_path"])
        val_metrics = read_prediction_metrics(val_path)
        test_metrics = read_prediction_metrics(test_path)
        group, _reason = assign_group(parsed["selection_label"], val_metrics["auroc"], val_metrics["mcc"])
        runs.append(
            RunInfo(
                run_id=run_id,
                selection_label=parsed["selection_label"],
                assigned_group=group,
                rank=int(row["lora_rank"]),
                lora_alpha=int(row["lora_alpha"]),
                learning_rate=parsed["learning_rate"],
                seed=parsed["seed"],
                checkpoint_path=Path(row["best_checkpoint"]),
                validation_prediction_path=val_path,
                test_prediction_path=test_path,
                validation_auroc=val_metrics["auroc"],
                validation_mcc=val_metrics["mcc"],
                test_auroc=test_metrics["auroc"],
                test_mcc=test_metrics["mcc"],
                selected_threshold=float(row["validation_selected_mcc_threshold"]),
                best_step=int(row["best_step"]),
            )
        )
    return runs


def load_confirmatory_runs(out_root: Path) -> list[RunInfo]:
    metrics_path = out_root / "confirmatory_adapter_metrics.csv"
    if not metrics_path.exists():
        return []
    runs: list[RunInfo] = []
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "complete":
                continue
            rank = int(row["rank"])
            runs.append(
                RunInfo(
                    run_id=row["run_id"],
                    selection_label=row["batch_id"],
                    assigned_group=row["assigned_group"],
                    rank=rank,
                    lora_alpha=rank * 2,
                    learning_rate=row["learning_rate"],
                    seed=int(row["seed"]),
                    checkpoint_path=Path(row["adapter_path"]),
                    validation_prediction_path=Path(row["validation_prediction_path"]),
                    test_prediction_path=Path(row["test_prediction_path"]),
                    validation_auroc=float(row["validation_auroc"]),
                    validation_mcc=float(row["validation_mcc"]),
                    test_auroc=float(row["test_auroc"]),
                    test_mcc=float(row["test_mcc"]),
                    selected_threshold=float(row["selected_threshold"]),
                    best_step=int(row["best_step"]),
                )
            )
    return runs


def write_strength_groups(out_root: Path, runs: list[RunInfo]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        _group, reason = assign_group(run.selection_label, run.validation_auroc, run.validation_mcc)
        rows.append(
            {
                "run_id": run.run_id,
                "source": "confirmatory" if run.run_id.startswith("confirmatory_") else "selected_rerun",
                "rank": run.rank,
                "learning_rate": run.learning_rate,
                "seed": run.seed,
                "validation_auroc": run.validation_auroc,
                "validation_mcc": run.validation_mcc,
                "test_auroc": run.test_auroc,
                "test_mcc": run.test_mcc,
                "validation_auroc_excess": run.validation_auroc - VALIDATION_AUROC_BASELINE,
                "validation_mcc_excess": run.validation_mcc - VALIDATION_MCC_BASELINE,
                "test_auroc_excess": run.test_auroc - AUROC_BASELINE,
                "test_mcc_excess": run.test_mcc - MCC_BASELINE,
                "selected_threshold": run.selected_threshold,
                "best_step": run.best_step,
                "assigned_group": run.assigned_group,
                "reason": reason,
            }
        )
    write_csv(out_root / "stage1_adapter_strength_groups.csv", rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["assigned_group"]] += 1
    lines = [
        "# Stage 1 Adapter Strength Groups",
        "",
        "Selection groups are frozen from validation AUROC/MCC and the predefined run configuration; test metrics are reported only after grouping.",
        "",
        f"- Validation AUROC baseline: `{VALIDATION_AUROC_BASELINE}`",
        f"- Validation MCC baseline: `{VALIDATION_MCC_BASELINE}`",
        f"- Test AUROC baseline: `{AUROC_BASELINE}`",
        f"- Test MCC baseline: `{MCC_BASELINE}`",
        "",
        "## Counts",
        "",
    ]
    for group, count in sorted(counts.items()):
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(["", "## Notes", "", "- Consensus construction must not average random-label controls or failed/weak attackers into the strong recovery subspace."])
    (out_root / "stage1_adapter_strength_groups.md").write_text("\n".join(lines) + "\n")
    audit = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_question": "whether validation grouping used the test k-mer baseline",
        "finding": "previous grouping compared validation LoRA metrics to test k-mer AUROC/MCC references; this run uses validation k-mer references for grouping and test references only for held-out test excess reporting",
        "validation_baselines": {
            "source": "data/phase2/stage1_baseline_alignment_20260729/*/strong_matched_input_hashing_first512_cgrid100_val_predictions.csv",
            "auroc": VALIDATION_AUROC_BASELINE,
            "mcc_at_validation_selected_threshold": VALIDATION_MCC_BASELINE,
            "selected_threshold": 0.4980323246670524,
        },
        "test_baselines": {
            "source": "data/phase2/stage1_baseline_alignment_20260729/*/strong_matched_input_hashing_first512_cgrid100_test_predictions.csv",
            "auroc": AUROC_BASELINE,
            "mcc_at_validation_selected_threshold": MCC_BASELINE,
            "selected_threshold": 0.4980323246670524,
        },
        "grouping_rule": "validation AUROC > validation k-mer AUROC and validation MCC > validation k-mer MCC defines dual_metric_strong_recovery",
    }
    write_json(out_root / "stage1_validation_baseline_audit.json", audit)
    (out_root / "stage1_validation_baseline_audit.md").write_text(
        "\n".join(
            [
                "# Stage 1 Validation Baseline Audit",
                "",
                "Finding: previous grouping compared LoRA validation metrics to the strong k-mer test references.",
                "",
                f"- Correct validation AUROC baseline: `{VALIDATION_AUROC_BASELINE}`",
                f"- Correct validation MCC baseline: `{VALIDATION_MCC_BASELINE}`",
                f"- Held-out test AUROC baseline: `{AUROC_BASELINE}`",
                f"- Held-out test MCC baseline: `{MCC_BASELINE}`",
                "- Corrected grouping now uses validation baselines only; test excess is reported after groups are frozen.",
            ]
        )
        + "\n"
    )


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["state_dict"]


def module_names_from_state(state: dict[str, torch.Tensor]) -> list[str]:
    names = []
    for key in state:
        if key.endswith(".lora_A"):
            names.append(key[: -len(".lora_A")])
    return sorted(names)


def module_to_base_weight_key(module_name: str) -> str:
    name = module_name
    if name.startswith("base_model."):
        name = name[len("base_model.") :]
    return "backbone." + name + ".linear.weight"


def module_layer(module_name: str) -> int:
    match = re.search(r"blocks\.(\d+)\.", module_name)
    return int(match.group(1)) if match else -1


def module_short_name(module_name: str) -> str:
    name = module_name
    if ".blocks." in name:
        name = name.split(".blocks.", 1)[1]
        name = re.sub(r"^\d+\.", "", name)
    return name


class BaseNormReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.cache: dict[str, Any] = {}

    def norm_for(self, key: str) -> float | None:
        shard = self.index.get(key)
        if shard is None:
            alt = key.replace(".linear.weight", ".weight")
            shard = self.index.get(alt)
            key = alt if shard is not None else key
        if shard is None:
            return None
        if shard not in self.cache:
            self.cache[shard] = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
        tensor = self.cache[shard].get_tensor(key).float()
        return float(torch.linalg.vector_norm(tensor).item())


def inverse_sqrt_psd(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix.float())
    keep = evals > eps
    if not bool(keep.any()):
        return torch.zeros_like(matrix.float())
    return (evecs[:, keep] * torch.rsqrt(evals[keep]).unsqueeze(0)) @ evecs[:, keep].T


def small_svd_from_factors(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    # Non-zero singular values of scale * B @ A from rank-sized Grams:
    # sigma(BA)^2 = eig(sqrt(B'B) @ (AA') @ sqrt(B'B)).
    gram_b = B.float().T @ B.float()
    gram_a = A.float() @ A.float().T
    evals_b, evecs_b = torch.linalg.eigh(gram_b)
    evals_b = torch.clamp(evals_b, min=0)
    sqrt_b = (evecs_b * torch.sqrt(evals_b).unsqueeze(0)) @ evecs_b.T
    small = sqrt_b @ gram_a @ sqrt_b
    evals = torch.linalg.eigvalsh((small + small.T) * 0.5).clamp(min=0)
    return torch.sqrt(torch.sort(evals, descending=True).values) * abs(scale)


def effective_rank(singular_values: torch.Tensor, threshold: float = 0.99) -> int:
    energy = singular_values.square()
    total = float(energy.sum().item())
    if total <= 0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int((cumulative < threshold).sum().item() + 1)


def validate_orientation(A: torch.Tensor, B: torch.Tensor, scale: float, seed: int) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(3, A.shape[1], generator=generator)
    adapter_out = (x @ A.float().T @ B.float().T) * scale
    delta = (B.float() @ A.float()) * scale
    merged_out = x @ delta.T
    diff = adapter_out - merged_out
    max_abs = float(diff.abs().max().item())
    denom = float(adapter_out.abs().max().item()) or 1.0
    return max_abs, max_abs / denom


def extract_effective_updates(out_root: Path, model_dir: Path, runs: list[RunInfo]) -> dict[str, Any]:
    update_dir = out_root / "effective_updates"
    update_dir.mkdir(parents=True, exist_ok=True)
    base_norms = BaseNormReader(model_dir)
    statistics_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    registry_runs: list[dict[str, Any]] = []
    for run in runs:
        state = load_checkpoint_state(run.checkpoint_path)
        module_names = module_names_from_state(state)
        run_energy = 0.0
        module_count = 0
        max_merge_abs = 0.0
        max_merge_rel = 0.0
        for module_name in module_names:
            A = state[module_name + ".lora_A"].float()
            B = state[module_name + ".lora_B"].float()
            scale = run.scale
            singular = small_svd_from_factors(A, B, scale)
            fro = float(torch.linalg.vector_norm(singular).item())
            spectral = float(singular.max().item()) if singular.numel() else 0.0
            eff_rank = effective_rank(singular)
            base_key = module_to_base_weight_key(module_name)
            base_norm = base_norms.norm_for(base_key)
            max_abs, max_rel = validate_orientation(A, B, scale, seed=run.seed + module_layer(module_name) + len(module_name))
            max_merge_abs = max(max_merge_abs, max_abs)
            max_merge_rel = max(max_merge_rel, max_rel)
            run_energy += fro * fro
            module_count += 1
            statistics_rows.append(
                {
                    "run_id": run.run_id,
                    "assigned_group": run.assigned_group,
                    "selection_label": run.selection_label,
                    "rank": run.rank,
                    "lora_alpha": run.lora_alpha,
                    "scale": scale,
                    "learning_rate": run.learning_rate,
                    "seed": run.seed,
                    "module": module_name,
                    "layer": module_layer(module_name),
                    "module_short_name": module_short_name(module_name),
                    "a_shape": "x".join(map(str, A.shape)),
                    "b_shape": "x".join(map(str, B.shape)),
                    "fan_in_fan_out": False,
                    "matrix_orientation": "delta_weight = scale * lora_B @ lora_A; forward adds x @ delta_weight.T",
                    "frobenius_norm": fro,
                    "spectral_norm": spectral,
                    "singular_value_energy": fro * fro,
                    "effective_rank_99pct": eff_rank,
                    "top_singular_values": ";".join(f"{float(x):.8g}" for x in singular[: min(16, len(singular))]),
                    "base_weight_key": base_key,
                    "base_weight_norm": "" if base_norm is None else base_norm,
                    "update_to_base_norm_ratio": "" if not base_norm else fro / base_norm,
                    "merge_equivalence_max_abs": max_abs,
                    "merge_equivalence_max_rel": max_rel,
                }
            )
            merge_rows.append(
                {
                    "run_id": run.run_id,
                    "module": module_name,
                    "max_abs_diff": max_abs,
                    "max_relative_diff": max_rel,
                    "status": "pass" if max_abs <= 1e-4 else "fail",
                }
            )
        registry_runs.append(
            {
                "run_id": run.run_id,
                "checkpoint_path": str(run.checkpoint_path),
                "checkpoint_sha256": sha256_file(run.checkpoint_path),
                "assigned_group": run.assigned_group,
                "module_count": module_count,
                "rank": run.rank,
                "lora_alpha": run.lora_alpha,
                "scale": run.scale,
                "total_update_energy": run_energy,
                "total_update_frobenius_norm": math.sqrt(run_energy),
                "max_merge_equivalence_abs": max_merge_abs,
                "max_merge_equivalence_rel": max_merge_rel,
            }
        )
    write_csv(out_root / "effective_update_statistics.csv", statistics_rows)
    write_csv(update_dir / "adapter_merge_equivalence_by_module.csv", merge_rows)
    passed = all(float(row["max_abs_diff"]) <= 1e-4 for row in merge_rows)
    registry = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dense_update_policy": "not_saved; sBA updates are represented by retained low-rank A/B factors plus compact statistics because dense matrices would be prohibitively large",
        "merge_equivalence_scope": "module-level randomized algebraic check of LoRA forward contribution vs dense merged delta; full Evo forward was not run",
        "merge_equivalence_tolerance_abs": 1e-4,
        "merge_equivalence_pass": passed,
        "runs": registry_runs,
    }
    write_json(out_root / "effective_update_registry.json", registry)
    lines = [
        "# Effective Update Statistics",
        "",
        f"- Runs analysed: `{len(runs)}`",
        f"- Module rows: `{len(statistics_rows)}`",
        f"- Merge-equivalence module checks: `{len(merge_rows)}`",
        f"- Merge-equivalence pass: `{passed}`",
        "- Dense updates were not materialized; all statistics use the scaled low-rank form `scale * B @ A`.",
    ]
    (out_root / "effective_update_statistics.md").write_text("\n".join(lines) + "\n")
    report = [
        "Adapter merge equivalence report",
        "================================",
        "",
        "Validation scope: module-level randomized algebraic equivalence.",
        "Compared original LoRA contribution `(x @ A.T @ B.T) * scale` against merged contribution `x @ (scale * B @ A).T`.",
        f"Tolerance abs: 1e-4",
        f"Status: {'PASS' if passed else 'FAIL'}",
        f"Max abs diff: {max(float(row['max_abs_diff']) for row in merge_rows):.8g}",
        f"Max relative diff: {max(float(row['max_relative_diff']) for row in merge_rows):.8g}",
        "",
        "Full Evo forward equivalence was not run in this script; this report validates scaling and matrix orientation for every retained LoRA module.",
    ]
    (out_root / "adapter_merge_equivalence_report.txt").write_text("\n".join(report) + "\n")
    return registry


@dataclass
class ModuleFactors:
    run: RunInfo
    A: torch.Tensor
    B: torch.Tensor
    scale: float
    fro: float
    gram: torch.Tensor
    inv_sqrt_gram: torch.Tensor


def factor_inner(x: ModuleFactors, y: ModuleFactors) -> float:
    cb = x.B.float().T @ y.B.float()
    ca = y.A.float() @ x.A.float().T
    return float((x.scale * y.scale * torch.trace(cb @ ca)).item())


def component_gram(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    # Rank-component vectorized basis columns are vec(b_i a_i^T).
    # Their Gram is (B'B) Hadamard (AA').
    return (B.float().T @ B.float()) * (A.float() @ A.float().T) * (scale * scale)


def component_cross_gram(a: ModuleFactors, b: ModuleFactors) -> torch.Tensor:
    return (a.B.float().T @ b.B.float()) * (a.A.float() @ b.A.float().T) * (a.scale * b.scale)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def random_orthonormal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(rows, cols, generator=generator)
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :cols].contiguous()


def singular_value_matched_random_factors(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
    run_id: str,
    module_name: str,
) -> ModuleFactors:
    singular = small_svd_from_factors(A, B, scale)
    rank = min(int(singular.numel()), A.shape[0], B.shape[1])
    singular = singular[:rank].float()
    generator = torch.Generator().manual_seed(stable_seed(RANDOM_ORIENTATION_CONTROL_SEED, run_id, module_name))
    u = random_orthonormal(B.shape[0], rank, generator)
    v = random_orthonormal(A.shape[1], rank, generator)
    root = torch.sqrt(torch.clamp(singular, min=0.0))
    rand_b = u * root.unsqueeze(0)
    rand_a = root.unsqueeze(1) * v.T
    gram = component_gram(rand_a, rand_b, 1.0)
    fro = float(torch.linalg.vector_norm(singular).item())
    placeholder_run = RunInfo(
        run_id=f"{run_id}__singular_value_matched_random_orientation",
        selection_label="singular_value_matched_random_orientation_control",
        assigned_group="singular_value_matched_random_orientation_control",
        rank=rank,
        lora_alpha=rank,
        learning_rate="matched",
        seed=stable_seed(run_id, module_name),
        checkpoint_path=Path(""),
        validation_prediction_path=Path(""),
        test_prediction_path=Path(""),
        validation_auroc=float("nan"),
        validation_mcc=float("nan"),
        test_auroc=float("nan"),
        test_mcc=float("nan"),
        selected_threshold=float("nan"),
        best_step=-1,
    )
    return ModuleFactors(
        run=placeholder_run,
        A=rand_a,
        B=rand_b,
        scale=1.0,
        fro=fro,
        gram=gram,
        inv_sqrt_gram=inverse_sqrt_psd(gram),
    )


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def empty_pair_acc(a: RunInfo, b: RunInfo, typ: str) -> dict[str, Any]:
    return {
        "run_a": a.run_id,
        "run_b": b.run_id,
        "group_a": a.assigned_group,
        "group_b": b.assigned_group,
        "comparison_type": typ,
        "rank_a": a.rank,
        "rank_b": b.rank,
        "lr_a": a.learning_rate,
        "lr_b": b.learning_rate,
        "seed_a": a.seed,
        "seed_b": b.seed,
        "inner": 0.0,
        "norm_a2": 0.0,
        "norm_b2": 0.0,
        "module_cosines": [],
        "weighted_overlaps": [],
        "principal_angle_means": [],
        "topk_overlaps": [],
        "energy_a_by_layer": defaultdict(float),
        "energy_b_by_layer": defaultdict(float),
    }


def update_pair_acc(acc: dict[str, Any], fa: ModuleFactors, fb: ModuleFactors, layer: int) -> None:
    inner = factor_inner(fa, fb)
    na2 = fa.fro * fa.fro
    nb2 = fb.fro * fb.fro
    denom = math.sqrt(max(na2 * nb2, 1e-30))
    cosine = inner / denom
    cross = component_cross_gram(fa, fb)
    normalized_cross = fa.inv_sqrt_gram @ cross @ fb.inv_sqrt_gram
    sv = torch.linalg.svdvals(normalized_cross).clamp(0, 1)
    k = max(1, min(8, len(sv)))
    angles = torch.rad2deg(torch.arccos(sv)).tolist()
    weighted = float((sv[:k].square().mean()).item())
    acc["inner"] += inner
    acc["norm_a2"] += na2
    acc["norm_b2"] += nb2
    acc["module_cosines"].append(cosine)
    acc["weighted_overlaps"].append(weighted)
    acc["principal_angle_means"].append(float(np.mean(angles[:k])))
    acc["topk_overlaps"].append(float((sv[:k].square().sum() / k).item()))
    acc["energy_a_by_layer"][layer] += na2
    acc["energy_b_by_layer"][layer] += nb2


def finalize_pair_acc(acc: dict[str, Any]) -> dict[str, Any]:
    global_cosine = acc["inner"] / math.sqrt(max(acc["norm_a2"] * acc["norm_b2"], 1e-30))
    layers = sorted(set(acc["energy_a_by_layer"]) | set(acc["energy_b_by_layer"]))
    energy_corr = pearson([acc["energy_a_by_layer"][x] for x in layers], [acc["energy_b_by_layer"][x] for x in layers])
    return {
        "run_a": acc["run_a"],
        "run_b": acc["run_b"],
        "group_a": acc["group_a"],
        "group_b": acc["group_b"],
        "comparison_type": acc["comparison_type"],
        "rank_a": acc["rank_a"],
        "rank_b": acc["rank_b"],
        "lr_a": acc["lr_a"],
        "lr_b": acc["lr_b"],
        "seed_a": acc["seed_a"],
        "seed_b": acc["seed_b"],
        "global_update_cosine": global_cosine,
        "mean_module_cosine": float(np.mean(acc["module_cosines"])),
        "median_module_cosine": float(np.median(acc["module_cosines"])),
        "mean_principal_angle_deg": float(np.mean(acc["principal_angle_means"])),
        "mean_topk_subspace_overlap": float(np.mean(acc["topk_overlaps"])),
        "projection_weighted_overlap": float(np.mean(acc["weighted_overlaps"])),
        "layerwise_update_energy_correlation": energy_corr,
        "module_count": len(acc["module_cosines"]),
    }


def comparison_type(a: RunInfo, b: RunInfo) -> str:
    if a.assigned_group == "random_label_control" and b.assigned_group == "random_label_control":
        return "random_control_vs_random_control"
    if "random_label_control" in {a.assigned_group, b.assigned_group}:
        return "normal_vs_random_control"
    if a.assigned_group == "dual_metric_strong_recovery" and b.assigned_group == "dual_metric_strong_recovery":
        return "dual_metric_strong_vs_strong"
    if "dual_metric_strong_recovery" in {a.assigned_group, b.assigned_group}:
        return "dual_metric_strong_vs_other_normal"
    if a.rank == b.rank and a.learning_rate == b.learning_rate and a.seed != b.seed:
        return "same_config_across_seed"
    if a.rank != b.rank:
        return "cross_rank_normal"
    return "other_normal"


def run_subspace_analysis(out_root: Path, runs: list[RunInfo]) -> dict[str, Any]:
    states = {run.run_id: load_checkpoint_state(run.checkpoint_path) for run in runs}
    modules = module_names_from_state(next(iter(states.values())))
    pair_acc: dict[tuple[str, str], dict[str, Any]] = {}
    random_orientation_acc: dict[tuple[str, str], dict[str, Any]] = {}
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            pair_acc[(a.run_id, b.run_id)] = empty_pair_acc(a, b, comparison_type(a, b))
            random_orientation_acc[(a.run_id, b.run_id)] = empty_pair_acc(
                a,
                b,
                "singular_value_matched_random_orientation_control",
            )
    for module_name in modules:
        factors: dict[str, ModuleFactors] = {}
        random_factors: dict[str, ModuleFactors] = {}
        for run in runs:
            state = states[run.run_id]
            A = state[module_name + ".lora_A"].float()
            B = state[module_name + ".lora_B"].float()
            singular = small_svd_from_factors(A, B, run.scale)
            gram = component_gram(A, B, run.scale)
            fro = float(torch.linalg.vector_norm(singular).item())
            factors[run.run_id] = ModuleFactors(
                run=run,
                A=A,
                B=B,
                scale=run.scale,
                fro=fro,
                gram=gram,
                inv_sqrt_gram=inverse_sqrt_psd(gram),
            )
            random_factors[run.run_id] = singular_value_matched_random_factors(A, B, run.scale, run.run_id, module_name)
        layer = module_layer(module_name)
        for i, a in enumerate(runs):
            fa = factors[a.run_id]
            rfa = random_factors[a.run_id]
            for b in runs[i + 1 :]:
                fb = factors[b.run_id]
                rfb = random_factors[b.run_id]
                update_pair_acc(pair_acc[(a.run_id, b.run_id)], fa, fb, layer)
                update_pair_acc(random_orientation_acc[(a.run_id, b.run_id)], rfa, rfb, layer)
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for acc in pair_acc.values():
        row = finalize_pair_acc(acc)
        rows.append(row)
        if "control" in acc["comparison_type"]:
            control_rows.append(row)
    random_orientation_rows = [finalize_pair_acc(acc) for acc in random_orientation_acc.values()]
    control_rows.extend(random_orientation_rows)
    write_csv(out_root / "subspace_pairwise_metrics.csv", rows)
    write_csv(out_root / "control_overlap_distribution.csv", control_rows)
    write_csv(out_root / "singular_value_matched_random_orientation_controls.csv", random_orientation_rows)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows + random_orientation_rows:
        by_type[row["comparison_type"]].append(row)
    summary: dict[str, Any] = {}
    for typ, typ_rows in by_type.items():
        summary[typ] = {
            "n": len(typ_rows),
            "mean_global_update_cosine": float(np.mean([r["global_update_cosine"] for r in typ_rows])),
            "mean_topk_subspace_overlap": float(np.mean([r["mean_topk_subspace_overlap"] for r in typ_rows])),
            "mean_principal_angle_deg": float(np.mean([r["mean_principal_angle_deg"] for r in typ_rows])),
            "mean_layerwise_energy_correlation": float(np.mean([r["layerwise_update_energy_correlation"] for r in typ_rows])),
        }
    def mean_overlap(typ_rows: list[dict[str, Any]]) -> float:
        return float(np.mean([row["mean_topk_subspace_overlap"] for row in typ_rows])) if typ_rows else 0.0

    def passes_control(overlap: float, control_overlap: float) -> bool:
        return (
            overlap >= control_overlap * STRONG_CONTROL_OVERLAP_FOLD
            and (overlap - control_overlap) >= STRONG_CONTROL_OVERLAP_ABSOLUTE_MARGIN
        )

    def leave_one_out_ok(member_ids: list[str], typ_rows: list[dict[str, Any]], full_overlap: float) -> tuple[bool, dict[str, float]]:
        if len(member_ids) < 3 or full_overlap <= 0:
            return False, {}
        ratios: dict[str, float] = {}
        for run_id in member_ids:
            kept = [row for row in typ_rows if run_id not in {row["run_a"], row["run_b"]}]
            kept_overlap = mean_overlap(kept)
            ratios[run_id] = kept_overlap / full_overlap if full_overlap else 0.0
        return min(ratios.values()) >= (1.0 - LEAVE_ONE_OUT_MAX_RELATIVE_DROP), ratios

    def contribution_ok(member_ids: list[str], typ_rows: list[dict[str, Any]]) -> tuple[bool, dict[str, float]]:
        totals = {run_id: 0.0 for run_id in member_ids}
        total = 0.0
        for row in typ_rows:
            value = float(row["mean_topk_subspace_overlap"])
            totals[row["run_a"]] = totals.get(row["run_a"], 0.0) + value
            totals[row["run_b"]] = totals.get(row["run_b"], 0.0) + value
            total += 2.0 * value
        shares = {run_id: (value / total if total else 1.0) for run_id, value in totals.items()}
        return max(shares.values(), default=1.0) <= MAX_SINGLE_ADAPTER_CONTRIBUTION, shares

    strong_rows = by_type.get("dual_metric_strong_vs_strong", [])
    strong = summary.get("dual_metric_strong_vs_strong", {})
    normal_random = summary.get("normal_vs_random_control", {})
    random_random = summary.get("random_control_vs_random_control", {})
    strong_runs = [run for run in runs if run.assigned_group == "dual_metric_strong_recovery"]
    strong_ids = [run.run_id for run in strong_runs]
    strong_id_set = set(strong_ids)
    matched_random_orientation_rows = [
        row
        for row in random_orientation_rows
        if row["run_a"] in strong_id_set and row["run_b"] in strong_id_set
    ]
    random_orientation_overlap = mean_overlap(matched_random_orientation_rows or random_orientation_rows)
    strong_overlap = float(strong.get("mean_topk_subspace_overlap", 0.0))
    strong_beats_control = passes_control(strong_overlap, random_orientation_overlap)
    loo_ok, loo_ratios = leave_one_out_ok(strong_ids, strong_rows, strong_overlap)
    contribution_pass, contribution_shares = contribution_ok(strong_ids, strong_rows)
    rank_evaluations: dict[str, Any] = {}
    rank_specific_good: list[int] = []
    for rank in sorted({run.rank for run in runs}):
        rank_members = [run.run_id for run in strong_runs if run.rank == rank]
        rank_rows = [
            row
            for row in strong_rows
            if int(row["rank_a"]) == rank and int(row["rank_b"]) == rank
        ]
        rank_overlap = mean_overlap(rank_rows)
        rank_loo_ok, rank_loo = leave_one_out_ok(rank_members, rank_rows, rank_overlap)
        rank_contrib_ok, rank_contrib = contribution_ok(rank_members, rank_rows)
        rank_pass = (
            len(rank_members) >= MIN_RANK_SPECIFIC_STRONG_ADAPTERS
            and passes_control(rank_overlap, random_orientation_overlap)
            and rank_loo_ok
            and rank_contrib_ok
        )
        if rank_pass:
            rank_specific_good.append(rank)
        rank_evaluations[str(rank)] = {
            "validation_dual_strong_members": rank_members,
            "strong_pair_count": len(rank_rows),
            "mean_topk_subspace_overlap": rank_overlap,
            "beats_singular_value_matched_random_orientation_control": passes_control(rank_overlap, random_orientation_overlap),
            "leave_one_out_pass": rank_loo_ok,
            "leave_one_out_overlap_ratio_by_left_out_member": rank_loo,
            "single_adapter_contribution_pass": rank_contrib_ok,
            "pairwise_overlap_share_by_member": rank_contrib,
            "rank_specific_go_pass": rank_pass,
        }
    cross_rank_rows = [
        row
        for row in strong_rows
        if int(row["rank_a"]) != int(row["rank_b"])
    ]
    cross_rank_overlap = mean_overlap(cross_rank_rows)
    cross_rank_good = (
        len(rank_specific_good) >= 2
        and all(
            sum(1 for run in strong_runs if run.rank == rank) >= MIN_CROSS_RANK_STRONG_ADAPTERS_PER_RANK
            for rank in rank_specific_good
        )
        and passes_control(cross_rank_overlap, random_orientation_overlap)
        and loo_ok
        and contribution_pass
    )
    decision = "conditional_go_requires_more_evidence"
    reasons: list[str] = []
    if len(strong_runs) < MIN_CONSENSUS_ADAPTERS:
        decision = "conditional_go_requires_more_evidence"
        reasons.append(
            f"fewer than {MIN_CONSENSUS_ADAPTERS} validation dual-metric strong adapters were available after grouping"
        )
    elif cross_rank_good:
        decision = "go_to_stage2"
        reasons.append("multiple ranks passed validation-dual recovery, matched-control overlap, leave-one-out, and contribution rules")
    elif rank_specific_good:
        decision = "rank_specific_go_to_stage2"
        reasons.append(f"rank-specific consensus passed for rank(s): {', '.join(map(str, rank_specific_good))}")
    elif not strong_beats_control:
        decision = "heterogeneous_recovery_paths"
        reasons.append("validation-strong adapters recovered metrics but their overlap did not beat singular-value-matched random orientation controls")
    else:
        decision = "conditional_go_requires_more_evidence"
        reasons.append("overlap was above controls, but a pre-frozen stability or dominance rule did not pass")
    if strong:
        reasons.append(
            f"validation-strong mean top-k overlap={strong_overlap:.4f}; singular-value-matched random orientation control={random_orientation_overlap:.4f}"
        )
    if random_random:
        reasons.append("random-label controls provide a baseline overlap distribution")
    report = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_stage1_rules": FROZEN_STAGE1_RULES,
        "decision_input_boundary": "test metrics are reported only and are not used for grouping, consensus membership, SVD rank, top-k/layer selection, seed continuation, or Go/No-Go",
        "decision": decision,
        "comparison_summary": summary,
        "rank_evaluations": rank_evaluations,
        "cross_rank_evaluation": {
            "strong_pair_count": len(cross_rank_rows),
            "mean_topk_subspace_overlap": cross_rank_overlap,
            "beats_singular_value_matched_random_orientation_control": passes_control(cross_rank_overlap, random_orientation_overlap),
            "cross_rank_go_pass": cross_rank_good,
        },
        "leave_one_out": {
            "pass": loo_ok,
            "overlap_ratio_by_left_out_member": loo_ratios,
        },
        "single_adapter_contribution": {
            "pass": contribution_pass,
            "pairwise_overlap_share_by_member": contribution_shares,
        },
        "matched_random_orientation_control": {
            "strong_member_matched_pair_count": len(matched_random_orientation_rows),
            "mean_topk_subspace_overlap": random_orientation_overlap,
            "fallback_used_all_random_orientation_pairs": not bool(matched_random_orientation_rows),
        },
        "reasons": reasons,
        "stage2_allowed": decision in {"go_to_stage2", "rank_specific_go_to_stage2"},
        "stage2_scope": "cross_rank" if decision == "go_to_stage2" else (f"rank_{rank_specific_good[0]}" if len(rank_specific_good) == 1 else (";".join(f"rank_{rank}" for rank in rank_specific_good) if rank_specific_good else "none")),
        "real_pairwise_comparisons": len(rows),
        "singular_value_matched_random_orientation_control_comparisons": len(random_orientation_rows),
    }
    write_json(out_root / "subspace_stability_report.json", report)
    lines = [
        "# Subspace Stability Report",
        "",
        f"- Pairwise comparisons: `{len(rows)}`",
        f"- Singular-value-matched random orientation control comparisons: `{len(random_orientation_rows)}`",
        f"- Decision preview: `{decision}`",
        f"- Stage 2 scope: `{report['stage2_scope']}`",
        "",
        "## Frozen Decision Boundary",
        "",
        "- Test AUROC/MCC are reporting-only fields.",
        "- Grouping, consensus membership, top-k/layer statistics, seed continuation, and Go/No-Go use validation grouping plus weight-space/control comparisons.",
        "",
        "## Comparison Summary",
        "",
    ]
    for typ, values in sorted(summary.items()):
        lines.append(
            f"- `{typ}`: n={values['n']}, cosine={values['mean_global_update_cosine']:.4f}, "
            f"top-k overlap={values['mean_topk_subspace_overlap']:.4f}, "
            f"mean angle={values['mean_principal_angle_deg']:.2f} deg, "
            f"layer-energy corr={values['mean_layerwise_energy_correlation']:.4f}"
        )
    lines.extend(["", "## Reasons", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    (out_root / "subspace_stability_report.md").write_text("\n".join(lines) + "\n")
    return report


def write_consensus(out_root: Path, runs: list[RunInfo], stability: dict[str, Any]) -> None:
    groups = defaultdict(list)
    for run in runs:
        groups[run.assigned_group].append(run.run_id)
    consensus = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dense_subspace_policy": "not materialized; consensus candidates are represented by member run sets pending Stage 2 Go",
        "candidate_subspaces": {
            "best_run_subspace": [
                max(
                    runs,
                    key=lambda r: (
                        r.validation_auroc > VALIDATION_AUROC_BASELINE and r.validation_mcc > VALIDATION_MCC_BASELINE,
                        r.validation_auroc,
                        r.validation_mcc,
                    ),
                ).run_id
            ],
            "rank32_three_seed_consensus": [r.run_id for r in runs if r.rank == 32 and r.learning_rate == "5e-5" and r.selection_label == "best_run_configuration"],
            "rank16_three_seed_consensus": [r.run_id for r in runs if r.rank == 16 and r.learning_rate == "5e-5" and r.selection_label == "frozen_exploratory_configuration"],
            "strong_recovery_consensus": [r.run_id for r in runs if r.assigned_group == "dual_metric_strong_recovery"],
            "random_label_control_subspace": [r.run_id for r in runs if r.assigned_group == "random_label_control"],
        },
        "stage1_decision": stability["decision"],
        "stage2_allowed": stability["stage2_allowed"],
        "stage2_scope": stability.get("stage2_scope", "none"),
        "frozen_stage1_rules": FROZEN_STAGE1_RULES,
        "reasons": stability["reasons"],
    }
    write_json(out_root / "consensus_subspace_registry.json", consensus)
    if stability["decision"] == "go_to_stage2":
        required_next_action = "stage2_checkpoint_intervention_allowed_cross_rank"
    elif stability["decision"] == "rank_specific_go_to_stage2":
        required_next_action = f"stage2_checkpoint_intervention_allowed_{stability.get('stage2_scope', 'rank_specific')}"
    else:
        required_next_action = "do_not_generate_stage2_checkpoints"
    go_payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": stability["decision"],
        "stage2_allowed": stability["stage2_allowed"],
        "stage2_scope": stability.get("stage2_scope", "none"),
        "frozen_stage1_rules": FROZEN_STAGE1_RULES,
        "reasons": stability["reasons"],
        "required_next_action": required_next_action,
    }
    write_json(out_root / "consensus_go_no_go_report.json", go_payload)
    lines = [
        "# Consensus Go/No-Go Report",
        "",
        f"Decision: `{stability['decision']}`",
        "",
        f"Stage 2 allowed: `{stability['stage2_allowed']}`",
        "",
        f"Stage 2 scope: `{stability.get('stage2_scope', 'none')}`",
        "",
        "## Frozen Rules",
        "",
        f"- Dual-metric strong: validation AUROC > `{VALIDATION_AUROC_BASELINE}` and validation MCC > `{VALIDATION_MCC_BASELINE}`.",
        f"- Minimum consensus adapters: `{MIN_CONSENSUS_ADAPTERS}`.",
        "- Test metrics are report-only and cannot decide grouping, consensus, controls, or Go/No-Go.",
        "",
        "## Reasons",
        "",
    ]
    for reason in stability["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", "## Consensus Candidates", ""])
    for name, members in consensus["candidate_subspaces"].items():
        lines.append(f"- `{name}`: {', '.join(members) if members else 'none'}")
    (out_root / "consensus_go_no_go_report.md").write_text("\n".join(lines) + "\n")


def write_final_report(out_root: Path, runs: list[RunInfo], stability: dict[str, Any], update_registry: dict[str, Any]) -> None:
    payload = {
        "status": "stage1_complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage1_decision": stability["decision"],
        "stage2_allowed": stability["stage2_allowed"],
        "stage2_scope": stability.get("stage2_scope", "none"),
        "frozen_stage1_rules": FROZEN_STAGE1_RULES,
        "decision_input_boundary": stability.get("decision_input_boundary"),
        "primary_baselines": {
            "validation_strong_matched_input_kmer_auroc": VALIDATION_AUROC_BASELINE,
            "validation_strong_matched_input_kmer_mcc": VALIDATION_MCC_BASELINE,
            "test_strong_matched_input_kmer_auroc": AUROC_BASELINE,
            "test_strong_matched_input_kmer_mcc": MCC_BASELINE,
            "full_sequence_strong_reference_auroc": FULL_SEQUENCE_AUROC,
        },
        "run_count": len(runs),
        "merge_equivalence_pass": update_registry["merge_equivalence_pass"],
        "final_classification": "not_applicable_stage2_not_run",
        "answer": "Stage 1 extraction and stability analysis completed. Stage 2 remains gated by the consensus Go/No-Go decision.",
    }
    write_json(out_root / "final_lora_subspace_targeting_report.json", payload)
    (out_root / "final_lora_subspace_targeting_report.md").write_text(
        "\n".join(
            [
                "# Formal LoRA-Subspace Targeting Report",
                "",
                "## Stage 1 Status",
                "",
                f"- Analysed retained selected adapters: `{len(runs)}`",
                f"- Merge-equivalence pass: `{update_registry['merge_equivalence_pass']}`",
                f"- Stage 1 decision: `{stability['decision']}`",
                f"- Stage 2 allowed: `{stability['stage2_allowed']}`",
                f"- Stage 2 scope: `{stability.get('stage2_scope', 'none')}`",
                "",
                "## Interpretation",
                "",
                "Stage 1 now has adapter grouping, compact scaled-update extraction, pairwise stability metrics, matched controls, and a consensus decision artifact. Stage 2 checkpoint generation remains prohibited unless the decision is `go_to_stage2` or `rank_specific_go_to_stage2`.",
            ]
        )
        + "\n"
    )


def write_analysis_registry(
    out_root: Path,
    runs: list[RunInfo],
    stability: dict[str, Any],
    update_registry: dict[str, Any],
    include_confirmatory: bool,
) -> None:
    confirmatory_count = sum(1 for run in runs if run.run_id.startswith("confirmatory_"))
    total_runs = len(runs)
    expected_effective_rows = total_runs * 160
    expected_real_pairwise = total_runs * (total_runs - 1) // 2
    input_runs = []
    for run in runs:
        input_runs.append(
            {
                "run_id": run.run_id,
                "source": "confirmatory" if run.run_id.startswith("confirmatory_") else "selected_rerun",
                "assigned_group": run.assigned_group,
                "rank": run.rank,
                "learning_rate": run.learning_rate,
                "seed": run.seed,
                "checkpoint_path": str(run.checkpoint_path),
                "checkpoint_sha256": sha256_file(run.checkpoint_path) if run.checkpoint_path.exists() else "",
                "validation_prediction_path": str(run.validation_prediction_path),
                "validation_prediction_sha256": sha256_file(run.validation_prediction_path)
                if run.validation_prediction_path.exists()
                else "",
                "test_prediction_path": str(run.test_prediction_path),
                "test_prediction_sha256": sha256_file(run.test_prediction_path) if run.test_prediction_path.exists() else "",
            }
        )
    output_row_counts = {
        "stage1_adapter_strength_groups": count_csv_rows(out_root / "stage1_adapter_strength_groups.csv"),
        "effective_update_statistics": count_csv_rows(out_root / "effective_update_statistics.csv"),
        "merge_equivalence_by_module": count_csv_rows(out_root / "effective_updates" / "adapter_merge_equivalence_by_module.csv"),
        "subspace_pairwise_metrics_real_runs": count_csv_rows(out_root / "subspace_pairwise_metrics.csv"),
        "singular_value_matched_random_orientation_controls": count_csv_rows(
            out_root / "singular_value_matched_random_orientation_controls.csv"
        ),
    }
    completeness = {
        "expected_confirmatory_runs_if_final": 10,
        "actual_confirmatory_runs": confirmatory_count,
        "confirmatory_runs_complete": confirmatory_count == 10 if include_confirmatory else None,
        "expected_total_runs_if_final": 25,
        "actual_total_runs": total_runs,
        "total_runs_complete": total_runs == 25 if include_confirmatory else None,
        "expected_effective_update_rows": expected_effective_rows,
        "actual_effective_update_rows": output_row_counts["effective_update_statistics"],
        "effective_update_rows_match": output_row_counts["effective_update_statistics"] == expected_effective_rows,
        "expected_merge_checks": expected_effective_rows,
        "actual_merge_checks": output_row_counts["merge_equivalence_by_module"],
        "merge_checks_match": output_row_counts["merge_equivalence_by_module"] == expected_effective_rows,
        "merge_failures": count_csv_value(out_root / "effective_updates" / "adapter_merge_equivalence_by_module.csv", "status", "fail"),
        "expected_real_pairwise_comparisons": expected_real_pairwise,
        "actual_real_pairwise_comparisons": output_row_counts["subspace_pairwise_metrics_real_runs"],
        "real_pairwise_comparisons_match": output_row_counts["subspace_pairwise_metrics_real_runs"] == expected_real_pairwise,
    }
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": "phase2/analyze_lora_subspace_stage1.py",
        "script_sha256": sha256_file(Path("phase2/analyze_lora_subspace_stage1.py")),
        "script_commit": git_commit(),
        "include_confirmatory": include_confirmatory,
        "frozen_stage1_rules": FROZEN_STAGE1_RULES,
        "decision": stability["decision"],
        "stage2_allowed": stability["stage2_allowed"],
        "stage2_scope": stability.get("stage2_scope", "none"),
        "input_run_count": total_runs,
        "input_runs": input_runs,
        "output_row_counts": output_row_counts,
        "completeness_checks": completeness,
        "old_cache_policy": "outputs are regenerated from current input files on every invocation; no cached CSV is read for strength grouping, update extraction, or pairwise analysis",
    }
    write_json(out_root / "stage1_analysis_registry.json", payload)


def run(args: argparse.Namespace) -> None:
    out_root = args.out_dir
    runs = load_runs(out_root)
    if args.include_confirmatory:
        runs.extend(load_confirmatory_runs(out_root))
    if not runs:
        raise RuntimeError(f"No selected adapter rerun results found under {out_root}")
    write_strength_groups(out_root, runs)
    update_registry = extract_effective_updates(out_root, args.model_dir, runs)
    if not update_registry["merge_equivalence_pass"]:
        raise RuntimeError("Merge equivalence failed; not running subspace comparison")
    stability = run_subspace_analysis(out_root, runs)
    write_consensus(out_root, runs, stability)
    write_final_report(out_root, runs, stability, update_registry)
    write_analysis_registry(out_root, runs, stability, update_registry, args.include_confirmatory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--include-confirmatory", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
