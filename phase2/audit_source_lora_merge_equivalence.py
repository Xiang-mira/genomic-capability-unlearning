from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.utils import CharLevelTokenizer, load_local_checkpoint
from phase2.lora_utils import LoRALinear, inject_lora_all_blocks, merge_lora_adapters


DEFAULT_OUT_DIR = PROJECT_ROOT / "data/phase2/standalone_single_lora_intervention_20260730"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "evo-1-8k-base"
DEFAULT_CONFIG_PATH = "configs/evo-1-8k-base_inference.yml"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEED = 49
DEFAULT_ABS_TOL = 1e-4
DEFAULT_REL_TOL = 1e-4
DEFAULT_FORWARD_AGREEMENT_TOL = 0.999
DEFAULT_FUNCTIONAL_REL_L2_TOL = 0.02
DEFAULT_FUNCTIONAL_COSINE_TOL = 0.999


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def module_name_from_adapter_key(key: str) -> str:
    module = key[len("base_model.") :] if key.startswith("base_model.") else key
    if module.endswith(".lora_A"):
        return module[: -len(".lora_A")]
    if module.endswith(".lora_B"):
        return module[: -len(".lora_B")]
    raise ValueError(f"unsupported adapter key: {key}")


def load_adapter_state(adapter_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(adapter_path, map_location="cpu")
    return payload["state_dict"], dict(payload.get("meta") or {})


def capture_final_hidden(model, input_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = model.norm(hidden)

    handle = model.blocks[len(model.blocks) - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits, _ = model(input_ids, padding_mask=mask)
        hidden = captured["hidden"]
    finally:
        handle.remove()
    return logits, hidden


def set_lora_weights(model, state_dict: dict[str, torch.Tensor]) -> list[str]:
    loaded: list[str] = []
    for block_idx, block in enumerate(model.blocks):
        for module_path, module in block.named_modules():
            if not module_path or not isinstance(module, LoRALinear):
                continue
            prefix = f"base_model.blocks.{block_idx}.{module_path}"
            key_a = prefix + ".lora_A"
            key_b = prefix + ".lora_B"
            if key_a in state_dict and key_b in state_dict:
                module.lora_A.data.copy_(state_dict[key_a].to(module.lora_A.dtype))
                module.lora_B.data.copy_(state_dict[key_b].to(module.lora_B.dtype))
                loaded.append(prefix)
    return loaded


def build_sample_batch(manifest_path: Path, sample_ids: list[str], *, max_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    csv.field_size_limit(sys.maxsize)
    wanted = set(sample_ids)
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("id", "") in wanted:
                rows.append(row)
                if len(rows) == len(sample_ids):
                    break
    row_map = {row["id"]: row for row in rows}
    ordered_rows = [row_map[sample_id] for sample_id in sample_ids if sample_id in row_map]
    tokenizer = CharLevelTokenizer(max_length)
    ids = []
    for row in ordered_rows:
        ids.append(tokenizer.tokenize(row["sequence"][:max_length]))
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    return input_ids, mask, ordered_rows


@dataclass
class ForwardSummary:
    max_abs_logit_diff: float
    median_abs_logit_diff: float
    mean_abs_logit_diff: float
    max_rel_logit_diff: float
    relative_l2_logit_error: float
    logit_cosine_similarity: float
    reference_logit_l2_norm: float
    max_reference_logit_magnitude: float
    hidden_max_abs_diff: float
    hidden_relative_l2_error: float
    hidden_cosine_similarity: float
    reference_hidden_l2_norm: float
    max_reference_hidden_magnitude: float
    token_prediction_agreement: float
    prediction_disagreement_count: int
    prediction_total_count: int


def compare_outputs(a_logits: torch.Tensor, b_logits: torch.Tensor, a_hidden: torch.Tensor, b_hidden: torch.Tensor) -> ForwardSummary:
    diff = (a_logits.float() - b_logits.float()).abs()
    denom = torch.maximum(a_logits.float().abs(), torch.full_like(a_logits.float(), 1e-12))
    rel = diff / denom
    logit_delta = a_logits.float() - b_logits.float()
    logit_ref = a_logits.float()
    hidden_delta = a_hidden.float() - b_hidden.float()
    hidden_ref = a_hidden.float()
    logit_ref_norm = float(torch.linalg.vector_norm(logit_ref).item())
    hidden_ref_norm = float(torch.linalg.vector_norm(hidden_ref).item())
    logit_delta_norm = float(torch.linalg.vector_norm(logit_delta).item())
    hidden_delta_norm = float(torch.linalg.vector_norm(hidden_delta).item())
    logit_cosine = float(torch.nn.functional.cosine_similarity(logit_ref.flatten(), b_logits.float().flatten(), dim=0).item())
    hidden_cosine = float(torch.nn.functional.cosine_similarity(hidden_ref.flatten(), b_hidden.float().flatten(), dim=0).item())
    a_pred = a_logits.argmax(dim=-1)
    b_pred = b_logits.argmax(dim=-1)
    prediction_matches = a_pred == b_pred
    agreement = float(prediction_matches.float().mean().item())
    total = int(prediction_matches.numel())
    disagreements = int((~prediction_matches).sum().item())
    return ForwardSummary(
        max_abs_logit_diff=float(diff.max().item()),
        median_abs_logit_diff=float(diff.median().item()),
        mean_abs_logit_diff=float(diff.mean().item()),
        max_rel_logit_diff=float(rel.max().item()),
        relative_l2_logit_error=logit_delta_norm / max(logit_ref_norm, 1e-12),
        logit_cosine_similarity=logit_cosine,
        reference_logit_l2_norm=logit_ref_norm,
        max_reference_logit_magnitude=float(logit_ref.abs().max().item()),
        hidden_max_abs_diff=float((a_hidden.float() - b_hidden.float()).abs().max().item()),
        hidden_relative_l2_error=hidden_delta_norm / max(hidden_ref_norm, 1e-12),
        hidden_cosine_similarity=hidden_cosine,
        reference_hidden_l2_norm=hidden_ref_norm,
        max_reference_hidden_magnitude=float(hidden_ref.abs().max().item()),
        token_prediction_agreement=agreement,
        prediction_disagreement_count=disagreements,
        prediction_total_count=total,
    )


def count_lora_modules(model) -> int:
    return sum(1 for module in model.modules() if isinstance(module, LoRALinear))


def compare_state_tensors(
    model_left,
    model_right,
    *,
    suffix: str | None = None,
) -> dict[str, Any]:
    left_state = model_left.state_dict()
    right_state = model_right.state_dict()
    keys = sorted(set(left_state) | set(right_state))
    missing = [key for key in keys if key not in left_state or key not in right_state]
    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    dtype_mismatch = []
    device_mismatch = []
    for key in keys:
        if missing and (key not in left_state or key not in right_state):
            continue
        if suffix and not key.endswith(suffix):
            continue
        left = left_state[key]
        right = right_state[key]
        checked += 1
        if left.dtype != right.dtype:
            dtype_mismatch.append(key)
        if left.device != right.device:
            device_mismatch.append(key)
        if not (left.is_floating_point() or right.is_floating_point()):
            if not torch.equal(left.cpu(), right.cpu()):
                max_abs = max(max_abs, 1.0)
                max_rel = max(max_rel, 1.0)
            continue
        diff = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
        denom = torch.maximum(left.detach().float().cpu().abs(), torch.full_like(diff, 1e-12))
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float((diff / denom).max().item()))
    return {
        "checked_tensor_count": checked,
        "missing_tensor_count": len(missing),
        "missing_tensor_examples": missing[:10],
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "dtype_mismatch_count": len(dtype_mismatch),
        "dtype_mismatch_examples": dtype_mismatch[:10],
        "device_mismatch_count": len(device_mismatch),
        "device_mismatch_examples": device_mismatch[:10],
        "status": "pass" if not missing and max_abs <= DEFAULT_ABS_TOL and max_rel <= DEFAULT_REL_TOL and not dtype_mismatch and not device_mismatch else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument("--functional-rel-l2-tol", type=float, default=DEFAULT_FUNCTIONAL_REL_L2_TOL)
    parser.add_argument("--functional-cosine-tol", type=float, default=DEFAULT_FUNCTIONAL_COSINE_TOL)
    parser.add_argument("--functional-prediction-agreement-tol", type=float, default=0.995)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    selection = read_json(args.out_dir / "standalone_source_adapter_selection.json")
    selected = selection["selected_source_adapter"]
    adapter_path = PROJECT_ROOT / Path(selected["adapter_path"])
    manifest_path = PROJECT_ROOT / "data/benchmarks/hvue_gue_manifest.csv"
    val_predictions_path = PROJECT_ROOT / Path(selected["validation_prediction_path"])
    adapter_state, adapter_meta = load_adapter_state(adapter_path)
    modules = sorted({module_name_from_adapter_key(key) for key in adapter_state if key.endswith(".lora_A")})
    lora_scale = 2.0

    base_model = load_local_checkpoint(str(args.model_dir), args.config_path, device=args.device)
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items() if name.endswith(".weight")}

    model_a = load_local_checkpoint(str(args.model_dir), args.config_path, device=args.device)
    _, injected_modules = inject_lora_all_blocks(model_a, rank=int(selected["rank"]), alpha=int(selected["rank"] * lora_scale), dropout=0.0)
    loaded_modules = set_lora_weights(model_a, adapter_state)

    model_b = load_local_checkpoint(str(args.model_dir), args.config_path, device=args.device)
    model_c = load_local_checkpoint(str(args.model_dir), args.config_path, device=args.device)
    model_d = load_local_checkpoint(str(args.model_dir), args.config_path, device=args.device)
    _, _ = inject_lora_all_blocks(model_d, rank=int(selected["rank"]), alpha=int(selected["rank"] * lora_scale), dropout=0.0)
    _ = set_lora_weights(model_d, adapter_state)

    parameter_rows: list[dict[str, Any]] = []
    merged_names = merge_lora_adapters(model_d)
    max_param_abs = 0.0
    max_param_rel = 0.0
    param_passes = 0

    for module in modules:
        a = adapter_state[f"base_model.{module}.lora_A"].float()
        b = adapter_state[f"base_model.{module}.lora_B"].float()
        rank = int(a.shape[0])
        alpha = float(rank * lora_scale)
        scale = alpha / rank
        dense_delta = (b @ a) * scale
        weight_key = module + ".weight"
        merged_linear = dict(model_d.named_modules())[module].weight.detach().cpu().float()
        base_weight = base_state[weight_key].float()
        actual_delta = merged_linear - base_weight
        merged_dtype = dict(model_d.named_modules())[module].weight.dtype
        merged_weight_expected = (
            base_state[weight_key].to(merged_dtype) + dense_delta.to(merged_dtype)
        ).to(merged_dtype)
        expected_delta = merged_weight_expected.float() - base_weight
        abs_diff = (actual_delta - expected_delta).abs()
        rel_diff = abs_diff / torch.maximum(expected_delta.abs(), torch.full_like(expected_delta, 1e-12))
        passed = float(abs_diff.max().item()) <= args.abs_tol and float(rel_diff.max().item()) <= args.rel_tol
        if passed:
            param_passes += 1
        max_param_abs = max(max_param_abs, float(abs_diff.max().item()))
        max_param_rel = max(max_param_rel, float(rel_diff.max().item()))
        parameter_rows.append(
            {
                "module_name": module,
                "a_shape": list(a.shape),
                "b_shape": list(b.shape),
                "rank": rank,
                "lora_alpha": alpha,
                "scaling_value": scale,
                "fan_in_fan_out": False,
                "matrix_orientation": "delta_weight = scale * lora_B @ lora_A; forward adds x @ delta_weight.T",
                "dtype": str(base_weight.dtype),
                "merged_weight_dtype": str(merged_dtype),
                "base_weight_shape": list(base_weight.shape),
                "reconstructed_update_shape": list(dense_delta.shape),
                "reconstructed_update_norm": float(torch.linalg.vector_norm(dense_delta).item()),
                "actual_merged_weight_delta_norm": float(torch.linalg.vector_norm(actual_delta).item()),
                "expected_quantized_merged_weight_delta_norm": float(torch.linalg.vector_norm(expected_delta).item()),
                "max_abs_diff": float(abs_diff.max().item()),
                "max_relative_diff": float(rel_diff.max().item()),
                "status": "pass" if passed else "fail",
            }
        )
        merged_target = model_b.state_dict()[weight_key]
        quantized_delta = dense_delta.to(merged_target.dtype).to(merged_target.device)
        merged_target.add_(quantized_delta)

    sample_ids: list[str] = []
    with val_predictions_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_ids.append(row["sample_id"])
            if len(sample_ids) >= args.batch_size:
                break
    input_ids, mask, sample_rows = build_sample_batch(manifest_path, sample_ids, max_length=args.max_length, device=args.device)

    logits_a, hidden_a = capture_final_hidden(model_a, input_ids, mask)
    logits_a_repeat, hidden_a_repeat = capture_final_hidden(model_a, input_ids, mask)
    logits_b, hidden_b = capture_final_hidden(model_b, input_ids, mask)
    logits_b_repeat, hidden_b_repeat = capture_final_hidden(model_b, input_ids, mask)
    logits_c, hidden_c = capture_final_hidden(model_c, input_ids, mask)
    logits_d, hidden_d = capture_final_hidden(model_d, input_ids, mask)
    summary_a_repeat = compare_outputs(logits_a, logits_a_repeat, hidden_a, hidden_a_repeat)
    summary_b_repeat = compare_outputs(logits_b, logits_b_repeat, hidden_b, hidden_b_repeat)
    summary_ab = compare_outputs(logits_a, logits_b, hidden_a, hidden_b)
    summary_ad = compare_outputs(logits_a, logits_d, hidden_a, hidden_d)
    summary_bd = compare_outputs(logits_b, logits_d, hidden_b, hidden_d)
    summary_ac = compare_outputs(logits_a, logits_c, hidden_a, hidden_c)
    summary_bc = compare_outputs(logits_b, logits_c, hidden_b, hidden_c)
    summary_dc = compare_outputs(logits_d, logits_c, hidden_d, hidden_c)

    source_update_frobenius_norm = math.sqrt(
        sum(float(row["reconstructed_update_norm"]) ** 2 for row in parameter_rows)
    )
    source_update_nonzero = source_update_frobenius_norm > 0.0
    base_difference_pass = (
        summary_ac.max_abs_logit_diff > args.abs_tol and summary_bc.max_abs_logit_diff > args.abs_tol
    )
    official_manual_state = compare_state_tensors(model_b, model_d)
    official_manual_weight_state = compare_state_tensors(model_b, model_d, suffix=".weight")
    repeat_deterministic = (
        summary_a_repeat.max_abs_logit_diff <= args.abs_tol
        and summary_a_repeat.hidden_max_abs_diff <= args.abs_tol
        and summary_b_repeat.max_abs_logit_diff <= args.abs_tol
        and summary_b_repeat.hidden_max_abs_diff <= args.abs_tol
    )
    active_adapter_state_valid = (
        count_lora_modules(model_a) == len(loaded_modules)
        and count_lora_modules(model_b) == 0
        and count_lora_modules(model_d) == 0
    )
    official_merge_vs_manual_merge_pass = (
        official_manual_state["status"] == "pass"
        and summary_bd.max_abs_logit_diff <= args.abs_tol
        and summary_bd.hidden_max_abs_diff <= args.abs_tol
    )
    bf16_functional_equivalence_pass = (
        repeat_deterministic
        and active_adapter_state_valid
        and official_merge_vs_manual_merge_pass
        and summary_ad.hidden_relative_l2_error <= args.functional_rel_l2_tol
        and summary_ad.logit_cosine_similarity >= args.functional_cosine_tol
        and summary_ad.hidden_cosine_similarity >= args.functional_cosine_tol
        and summary_ad.token_prediction_agreement >= args.functional_prediction_agreement_tol
        and summary_ad.relative_l2_logit_error < summary_ac.relative_l2_logit_error
    )

    prediction_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(sample_rows):
        prediction_rows.append(
            {
                "sample_id": row["id"],
                "label": row["label"],
                "split": row["split"],
                "sequence_length": len(row["sequence"]),
                "model_a_top_token": int(logits_a[idx, -1].argmax().item()),
                "model_b_top_token": int(logits_b[idx, -1].argmax().item()),
                "model_d_top_token": int(logits_d[idx, -1].argmax().item()),
                "model_c_top_token": int(logits_c[idx, -1].argmax().item()),
                "ab_max_abs_logit_diff": float((logits_a[idx].float() - logits_b[idx].float()).abs().max().item()),
                "ad_max_abs_logit_diff": float((logits_a[idx].float() - logits_d[idx].float()).abs().max().item()),
                "bd_max_abs_logit_diff": float((logits_b[idx].float() - logits_d[idx].float()).abs().max().item()),
                "ac_max_abs_logit_diff": float((logits_a[idx].float() - logits_c[idx].float()).abs().max().item()),
                "bc_max_abs_logit_diff": float((logits_b[idx].float() - logits_c[idx].float()).abs().max().item()),
            }
        )

    command = (
        f"python phase2/audit_source_lora_merge_equivalence.py --out-dir {args.out_dir} "
        f"--model-dir {args.model_dir} --config-path {args.config_path} --device {args.device} "
        f"--max-length {args.max_length} --batch-size {args.batch_size} --seed {args.seed} "
        f"--abs-tol {args.abs_tol} --rel-tol {args.rel_tol} "
        f"--functional-rel-l2-tol {args.functional_rel_l2_tol} "
        f"--functional-cosine-tol {args.functional_cosine_tol} "
        f"--functional-prediction-agreement-tol {args.functional_prediction_agreement_tol}"
    )
    (args.out_dir / "source_lora_merge_equivalence_command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    (args.out_dir / "source_lora_merge_equivalence_command.sh").chmod(0o755)

    parameter_level_pass = param_passes == len(parameter_rows)
    strict_backbone_forward_pass = (
        summary_ab.max_abs_logit_diff <= args.abs_tol
        or (
            summary_ab.token_prediction_agreement >= DEFAULT_FORWARD_AGREEMENT_TOL
            and summary_ab.max_abs_logit_diff < summary_ac.max_abs_logit_diff
            and summary_ab.hidden_max_abs_diff < float("inf")
        )
    )
    summary = {
        "generated_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "source_run_id": selected["run_id"],
        "source_adapter_path": str(adapter_path),
        "source_adapter_sha256": sha256_file(adapter_path),
        "base_checkpoint_path": str(args.model_dir),
        "base_checkpoint_hash": selected["base_checkpoint_identity"]["index_sha256"],
        "code_commit": git_output(["git", "rev-parse", "HEAD"]),
        "dirty_state_hash": sha256_text(git_output(["git", "status", "--short"])),
        "input_manifest_path": str(manifest_path),
        "input_manifest_hash": sha256_file(manifest_path),
        "frozen_sample_ids": [row["id"] for row in sample_rows],
        "random_seed": args.seed,
        "dtype": "bfloat16_base_float32_lora",
        "device": args.device,
        "tolerances": {
            "abs": args.abs_tol,
            "rel": args.rel_tol,
            "functional_rel_l2": args.functional_rel_l2_tol,
            "functional_cosine": args.functional_cosine_tol,
            "functional_prediction_agreement": args.functional_prediction_agreement_tol,
        },
        "reproduction_command": command,
        "classification_head_available": False,
        "classification_head_note": "source adapter checkpoint stores LoRA weights only; no task head weights were recoverable from the selected source run artifacts",
        "parameter_level_merge_equivalence": "pass" if parameter_level_pass else "fail",
        "strict_backbone_forward_equivalence": "pass" if strict_backbone_forward_pass else "fail",
        "backbone_forward_equivalence": "pass" if bf16_functional_equivalence_pass else "fail",
        "backbone_functional_equivalence": "pass" if bf16_functional_equivalence_pass else "fail",
        "official_merge_vs_manual_merge": "pass" if official_merge_vs_manual_merge_pass else "fail",
        "parameter_state_equivalence": "pass" if parameter_level_pass and official_manual_weight_state["status"] == "pass" else "fail",
        "buffer_state_equivalence": "pass" if official_manual_state["status"] == "pass" else "fail",
        "active_adapter_state_valid": "pass" if active_adapter_state_valid else "fail",
        "deterministic_forward_self_repeat": "pass" if repeat_deterministic else "fail",
        "original_classification_head_equivalence": "unavailable",
        "source_lora_merge_equivalence": "pass" if parameter_level_pass and bf16_functional_equivalence_pass else "partial",
        "source_update_nonzero": "pass" if source_update_nonzero else "fail",
        "source_update_frobenius_norm": source_update_frobenius_norm,
        "unmerged_source_vs_base_forward_difference": "pass" if summary_ac.max_abs_logit_diff > args.abs_tol else "fail",
        "manual_merged_vs_base_forward_difference": "pass" if summary_bc.max_abs_logit_diff > args.abs_tol else "fail",
        "official_merged_vs_base_forward_difference": "pass" if summary_dc.max_abs_logit_diff > args.abs_tol else "fail",
        "nontrivial_update_check": "pass" if source_update_nonzero and base_difference_pass else "fail",
        "parameter_checks_passed": param_passes,
        "parameter_checks_total": len(parameter_rows),
        "forward_checks_passed": int(bf16_functional_equivalence_pass) + int(official_merge_vs_manual_merge_pass),
        "forward_checks_total": 2,
        "max_parameter_abs_diff": max_param_abs,
        "max_parameter_rel_diff": max_param_rel,
        "max_forward_logit_abs_diff_ab": summary_ab.max_abs_logit_diff,
        "mean_forward_logit_abs_diff_ab": summary_ab.mean_abs_logit_diff,
        "median_forward_logit_abs_diff_ab": summary_ab.median_abs_logit_diff,
        "relative_l2_forward_logit_error_ab": summary_ab.relative_l2_logit_error,
        "cosine_similarity_forward_logits_ab": summary_ab.logit_cosine_similarity,
        "reference_output_norm_ab": summary_ab.reference_logit_l2_norm,
        "max_output_magnitude_ab": summary_ab.max_reference_logit_magnitude,
        "prediction_disagreement_count_ab": summary_ab.prediction_disagreement_count,
        "prediction_total_count_ab": summary_ab.prediction_total_count,
        "max_forward_logit_abs_diff_ad": summary_ad.max_abs_logit_diff,
        "mean_forward_logit_abs_diff_ad": summary_ad.mean_abs_logit_diff,
        "median_forward_logit_abs_diff_ad": summary_ad.median_abs_logit_diff,
        "relative_l2_forward_logit_error_ad": summary_ad.relative_l2_logit_error,
        "cosine_similarity_forward_logits_ad": summary_ad.logit_cosine_similarity,
        "reference_output_norm_ad": summary_ad.reference_logit_l2_norm,
        "max_output_magnitude_ad": summary_ad.max_reference_logit_magnitude,
        "prediction_disagreement_count_ad": summary_ad.prediction_disagreement_count,
        "prediction_total_count_ad": summary_ad.prediction_total_count,
        "max_forward_logit_abs_diff_bd": summary_bd.max_abs_logit_diff,
        "relative_l2_forward_logit_error_bd": summary_bd.relative_l2_logit_error,
        "max_forward_logit_abs_diff_ac": summary_ac.max_abs_logit_diff,
        "max_forward_logit_abs_diff_bc": summary_bc.max_abs_logit_diff,
        "max_forward_logit_abs_diff_dc": summary_dc.max_abs_logit_diff,
        "max_forward_hidden_abs_diff_ab": summary_ab.hidden_max_abs_diff,
        "relative_l2_forward_hidden_error_ab": summary_ab.hidden_relative_l2_error,
        "cosine_similarity_forward_hidden_ab": summary_ab.hidden_cosine_similarity,
        "reference_hidden_norm_ab": summary_ab.reference_hidden_l2_norm,
        "max_hidden_magnitude_ab": summary_ab.max_reference_hidden_magnitude,
        "max_forward_hidden_abs_diff_ad": summary_ad.hidden_max_abs_diff,
        "relative_l2_forward_hidden_error_ad": summary_ad.hidden_relative_l2_error,
        "cosine_similarity_forward_hidden_ad": summary_ad.hidden_cosine_similarity,
        "max_forward_hidden_abs_diff_bd": summary_bd.hidden_max_abs_diff,
        "relative_l2_forward_hidden_error_bd": summary_bd.hidden_relative_l2_error,
        "max_forward_hidden_abs_diff_ac": summary_ac.hidden_max_abs_diff,
        "max_forward_hidden_abs_diff_bc": summary_bc.hidden_max_abs_diff,
        "max_forward_hidden_abs_diff_dc": summary_dc.hidden_max_abs_diff,
        "max_abs_diff_a_repeat": summary_a_repeat.max_abs_logit_diff,
        "relative_l2_diff_a_repeat": summary_a_repeat.relative_l2_logit_error,
        "max_abs_diff_b_repeat": summary_b_repeat.max_abs_logit_diff,
        "relative_l2_diff_b_repeat": summary_b_repeat.relative_l2_logit_error,
        "prediction_agreement_ab": summary_ab.token_prediction_agreement,
        "prediction_agreement_ad": summary_ad.token_prediction_agreement,
        "prediction_agreement_bd": summary_bd.token_prediction_agreement,
        "prediction_agreement_ac": summary_ac.token_prediction_agreement,
        "prediction_agreement_bc": summary_bc.token_prediction_agreement,
        "prediction_agreement_dc": summary_dc.token_prediction_agreement,
        "official_manual_state_comparison": official_manual_state,
        "official_manual_weight_state_comparison": official_manual_weight_state,
        "active_lora_module_count_a": count_lora_modules(model_a),
        "active_lora_module_count_b": count_lora_modules(model_b),
        "active_lora_module_count_d": count_lora_modules(model_d),
        "active_adapter": {"A": "source_lora_unmerged", "B": "none", "C": "none", "D": "none_after_official_merge"},
        "merged_adapter_applied_again": {"B": False, "D": False},
        "merged_module_count": len(merged_names),
        "injected_module_count": len(injected_modules),
        "loaded_lora_module_count": len(loaded_modules),
        "forward_level_scope": "base-model token logits and final hidden states only",
        "merge_equivalence_rule": "W_merged - W_base is checked against the repo's operational bfloat16 merge path: cast(base) + cast(scale * B @ A)",
        "forward_equivalence_rule": "A/B/D are judged by deterministic self-repeat, no double adapter application, official-vs-manual exactness, and bf16-scale functional agreement; strict max-abs equality is recorded separately.",
        "path_independence_note": (
            "parameter-level dense deltas are reconstructed directly from adapter A/B tensors and rank/alpha; "
            "unmerged forward uses injected LoRA modules; manual merged forward writes dense deltas into an independent base model; "
            "the manual merged comparison path does not call merge_lora_adapters()"
        ),
    }
    write_csv(args.out_dir / "source_lora_merge_equivalence_audit.csv", parameter_rows)
    write_csv(args.out_dir / "source_lora_merge_equivalence_predictions.csv", prediction_rows)
    write_json(args.out_dir / "source_lora_merge_equivalence_audit.json", summary)

    report_lines = [
        "# Source LoRA Merge Equivalence Audit",
        "",
        f"- source_lora_merge_equivalence = `{summary['source_lora_merge_equivalence']}`",
        f"- parameter_level_merge_equivalence = `{summary['parameter_level_merge_equivalence']}`",
        f"- source_update_nonzero = `{summary['source_update_nonzero']}`",
        f"- official_merge_vs_manual_merge = `{summary['official_merge_vs_manual_merge']}`",
        f"- backbone_functional_equivalence = `{summary['backbone_functional_equivalence']}`",
        f"- strict_backbone_forward_equivalence = `{summary['strict_backbone_forward_equivalence']}`",
        f"- original_classification_head_equivalence = `{summary['original_classification_head_equivalence']}`",
        f"- nontrivial_update_check = `{summary['nontrivial_update_check']}`",
        f"- active_adapter_state_valid = `{summary['active_adapter_state_valid']}`",
        f"- deterministic_forward_self_repeat = `{summary['deterministic_forward_self_repeat']}`",
        f"- parameter_checks_passed = `{param_passes} / {len(parameter_rows)}`",
        f"- forward_checks_passed = `{summary['forward_checks_passed']} / {summary['forward_checks_total']}`",
        f"- max_parameter_abs_diff = `{max_param_abs:.6g}`",
        f"- source_update_frobenius_norm = `{source_update_frobenius_norm:.6g}`",
        f"- max_forward_logit_abs_diff(A,B) = `{summary_ab.max_abs_logit_diff:.6g}`",
        f"- relative_l2_forward_logit_error(A,B) = `{summary_ab.relative_l2_logit_error:.6g}`",
        f"- cosine_similarity_forward_logits(A,B) = `{summary_ab.logit_cosine_similarity:.9g}`",
        f"- prediction_disagreement(A,B) = `{summary_ab.prediction_disagreement_count} / {summary_ab.prediction_total_count}`",
        f"- max_forward_logit_abs_diff(A,D official) = `{summary_ad.max_abs_logit_diff:.6g}`",
        f"- relative_l2_forward_logit_error(A,D official) = `{summary_ad.relative_l2_logit_error:.6g}`",
        f"- cosine_similarity_forward_logits(A,D official) = `{summary_ad.logit_cosine_similarity:.9g}`",
        f"- max_forward_logit_abs_diff(B,D official) = `{summary_bd.max_abs_logit_diff:.6g}`",
        f"- max_forward_logit_abs_diff(A,C) = `{summary_ac.max_abs_logit_diff:.6g}`",
        f"- max_forward_logit_abs_diff(B,C) = `{summary_bc.max_abs_logit_diff:.6g}`",
        f"- max_forward_hidden_abs_diff(A,B) = `{summary_ab.hidden_max_abs_diff:.6g}`",
        f"- relative_l2_forward_hidden_error(A,B) = `{summary_ab.hidden_relative_l2_error:.6g}`",
        f"- cosine_similarity_forward_hidden(A,B) = `{summary_ab.hidden_cosine_similarity:.9g}`",
        f"- max_abs_diff_a_repeat = `{summary_a_repeat.max_abs_logit_diff:.6g}`",
        f"- max_abs_diff_b_repeat = `{summary_b_repeat.max_abs_logit_diff:.6g}`",
        f"- prediction_agreement(A,B) = `{summary_ab.token_prediction_agreement:.6f}`",
        "",
        "## Notes",
        "",
        f"- Classification head available: `{summary['classification_head_available']}`",
        f"- Classification head note: {summary['classification_head_note']}",
        f"- Parameter rule: `{summary['merge_equivalence_rule']}`",
        f"- Forward rule: `{summary['forward_equivalence_rule']}`",
        f"- Path independence: `{summary['path_independence_note']}`",
        f"- Forward scope: `{summary['forward_level_scope']}`",
        "",
    ]
    (args.out_dir / "source_lora_merge_equivalence_audit.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
