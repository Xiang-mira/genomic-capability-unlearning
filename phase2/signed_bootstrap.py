from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def summarize_signed_deltas(
    deltas: Sequence[float] | np.ndarray,
    *,
    observed_delta: float,
    status: str,
    valid_replicates: int,
    invalid_replicates: int,
    attempted_replicates: int,
    delta_prefix: str = "",
    bootstrap_unit: str | None = None,
    requested_valid_replicates: int | None = None,
    invalid_reason: str | None = None,
) -> dict[str, Any]:
    delta_prefix = f"{delta_prefix}_" if delta_prefix else ""
    arr = np.asarray(list(deltas), dtype=float)
    summary: dict[str, Any] = {
        "status": status,
        "observed_delta": float(observed_delta),
        "valid_bootstrap_replicates": int(valid_replicates),
        "invalid_bootstrap_replicates": int(invalid_replicates),
        "attempted_bootstrap_replicates": int(attempted_replicates),
    }
    if bootstrap_unit is not None:
        summary["bootstrap_unit"] = bootstrap_unit
    if requested_valid_replicates is not None:
        summary["requested_valid_replicates"] = int(requested_valid_replicates)
    if invalid_reason is not None:
        summary["invalid_reason"] = invalid_reason
    if arr.size == 0:
        summary.update(
            {
                f"{delta_prefix}mean_delta": None,
                f"{delta_prefix}median_delta": None,
                f"{delta_prefix}ci95_low": None,
                f"{delta_prefix}ci95_high": None,
                f"{delta_prefix}p_delta_gt_0": None,
                f"{delta_prefix}p_delta_lt_0": None,
                f"{delta_prefix}p_delta_eq_0": None,
            }
        )
        return summary

    summary.update(
        {
            f"{delta_prefix}mean_delta": float(np.mean(arr)),
            f"{delta_prefix}median_delta": float(np.median(arr)),
            f"{delta_prefix}ci95_low": float(np.quantile(arr, 0.025)),
            f"{delta_prefix}ci95_high": float(np.quantile(arr, 0.975)),
            f"{delta_prefix}p_delta_gt_0": float(np.mean(arr > 0)),
            f"{delta_prefix}p_delta_lt_0": float(np.mean(arr < 0)),
            f"{delta_prefix}p_delta_eq_0": float(np.mean(arr == 0)),
        }
    )
    assert_signed_delta_summary(summary, delta_prefix=delta_prefix, direct_observed_delta=observed_delta)
    return summary


def assert_signed_delta_summary(
    summary: dict[str, Any],
    *,
    delta_prefix: str = "",
    direct_observed_delta: float | None = None,
    atol: float = 1e-9,
) -> None:
    delta_prefix = f"{delta_prefix}_" if delta_prefix else ""
    ci_low = summary.get(f"{delta_prefix}ci95_low")
    ci_high = summary.get(f"{delta_prefix}ci95_high")
    median = summary.get(f"{delta_prefix}median_delta")
    p_gt = summary.get(f"{delta_prefix}p_delta_gt_0")
    p_lt = summary.get(f"{delta_prefix}p_delta_lt_0")
    p_eq = summary.get(f"{delta_prefix}p_delta_eq_0")
    observed = summary.get("observed_delta")
    if direct_observed_delta is not None and observed is not None:
        if not np.isclose(float(observed), float(direct_observed_delta), atol=atol):
            raise AssertionError(
                f"observed_delta={observed} disagrees with direct arithmetic {direct_observed_delta}"
            )
    if median is not None and ci_low is not None and ci_high is not None:
        if not (float(ci_low) - atol <= float(median) <= float(ci_high) + atol):
            raise AssertionError(
                f"median delta {median} lies outside CI [{ci_low}, {ci_high}]"
            )
    if p_gt is not None and p_lt is not None and p_eq is not None:
        total = float(p_gt) + float(p_lt) + float(p_eq)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise AssertionError(f"p(delta>0)+p(delta<0)+p(delta=0)={total}, expected 1")
        nonzero_mass = float(p_gt) + float(p_lt)
        if float(p_eq) <= 1e-6 and not np.isclose(nonzero_mass, 1.0, atol=1e-6):
            raise AssertionError(f"p(delta>0)+p(delta<0)={nonzero_mass}, expected 1 when ties absent")


def paired_grouped_prediction_bootstrap(
    rows: pd.DataFrame,
    *,
    group_col: str,
    true_col: str,
    model_pred_col: str,
    baseline_pred_col: str,
    labels: Sequence[str],
    scorer: Callable[[Sequence[str], Sequence[str], Sequence[str]], float],
    n_valid: int,
    max_attempts: int,
    seed: int,
    model_score_key: str,
    baseline_score_key: str,
    delta_key: str,
    bootstrap_unit: str,
    invalid_reason: str,
    extra_sample_fields: dict[str, Callable[[pd.DataFrame, np.ndarray], Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = sorted(rows[group_col].astype(str).unique().tolist())
    by_group = {group: rows[rows[group_col].astype(str) == group] for group in groups}
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    invalid = 0
    attempts = 0
    while len(samples) < n_valid and attempts < max_attempts:
        attempts += 1
        chosen = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[group] for group in chosen], ignore_index=True)
        present = set(sample[true_col].astype(str))
        if set(labels) - present:
            invalid += 1
            continue
        model_score = float(
            scorer(sample[true_col].astype(str), sample[model_pred_col].astype(str), labels)
        )
        baseline_score = float(
            scorer(sample[true_col].astype(str), sample[baseline_pred_col].astype(str), labels)
        )
        row: dict[str, Any] = {
            "replicate": len(samples) + 1,
            model_score_key: model_score,
            baseline_score_key: baseline_score,
            delta_key: model_score - baseline_score,
        }
        if extra_sample_fields:
            for key, fn in extra_sample_fields.items():
                row[key] = fn(sample, chosen)
        samples.append(row)
    observed_model_score = float(
        scorer(rows[true_col].astype(str), rows[model_pred_col].astype(str), labels)
    )
    observed_baseline_score = float(
        scorer(rows[true_col].astype(str), rows[baseline_pred_col].astype(str), labels)
    )
    observed_delta = observed_model_score - observed_baseline_score
    deltas = np.asarray([row[delta_key] for row in samples], dtype=float)
    summary = summarize_signed_deltas(
        deltas,
        observed_delta=observed_delta,
        status="complete" if len(samples) == n_valid else "partial",
        valid_replicates=len(samples),
        invalid_replicates=invalid,
        attempted_replicates=attempts,
        bootstrap_unit=bootstrap_unit,
        requested_valid_replicates=n_valid,
        invalid_reason=invalid_reason,
    )
    return samples, summary
