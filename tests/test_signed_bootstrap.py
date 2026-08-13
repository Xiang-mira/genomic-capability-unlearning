from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from phase2.signed_bootstrap import (
    assert_signed_delta_summary,
    paired_grouped_prediction_bootstrap,
    summarize_signed_deltas,
)


def test_summarize_signed_deltas_preserves_distribution_signs() -> None:
    deltas = np.array([0.3, 0.1, -0.2, 0.0], dtype=float)
    summary = summarize_signed_deltas(
        deltas,
        observed_delta=0.2,
        status="complete",
        valid_replicates=4,
        invalid_replicates=0,
        attempted_replicates=4,
    )
    assert summary["observed_delta"] == pytest.approx(0.2)
    assert summary["mean_delta"] == pytest.approx(np.mean(deltas))
    assert summary["median_delta"] == pytest.approx(np.median(deltas))
    assert summary["ci95_low"] <= summary["median_delta"] <= summary["ci95_high"]
    assert summary["p_delta_gt_0"] == pytest.approx(0.5)
    assert summary["p_delta_lt_0"] == pytest.approx(0.25)
    assert summary["p_delta_eq_0"] == pytest.approx(0.25)
    assert summary["p_delta_gt_0"] + summary["p_delta_lt_0"] + summary["p_delta_eq_0"] == pytest.approx(1.0)


def test_assert_signed_delta_summary_rejects_bad_observed_delta() -> None:
    summary = summarize_signed_deltas(
        [0.1, 0.2, 0.3],
        observed_delta=0.2,
        status="complete",
        valid_replicates=3,
        invalid_replicates=0,
        attempted_replicates=3,
    )
    with pytest.raises(AssertionError):
        assert_signed_delta_summary(summary, direct_observed_delta=-0.2)


def test_paired_grouped_prediction_bootstrap_uses_model_minus_baseline() -> None:
    rows = pd.DataFrame(
        [
            {"group": "g1", "true": "A", "model": "A", "baseline": "B"},
            {"group": "g1", "true": "B", "model": "B", "baseline": "B"},
            {"group": "g2", "true": "A", "model": "A", "baseline": "A"},
            {"group": "g2", "true": "B", "model": "A", "baseline": "A"},
        ]
    )
    labels = ["A", "B"]
    samples, summary = paired_grouped_prediction_bootstrap(
        rows,
        group_col="group",
        true_col="true",
        model_pred_col="model",
        baseline_pred_col="baseline",
        labels=labels,
        scorer=lambda y_true, y_pred, lab: f1_score(
            y_true, y_pred, labels=list(lab), average="macro", zero_division=0
        ),
        n_valid=20,
        max_attempts=400,
        seed=7,
        model_score_key="model_macro_f1",
        baseline_score_key="baseline_macro_f1",
        delta_key="delta_model_minus_baseline",
        bootstrap_unit="group",
        invalid_reason="missing class",
    )
    direct_model = f1_score(rows["true"], rows["model"], labels=labels, average="macro", zero_division=0)
    direct_base = f1_score(rows["true"], rows["baseline"], labels=labels, average="macro", zero_division=0)
    assert summary["observed_delta"] == pytest.approx(direct_model - direct_base)
    assert summary["valid_bootstrap_replicates"] == 20
    assert all(
        sample["delta_model_minus_baseline"]
        == pytest.approx(sample["model_macro_f1"] - sample["baseline_macro_f1"])
        for sample in samples
    )
