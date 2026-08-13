from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_legacy_saved_model(model_path: str | Path) -> Any:
    model_path = str(model_path)
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    try:
        import tf_keras

        return tf_keras.models.load_model(model_path)
    except Exception:
        from tensorflow import keras

        return keras.models.load_model(model_path)


def predict_probabilities(model_path: str | Path, embeddings: np.ndarray) -> np.ndarray:
    model = load_legacy_saved_model(model_path)
    outputs = model.predict(embeddings, verbose=0)
    return np.asarray(outputs, dtype=np.float32)


def load_label_classes(classes_path: str | Path) -> list[str]:
    with Path(classes_path).open("rb") as handle:
        classes = pickle.load(handle)
    if hasattr(classes, "classes_"):
        return [str(x) for x in classes.classes_]
    raise TypeError(f"unsupported classes object in {classes_path}")


def format_probability_frame(
    probabilities: np.ndarray,
    *,
    classes_path: str | Path,
    identifiers: list[str],
) -> pd.DataFrame:
    classes = load_label_classes(classes_path)
    return pd.DataFrame(probabilities, columns=classes, index=identifiers)


def format_prediction_frame(
    probabilities: np.ndarray,
    *,
    classes_path: str | Path,
    identifiers: list[str],
    calibration_thresholds: dict[str, float] | None = None,
    unknown_output_label: str = "unknown function",
) -> pd.DataFrame:
    classes = load_label_classes(classes_path)
    rows: list[tuple[str, str, float]] = []
    if calibration_thresholds is None:
        for idx, ident in enumerate(identifiers):
            pred_idx = int(np.argmax(probabilities[idx]))
            rows.append((ident, classes[pred_idx], float(probabilities[idx][pred_idx])))
        return pd.DataFrame(rows, columns=["protein_id", "class_phrog", "phog_model_score"])

    threshold_vector = [calibration_thresholds[cat] for cat in classes[:-1]]
    for idx, ident in enumerate(identifiers):
        thresholded = probabilities[idx][:-1] > threshold_vector
        indices = np.where(thresholded)[0]
        if len(indices) < 1:
            rows.append((ident, unknown_output_label, float(probabilities[idx][-1])))
            continue
        for pred_idx in indices:
            rows.append((ident, classes[int(pred_idx)], float(probabilities[idx][int(pred_idx)])))
    return pd.DataFrame(rows, columns=["protein_id", "class_phrog", "phog_model_score"])
