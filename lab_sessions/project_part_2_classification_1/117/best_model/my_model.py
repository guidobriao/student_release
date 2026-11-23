from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from utils import build_explicit_features, parse_input_dataframe


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Return the fitted predictor and plain numerical preprocessing state."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: dict[str, Any], day_and_lag_features_df: pd.DataFrame) -> np.ndarray:
    """Return a finite NumPy array of shape (1,) with probability of class 1 (HAB == 1)."""
    if not isinstance(day_and_lag_features_df, pd.DataFrame) or day_and_lag_features_df.empty:
        raise ValueError("day_and_lag_features_df must be a non-empty pandas DataFrame")

    df_copy = day_and_lag_features_df.copy(deep=True)
    X_raw, _, _ = build_explicit_features(df_copy, fill_stats=model["fill_stats"])

    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    X_scaled = (X_raw - mean) / scale

    estimator = model["estimator"]
    probs = estimator.predict_proba(X_scaled)[:, 1]
    prob_val = float(np.clip(np.asarray(probs, dtype=float)[-1], 0.0, 1.0))

    out = np.asarray([prob_val], dtype=float)
    if out.shape != (1,) or not np.isfinite(out).all():
        raise ValueError("Prediction must be a finite array of shape (1,)")
    return out


def predict_from_xlsx(model: dict[str, Any], xlsx_path: str | Path) -> np.ndarray:
    """Parse an .xlsx file (without requiring HAB) and run prediction."""
    df = parse_input_dataframe(xlsx_path)
    return predict(model, df)