from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_MODEL_DIR = PROJECT_ROOT / "best_model"
if str(BEST_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BEST_MODEL_DIR))

from utils import build_explicit_features, parse_input_dataframe


def compute_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def build_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-2,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            class_weight="balanced",
            random_state=42,
        ),
    }


def main() -> None:
    data_candidates = [
        PROJECT_ROOT / "data" / "classification_task1_train.xlsx",
        PROJECT_ROOT / "classification_task1_train.xlsx",
        PROJECT_ROOT / "data" / "classification_task1_train.csv",
        PROJECT_ROOT / "classification_task1_train.csv",
        PROJECT_ROOT.parent / "data" / "classification_task1_train.xlsx",
        PROJECT_ROOT.parent / "classification_task1_train.xlsx",
        PROJECT_ROOT.parent / "data" / "classification_task1_train.csv",
        PROJECT_ROOT.parent / "classification_task1_train.csv",
    ]
    data_path = next((p for p in data_candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not locate classification_task1_train.{xlsx,csv}")

    df = parse_input_dataframe(data_path)
    if "HAB" not in df.columns:
        raise ValueError("Training dataset is missing target column 'HAB'")

    # Chronological validation split (2013-2018 train, 2019-2020 validation)
    train_mask = df["Year"] <= 2018
    val_mask = df["Year"] > 2018

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    y_train = df_train["HAB"].to_numpy(dtype=int)
    y_val = df_val["HAB"].to_numpy(dtype=int)

    X_train_raw, feature_names, chl_fill_values = build_explicit_features(df_train, chl_fill_values=None)
    mean, scale = compute_standardizer(X_train_raw)
    X_train = (X_train_raw - mean) / scale

    X_val_raw, _, _ = build_explicit_features(df_val, chl_fill_values=chl_fill_values)
    X_val = (X_val_raw - mean) / scale
    val_complete_mask = df_val[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()

    candidates = build_candidates()
    scores: dict[str, float] = {}

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        probs_all = clf.predict_proba(X_val)[:, 1]
        preds_all = (probs_all >= 0.5).astype(int)

        auc_all = roc_auc_score(y_val, probs_all)
        auc_complete = roc_auc_score(y_val[val_complete_mask], probs_all[val_complete_mask])
        f1 = f1_score(y_val, preds_all, zero_division=0)
        prec = precision_score(y_val, preds_all, zero_division=0)
        rec = recall_score(y_val, preds_all, zero_division=0)

        # Select using complete-row validation AUC (matching complete 2021-2023 test set)
        scores[name] = auc_complete
        print(
            f"[{name}] AUC(all)={auc_all:.4f} | AUC(complete)={auc_complete:.4f} | "
            f"F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}"
        )

    best_name = max(scores, key=scores.get)
    print(f"\nSelected best model: {best_name} (complete-row val AUC={scores[best_name]:.4f})")

    # Refit selected model on full 2013-2020 dataset
    X_full_raw, feature_names, final_chl_fills = build_explicit_features(df, chl_fill_values=None)
    final_mean, final_scale = compute_standardizer(X_full_raw)
    X_full = (X_full_raw - final_mean) / final_scale

    final_clf = build_candidates()[best_name]
    final_clf.fit(X_full, df["HAB"].to_numpy(dtype=int))

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_name,
        "estimator": final_clf,
        "scaler_mean": final_mean,
        "scaler_scale": final_scale,
        "chl_fill_values": final_chl_fills,
        "feature_names": feature_names,
    }
    out_path = BEST_MODEL_DIR / "model.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    print(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    main()