from __future__ import annotations

from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

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
            C=2.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "mlp_no_early_stopping": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.5,
            early_stopping=False,
            max_iter=500,
            random_state=42,
        ),
        "svc_rbf": SVC(
            C=0.5, kernel="rbf", probability=True, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=20,
            max_iter=150,
            random_state=42,
        ),
    }


def _complete_rows_mask(df: pd.DataFrame) -> pd.Series:
    required = [col for col in ("CHL", "CHL_uncertainty") if col in df.columns]
    if not required:
        return pd.Series(True, index=df.index)
    return df[required].notna().all(axis=1)


def main() -> None:
    data_candidates = [
        PROJECT_ROOT.parent / "data" / "classification_task1_train.xlsx",
        PROJECT_ROOT / "data" / "classification_task1_train.xlsx",
        PROJECT_ROOT.parent / "data" / "classification_task1_train.csv",
        PROJECT_ROOT / "data" / "classification_task1_train.csv",
    ]
    data_path = next((p for p in data_candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not locate classification_task1_train.{xlsx,csv}")

    df = parse_input_dataframe(data_path)
    if "HAB" not in df.columns:
        raise ValueError("Training dataset is missing target column 'HAB'")

    folds = [
        (df["Year"] <= 2016, (df["Year"] >= 2017) & (df["Year"] <= 2018)),
        (df["Year"] <= 2018, df["Year"] == 2019),
        (df["Year"] <= 2019, df["Year"] == 2020),
    ]

    candidate_names = list(build_candidates().keys())
    cv_scores: dict[str, float] = {}

    for name in candidate_names:
        fold_aucs_comp: list[float] = []
        fold_aucs_all: list[float] = []
        for tr_mask, va_mask in folds:
            tr_df = df.loc[tr_mask & _complete_rows_mask(df)].copy()
            va_df = df.loc[va_mask].copy()

            X_tr_raw, _, fill_stats = build_explicit_features(tr_df, fill_stats=None)
            mean, scale = compute_standardizer(X_tr_raw)
            X_tr = (X_tr_raw - mean) / scale
            y_tr = tr_df["HAB"].to_numpy(dtype=int)

            X_va_raw, _, _ = build_explicit_features(va_df, fill_stats=fill_stats)
            X_va = (X_va_raw - mean) / scale
            y_va = va_df["HAB"].to_numpy(dtype=int)

            clf = build_candidates()[name]
            clf.fit(X_tr, y_tr)
            probs = (
                clf.predict_proba(X_va)[:, 1]
                if hasattr(clf, "predict_proba")
                else clf.decision_function(X_va)
            )

            fold_aucs_all.append(float(roc_auc_score(y_va, probs)))
            comp_mask = _complete_rows_mask(va_df).to_numpy()
            fold_aucs_comp.append(float(roc_auc_score(y_va[comp_mask], probs[comp_mask])))

        mean_all = float(np.mean(fold_aucs_all))
        mean_comp = float(np.mean(fold_aucs_comp))
        cv_scores[name] = mean_comp
        print(f"[{name}] expanding CV AUC(all)={mean_all:.4f} | AUC(complete)={mean_comp:.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nSelected best model: {best_name} (expanding CV complete AUC={cv_scores[best_name]:.4f})")

    # Fit on complete training rows (matching Task 1 complete 2021-2023 test set)[cite: 14]
    complete_df = df.loc[_complete_rows_mask(df)].copy()
    X_full_raw, feature_names, final_fill_stats = build_explicit_features(
        complete_df, fill_stats=None
    )
    final_mean, final_scale = compute_standardizer(X_full_raw)
    X_full = (X_full_raw - final_mean) / final_scale

    final_clf = build_candidates()[best_name]
    final_clf.fit(X_full, complete_df["HAB"].to_numpy(dtype=int))

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_name,
        "estimator": final_clf,
        "scaler_mean": final_mean,
        "scaler_scale": final_scale,
        "fill_stats": final_fill_stats,
        "feature_names": feature_names,
    }
    out_path = BEST_MODEL_DIR / "model.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    print(f"Saved model artifact to {out_path}")


if __name__ == "__main__":
    main()