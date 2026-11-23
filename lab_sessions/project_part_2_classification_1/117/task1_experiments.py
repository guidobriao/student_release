from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_MODEL_DIR = SCRIPT_DIR / "best_model"
if str(BEST_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BEST_MODEL_DIR))

from utils import CLUSTER_CATEGORIES, EPS, SEASON_CATEGORIES, parse_input_dataframe


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    work = parse_input_dataframe(df)
    dt = pd.to_datetime(work["Date"], errors="coerce")
    doy = dt.dt.dayofyear.to_numpy(dtype=float)
    work["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.0)

    chl = pd.to_numeric(work["CHL"], errors="coerce").to_numpy(dtype=float)
    chl_unc = pd.to_numeric(work["CHL_uncertainty"], errors="coerce").to_numpy(dtype=float)
    work["log1p_chl"] = np.log1p(np.clip(chl, a_min=0.0, a_max=None))
    work["log1p_chl_unc"] = np.log1p(np.clip(chl_unc, a_min=0.0, a_max=None))
    work["chl_snr"] = chl / (chl_unc + EPS)
    return work


def build_feature_matrix(
    sub_df: pd.DataFrame,
    feature_set: str,
    fill_stats: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    imputable_cols = ["CHL", "CHL_uncertainty", "log1p_chl", "log1p_chl_unc", "chl_snr"]
    if fill_stats is None:
        fill_stats = {}
        for col in imputable_cols:
            s = pd.to_numeric(sub_df[col], errors="coerce")
            fill_stats[col] = float(s.median()) if s.notna().any() else 0.0

    cols: list[pd.Series] = [pd.to_numeric(sub_df["CHL"], errors="coerce").isna().astype(float)]

    if feature_set in ("fe_trig_log", "fe_full"):
        for col in ("CHL", "CHL_uncertainty"):
            cols.append(pd.to_numeric(sub_df[col], errors="coerce").fillna(fill_stats[col]).astype(float))
        cols.append(pd.to_numeric(sub_df["log1p_chl"], errors="coerce").fillna(fill_stats["log1p_chl"]).astype(float))

    if feature_set == "fe_full":
        cols.append(pd.to_numeric(sub_df["log1p_chl_unc"], errors="coerce").fillna(fill_stats["log1p_chl_unc"]).astype(float))
        cols.append(pd.to_numeric(sub_df["chl_snr"], errors="coerce").fillna(fill_stats["chl_snr"]).astype(float))

    if feature_set in ("fe_trig_log", "fe_full"):
        cols.append(pd.to_numeric(sub_df["sin_doy"], errors="coerce").fillna(0.0).astype(float))

    for col in ("SST", "UI", "YDOY"):
        cols.append(pd.to_numeric(sub_df[col], errors="coerce").fillna(0.0).astype(float))

    season_s = sub_df["Season"].astype(str)
    for cat in SEASON_CATEGORIES:
        cols.append((season_s == cat).astype(float))

    for cl in CLUSTER_CATEGORIES:
        cols.append((sub_df["spatial_cluster"] == cl).astype(float))

    return pd.concat(cols, axis=1).to_numpy(dtype=float), fill_stats


def standardize_fit_apply(X_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=0)
    scale = np.where(X_tr.std(axis=0) == 0.0, 1.0, X_tr.std(axis=0))
    return (X_tr - mean) / scale, (X_va - mean) / scale


def get_model_builders() -> dict[str, callable]:
    return {
        "LogReg (C=0.1, balanced)": lambda: LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "LogReg (C=0.5, balanced)": lambda: LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "LogReg (C=2.0, balanced)": lambda: LogisticRegression(
            C=2.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "MLP ((32, 16), early_stopping=False, alpha=0.5)": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.5, early_stopping=False, max_iter=500, random_state=42
        ),
        "SVC (RBF, C=0.5, probability=True, balanced)": lambda: SVC(
            C=0.5, kernel="rbf", probability=True, class_weight="balanced", random_state=42
        ),
        "HistGradientBoosting (lr=0.03, max_depth=2)": lambda: HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=2, min_samples_leaf=20, max_iter=150, random_state=42
        ),
        "RandomForest (max_depth=3, n=200, balanced)": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42
        ),
    }


def main() -> None:
    candidates = [
        SCRIPT_DIR.parent / "data" / "classification_task1_train.csv",
        SCRIPT_DIR.parent / "data" / "classification_task1_train.xlsx",
        SCRIPT_DIR / "data" / "classification_task1_train.csv",
        SCRIPT_DIR / "data" / "classification_task1_train.xlsx",
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not locate classification_task1_train.{csv,xlsx}")

    df = add_engineered_features(parse_input_dataframe(data_path))
    folds = [
        (df["Year"] <= 2016, (df["Year"] >= 2017) & (df["Year"] <= 2018)),
        (df["Year"] <= 2018, df["Year"] == 2019),
        (df["Year"] <= 2019, df["Year"] == 2020),
    ]
    scenarios = [
        ("fe_trig_log All Train", "fe_trig_log", False),
        ("fe_trig_log Complete Train", "fe_trig_log", True),
        ("fe_full All Train", "fe_full", False),
        ("fe_full Complete Train", "fe_full", True),
    ]
    builders = get_model_builders()

    print("=== PART A: EXPANDING CHRONOLOGICAL CV (<=2016->2017-2018, <=2018->2019, <=2019->2020) ===")
    for model_name, build_fn in builders.items():
        print(f"\nModel: {model_name}")
        for scen_label, fs, comp_only in scenarios:
            aucs_all: list[float] = []
            aucs_comp: list[float] = []
            for tr_mask, va_mask in folds:
                tr_df = df.loc[tr_mask].copy()
                if comp_only:
                    tr_df = tr_df.loc[tr_df[["CHL", "CHL_uncertainty"]].notna().all(axis=1)].copy()
                va_df = df.loc[va_mask].copy()

                X_tr_r, st = build_feature_matrix(tr_df, fs, None)
                X_va_r, _ = build_feature_matrix(va_df, fs, st)
                X_tr, X_va = standardize_fit_apply(X_tr_r, X_va_r)

                clf = build_fn()
                clf.fit(X_tr, tr_df["HAB"].to_numpy(dtype=int))
                probs = clf.predict_proba(X_va)[:, 1]
                yv = va_df["HAB"].to_numpy(dtype=int)
                cm = va_df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()

                aucs_all.append(float(roc_auc_score(yv, probs)))
                aucs_comp.append(float(roc_auc_score(yv[cm], probs[cm])))
            print(
                f"  {scen_label:28s} -> AUC (All / Complete) = "
                f"{np.mean(aucs_all):.4f} / {np.mean(aucs_comp):.4f}"
            )

    va_single = df[df["Year"] > 2018].copy()
    yv_single = va_single["HAB"].to_numpy(dtype=int)
    cm_single = va_single[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()

    print(f"\n=== PART B: SINGLE CHRONOLOGICAL SPLIT (<=2018 -> 2019-2020, n={len(va_single)}) ===")
    for model_name, build_fn in builders.items():
        print(f"\nModel: {model_name}")
        for scen_label, fs, comp_only in scenarios:
            tr_single = df[df["Year"] <= 2018].copy()
            if comp_only:
                tr_single = tr_single.loc[tr_single[["CHL", "CHL_uncertainty"]].notna().all(axis=1)].copy()

            X_tr_r, st = build_feature_matrix(tr_single, fs, None)
            X_va_r, _ = build_feature_matrix(va_single, fs, st)
            X_tr, X_va = standardize_fit_apply(X_tr_r, X_va_r)

            clf = build_fn()
            clf.fit(X_tr, tr_single["HAB"].to_numpy(dtype=int))
            probs = clf.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)

            auc_all = roc_auc_score(yv_single, probs)
            auc_comp = roc_auc_score(yv_single[cm_single], probs[cm_single])
            f1 = f1_score(yv_single, preds, zero_division=0)
            prec = precision_score(yv_single, preds, zero_division=0)
            rec = recall_score(yv_single, preds, zero_division=0)
            print(
                f"  {scen_label:28s} -> AUC (All / Complete) = {auc_all:.4f} / {auc_comp:.4f} "
                f"| F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f}"
            )


if __name__ == "__main__":
    main()