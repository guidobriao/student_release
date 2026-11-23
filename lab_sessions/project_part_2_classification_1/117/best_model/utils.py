from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEASON_CATEGORIES = ["Winter", "Spring", "Summer", "Fall"]
CLUSTER_CATEGORIES = [0, 1, 2, 3, 4, 5]
EPS = 1e-3


def parse_input_dataframe(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a .csv or .xlsx file (or DataFrame) and sort explicitly by Date and spatial_cluster."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            xls = pd.ExcelFile(path, engine="openpyxl")
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        sort_cols = ["Date", "spatial_cluster"] if "spatial_cluster" in df.columns else ["Date"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "Date" in work.columns:
        dt = pd.to_datetime(work["Date"], errors="coerce")
        doy = dt.dt.dayofyear.to_numpy(dtype=float)
        work["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.0)
    else:
        work["sin_doy"] = 0.0

    chl = pd.to_numeric(work["CHL"], errors="coerce").to_numpy(dtype=float) if "CHL" in work.columns else np.full(len(work), np.nan)
    chl_unc = (
        pd.to_numeric(work["CHL_uncertainty"], errors="coerce").to_numpy(dtype=float)
        if "CHL_uncertainty" in work.columns
        else np.full(len(work), np.nan)
    )

    work["log1p_chl"] = np.log1p(np.clip(chl, a_min=0.0, a_max=None))
    work["log1p_chl_unc"] = np.log1p(np.clip(chl_unc, a_min=0.0, a_max=None))
    work["chl_snr"] = chl / (chl_unc + EPS)
    return work


def build_explicit_features(
    df: pd.DataFrame,
    fill_stats: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Explicitly construct numeric feature matrix X using Pandas and NumPy only."""
    work = parse_input_dataframe(df)
    work = _compute_derived_columns(work)

    imputable_cols = ["CHL", "CHL_uncertainty", "log1p_chl", "log1p_chl_unc", "chl_snr"]
    if fill_stats is None:
        fill_stats = {}
        for col in imputable_cols:
            s = pd.to_numeric(work[col], errors="coerce")
            fill_stats[col] = float(s.median()) if s.notna().any() else 0.0

    columns: list[pd.Series] = []
    feature_names: list[str] = []

    chl_raw = pd.to_numeric(work["CHL"], errors="coerce") if "CHL" in work.columns else pd.Series(np.nan, index=work.index)
    columns.append(chl_raw.isna().astype(float))
    feature_names.append("CHL_is_missing")

    for col in imputable_cols:
        s = pd.to_numeric(work[col], errors="coerce") if col in work.columns else pd.Series(np.nan, index=work.index)
        columns.append(s.fillna(fill_stats[col]).astype(float))
        feature_names.append(f"{col}_clean")

    columns.append(pd.to_numeric(work["sin_doy"], errors="coerce").fillna(0.0).astype(float))
    feature_names.append("sin_doy")

    for col in ["SST", "UI", "YDOY"]:
        s = pd.to_numeric(work[col], errors="coerce") if col in work.columns else pd.Series(0.0, index=work.index)
        columns.append(s.fillna(0.0).astype(float))
        feature_names.append(col)

    season_s = work["Season"].astype(str) if "Season" in work.columns else pd.Series("", index=work.index)
    for cat in SEASON_CATEGORIES:
        columns.append((season_s == cat).astype(float))
        feature_names.append(f"season_{cat}")

    cluster_s = work["spatial_cluster"] if "spatial_cluster" in work.columns else pd.Series(-1, index=work.index)
    for cl in CLUSTER_CATEGORIES:
        columns.append((cluster_s == cl).astype(float))
        feature_names.append(f"cluster_{cl}")

    X_df = pd.concat(columns, axis=1)
    X_df.columns = feature_names
    return X_df.to_numpy(dtype=float), feature_names, fill_stats