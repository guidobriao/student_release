from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_BASE_COLS = ["SST", "UI", "YDOY"]
CHL_COLS = ["CHL", "CHL_uncertainty"]
SEASON_CATEGORIES = ["Winter", "Spring", "Summer", "Fall"]
CLUSTER_CATEGORIES = [0, 1, 2, 3, 4, 5]


def parse_input_dataframe(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load an .xlsx file (or accept a DataFrame) and sort explicitly by Date."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        sort_cols = ["Date", "spatial_cluster"] if "spatial_cluster" in df.columns else ["Date"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def build_explicit_features(
    df: pd.DataFrame,
    chl_fill_values: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Explicitly construct numeric feature matrix X using Pandas and NumPy only."""
    work = df.copy()
    if "Date" in work.columns:
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        sort_cols = ["Date", "spatial_cluster"] if "spatial_cluster" in work.columns else ["Date"]
        work = work.sort_values(sort_cols).reset_index(drop=True)

    if chl_fill_values is None:
        chl_fill_values = {}
        for col in CHL_COLS:
            med = float(work[col].median()) if col in work.columns and work[col].notna().any() else 0.0
            chl_fill_values[col] = med

    columns: list[pd.Series] = []
    feature_names: list[str] = []

    # Missingness indicators + explicit median fill for CHL and CHL_uncertainty
    for col in CHL_COLS:
        raw_series = work[col] if col in work.columns else pd.Series(np.nan, index=work.index, dtype=float)
        is_missing = raw_series.isna().astype(float)
        filled = raw_series.fillna(chl_fill_values[col]).astype(float)
        columns.extend([is_missing, filled])
        feature_names.extend([f"{col}_is_missing", f"{col}_filled"])

    # Continuous / cyclic numeric features
    for col in NUMERIC_BASE_COLS:
        s = work[col].astype(float) if col in work.columns else pd.Series(0.0, index=work.index, dtype=float)
        s = s.fillna(0.0)
        columns.append(s)
        feature_names.append(col)

    # Explicit one-hot encoding for Season and spatial_cluster
    season_series = work["Season"].astype(str) if "Season" in work.columns else pd.Series("", index=work.index)
    for cat in SEASON_CATEGORIES:
        columns.append((season_series == cat).astype(float))
        feature_names.append(f"season_{cat}")

    cluster_series = work["spatial_cluster"] if "spatial_cluster" in work.columns else pd.Series(-1, index=work.index)
    for cl in CLUSTER_CATEGORIES:
        columns.append((cluster_series == cl).astype(float))
        feature_names.append(f"cluster_{cl}")

    X_df = pd.concat(columns, axis=1)
    X_df.columns = feature_names
    X = X_df.to_numpy(dtype=float)
    return X, feature_names, chl_fill_values