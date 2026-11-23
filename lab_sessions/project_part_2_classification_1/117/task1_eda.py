from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_task1_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        xls = pd.ExcelFile(path)
        print("Excel sheet names:", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    else:
        df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values(["Date", "spatial_cluster"]).reset_index(drop=True)
    return df


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [
        script_dir.parent / "data" / "classification_task1_train.xlsx",
        script_dir.parent / "data" / "classification_task1_train.csv",
        script_dir / "data" / "classification_task1_train.xlsx",
        script_dir / "data" / "classification_task1_train.csv",
        cwd / "data" / "classification_task1_train.xlsx",
        cwd / "data" / "classification_task1_train.csv",
        cwd / "lab_sessions" / "project_part_2_classification_1" / "data" / "classification_task1_train.xlsx",
        cwd / "lab_sessions" / "project_part_2_classification_1" / "data" / "classification_task1_train.csv",
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(f"Could not locate Task 1 data in {[str(c) for c in candidates]}")

    df = load_task1_data(data_path)

    print(f"Shape: {df.shape}")
    print("\nDate range:", df["Date"].min(), "->", df["Date"].max())
    print("Rows per year:\n", df["Year"].value_counts().sort_index())

    # 1. Class balance
    if "HAB" in df.columns:
        counts = df["HAB"].value_counts(dropna=False)
        props = df["HAB"].value_counts(normalize=True, dropna=False)
        print("\nHAB class balance:")
        for k in counts.index:
            print(f"  HAB={k}: {counts[k]} ({props[k]:.2%})")

    # 2. Missingness & data types
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_n": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print("\nColumn types and missingness:")
    print(summary)

    # 3. Calendar redundancy & feature ranges/scales
    reconstructed = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]))
    print("\nExact Date vs (Year, Month, Day) match rate:", (df["Date"] == reconstructed).mean())
    print("Seasons:", df["Season"].value_counts(dropna=False).to_dict())
    print("Spatial clusters:", df["spatial_cluster"].value_counts(dropna=False).to_dict())

    numeric_cols = ["SST", "UI", "CHL", "CHL_uncertainty", "YDOY"]
    print("\nNumeric summary (count, mean, std, min, 50%, max):")
    print(df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3))

    # Check completeness by year (handout notes 2021-2023 test set has complete daily samples)
    complete_mask = df.notna().all(axis=1)
    print("\nComplete-row rate by year:")
    print((complete_mask.groupby(df["Year"]).mean() * 100).round(2))


if __name__ == "__main__":
    main()