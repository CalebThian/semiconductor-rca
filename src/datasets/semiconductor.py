"""Loader for semiconductor wafer-level sensor CSV.

Format:
    LOT_ID, WFR_ID, START_DATETIME, Symbol Define, F1, F2, ..., F26
    each row = 1 wafer (aggregated features)

We sort by START_DATETIME to get the temporal sequence, then output
(T, N) array where T = number of wafers, N = number of features (26).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


META_COLS = ["LOT_ID", "WFR_ID", "START_DATETIME", "Symbol Define"]


@dataclass
class SemiconductorMeta:
    n_wafers: int
    n_features: int
    feature_names: list[str]
    symbol_counts: dict
    date_min: str
    date_max: str
    has_missing: bool


def load_semiconductor_csv(
    csv_path: Path,
    filter_symbol: Optional[str] = None,
) -> tuple[np.ndarray, SemiconductorMeta]:
    """Load wafer-level CSV and return (T, N) time series + metadata.

    Args:
        csv_path: path to CSV.
        filter_symbol: if "Normal" / "Abnormal", filter to that subset.
                       If None, use all wafers.

    Returns:
        data: (T, N) float32 array, sorted by START_DATETIME.
        meta: SemiconductorMeta with details.
    """
    df = pd.read_csv(csv_path)

    # Sanity check columns
    for col in META_COLS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in CSV")

    # Parse datetime and sort
    df["START_DATETIME"] = pd.to_datetime(df["START_DATETIME"], errors="coerce")
    n_bad = df["START_DATETIME"].isna().sum()
    if n_bad > 0:
        print(f"⚠ Dropped {n_bad} rows with invalid START_DATETIME")
        df = df.dropna(subset=["START_DATETIME"])

    df = df.sort_values("START_DATETIME").reset_index(drop=True)

    # Filter by Symbol Define if requested
    if filter_symbol is not None:
        if filter_symbol not in df["Symbol Define"].unique():
            raise ValueError(
                f"Symbol Define='{filter_symbol}' not found. "
                f"Available: {df['Symbol Define'].unique().tolist()}"
            )
        df = df[df["Symbol Define"] == filter_symbol].reset_index(drop=True)
        print(f"  Filtered to Symbol Define={filter_symbol}: {len(df)} wafers")

    # Get feature columns (everything except meta)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Handle missing values: forward-fill then backward-fill
    feature_data = df[feature_cols].copy()
    has_missing = feature_data.isna().any().any()
    if has_missing:
        n_missing = feature_data.isna().sum().sum()
        print(f"⚠ Found {n_missing} missing values, applying ffill+bfill")
        feature_data = feature_data.ffill().bfill()

    data = feature_data.values.astype(np.float32)

    meta = SemiconductorMeta(
        n_wafers=len(df),
        n_features=len(feature_cols),
        feature_names=feature_cols,
        symbol_counts=df["Symbol Define"].value_counts().to_dict(),
        date_min=str(df["START_DATETIME"].min()),
        date_max=str(df["START_DATETIME"].max()),
        has_missing=has_missing,
    )
    return data, meta


if __name__ == "__main__":
    # Self-test: assumes CSV is in data/raw/semiconductor/
    csv_files = list(Path("data/raw/semiconductor").glob("*.csv"))
    if not csv_files:
        print("❌ No CSV found in data/raw/semiconductor/")
        print("Copy your CSV there first, then re-run.")
        exit(1)

    csv_path = csv_files[0]
    print(f"Testing with: {csv_path}\n")

    print("=== All wafers ===")
    data, meta = load_semiconductor_csv(csv_path)
    print(f"  shape: {data.shape}")
    print(f"  features: {meta.feature_names[:3]}... (showing first 3)")
    print(f"  Symbol Define: {meta.symbol_counts}")
    print(f"  date: {meta.date_min} to {meta.date_max}")
    print(f"  missing: {meta.has_missing}")

    print(f"\n=== Normal only ===")
    try:
        data_n, _ = load_semiconductor_csv(csv_path, filter_symbol="Normal")
        print(f"  shape: {data_n.shape}")
    except ValueError as e:
        print(f"  ⚠ {e}")