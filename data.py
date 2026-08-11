from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple


def load_mackey_glass_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" not in df.columns:
        raise ValueError("mackey_glass.csv must contain columns: index,value")
    return df


def load_specimens_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    # normalize columns (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"YEAR", "WEEK", "TOTAL SPECIMENS", "TOTAL A", "TOTAL B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Specimens-Train.xlsx missing required columns: {missing}")
    return df


def train_test_split_series(
    values: List[float], train_ratio: float
) -> Tuple[List[float], List[float]]:
    n = len(values)
    if n < 10:
        raise ValueError("Time series too short for train/test split.")
    split = int(n * train_ratio)
    # ensure test has at least 1 point beyond order
    if split >= n - 1:
        split = n - 2
    return values[:split], values[split:]


def extract_series_from_dataset(
    df: pd.DataFrame, kind: str, target_columns: List[str]
) -> Dict[str, List[float]]:
    """
    Returns dict: series_name -> list of floats in time order.
    """
    series = {}

    if kind == "mackey_glass":
        series["value"] = df["value"].astype(float).tolist()
        return series

    if kind == "specimens":
        for col in target_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in specimens dataset.")
            series[col] = df[col].astype(float).tolist()
        return series

    raise ValueError(f"Unknown dataset kind: {kind}")
