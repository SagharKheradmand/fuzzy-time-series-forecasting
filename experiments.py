from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import pandas as pd

from config import ExperimentConfig, ForecastConfig
from fuzzy_sets import build_partition
from fts_model import fit_fts, forecast_series
from metrics import all_metrics


@dataclass
class ExperimentResult:
    order: int
    partitions: int
    metrics: Dict[str, float]


def run_grid_search(
    values: List[float],
    exp_cfg: ExperimentConfig,
    fc_cfg: ForecastConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      - results_df: rows for each (order, partitions) with metrics
      - best: dict with best config and fitted model for RMSE
    """
    n = len(values)
    train_size = int(n * exp_cfg.train_ratio)
    if train_size < 10:
        raise ValueError("Train set too small; need more data.")
    train_values = values[:train_size]

    rows = []
    best_rmse = float("inf")
    best_bundle: Dict[str, Any] = {}

    for p in exp_cfg.partitions:
        part = build_partition(
            train_values,
            n_partitions=p,
            mf_type=exp_cfg.mf_type,
            padding=exp_cfg.universe_padding,
        )

        for order in exp_cfg.orders:
            if order >= train_size:
                continue

            model = fit_fts(train_values, part, order)

            y_true, y_pred = forecast_series(
                model=model,
                full_values=values,
                train_size=train_size,
                fallback=fc_cfg.fallback,
            )

            if len(y_true) == 0:
                continue

            m = all_metrics(y_true, y_pred)
            rows.append({"order": order, "partitions": p, **m})

            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                best_bundle = {
                    "order": order,
                    "partitions": p,
                    "model": model,
                    "train_size": train_size,
                    "metrics": m,
                }

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["RMSE", "MAE", "MAPE"], ascending=True)
        .reset_index(drop=True)
    )
    return results_df, best_bundle
