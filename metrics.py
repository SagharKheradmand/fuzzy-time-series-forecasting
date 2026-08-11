import numpy as np
from typing import List, Dict


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    a = np.array(y_true, dtype=float)
    b = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    a = np.array(y_true, dtype=float)
    b = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(a - b)))


def mape(y_true: List[float], y_pred: List[float], eps: float = 1e-9) -> float:
    a = np.array(y_true, dtype=float)
    b = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((a - b) / denom)) * 100.0)


def all_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
