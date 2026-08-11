from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DatasetSpec:
    name: str
    path: str
    kind: str  # "mackey_glass" or "specimens"
    target_columns: List[str]  # which columns to forecast for this dataset


@dataclass
class ExperimentConfig:
    train_ratio: float = 0.8
    orders: Tuple[int, ...] = (1, 2, 3, 4, 5)
    partitions: Tuple[int, ...] = (5, 7, 9, 11, 13, 15)
    mf_type: str = "triangular"  # triangular | trapezoidal | gaussian
    universe_padding: float = 0.05  # expand min/max by 5%
    random_seed: int = 42


@dataclass
class ForecastConfig:
    # If unseen antecedent occurs, fallback to global mean of training data
    fallback: str = "global_mean"  # global_mean | last_value
