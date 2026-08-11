from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from fuzzy_sets import Partition, fuzzify

Antecedent = Tuple[int, ...]  # tuple of fuzzy set indices
Consequents = List[int]  # list of fuzzy set indices


@dataclass
class FLRGModel:
    order: int
    partition: Partition
    rules: Dict[Antecedent, Consequents]
    global_mean: float

    def _defuzzify_consequents(self, cons: Consequents) -> float:
        # average of midpoints of consequent sets
        mids = [self.partition.midpoint(i) for i in cons]
        return float(np.mean(mids)) if mids else self.global_mean

    def predict_next_from_history(
        self, history_values: List[float], fallback: str = "global_mean"
    ) -> float:
        """
        history_values: last 'order' crisp values (length >= order)
        """
        if len(history_values) < self.order:
            raise ValueError(
                f"Need at least {self.order} historical values for order={self.order}"
            )

        ants = tuple(fuzzify(self.partition, v) for v in history_values[-self.order :])
        if ants in self.rules and len(self.rules[ants]) > 0:
            return self._defuzzify_consequents(self.rules[ants])

        # fallback
        if fallback == "last_value":
            return float(history_values[-1])
        return float(self.global_mean)


def build_flrg(
    values_train: List[float], partition: Partition, order: int
) -> Dict[Antecedent, Consequents]:
    """
    Builds FLRG mapping for given order:
      (F(t-order),...,F(t-1)) -> F(t)
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    fuzzy_idx = [fuzzify(partition, v) for v in values_train]

    rules: Dict[Antecedent, Consequents] = {}
    for t in range(order, len(fuzzy_idx)):
        ant = tuple(fuzzy_idx[t - order : t])
        cons = fuzzy_idx[t]
        rules.setdefault(ant, []).append(cons)

    # Optionally, remove duplicates in consequents to make "group"
    for ant, cons_list in rules.items():
        # keep order but unique
        seen = set()
        uniq = []
        for c in cons_list:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        rules[ant] = uniq

    return rules


def fit_fts(values_train: List[float], partition: Partition, order: int) -> FLRGModel:
    rules = build_flrg(values_train, partition, order)
    global_mean = float(np.mean(values_train))
    return FLRGModel(
        order=order, partition=partition, rules=rules, global_mean=global_mean
    )


def forecast_series(
    model: FLRGModel,
    full_values: List[float],
    train_size: int,
    fallback: str = "global_mean",
) -> Tuple[List[float], List[float]]:
    """
    One-step-ahead rolling forecast on the test segment.
    Uses historical *true* values (not recursive multi-step).
    Returns y_true_test, y_pred_test aligned.
    """
    order = model.order
    y_true: List[float] = []
    y_pred: List[float] = []

    # Predict for t in [train_size, len(full_values)-1] using past values
    for t in range(train_size, len(full_values)):
        hist_start = max(0, t - order)
        history = full_values[hist_start:t]
        if len(history) < order:
            # skip early points if not enough history
            continue
        pred = model.predict_next_from_history(history, fallback=fallback)
        y_true.append(float(full_values[t]))
        y_pred.append(float(pred))

    return y_true, y_pred
