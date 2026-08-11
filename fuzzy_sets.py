from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional


@dataclass(frozen=True)
class FuzzySet:
    name: str
    # parameters depend on MF type
    params: Tuple[float, ...]


@dataclass
class Partition:
    universe: Tuple[float, float]
    sets: List[FuzzySet]
    mf_type: str
    intervals: List[
        Tuple[float, float]
    ]  # crisp intervals corresponding to fuzzy sets (useful for midpoints)

    def midpoint(self, idx: int) -> float:
        lo, hi = self.intervals[idx]
        return (lo + hi) / 2.0


def _triangular_mf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)


def _trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a + 1e-12)
    return (d - x) / (d - c + 1e-12)


def _gaussian_mf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-12)
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def build_partition(
    values: List[float],
    n_partitions: int,
    mf_type: str = "triangular",
    padding: float = 0.05,
) -> Partition:
    """
    Creates n_partitions fuzzy sets over an expanded universe.
    Intervals are equal-width across the universe.
    """
    if n_partitions < 3:
        raise ValueError("n_partitions must be >= 3")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span <= 0:
        span = 1.0

    umin = vmin - padding * span
    umax = vmax + padding * span

    edges = np.linspace(umin, umax, n_partitions + 1)
    intervals = [(float(edges[i]), float(edges[i + 1])) for i in range(n_partitions)]

    sets: List[FuzzySet] = []
    if mf_type == "triangular":
        # Triangles centered at midpoints with overlaps
        centers = [(lo + hi) / 2.0 for lo, hi in intervals]
        # define triangles so that each center is peak; adjacent centers define feet
        for i in range(n_partitions):
            b = centers[i]
            if i == 0:
                a = intervals[i][0]
            else:
                a = centers[i - 1]
            if i == n_partitions - 1:
                c = intervals[i][1]
            else:
                c = centers[i + 1]
            sets.append(FuzzySet(name=f"A{i+1}", params=(a, b, c)))

    elif mf_type == "trapezoidal":
        # Trapezoids with flat top around midpoint; overlap with neighbors
        centers = [(lo + hi) / 2.0 for lo, hi in intervals]
        width = (umax - umin) / n_partitions
        for i in range(n_partitions):
            mu = centers[i]
            a = mu - width
            b = mu - width / 2.0
            c = mu + width / 2.0
            d = mu + width
            sets.append(FuzzySet(name=f"A{i+1}", params=(a, b, c, d)))

    elif mf_type == "gaussian":
        centers = [(lo + hi) / 2.0 for lo, hi in intervals]
        sigma = (umax - umin) / n_partitions / 2.0
        for i in range(n_partitions):
            mu = centers[i]
            sets.append(FuzzySet(name=f"A{i+1}", params=(mu, sigma)))

    else:
        raise ValueError("mf_type must be triangular | trapezoidal | gaussian")

    return Partition(
        universe=(umin, umax), sets=sets, mf_type=mf_type, intervals=intervals
    )


def membership(part: Partition, set_idx: int, x: float) -> float:
    fs = part.sets[set_idx]
    if part.mf_type == "triangular":
        a, b, c = fs.params
        return _triangular_mf(x, a, b, c)
    if part.mf_type == "trapezoidal":
        a, b, c, d = fs.params
        return _trapezoidal_mf(x, a, b, c, d)
    if part.mf_type == "gaussian":
        mu, sigma = fs.params
        return _gaussian_mf(x, mu, sigma)
    raise ValueError("Unknown mf_type")


def fuzzify(part: Partition, x: float) -> int:
    """
    Returns the index of fuzzy set with maximum membership.
    """
    mus = [membership(part, i, x) for i in range(len(part.sets))]
    best = int(np.argmax(mus))
    return best
