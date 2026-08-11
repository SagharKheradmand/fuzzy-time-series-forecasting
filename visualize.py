import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Dict, Any

from fuzzy_sets import Partition


def plot_predictions(
    y_true: List[float],
    y_pred: List[float],
    title: str,
    save_path: Optional[str] = None,
) -> None:
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Test time index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_membership_functions(part: Partition, save_path: Optional[str] = None) -> None:
    xs = np.linspace(part.universe[0], part.universe[1], 500)
    plt.figure()

    # compute membership for each set using internal membership via fuzzy_sets.membership
    from fuzzy_sets import membership

    for i, fs in enumerate(part.sets):
        ys = [membership(part, i, float(x)) for x in xs]
        plt.plot(xs, ys, label=fs.name)

    plt.title(f"Membership Functions ({part.mf_type})")
    plt.xlabel("Universe")
    plt.ylabel("Membership")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_rmse_heatmap(
    results_df: pd.DataFrame, title: str, save_path: Optional[str] = None
) -> None:
    if results_df.empty:
        return
    pivot = results_df.pivot_table(
        index="order", columns="partitions", values="RMSE", aggfunc="min"
    )
    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xlabel("Partitions (columns)")
    plt.ylabel("Order (rows)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="RMSE")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
