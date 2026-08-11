from __future__ import annotations
import os

from config import DatasetSpec, ExperimentConfig, ForecastConfig
from data import load_mackey_glass_csv, load_specimens_xlsx, extract_series_from_dataset
from experiments import run_grid_search
from fts_model import forecast_series
from visualize import plot_predictions, plot_membership_functions, plot_rmse_heatmap
from cli import run_cli


def main():
    # === EDIT THESE PATHS to your local file locations ===
    datasets = [
        DatasetSpec(
            name="mackey_glass",
            path="mackey_glass.csv",
            kind="mackey_glass",
            target_columns=["value"],
        ),
        DatasetSpec(
            name="specimens",
            path="Specimens-Train.xlsx",
            kind="specimens",
            target_columns=["TOTAL SPECIMENS", "TOTAL A", "TOTAL B"],
        ),
    ]

    exp_cfg = ExperimentConfig(
        train_ratio=0.8,
        orders=(1, 2, 3, 4, 5),
        partitions=(5, 7, 9, 11, 13, 15),
        mf_type="triangular",
        universe_padding=0.05,
    )
    fc_cfg = ForecastConfig(fallback="global_mean")

    os.makedirs("outputs", exist_ok=True)

    for ds in datasets:
        print(f"\n=== Dataset: {ds.name} ===")
        if ds.kind == "mackey_glass":
            df = load_mackey_glass_csv(ds.path)
        else:
            df = load_specimens_xlsx(ds.path)

        series_map = extract_series_from_dataset(df, ds.kind, ds.target_columns)

        for series_name, values in series_map.items():
            print(f"\n--- Series: {series_name} ---")
            results_df, best = run_grid_search(values, exp_cfg, fc_cfg)

            if results_df.empty:
                print("No results produced (maybe series too short for chosen orders).")
                continue

            # Save results table
            out_csv = f"outputs/{ds.name}__{series_name}__grid_results.csv".replace(
                " ", "_"
            )
            results_df.to_csv(out_csv, index=False)
            print(f"Saved grid results: {out_csv}")

            # Best config
            print("Best config (by RMSE):")
            print(f"  order={best['order']} partitions={best['partitions']}")
            print(f"  metrics={best['metrics']}")

            # Evaluate best model to plot prediction curve
            model = best["model"]
            train_size = best["train_size"]
            y_true, y_pred = forecast_series(
                model, values, train_size=train_size, fallback=fc_cfg.fallback
            )

            # Visualizations
            plot_rmse_heatmap(
                results_df,
                title=f"RMSE Heatmap: {ds.name} / {series_name}",
                save_path=f"outputs/{ds.name}__{series_name}__rmse_heatmap.png".replace(
                    " ", "_"
                ),
            )
            plot_membership_functions(
                model.partition,
                save_path=f"outputs/{ds.name}__{series_name}__membership.png".replace(
                    " ", "_"
                ),
            )
            plot_predictions(
                y_true,
                y_pred,
                title=f"Predictions: {ds.name} / {series_name} (best)",
                save_path=f"outputs/{ds.name}__{series_name}__predictions.png".replace(
                    " ", "_"
                ),
            )

            # Optional: interactive CLI for the best model
            print(
                "\nDo you want to run the interactive predictor for this best model? (y/n)"
            )
            ans = input("> ").strip().lower()
            if ans == "y":
                run_cli(model)


if __name__ == "__main__":
    main()
