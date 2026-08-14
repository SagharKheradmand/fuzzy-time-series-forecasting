# Fuzzy Time Series Forecasting from Scratch

This repository implements first-order and high-order Fuzzy Time Series (FTS) forecasting models from scratch for a Fuzzy Sets and Systems course project. It forecasts one-step-ahead values for chaotic Mackey-Glass data and influenza specimen count series, compares model orders and fuzzy partition counts, and saves evaluation tables and visual results.

The project is a machine learning / time series forecasting project focused on interpretable fuzzy-rule modeling rather than deep learning. It builds fuzzy sets, fuzzifies numeric time series values, derives Fuzzy Logical Relationship Groups (FLRGs), defuzzifies predictions, and evaluates each configuration with RMSE, MAE, and MAPE.

## Project Highlights

- Implements First-Order FTS (FOFTS) and High-Order FTS (HOFTS) without prebuilt fuzzy time-series libraries.
- Supports triangular, trapezoidal, and Gaussian membership functions in the fuzzy partitioning module.
- Runs a grid search across model orders and partition counts.
- Evaluates four forecasting targets from two datasets.
- Saves reproducible CSV result tables, RMSE heatmaps, membership-function plots, and prediction plots.
- Includes a simple interactive CLI for entering historical values and predicting the next value with the selected best model.

## Problem Statement

The project predicts the next value in a time series using historical observations represented as fuzzy linguistic states. This is useful for studying forecasting problems where interpretable fuzzy rules are preferred over black-box models.

Input:

- A univariate time series loaded from a CSV or Excel file.
- Experiment parameters such as model order, number of fuzzy partitions, membership function type, and train/test split ratio.

Output:

- One-step-ahead test predictions.
- Error metrics for each configuration.
- Best configuration selected by lowest RMSE.
- Visualizations of fuzzy sets, prediction curves, and RMSE across the parameter grid.

## Datasets

### Mackey-Glass Time Series

| Item | Description |
| --- | --- |
| Repository file | `mackey_glass.csv` |
| Dataset type | Numeric chaotic time series |
| Source in project files | Local CSV file |
| External reference | Mackey-Glass systems are widely used as chaotic forecasting benchmarks. See the original paper reference: [Mackey and Glass, 1977](https://doi.org/10.1126/science.267326). |
| Shape | 1,000 rows, 2 columns |
| Columns | `index`, `value` |
| Forecast target | `value` |
| Missing values | None found in the current file |
| Input format | CSV |
| Output format | Forecasted numeric values and metric/plot files in `outputs/` |

Preprocessing:

- `data.py` reads the CSV with `pandas.read_csv`.
- The loader validates that the `value` column exists.
- The `value` column is converted to `float` and used as the time series.
- No missing-value handling is applied because no missing values were found in the current file.

### Influenza Specimens Dataset

| Item | Description |
| --- | --- |
| Repository file | `Specimens-Train.xlsx` |
| Dataset type | Weekly influenza/specimen count table |
| Dataset source | Not specified in the current project files. |
| Excel sheet | `Speciment` |
| Shape | 238 rows, 8 columns |
| Columns | `YEAR`, `WEEK`, `TOTAL SPECIMENS`, `TOTAL A`, `TOTAL B`, `PERCENT POSITIVE`, `PERCENT A`, `PERCENT B` |
| Forecast targets | `TOTAL SPECIMENS`, `TOTAL A`, `TOTAL B` |
| Missing values | None found in the current file |
| Input format | Excel `.xlsx` |
| Output format | Forecasted numeric values and metric/plot files in `outputs/` |

Preprocessing:

- `data.py` reads the workbook with `pandas.read_excel(path, header=1)`.
- Column names are stripped of leading/trailing whitespace.
- The loader validates the required columns: `YEAR`, `WEEK`, `TOTAL SPECIMENS`, `TOTAL A`, and `TOTAL B`.
- Each selected target column is converted to a float time series.
- No normalization, scaling, encoding, or augmentation is applied in the current code.

## Methodology

The workflow is defined across `main.py`, `data.py`, `fuzzy_sets.py`, `fts_model.py`, `experiments.py`, `metrics.py`, and `visualize.py`.

1. Load each dataset.
2. Extract one or more target time series.
3. Split each series chronologically using an 80% training ratio and a 20% test segment.
4. Build an expanded universe of discourse from the training values using 5% padding.
5. Partition the universe into equal-width fuzzy intervals.
6. Create membership functions. The default experiment uses triangular membership functions.
7. Fuzzify each training value by assigning it to the fuzzy set with maximum membership.
8. Build FLRG rules:
   - Order 1: `(F(t-1)) -> F(t)`
   - Higher order: `(F(t-k), ..., F(t-1)) -> F(t)`
9. Forecast the test segment using one-step-ahead rolling prediction with true historical values.
10. Defuzzify consequent fuzzy sets by averaging their interval midpoints.
11. Use the training global mean as the default fallback for unseen antecedents.
12. Evaluate each configuration with RMSE, MAE, and MAPE.
13. Select the best configuration by lowest RMSE.
14. Save grid results and plots.

Default experiment settings from `main.py`:

| Parameter | Value |
| --- | --- |
| Train/test split | 80% / 20% |
| Model orders | 1, 2, 3, 4, 5 |
| Partition counts | 5, 7, 9, 11, 13, 15 |
| Membership function | Triangular |
| Universe padding | 0.05 |
| Fallback strategy | Training global mean |

## Results

The repository includes generated result files in `outputs/`. The best saved configurations are selected by the first row of each sorted grid-result CSV.

| Series | Best order | Best partitions | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mackey_glass / value` | 3 | 15 | 0.0538 | 0.0433 | 6.9028 |
| `specimens / TOTAL SPECIMENS` | 1 | 7 | 9281.4843 | 7586.1808 | 10.7146 |
| `specimens / TOTAL A` | 1 | 7 | 10285.1324 | 6584.2667 | 1056.4019 |
| `specimens / TOTAL B` | 1 | 11 | 405.0040 | 337.5056 | 254.6614 |

High MAPE values for some influenza targets are present in the saved outputs. This can happen when actual values are small, because percentage error becomes very large when the denominator is near zero.

## Visual Results

### Mackey-Glass Forecasting

![Mackey-Glass predictions](outputs/mackey_glass__value__predictions.png)

The prediction plot compares the true Mackey-Glass test values with the best saved FTS model predictions.

![Mackey-Glass RMSE heatmap](outputs/mackey_glass__value__rmse_heatmap.png)

The heatmap shows RMSE across model orders and partition counts for the Mackey-Glass series.

![Mackey-Glass membership functions](outputs/mackey_glass__value__membership.png)

The membership plot shows the triangular fuzzy sets used by the best Mackey-Glass model.

### Influenza Specimens Forecasting

![Total specimens predictions](outputs/specimens__TOTAL_SPECIMENS__predictions.png)

The plot compares true and predicted values for `TOTAL SPECIMENS`.

![Total A predictions](outputs/specimens__TOTAL_A__predictions.png)

The plot compares true and predicted values for `TOTAL A`.

![Total B predictions](outputs/specimens__TOTAL_B__predictions.png)

The plot compares true and predicted values for `TOTAL B`.

![Total specimens RMSE heatmap](outputs/specimens__TOTAL_SPECIMENS__rmse_heatmap.png)

The heatmap shows how RMSE changes across order and partition settings for `TOTAL SPECIMENS`.

![Total A RMSE heatmap](outputs/specimens__TOTAL_A__rmse_heatmap.png)

The heatmap shows how RMSE changes across order and partition settings for `TOTAL A`.

![Total B RMSE heatmap](outputs/specimens__TOTAL_B__rmse_heatmap.png)

The heatmap shows how RMSE changes across order and partition settings for `TOTAL B`.

Additional membership-function plots are available in `outputs/` for all influenza targets.

## Project Structure

```text
Project1/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── data.py
├── fuzzy_sets.py
├── fts_model.py
├── experiments.py
├── metrics.py
├── visualize.py
├── cli.py
├── mackey_glass.csv
├── Specimens-Train.xlsx
├── outputs/
│   ├── mackey_glass__value__grid_results.csv
│   ├── mackey_glass__value__membership.png
│   ├── mackey_glass__value__predictions.png
│   ├── mackey_glass__value__rmse_heatmap.png
│   ├── specimens__TOTAL_SPECIMENS__grid_results.csv
│   ├── specimens__TOTAL_SPECIMENS__membership.png
│   ├── specimens__TOTAL_SPECIMENS__predictions.png
│   ├── specimens__TOTAL_SPECIMENS__rmse_heatmap.png
│   ├── specimens__TOTAL_A__grid_results.csv
│   ├── specimens__TOTAL_A__membership.png
│   ├── specimens__TOTAL_A__predictions.png
│   ├── specimens__TOTAL_A__rmse_heatmap.png
│   ├── specimens__TOTAL_B__grid_results.csv
│   ├── specimens__TOTAL_B__membership.png
│   ├── specimens__TOTAL_B__predictions.png
│   └── specimens__TOTAL_B__rmse_heatmap.png
├── Fuzzy-1404-1-Prj1.pdf
├── Fuzzy-Project1-Doc.pdf
├── Fuzzy-Project1-Doc.docx
└── Mehdi_Mortazavian_40435074.zip
```

Important files:

- `main.py`: Entry point that loads datasets, runs experiments, saves results, generates plots, and optionally starts the interactive predictor.
- `config.py`: Dataclasses for dataset, experiment, and forecasting configuration.
- `data.py`: Dataset loaders and time-series extraction utilities.
- `fuzzy_sets.py`: Fuzzy set definitions, partition construction, membership functions, and fuzzification.
- `fts_model.py`: FLRG construction, model fitting, defuzzification, and rolling forecasts.
- `experiments.py`: Grid search over order and partition combinations.
- `metrics.py`: RMSE, MAE, and MAPE calculations.
- `visualize.py`: Prediction plots, membership-function plots, and RMSE heatmaps.
- `cli.py`: Interactive command-line predictor for a fitted best model.
- `outputs/`: Generated result tables and visualizations.
- `Fuzzy-1404-1-Prj1.pdf`: Original assignment description.
- `Fuzzy-Project1-Doc.pdf` and `Fuzzy-Project1-Doc.docx`: Project report files.
- `Mehdi_Mortazavian_40435074.zip`: Submission archive containing source files, documentation, datasets, and outputs.

## Installation

Python version is not specified in the current project files. The local virtual environment in this repository uses Python 3.13, and the code uses standard modern Python syntax.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies listed in `requirements.txt`:

- `numpy`
- `pandas`
- `openpyxl`
- `matplotlib`

## Usage

Run the full experiment pipeline:

```bash
python main.py
```

The script processes all configured datasets and target columns, writes outputs to `outputs/`, and prompts after each target series:

```text
Do you want to run the interactive predictor for this best model? (y/n)
```

Enter `n` to continue to the next series without opening the predictor. Enter `y` to start the interactive CLI for the current best model.

Interactive predictor input example:

```text
history> [120, 125, 128]
```

The list must contain at least as many historical values as the selected model order.

To run without entering the interactive predictor for any series:

```bash
printf "n\nn\nn\nn\n" | python main.py
```

## Generated Outputs

For each target series, the pipeline saves:

- `*_grid_results.csv`: Metrics for each order/partition configuration.
- `*_rmse_heatmap.png`: RMSE comparison across the grid.
- `*_membership.png`: Membership functions for the best model partition.
- `*_predictions.png`: True vs. predicted test values for the best model.

The output paths are generated by `main.py` and saved under `outputs/`.

## Technologies Used

- Python
- NumPy
- Pandas
- OpenPyXL
- Matplotlib
- CSV and Excel data files
- Fuzzy Time Series modeling
- Grid-search experimentation

## Future Improvements

- Add command-line arguments for dataset paths, orders, partitions, membership type, and fallback strategy.
- Save best-model metadata as JSON for easier reproducibility.
- Add automated tests for data loading, fuzzification, FLRG generation, and metric calculations.
- Move source files into a `src/` package if the project grows beyond a course-project layout.
- Add a license file before publishing if the intended license is known.

## License

Not specified in the current project files.
