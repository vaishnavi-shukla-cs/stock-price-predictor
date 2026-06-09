"""Train a linear regression model for next-day stock close prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


@dataclass
class StockRow:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Dataset:
    dates: list[str]
    feature_names: list[str]
    x: np.ndarray
    y: np.ndarray


@dataclass
class TrainedModel:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list[str]


def parse_float(value: str, column: str, row_number: int) -> float:
    try:
        return float(value.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid {column!r} value on CSV row {row_number}: {value!r}") from exc


def load_stock_rows(path: Path) -> list[StockRow]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        normalized = {name.lower().strip(): name for name in reader.fieldnames}
        missing = REQUIRED_COLUMNS - set(normalized)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        rows: list[StockRow] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                StockRow(
                    date=row[normalized["date"]].strip(),
                    open=parse_float(row[normalized["open"]], "Open", row_number),
                    high=parse_float(row[normalized["high"]], "High", row_number),
                    low=parse_float(row[normalized["low"]], "Low", row_number),
                    close=parse_float(row[normalized["close"]], "Close", row_number),
                    volume=parse_float(row[normalized["volume"]], "Volume", row_number),
                )
            )

    rows.sort(key=lambda item: item.date)
    if len(rows) < 80:
        raise ValueError("At least 80 rows are recommended for train/test splitting.")
    return rows


def rolling_mean(values: np.ndarray, start: int, window: int) -> float:
    return float(np.mean(values[start - window : start]))


def rolling_std(values: np.ndarray, start: int, window: int) -> float:
    return float(np.std(values[start - window : start]))


def build_dataset(rows: list[StockRow], horizon: int) -> Dataset:
    closes = np.array([row.close for row in rows], dtype=float)
    volumes = np.array([row.volume for row in rows], dtype=float)
    highs = np.array([row.high for row in rows], dtype=float)
    lows = np.array([row.low for row in rows], dtype=float)
    opens = np.array([row.open for row in rows], dtype=float)

    feature_names = [
        "close_lag_1",
        "close_lag_2",
        "close_lag_5",
        "close_lag_10",
        "return_1d",
        "return_5d",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_10",
        "volume_change_1d",
        "high_low_spread",
        "open_close_spread",
    ]

    lookback = 20
    features: list[list[float]] = []
    targets: list[float] = []
    dates: list[str] = []

    for i in range(lookback, len(rows) - horizon):
        volume_change = (volumes[i] - volumes[i - 1]) / max(volumes[i - 1], 1.0)
        feature_row = [
            closes[i],
            closes[i - 1],
            closes[i - 4],
            closes[i - 9],
            (closes[i] - closes[i - 1]) / closes[i - 1],
            (closes[i] - closes[i - 5]) / closes[i - 5],
            rolling_mean(closes, i + 1, 5),
            rolling_mean(closes, i + 1, 10),
            rolling_mean(closes, i + 1, 20),
            rolling_std(closes, i + 1, 10),
            volume_change,
            (highs[i] - lows[i]) / closes[i],
            (closes[i] - opens[i]) / opens[i],
        ]
        features.append(feature_row)
        targets.append(closes[i + horizon])
        dates.append(rows[i + horizon].date)

    return Dataset(dates=dates, feature_names=feature_names, x=np.array(features), y=np.array(targets))


def chronological_split(dataset: Dataset, test_size: float) -> tuple[Dataset, Dataset]:
    if not 0 < test_size < 0.8:
        raise ValueError("--test-size must be greater than 0 and less than 0.8")

    split_index = int(len(dataset.y) * (1 - test_size))
    if split_index < 30 or len(dataset.y) - split_index < 10:
        raise ValueError("Not enough rows after feature generation for the requested split.")

    train = Dataset(
        dates=dataset.dates[:split_index],
        feature_names=dataset.feature_names,
        x=dataset.x[:split_index],
        y=dataset.y[:split_index],
    )
    test = Dataset(
        dates=dataset.dates[split_index:],
        feature_names=dataset.feature_names,
        x=dataset.x[split_index:],
        y=dataset.y[split_index:],
    )
    return train, test


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> TrainedModel:
    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    x_scaled = (x_train - feature_mean) / feature_std
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

    penalty = np.eye(x_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(x_design.T @ x_design + penalty) @ x_design.T @ y_train
    return TrainedModel(weights=weights, feature_mean=feature_mean, feature_std=feature_std, feature_names=[])


def predict(model: TrainedModel, x: np.ndarray) -> np.ndarray:
    x_scaled = (x - model.feature_mean) / model.feature_std
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return x_design @ model.weights


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(math.sqrt(np.mean(residuals**2)))
    mape = float(np.mean(np.abs(residuals / y_true)) * 100)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "r2": float(r2)}


def write_predictions(path: Path, dates: Iterable[str], actual: np.ndarray, predicted: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Date", "ActualClose", "PredictedClose", "Error"])
        for date, actual_value, predicted_value in zip(dates, actual, predicted):
            writer.writerow([date, f"{actual_value:.4f}", f"{predicted_value:.4f}", f"{actual_value - predicted_value:.4f}"])


def write_weights(path: Path, model: TrainedModel, feature_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Feature", "Coefficient"])
        writer.writerow(["intercept", f"{model.weights[0]:.8f}"])
        for name, weight in zip(feature_names, model.weights[1:]):
            writer.writerow([name, f"{weight:.8f}"])


def latest_feature_row(rows: list[StockRow], horizon: int) -> tuple[np.ndarray, str]:
    dataset = build_dataset(rows + [rows[-1]], horizon=1)
    return dataset.x[-1:].copy(), rows[-1].date


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict stock close prices from historical OHLCV data.")
    parser.add_argument("--csv", type=Path, required=True, help="Historical stock CSV path.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in trading days.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Chronological test split fraction.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Ridge regularization strength.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory.")
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1")

    rows = load_stock_rows(args.csv)
    dataset = build_dataset(rows, args.horizon)
    train, test = chronological_split(dataset, args.test_size)
    model = fit_linear_regression(train.x, train.y, alpha=args.alpha)
    predictions = predict(model, test.x)
    metrics = regression_metrics(test.y, predictions)

    next_x, latest_date = latest_feature_row(rows, args.horizon)
    next_forecast = float(predict(model, next_x)[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out_dir / "predictions.csv", test.dates, test.y, predictions)
    write_weights(args.out_dir / "model_weights.csv", model, dataset.feature_names)

    metrics_payload = {
        **metrics,
        "rows_loaded": len(rows),
        "training_rows": len(train.y),
        "test_rows": len(test.y),
        "horizon_trading_days": args.horizon,
        "latest_input_date": latest_date,
        "next_close_forecast": next_forecast,
        "features": dataset.feature_names,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(metrics_payload, json_file, indent=2)

    print("Model trained successfully.")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape_percent']:.2f}%")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Next close forecast from {latest_date}: {next_forecast:.4f}")
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
