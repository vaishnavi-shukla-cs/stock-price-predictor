# Stock Price Predictor

A small, dependency-light Python project that predicts the next trading day's
stock close price from historical OHLCV data.

The default model is linear regression implemented with NumPy, so the project
can run without scikit-learn. It builds practical time-series features such as
lagged closes, rolling means, daily returns, volatility, and volume changes.

> Note: This is an educational forecasting model, not financial advice. Stock
> prices are noisy and can change because of news, earnings, macro events, and
> market behavior that historical prices alone may not capture.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|   |-- generate_sample_data.py
|   `-- stock_predictor.py
`-- outputs/
```

## Quick Start

Use the bundled Codex Python runtime:

```powershell
& 'C:\Users\vaish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\generate_sample_data.py
& 'C:\Users\vaish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\stock_predictor.py --csv work\sample_stock_prices.csv
```

If Python is installed on your PATH, you can use:

```powershell
python src\generate_sample_data.py
python src\stock_predictor.py --csv work\sample_stock_prices.csv
```

## Input CSV Format

The predictor accepts a CSV with these columns:

```text
Date,Open,High,Low,Close,Volume
```

Column names are case-insensitive. The script sorts rows by `Date`, uses past
data only for features, and predicts the next row's `Close`.

Example:

```powershell
python src\stock_predictor.py --csv path\to\AAPL.csv --horizon 1 --test-size 0.2
```

## Outputs

Running the model writes:

- `outputs/metrics.json`: MAE, RMSE, MAPE, R2, rows, feature names, and next forecast.
- `outputs/predictions.csv`: date-by-date actual vs predicted values for the test period.
- `outputs/model_weights.csv`: learned linear regression coefficients.

## How It Works

1. Loads and validates historical stock data.
2. Creates lag and rolling features from past close/volume values.
3. Splits data chronologically into train and test sets.
4. Standardizes features using training data only.
5. Fits ridge-regularized linear regression via the normal equation.
6. Evaluates on the holdout test period.
7. Forecasts the next close using the latest available row.

## Optional Improvements

For a stronger production-style predictor, consider adding:

- More data: market indices, sector ETFs, rates, earnings dates, and sentiment.
- More models: random forests, gradient boosting, LSTM/Transformer models.
- Better validation: walk-forward cross-validation instead of one holdout split.
- Risk outputs: prediction intervals and confidence bands.
