"""Generate synthetic stock-like OHLCV data for demos and smoke tests."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
import random


def trading_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_rows(rows: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)
    price = 125.0
    volume = 2_500_000
    output: list[dict[str, str]] = []

    for i, day in enumerate(trading_days(date(2022, 1, 3), rows)):
        trend = 0.00045
        seasonal = 0.004 * ((i % 21) - 10) / 10
        shock = random.gauss(0, 0.018)
        return_pct = trend + seasonal + shock

        open_price = price * (1 + random.gauss(0, 0.004))
        close_price = max(1.0, price * (1 + return_pct))
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.006)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.006)))
        volume = max(100_000, int(volume * (1 + random.gauss(0, 0.08))))

        output.append(
            {
                "Date": day.isoformat(),
                "Open": f"{open_price:.2f}",
                "High": f"{high_price:.2f}",
                "Low": f"{low_price:.2f}",
                "Close": f"{close_price:.2f}",
                "Volume": str(volume),
            }
        )
        price = close_price

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample OHLCV stock data.")
    parser.add_argument("--rows", type=int, default=520, help="Number of trading rows.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("work") / "sample_stock_prices.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(args.rows, args.seed)
    with args.out.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
