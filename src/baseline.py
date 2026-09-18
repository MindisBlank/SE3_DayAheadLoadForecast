"""Seasonal-naive baseline: forecast for hour t = actual load at t - 168h (same hour last week).

This is a legitimate day-ahead forecast: when the forecast for day D is made on D-1,
all of D-7 is already known. (Same hour *yesterday* is NOT fully available day-ahead,
since D-1's afternoon/evening hasn't happened yet when you bid at noon.)

Usage: python src/baseline.py [--test-start 2025-09-01] [--test-end 2026-09-01]
"""
import argparse

import pandas as pd

from features import load_hourly


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    m = actual.notna() & pred.notna()
    err = (pred[m] - actual[m])
    return {
        "hours": int(m.sum()),
        "MAE_MW": err.abs().mean(),
        "MAPE_%": (err.abs() / actual[m]).mean() * 100,
        "bias_MW": err.mean(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2025-09-01")
    p.add_argument("--test-end", default="2026-09-01")
    a = p.parse_args()

    y = load_hourly()
    pred = y.shift(168, freq="h").reindex(y.index)

    test = (y.index >= pd.Timestamp(a.test_start, tz="UTC")) & (y.index < pd.Timestamp(a.test_end, tz="UTC"))
    res = metrics(y[test], pred[test])
    print(f"Baseline (same hour last week), test {a.test_start} -> {a.test_end}")
    for k, v in res.items():
        print(f"  {k:8s} {v:,.2f}" if isinstance(v, float) else f"  {k:8s} {v}")

    # Where does it hurt? Error by month, to see holidays and cold snaps.
    err = (pred[test] - y[test]).abs()
    by_month = err.groupby(err.index.tz_localize(None).to_period("M")).mean().round(0)
    print("\nMAE by month (MW):")
    print(by_month.to_string())


if __name__ == "__main__":
    main()
