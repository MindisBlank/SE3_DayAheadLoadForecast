"""Error of Svenska kraftnät's official day-ahead load forecast for SE3, on the same
hours as the backtest, side by side with the baseline and the model.

Usage (from the repo root):
    python src/compare_tso.py                                          # 2025-09-01 -> 2026-09-01
    python src/compare_tso.py --test-start 2024-09-01 --test-end 2025-09-01

Needs the backtest file for the same period (run `python src/model.py` first) and the
ENTSOE_API_TOKEN in .env. Note: the model's backtest uses observed temperature, so the
model column is optimistic; the TSO and baseline columns are real day-ahead forecasts.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from baseline import metrics
from features import to_hourly

ROOT = Path(__file__).resolve().parent.parent


def fetch_tso_forecast(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    load_dotenv()
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_TOKEN"])
    chunks, t = [], start
    while t < end:
        t_end = min(t + pd.DateOffset(years=1), end)
        print(f"Fetching TSO forecast {t:%Y-%m-%d} -> {t_end:%Y-%m-%d}")
        df = client.query_load_forecast("SE_3", start=t, end=t_end)
        chunks.append(df.iloc[:, 0])
        t = t_end
    s = pd.concat(chunks)
    s.index = s.index.tz_convert("UTC")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return to_hourly(s)  # same 15-min -> hourly rule as the actual load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2025-09-01")
    p.add_argument("--test-end", default="2026-09-01")
    a = p.parse_args()
    t0, t1 = pd.Timestamp(a.test_start, tz="UTC"), pd.Timestamp(a.test_end, tz="UTC")

    bt_file = ROOT / "backtest" / f"backtest_{t0:%Y%m%d}_{t1:%Y%m%d}.csv"
    if not bt_file.exists():
        raise SystemExit(f"{bt_file} missing - run: python src/model.py --test-start {a.test_start} --test-end {a.test_end}")
    bt = pd.read_csv(bt_file, parse_dates=["time_utc"], index_col="time_utc")

    tso = fetch_tso_forecast(t0, t1)
    bt["tso_mw"] = tso.reindex(bt.index)
    both = bt.dropna()
    print(f"\nHours compared: {len(both)} of {len(bt)} (hours where the TSO forecast exists)\n")

    y = both["actual_mw"]
    for name, col in [("Baseline (last week)", "baseline_mw"),
                      ("Svenska kraftnät (TSO)", "tso_mw"),
                      ("Model (backtest*)", "model_mw")]:
        m = metrics(y, both[col])
        print(f"{name:24s} MAE {m['MAE_MW']:6.1f} MW   MAPE {m['MAPE_%']:5.2f} %   bias {m['bias_MW']:+6.1f} MW")
    print("\n* observed temperature used as a perfect forecast, so optimistic")

    month = y.index.tz_convert(None).to_period("M")
    print("\nMAE by month (MW):")
    print(pd.DataFrame({
        "baseline": (both.baseline_mw - y).abs().groupby(month).mean(),
        "tso": (both.tso_mw - y).abs().groupby(month).mean(),
        "model": (both.model_mw - y).abs().groupby(month).mean(),
    }).round(0).to_string())


if __name__ == "__main__":
    main()
