"""LightGBM day-ahead load model, backtested side by side with the seasonal-naive baseline.

Usage:
    python src/model.py                        # backtest v2 (regional temperature) on the held-out year
    python src/model.py --temp stockholm       # backtest v1 (Stockholm temperature only)
    python src/model.py --test-start 2024-09-01 --test-end 2025-09-01   # a second year
    python src/model.py --final                # train on ALL data, save models/lgbm_load.txt

Honesty notes (go into the README):
- Train strictly before test-start; test is never seen during training.
- The backtest uses OBSERVED temperature as if it were a perfect forecast, so it is
  optimistic. Live runs use SMHI's forecast; the gap between the two is measurable
  once the daily forecast archive has built up.
- Backtest predictions are written to backtest/, never to forecasts/. forecasts/ is
  only for predictions made before the outcome was known.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from baseline import metrics
from features import build_dataset

ROOT = Path(__file__).resolve().parent.parent
TARGET = "load_mw"
# Model versions: same features and settings, different temperature input.
# Every published forecast records which one produced it.
VERSIONS = {"stockholm": "v1", "regional": "v2"}
FEATURES = [
    "temp_c", "temp_24h_mean", "hdh",
    "load_lag_48", "load_lag_168",
    "hour", "dow", "month", "doy",
    "is_weekend", "is_holiday", "is_eve", "is_bridge", "xmas_period",
]
NUM_ROUNDS = 800
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "seed": 42,
    "verbosity": -1,
}


def train(df: pd.DataFrame) -> lgb.Booster:
    """Native LightGBM API (no scikit-learn needed)."""
    d = df.dropna(subset=FEATURES + [TARGET])
    return lgb.train(PARAMS, lgb.Dataset(d[FEATURES], label=d[TARGET]), num_boost_round=NUM_ROUNDS)


def fmt(res: dict) -> str:
    return f"MAE {res['MAE_MW']:7.1f} MW   MAPE {res['MAPE_%']:5.2f} %   bias {res['bias_MW']:+6.1f} MW"


def backtest(df: pd.DataFrame, start: str, end: str, suffix: str = "") -> None:
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    train_df = df[df.index < t0]
    test_df = df[(df.index >= t0) & (df.index < t1)].dropna(subset=FEATURES + [TARGET])

    print(f"Train: {train_df.index.min():%Y-%m-%d} -> {t0:%Y-%m-%d}  ({len(train_df.dropna(subset=FEATURES + [TARGET]))} h)")
    print(f"Test:  {start} -> {end}  ({len(test_df)} h)\n")

    model = train(train_df)
    pred = pd.Series(model.predict(test_df[FEATURES]), index=test_df.index)
    base = test_df["load_lag_168"]
    y = test_df[TARGET]

    m_model, m_base = metrics(y, pred), metrics(y, base)
    print(f"Baseline (same hour last week): {fmt(m_base)}")
    print(f"LightGBM:                       {fmt(m_model)}")
    print(f"Improvement in MAE:             {(1 - m_model['MAE_MW'] / m_base['MAE_MW']) * 100:.1f} %\n")

    month = y.index.tz_convert(None).to_period("M")
    by_month = pd.DataFrame({
        "baseline": (base - y).abs().groupby(month).mean(),
        "model": (pred - y).abs().groupby(month).mean(),
    }).round(0)
    print("MAE by month (MW):")
    print(by_month.to_string(), "\n")

    days_off = (test_df.is_holiday == 1) | (test_df.is_eve == 1) | (test_df.is_bridge == 1)
    print("MAE on holidays/eves/bridge days vs normal days (MW):")
    for name, mask in [("days off", days_off), ("normal", ~days_off)]:
        print(f"  {name:9s} baseline {(base - y)[mask].abs().mean():6.0f}   model {(pred - y)[mask].abs().mean():6.0f}")

    imp = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES)
    print("\nFeature importance (share of gain):")
    print((imp / imp.sum() * 100).sort_values(ascending=False).round(1).to_string())

    out_dir = ROOT / "backtest"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"backtest_{t0:%Y%m%d}_{t1:%Y%m%d}{suffix}.csv"
    pd.DataFrame({"actual_mw": y, "model_mw": pred.round(1), "baseline_mw": base}).to_csv(out)
    print(f"\nSaved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-start", default="2025-09-01")
    p.add_argument("--test-end", default="2026-09-01")
    p.add_argument("--final", action="store_true", help="train on all data and save the model")
    p.add_argument("--temp", choices=list(VERSIONS), default="regional",
                   help="temperature input: regional (v2, live) or stockholm (v1)")
    a = p.parse_args()

    print(f"Model {VERSIONS[a.temp]} ({a.temp} temperature)")
    df = build_dataset(a.temp)
    if a.final:
        model = train(df)
        out = ROOT / "models" / "lgbm_load.txt"
        out.parent.mkdir(exist_ok=True)
        model.save_model(str(out))
        print(f"Trained on {df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d}; saved {out}")
    else:
        # v2 keeps the plain file name (compare_tso.py reads it); v1 gets a suffix
        backtest(df, a.test_start, a.test_end, suffix="" if a.temp == "regional" else "_v1")


if __name__ == "__main__":
    main()
