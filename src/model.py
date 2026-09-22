"""LightGBM day-ahead load model, backtested side by side with the seasonal-naive baseline.

Usage:
    python src/model.py                        # backtest v2 (regional temperature) on the held-out year
    python src/model.py --temp stockholm       # backtest v1 (Stockholm temperature only)
    python src/model.py --test-start 2024-09-01 --test-end 2025-09-01   # a second year
    python src/model.py --final                # train on ALL data, save models/lgbm_load.txt

Forecast range (P10-P90, an 80 % range): built from OUT-OF-SAMPLE errors of the point model.
The training history is cut into 4 time blocks; for each block the model is trained on the
other 3 and its errors on the held-out block are recorded. The 10th and 90th percentiles of
those errors, per bucket (season x time of day, plus a separate bucket for days off), are
added to the point forecast. (A first attempt with LightGBM quantile models gave a range
that held only ~52 % of hours in the backtest: errors seen on training data are smaller
than real future errors, so the band came out far too narrow.) The backtest reports
coverage (target 80 %), width and pinball loss.

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
import numpy as np
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


RANGE = (0.10, 0.90)   # P10 and P90: an 80 % range
RESID_FOLDS = 4


def train(df: pd.DataFrame) -> lgb.Booster:
    """Native LightGBM API (no scikit-learn needed)."""
    d = df.dropna(subset=FEATURES + [TARGET])
    return lgb.train(PARAMS, lgb.Dataset(d[FEATURES], label=d[TARGET]), num_boost_round=NUM_ROUNDS)


def error_bucket(df: pd.DataFrame) -> pd.Series:
    """Errors differ by season, time of day and day type; the range is sized per bucket."""
    season = np.select([df["month"].isin([12, 1, 2]), df["month"].isin([6, 7, 8])],
                       ["winter", "summer"], "shoulder")
    part = np.array(["night", "morning", "afternoon", "evening"])[(df["hour"] // 6).to_numpy()]
    off = (df[["is_holiday", "is_eve", "is_bridge"]].sum(axis=1) > 0).to_numpy()
    return pd.Series(np.where(off, "day_off", np.char.add(np.char.add(season, "_"), part)), index=df.index)


def out_of_sample_errors(train_df: pd.DataFrame) -> pd.Series:
    """actual - forecast for every training hour, each predicted by a model that did not see it."""
    d = train_df.dropna(subset=FEATURES + [TARGET])
    blocks = np.array_split(np.arange(len(d)), RESID_FOLDS)       # contiguous in time
    errs = []
    for idx in blocks:
        held = d.iloc[idx]
        model = train(d.drop(held.index))
        errs.append(held[TARGET] - model.predict(held[FEATURES]))
    return pd.concat(errs)


def error_quantiles(train_df: pd.DataFrame) -> pd.DataFrame:
    """10th/90th percentile of out-of-sample error per bucket, plus an 'all' fallback row."""
    err = out_of_sample_errors(train_df)
    b = error_bucket(train_df.loc[err.index])
    q = err.groupby(b).quantile(list(RANGE)).unstack()
    q.loc["all"] = err.quantile(list(RANGE)).values
    q.columns = ["lo", "hi"]
    return q


def fit_predict_all(train_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Point forecast plus P10/P90 for the rows of X. Used by the backtest AND the live forecast."""
    point = pd.Series(train(train_df).predict(X[FEATURES]), index=X.index)
    q = error_quantiles(train_df)
    b = error_bucket(X)
    lo = b.map(q["lo"]).fillna(q.loc["all", "lo"])
    hi = b.map(q["hi"]).fillna(q.loc["all", "hi"])
    out = pd.DataFrame({"forecast_mw": point, "p10_mw": point + lo, "p90_mw": point + hi})
    # never show a band that misses its own line
    out["p10_mw"] = out[["p10_mw", "forecast_mw"]].min(axis=1)
    out["p90_mw"] = out[["p90_mw", "forecast_mw"]].max(axis=1)
    return out


def interval_metrics(y: pd.Series, lo: pd.Series, hi: pd.Series) -> dict:
    """Coverage of the 80 % range, its mean width, and the mean pinball loss of P10/P90."""
    def pinball(p, q):
        e = y - p
        return np.maximum(q * e, (q - 1) * e).mean()
    return {
        "coverage_%": ((y >= lo) & (y <= hi)).mean() * 100,
        "width_MW": (hi - lo).mean(),
        "below_p10_%": (y < lo).mean() * 100,
        "above_p90_%": (y > hi).mean() * 100,
        "pinball_MW": (pinball(lo, 0.10) + pinball(hi, 0.90)) / 2,
    }


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
    rng = fit_predict_all(train_df, test_df)            # point model is retrained too (same seed)
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

    iv = interval_metrics(y, rng["p10_mw"], rng["p90_mw"])
    print("80 % range (P10-P90):")
    print(f"  coverage {iv['coverage_%']:.1f} % (target 80)   below P10 {iv['below_p10_%']:.1f} %   "
          f"above P90 {iv['above_p90_%']:.1f} % (targets 10 / 10)")
    print(f"  mean width {iv['width_MW']:.0f} MW   pinball loss {iv['pinball_MW']:.1f} MW")
    inside = (y >= rng["p10_mw"]) & (y <= rng["p90_mw"])
    cov_month = (inside.groupby(month).mean() * 100).round(0)
    print("  coverage by month (%): " + "  ".join(f"{str(m)[-2:]}:{v:.0f}" for m, v in cov_month.items()) + "\n")

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
    pd.DataFrame({"actual_mw": y, "model_mw": pred.round(1), "baseline_mw": base,
                  "p10_mw": rng["p10_mw"].round(1), "p90_mw": rng["p90_mw"].round(1)}).to_csv(out)
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
