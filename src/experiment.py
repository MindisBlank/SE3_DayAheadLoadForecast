"""Backtest several model variants side by side, on the same hours, one change at a time.

    python src/experiment.py                   # both test years, all variants
    python src/experiment.py --years 2025      # only Sep 2025 - Aug 2026

Variants (each adds ONE idea to the one before, except where noted):
    v1        the live model today
    delta     predict load - load_lag_168 (the change vs last week), then add it back
    delta+lvl delta + recent level: mean load over the 7 days ending 48 h before
    delta+w   delta + recent data weighted more (half-life 1 year)
    regional  v1 but with population-weighted temperature over 9 SE3 cities
    best?     delta + level + regional  (only if regional data exists)

The multi-station variants need `python src/fetch_weather.py --regional` first.
All variants use observed temperature, so absolute numbers are optimistic; the point
is the COMPARISON between variants. Nothing here touches the live pipeline.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import DATA_DIR, build_dataset, regional_obs_hourly, weighted_temperature
from model import FEATURES, NUM_ROUNDS, PARAMS

TEMP_COLS = ["temp_c", "temp_24h_mean", "hdh"]


# ------------------------------------------------------------------ extra features
def regional_temperature() -> pd.Series | None:
    """Same weighted temperature the live v2 model uses (features.weighted_temperature)."""
    if not (DATA_DIR / "regional_stations.csv").exists():
        return None
    return weighted_temperature(regional_obs_hourly())[0]


def add_extras(df: pd.DataFrame, t_reg: pd.Series | None) -> pd.DataFrame:
    df = df.copy()
    load = df["load_mw"]
    # Mean of the 7 days ending 48 h before t: every value is known at forecast time
    df["load_level_7d"] = load.shift(48, freq="h").rolling("168h", min_periods=120).mean().reindex(df.index)
    if t_reg is not None:
        t = t_reg.reindex(df.index)
        df["reg_temp_c"] = t
        df["reg_temp_24h_mean"] = t.rolling(24, min_periods=18).mean()
        df["reg_hdh"] = (17 - t).clip(lower=0)
    return df


# ------------------------------------------------------------------ variants
def swap_regional(feats):
    return [f"reg_{c}" if c in TEMP_COLS else c for c in feats]


VARIANTS = {
    #  name        features                              delta target  recency weight  needs regional
    "v1":        (FEATURES,                              False,        False,          False),
    "delta":     (FEATURES,                              True,         False,          False),
    "delta+lvl": (FEATURES + ["load_level_7d"],          True,         False,          False),
    "delta+w":   (FEATURES,                              True,         True,           False),
    "regional":  (swap_regional(FEATURES),               False,        False,          True),
    "best?":     (swap_regional(FEATURES) + ["load_level_7d"], True,   False,          True),
}


def fit_predict(train_df, test_df, feats, delta, recency):
    d = train_df.dropna(subset=feats + ["load_mw", "load_lag_168"])
    y = d["load_mw"] - d["load_lag_168"] if delta else d["load_mw"]
    weight = None
    if recency:
        # .to_numpy() first: pandas returns an Index here, which LightGBM rejects as a weight
        age_days = (d.index.max() - d.index).total_seconds().to_numpy(dtype=float) / 86400
        weight = np.power(0.5, age_days / 365)   # half-life 1 year; plain float ndarray
    model = lgb.train(PARAMS, lgb.Dataset(d[feats], label=y, weight=weight), num_boost_round=NUM_ROUNDS)
    pred = model.predict(test_df[feats])
    return pd.Series(pred + (test_df["load_lag_168"].values if delta else 0), index=test_df.index)


def summarise(y, pred, test_df) -> dict:
    e = pred - y
    local = y.index.tz_convert("Europe/Stockholm")
    winter = local.month.isin([12, 1, 2])
    weekend = test_df["is_weekend"] == 1
    off = (test_df[["is_holiday", "is_eve", "is_bridge"]].sum(axis=1) > 0)
    return {
        "MAE": e.abs().mean(),
        "MAPE%": (e.abs() / y).mean() * 100,
        "bias": e.mean(),
        "MAE winter": e[winter].abs().mean(),
        "MAE days off": e[off].abs().mean(),
        "bias weekday": e[~weekend].mean(),
        "bias weekend": e[weekend].mean(),
    }


def run_year(df, start, end, variants) -> pd.DataFrame:
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    train_df = df[df.index < t0]
    # Same test hours for every variant: drop rows missing ANY variant's features
    all_feats = sorted({f for v in variants.values() for f in v[0]})
    test_df = df[(df.index >= t0) & (df.index < t1)].dropna(subset=all_feats + ["load_mw", "load_lag_168"])
    y = test_df["load_mw"]

    rows = {"baseline": summarise(y, test_df["load_lag_168"], test_df)}
    for name, (feats, delta, recency, _) in variants.items():
        print(f"  training {name} ...")
        rows[name] = summarise(y, fit_predict(train_df, test_df, feats, delta, recency), test_df)
    out = pd.DataFrame(rows).T.round(1)
    out.attrs["hours"] = len(test_df)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", default=["2024", "2025"],
                   help="test years by start: 2024 = Sep 2024-Aug 2025, 2025 = Sep 2025-Aug 2026")
    a = p.parse_args()

    t_reg = regional_temperature()
    df = add_extras(build_dataset("stockholm"), t_reg)   # v1 base; regional columns added alongside
    variants = {k: v for k, v in VARIANTS.items() if t_reg is not None or not v[3]}
    if t_reg is None:
        print("No regional data (run: python src/fetch_weather.py --regional) - skipping regional variants\n")
    else:
        stations = pd.read_csv(DATA_DIR / "regional_stations.csv")
        print("Regional temperature from: " + ", ".join(f"{c} ({w:.0%})" for c, w in zip(stations.city, stations.weight)) + "\n")

    results = []
    for yr in a.years:
        start, end = f"{yr}-09-01", f"{int(yr) + 1}-09-01"
        print(f"Test year {start} -> {end}")
        res = run_year(df, start, end, variants)
        print(f"\n{res.to_string()}\n  ({res.attrs['hours']} test hours, all errors in MW)\n")
        results.append(res.assign(test_year=yr))

    if len(results) > 1:
        both = pd.concat(results).groupby(level=0, sort=False).mean(numeric_only=True).round(1)
        print("Average over both test years:")
        print(both.to_string())

    out = Path(__file__).resolve().parent.parent / "backtest" / "experiment_results.csv"
    out.parent.mkdir(exist_ok=True)
    pd.concat(results).to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
