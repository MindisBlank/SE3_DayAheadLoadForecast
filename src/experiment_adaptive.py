"""Walk-forward test of candidate v3 models, including ones that adapt to their own errors.

    python src/experiment_adaptive.py                    # both test years, every variant
    python src/experiment_adaptive.py --years 2025 --retrain-days 7
    python src/experiment_adaptive.py --days 60          # quick smoke test
    python src/experiment_adaptive.py --ablation         # which weather variable earns its place

Why a walk-forward test: a bias filter uses the model's recent errors, so the test has to
reproduce the live loop - forecast a day, learn what the error was, then forecast the next.
A plain backtest would let it see errors it could not have known.

Each day D is forecast with data available on the morning of D-1 (load ends ~28 h before D
starts), the way the live job works. The model is retrained every `--retrain-days` days
(live retrains daily; weekly keeps the test to ~30 min).

With --ablation the base variants are instead v2 plus ONE weather variable at a time
(wind, cloud, humidity, rain), plus wind+cloud and the full set, to see which ones
actually carry the gain.

Base variants (each trains its own model):
    v2        the live model: level target, full history
    ratio     predicts load / recent level, multiplied back (level comes from recent data)
    short18   trained on the last 18 months only
    short24   trained on the last 24 months only
    weather   v2 + wind / cloud / humidity / rain      (needs fetch_weather.py --extra)
    wx_ratio  weather + ratio target

Bias filters, applied on top of every base variant for free (no extra training):
    none      as-is
    ewma<h>   shift by an exponentially weighted mean of past daily errors, half-life h days
    half<h>   the same shift, damped by 0.5 (hedges against chasing noise)

All variants use OBSERVED temperature, so absolute numbers are optimistic in the same way
for everyone; what matters is the comparison. Nothing here touches the live pipeline.
"""
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_dataset, extra_weather
from model import FEATURES, PARAMS, NUM_ROUNDS, TARGET, required
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
LOCAL_TZ = "Europe/Stockholm"
DATA_LAG_H = 28          # load available at forecast time ends this many hours before day D starts
HALF_LIVES = [3, 7, 14, 30]


# ------------------------------------------------------------------ variants
def wx_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("wx_")]


WX_GROUPS = {
    "wind": ["wx_wind", "wx_wind_chill", "wx_wind_24h"],
    "cloud": ["wx_cloud", "wx_cloud_24h"],
    "humidity": ["wx_humidity"],
    "precip": ["wx_precip"],
}


def ablation_variants(df: pd.DataFrame) -> dict:
    """v2 plus one weather variable at a time; same features, same everything else."""
    have = set(wx_features(df))
    groups = {k: [c for c in v if c in have] for k, v in WX_GROUPS.items()}
    groups = {k: v for k, v in groups.items() if v}
    out = {"v2": (FEATURES, None, False)}
    for name, cols in groups.items():
        out[f"+{name}"] = (FEATURES + cols, None, False)
    if "wind" in groups and "cloud" in groups:
        out["+wind+cloud"] = (FEATURES + groups["wind"] + groups["cloud"], None, False)
    out["weather (all)"] = (FEATURES + sorted(have), None, False)
    return out


def variants(df: pd.DataFrame) -> dict:
    """name -> (features, months of history or None, ratio target?)"""
    v = {
        "v2":      (FEATURES, None, False),
        "ratio":   (FEATURES, None, True),
        "short18": (FEATURES, 18, False),
        "short24": (FEATURES, 24, False),
    }
    wx = wx_features(df)
    if wx:
        v["weather"] = (FEATURES + wx, None, False)
        v["wx_ratio"] = (FEATURES + wx, None, True)
    return v


def level_series(load: pd.Series) -> pd.Series:
    """Mean load over the 7 days ending 48 h before t: known at forecast time."""
    return load.shift(48, freq="h").rolling("168h", min_periods=120).mean()


# ------------------------------------------------------------------ walk-forward
def run_variant(df: pd.DataFrame, days: pd.DatetimeIndex, feats: list[str], months, ratio: bool,
                retrain_days: int, name: str) -> pd.Series:
    """Predictions for every hour of every day in `days`, each from a model that only saw
    data available on the morning before."""
    level = level_series(df[TARGET]) if ratio else None
    preds, model, trained_for = [], None, None

    for i, day in enumerate(days):
        # calendar day, not "+24 h": local days are 23 or 25 hours long at the DST changes,
        # and adding 24 h spills into the next day (which then has duplicate hours)
        start = pd.Timestamp(day.date(), tz=LOCAL_TZ)
        end = pd.Timestamp(day.date() + timedelta(days=1), tz=LOCAL_TZ)
        hours = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
        cutoff = start.tz_convert("UTC") - pd.Timedelta(hours=DATA_LAG_H)

        if model is None or (i - trained_for) >= retrain_days:
            tr = df[df.index < cutoff]
            if months:
                tr = tr[tr.index >= cutoff - pd.DateOffset(months=months)]
            # weather columns are not required (see model.required): rain starts in 2024 and
            # demanding it would cost 40 % of the training history. LightGBM handles the gaps.
            need = [c for c in required(feats) if c != TARGET]
            if ratio:
                tr = tr.assign(_y=tr[TARGET] / level.reindex(tr.index)).dropna(subset=need + ["_y"])
                label = tr["_y"]
            else:
                tr = tr.dropna(subset=need + [TARGET])
                label = tr[TARGET]
            model = lgb.train(PARAMS, lgb.Dataset(tr[feats], label=label), num_boost_round=NUM_ROUNDS)
            trained_for = i
            print(f"  {name}: retrained at {day} on {len(tr)} h", flush=True)

        X = df.reindex(hours)
        ok = X[feats].notna().all(axis=1)
        p = pd.Series(np.nan, index=hours)
        if ok.any():
            raw = model.predict(X.loc[ok, feats])
            p.loc[ok] = raw * level.reindex(hours)[ok] if ratio else raw
        preds.append(p)
    return pd.concat(preds)


def apply_filter(pred: pd.Series, actual: pd.Series, half_life: float, damping: float) -> pd.Series:
    """Shift each day by an EWMA of PAST daily mean errors (days up to D-2 only)."""
    err = (pred - actual).dropna()
    daily = err.groupby(err.index.tz_convert(LOCAL_TZ).date).mean()
    # ewm uses days up to and including d; shift(2) leaves days <= d-2, which is what a
    # forecast made on the morning of d-1 could have scored
    shift = daily.ewm(halflife=half_life).mean().shift(2) * damping
    day_of = pd.Index(pred.index.tz_convert(LOCAL_TZ).date)
    return pred - pd.Series(day_of.map(shift).to_numpy(), index=pred.index).fillna(0.0)


# ------------------------------------------------------------------ metrics
def metrics(pred: pd.Series, df: pd.DataFrame) -> dict:
    pred = pred[~pred.index.duplicated(keep="first")].sort_index()
    y = df[TARGET].reindex(pred.index)
    e = (pred - y).dropna()
    y = y.reindex(e.index)
    local = e.index.tz_convert(LOCAL_TZ)
    off = (df.reindex(e.index)[["is_holiday", "is_eve", "is_bridge"]].sum(axis=1) > 0)
    daily_mae = e.abs().groupby(local.date).mean()
    return {
        "MAE": e.abs().mean(),
        "MAPE%": (e.abs() / y).mean() * 100,
        "bias": e.mean(),
        "MAE winter": e[local.month.isin([12, 1, 2])].abs().mean(),
        "MAE days off": e[off.to_numpy()].abs().mean(),
        "worst day": daily_mae.max(),
        "days": len(daily_mae),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", default=["2024", "2025"])
    p.add_argument("--retrain-days", type=int, default=7)
    p.add_argument("--days", type=int, help="only the first N days of each year (smoke test)")
    p.add_argument("--ablation", action="store_true", help="test one weather variable at a time")
    a = p.parse_args()

    df = build_dataset("regional", weather=True)
    wx = wx_features(df)
    print(f"Extra weather: {', '.join(wx) if wx else 'none (run: python src/fetch_weather.py --extra)'}\n")

    rows = []
    for yr in a.years:
        days = pd.date_range(f"{yr}-09-01", f"{int(yr) + 1}-09-01", freq="D", inclusive="left")
        if a.days:
            days = days[:a.days]
        print(f"=== test year {days[0].date()} -> {days[-1].date()} ({len(days)} days)")
        chosen = ablation_variants(df) if a.ablation else variants(df)
        for name, (feats, months, ratio) in chosen.items():
            pred = run_variant(df, days, feats, months, ratio, a.retrain_days, name)
            actual = df[TARGET].reindex(pred.index)
            for fname, hl, damp in ([("none", None, 1.0)] +
                                    [(f"ewma{h}", h, 1.0) for h in HALF_LIVES] +
                                    [(f"half{h}", h, 0.5) for h in HALF_LIVES]):
                p2 = pred if hl is None else apply_filter(pred, actual, hl, damp)
                rows.append({"test_year": yr, "variant": name, "filter": fname, **metrics(p2, df)})
        # the naive baseline over the same hours, for scale
        hours = pd.date_range(pd.Timestamp(days[0], tz=LOCAL_TZ),
                              pd.Timestamp(days[-1], tz=LOCAL_TZ) + pd.Timedelta(days=1),
                              freq="h", inclusive="left").tz_convert("UTC")
        rows.append({"test_year": yr, "variant": "baseline", "filter": "none",
                     **metrics(df["load_lag_168"].reindex(hours), df)})

    res = pd.DataFrame(rows).round(1)
    out = ROOT / "backtest" / ("ablation_results.csv" if a.ablation else "adaptive_results.csv")
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)

    print("\n================ MAE by variant and filter (MW) ================")
    for yr in a.years:
        print(f"\ntest year {yr}:")
        piv = res[res.test_year == yr].pivot(index="variant", columns="filter", values="MAE")
        order = ["none"] + [f"ewma{h}" for h in HALF_LIVES] + [f"half{h}" for h in HALF_LIVES]
        print(piv[[c for c in order if c in piv]].to_string())
    print("\n================ best 12 rows, averaged over the test years ================")
    avg = (res[res.variant != "baseline"].groupby(["variant", "filter"])
           .mean(numeric_only=True).round(1).sort_values("MAE"))
    print(avg.head(12).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
