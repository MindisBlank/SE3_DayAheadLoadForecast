"""Sweep bias-correction filters over a cached set of walk-forward predictions.

    python src/filter_sweep.py                 # first run: predicts, caches, sweeps
    python src/filter_sweep.py --refresh       # throw the cache away and predict again
    python src/filter_sweep.py --top 25

Why this exists: a bias filter is pure post-processing, so once the day-by-day predictions
exist, trying 300 filter settings costs seconds. The expensive part - the walk-forward
predictions - is done once per period and cached in backtest/preds_<period>.csv.

Periods:
    2024, 2025   the two historical test years        -> used to CHOOSE the filter
    live2026     1 Sep 2026 to the end of the data    -> HELD OUT, used to check the choice

Choosing on the historical years and reporting the live window separately matters: with
hundreds of settings, the best number on the data you selected on is partly luck.

The filter: shift every hour of day D by a weighted average of the model's daily mean errors
up to D-2 (the last day fully scored when the forecast is made). Settings swept:
    half-life    1 ... 60 days      how fast it follows a drift
    damping      0.1 ... 1.0        fraction of the measured bias actually applied
    statistic    mean | median      median ignores a single wild day
    cap          none | 150 | 300   MW limit on the shift
    days off     apply | skip       holidays/eves/bridge days behave differently

Selection: the lowest error, full stop. "Error" is the MAE pooled over all selection hours
at once, not the average of the two years' MAEs - 2024 has 257 scored days and 2025 has 347,
and averaging two MAEs quietly weights a day in the thinner year more heavily.

Kill switch: one, and it does the only job a kill switch has - throwing out a setting whose
error SATURATES well above the baseline, judged across periods rather than on one reading.
A setting is discarded only if it is more than --max-regression per cent worse than no
filter on BOTH selection periods. Anything that is merely worse on one period, or worse on a
single bad day, stays in and is ranked on its error like everything else; a safety check that
rejects on one measurement is a tie-breaker in disguise, and it will happily throw away the
best setting because of one unlucky day.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_adaptive import (DATA_LAG_H, LOCAL_TZ, metrics, run_variant, variants, wx_features)
from features import build_dataset
from model import FEATURES, TARGET, bias_shift

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "backtest"

HALF_LIVES = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]
DAMPINGS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
STATS = ["mean", "median"]
CAPS = [None, 150, 300]
SKIP_OFF = [False, True]


def periods(df: pd.DataFrame) -> dict:
    last = df.index.max().tz_convert(LOCAL_TZ).date()
    return {
        "2024": pd.date_range("2024-09-01", "2025-09-01", freq="D", inclusive="left"),
        "2025": pd.date_range("2025-09-01", "2026-09-01", freq="D", inclusive="left"),
        "live2026": pd.date_range("2026-09-01", str(last), freq="D", inclusive="left"),
    }


def predictions(df: pd.DataFrame, name: str, days, retrain_days: int, refresh: bool) -> pd.Series:
    """Walk-forward predictions, cached on disk: the sweep itself is then free."""
    f = CACHE / f"preds_{name}.csv"
    if f.exists() and not refresh:
        s = pd.read_csv(f, index_col=0, parse_dates=[0])["pred"]
        s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
        print(f"  using cached {f.name} ({len(s)} h)")
        return s
    feats = FEATURES + wx_features(df)
    print(f"  predicting {name}: {len(days)} days, retrain every {retrain_days} d")
    s = run_variant(df, days, feats, None, False, retrain_days, name).rename("pred")
    CACHE.mkdir(exist_ok=True)
    s.to_csv(f, index_label="time_utc")
    return s


# ------------------------------------------------------------------ the filter
def shift_for(pred: pd.Series, actual: pd.Series, off_days: pd.Series,
              half_life, damping, stat, cap, skip_off) -> pd.Series:
    """Per-hour shift to subtract, built only from days scored before the forecast was made.

    "Before the forecast was made" means every day DATED D-2 or earlier, which is what the
    live job has: it forecasts D on the morning of D-1, when D-2 is the newest finished day.
    An earlier version of this took the smoothed series and shifted it two POSITIONS, which
    is the same thing only when every day in between has a prediction. It does not here -
    2024 has predictions for 257 of 365 days, because SMHI's rain series starts in 2024 - so
    position-shifting silently measured a filter that lags differently from the one that
    ships. It now reproduces model.bias_shift() exactly; src/filter_sweep.py --check proves it.

    Vectorised by noticing that the history only grows: for target day D the filter sees the
    first k(D) entries of `daily`, so one cumulative pass per (stat, half-life) serves every
    day, and each day just reads its own position."""
    err = (pred - actual).dropna()
    daily = err.groupby(err.index.tz_convert(LOCAL_TZ).date).mean()
    if stat == "mean":
        smooth = daily.ewm(halflife=half_life).mean()                     # value at i uses daily[:i+1]
    else:
        smooth = daily.rolling(int(2 * half_life) + 1, min_periods=1).median()

    # k(D) - 1 = index of the newest day dated <= D-2, for every target day
    days = pd.Index(sorted(set(pred.index.tz_convert(LOCAL_TZ).date)))
    cutoff = days - pd.Timedelta(days=2)
    pos = np.searchsorted(daily.index.to_numpy(), cutoff.to_numpy(), side="right") - 1
    val = np.where(pos >= 0, smooth.to_numpy()[pos.clip(0)], np.nan) * damping
    if cap:
        val = np.clip(val, -cap, cap)

    shift = pd.Series(val, index=days)
    day_of = pd.Index(pred.index.tz_convert(LOCAL_TZ).date)
    out = pd.Series(day_of.map(shift).to_numpy(), index=pred.index, dtype=float).fillna(0.0)
    if skip_off:
        out[off_days.reindex(pred.index).fillna(False).to_numpy()] = 0.0
    return out


def check(df: pd.DataFrame, off: pd.Series, preds: dict) -> None:
    """A filter is only worth selecting if the thing selected is the thing that ships.
    This recomputes every day's shift with model.bias_shift() - the function forecast.py
    calls - and compares it with what shift_for() produced. Any mismatch means the sweep is
    measuring a filter that does not exist. (It caught one: see shift_for's docstring.)"""
    bad = total = 0
    for stat in ("mean", "median"):
        for hl in (1, 2, 14):
            cfg = {"half_life": hl, "damping": 0.25, "stat": stat, "cap": 150, "skip_days_off": False}
            for name, pred in preds.items():
                actual = df[TARGET].reindex(pred.index)
                err = (pred - actual).dropna()
                daily = err.groupby(err.index.tz_convert(LOCAL_TZ).date).mean()
                mine = shift_for(pred, actual, off, hl, 0.25, stat, 150, False)
                first = {}
                for t, d in zip(pred.index, pred.index.tz_convert(LOCAL_TZ).date):
                    first.setdefault(d, t)
                for d, t in first.items():
                    hist = daily[daily.index <= d - pd.Timedelta(days=2)]
                    if hist.empty:
                        continue
                    total += 1
                    bad += abs(bias_shift(hist, cfg) - mine[t]) > 1e-6
    print(f"\nharness vs shipped model.bias_shift(): {bad} mismatches over {total} day-checks "
          f"-> {'OK' if not bad else 'THE SWEEP IS MEASURING THE WRONG FILTER'}")


def pooled(errs: list) -> float:
    """MAE over all the given hours at once. Averaging each period's MAE instead would weight
    a day in the thinner period more: 2024 has 257 scored days to 2025's 347."""
    e = pd.concat(errs)
    return float(e.abs().mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retrain-days", type=int, default=14)
    p.add_argument("--refresh", action="store_true", help="recompute the cached predictions")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--max-regression", type=float, default=3.0,
                   help="%% worse than no filter, on BOTH selection periods, before a setting is killed")
    p.add_argument("--check", action="store_true",
                   help="prove this harness computes the same shift as the shipped filter, then exit")
    a = p.parse_args()

    df = build_dataset("regional", weather=True)
    off = (df[["is_holiday", "is_eve", "is_bridge"]].sum(axis=1) > 0)
    print(f"Weather features: {len(wx_features(df))}\n")

    preds = {}
    for name, days in periods(df).items():
        print(f"[{name}]")
        preds[name] = predictions(df, name, days, a.retrain_days, a.refresh)

    if a.check:
        check(df, off, preds)
        return

    # reference: no filter at all
    sel = [n for n in preds if not n.startswith("live")]
    base = {k: metrics(v, df) for k, v in preds.items()}
    base_err = {k: (v - df[TARGET].reindex(v.index)).dropna() for k, v in preds.items()}
    base_sel = pooled([base_err[n] for n in sel])
    print("\nNo filter (the model as it would ship without a correction):")
    for k, m in base.items():
        print(f"  {k:9s} MAE {m['MAE']:6.1f}  bias {m['bias']:+7.1f}  winter {m['MAE winter']:6.1f}  worst {m['worst day']:6.0f}")
    print(f"  pooled over the selection periods: MAE {base_sel:.1f}")

    rows = []
    grid = list(itertools.product(HALF_LIVES, DAMPINGS, STATS, CAPS, SKIP_OFF))
    print(f"\nSweeping {len(grid)} filter settings over {len(preds)} periods ...")
    for hl, damp, stat, cap, skip in grid:
        r = {"half_life": hl, "damping": damp, "stat": stat, "cap": cap or 0, "skip_days_off": skip}
        errs = {}
        for name, pred in preds.items():
            actual = df[TARGET].reindex(pred.index)
            shifted = pred - shift_for(pred, actual, off, hl, damp, stat, cap, skip)
            errs[name] = (shifted - actual).dropna()
            m = metrics(shifted, df)
            r |= {f"{name}_MAE": m["MAE"], f"{name}_bias": m["bias"],
                  f"{name}_winter": m["MAE winter"], f"{name}_worst": m["worst day"]}
        r["sel_MAE"] = pooled([errs[n] for n in sel])
        # The only kill switch: error saturating above the baseline across BOTH selection
        # periods. Not one period, and never one day - see the module docstring.
        limit = 1 + a.max_regression / 100
        r["killed"] = (f"diverges: >{a.max_regression:g} % worse than no filter on every selection period"
                       if all(r[f"{n}_MAE"] > base[n]["MAE"] * limit for n in sel) else "")
        rows.append(r)

    res = pd.DataFrame(rows)
    out = CACHE / "filter_sweep.csv"
    res.round(2).to_csv(out, index=False)

    alive = res[res.killed == ""].sort_values("sel_MAE")
    print(f"\n{len(alive)} of {len(res)} settings survived the divergence guard "
          f"({(res.killed != '').sum()} discarded)")
    if alive.empty:
        print("Every setting diverges - do not ship a filter.")
        return

    cols = ["half_life", "damping", "stat", "cap", "skip_days_off",
            "2024_MAE", "2025_MAE", "sel_MAE", "live2026_MAE", "live2026_bias"]
    print("\nLowest pooled error on the selection periods (live2026 is held out):")
    print(alive[cols].head(a.top).round(2).to_string(index=False))
    print("\nBest pooled selection error per damping (is the best value interior, "
          "or at the edge of the grid?):")
    g = alive.groupby("damping")["sel_MAE"].min()
    for d in DAMPINGS:
        print(f"  damping {d:<5} {g.get(d, float('nan')):.2f}")

    best = alive.iloc[0]
    print("\n--- pick ---")
    print(f"half-life {best.half_life} d, damping {best.damping}, {best.stat}, "
          f"cap {int(best.cap) or 'none'}, skip days off: {bool(best.skip_days_off)}")
    for name in preds:
        bs, m = base[name], best
        print(f"  {name:9s} MAE {m[name + '_MAE']:6.1f} vs {bs['MAE']:6.1f} no filter "
              f"({(m[name + '_MAE'] / bs['MAE'] - 1) * 100:+.1f} %)   bias {m[name + '_bias']:+7.1f} vs {bs['bias']:+7.1f}"
              f"   worst day {m[name + '_worst']:5.0f} vs {bs['worst day']:5.0f}")
    print(f"  pooled selection MAE {best['sel_MAE']:.1f} vs {base_sel:.1f} no filter "
          f"({(best['sel_MAE'] / base_sel - 1) * 100:+.1f} %)")
    if best["live2026_MAE"] > base["live2026"]["MAE"]:
        print("  WARNING: the chosen setting is worse than no filter on the held-out live window.")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
