"""Score every published forecast against what actually happened.

Rebuilt from scratch on every run from the immutable files in forecasts/, so the
scores can always be reproduced and a crashed run never leaves a half-written table.

Outputs:
    scores/hourly.csv   one row per forecast hour with an actual value
    scores/daily.csv    one row per fully-observed target day: MAE/MAPE model vs baseline
Prints overall and rolling 7-day MAE.
"""
from pathlib import Path

import pandas as pd

from features import load_hourly

ROOT = Path(__file__).resolve().parent.parent
FCST_DIR = ROOT / "forecasts"
SCORE_DIR = ROOT / "scores"


def main() -> None:
    files = sorted(f for f in FCST_DIR.glob("*.csv") if f.name != "run_log.csv")
    if not files:
        print("No forecasts published yet.")
        return
    fc = pd.concat(
        pd.read_csv(f, parse_dates=["time_utc"], index_col="time_utc").assign(target_date=f.stem)
        for f in files
    )
    actual = load_hourly()
    fc["actual_mw"] = actual.reindex(fc.index)
    fc["err_model"] = fc["forecast_mw"] - fc["actual_mw"]
    fc["err_baseline"] = fc["baseline_mw"] - fc["actual_mw"]

    SCORE_DIR.mkdir(exist_ok=True)
    hourly = fc.dropna(subset=["actual_mw"])
    hourly.to_csv(SCORE_DIR / "hourly.csv")

    # A day counts only once every one of its hours has an actual value
    g = fc.groupby("target_date")
    complete = g["actual_mw"].apply(lambda s: s.notna().all())
    days = [d for d, ok in complete.items() if ok]
    if not days:
        print(f"{len(files)} forecast(s) published; none fully observed yet.")
        return

    rows = []
    for d in days:
        x = fc[fc.target_date == d]
        rows.append({
            "date": d,
            "method": x["method"].iloc[0],
            "hours": len(x),
            "mae_model": x.err_model.abs().mean(),
            "mae_baseline": x.err_baseline.abs().mean(),
            "mape_model": (x.err_model.abs() / x.actual_mw).mean() * 100,
            "mape_baseline": (x.err_baseline.abs() / x.actual_mw).mean() * 100,
            "bias_model": x.err_model.mean(),
        })
    daily = pd.DataFrame(rows).round(2)
    daily.to_csv(SCORE_DIR / "daily.csv", index=False)

    h = hourly[hourly.target_date.isin(days)]
    last7 = h[h.target_date.isin(days[-7:])]
    print(f"Scored {len(days)} day(s): {days[0]} -> {days[-1]}")
    print(f"  all days   MAE model {h.err_model.abs().mean():6.1f} MW | baseline {h.err_baseline.abs().mean():6.1f} MW")
    print(f"  last 7     MAE model {last7.err_model.abs().mean():6.1f} MW | baseline {last7.err_baseline.abs().mean():6.1f} MW")
    print(f"  fallbacks  {(daily.method != 'model').sum()} of {len(daily)} day(s)")


if __name__ == "__main__":
    main()
