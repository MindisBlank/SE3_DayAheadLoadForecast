"""Score every published forecast against what actually happened.

Rebuilt from scratch on every run from the immutable files in forecasts/, so the
scores can always be reproduced and a crashed run never leaves a half-written table.

Outputs:
    scores/hourly.csv   one row per forecast hour with an actual value
    scores/daily.csv    one row per fully-observed target day: MAE/MAPE model vs baseline
    docs/data.json      what the GitHub Pages chart reads
Prints overall and rolling 7-day MAE.
"""
import json
from pathlib import Path

import pandas as pd

from features import load_hourly

ROOT = Path(__file__).resolve().parent.parent
FCST_DIR = ROOT / "forecasts"
SCORE_DIR = ROOT / "scores"
DOCS_DIR = ROOT / "docs"
CHART_DAYS = 14


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN -> null)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def write_site_data(fc: pd.DataFrame, daily: pd.DataFrame) -> None:
    now = pd.Timestamp.now(tz="UTC")
    dates = sorted(fc.target_date.unique())[-(CHART_DAYS + 1):]   # last 14 days + tomorrow
    hourly = fc[fc.target_date.isin(dates)].reset_index()[
        ["time_utc", "target_date", "actual_mw", "forecast_mw", "baseline_mw", "method", "model_version"]]

    summary = {"days_scored": int(len(daily)), "days_published": int(fc.target_date.nunique())}
    if len(daily):
        last7 = daily.tail(7)
        summary.update({
            "first_day": daily.date.iloc[0], "last_day": daily.date.iloc[-1],
            "mae_model_7d": round(float(last7.mae_model.mean()), 1),
            "mae_baseline_7d": round(float(last7.mae_baseline.mean()), 1),
            "mae_model_all": round(float(daily.mae_model.mean()), 1),
            "mae_baseline_all": round(float(daily.mae_baseline.mean()), 1),
            "fallback_days": int((daily.method != "model").sum()),
            "current_version": str(fc.sort_values("target_date")["model_version"].iloc[-1]),
        })
        # Error per model version, so a version change is visible in the numbers
        summary["by_version"] = {
            v: {"days": int(len(g)), "mae_model": round(float(g.mae_model.mean()), 1),
                "mae_baseline": round(float(g.mae_baseline.mean()), 1)}
            for v, g in daily.groupby("model_version")
        }

    log = FCST_DIR / "run_log.csv"
    runs = pd.read_csv(log).tail(10).iloc[::-1] if log.exists() else pd.DataFrame()

    DOCS_DIR.mkdir(exist_ok=True)
    payload = {
        "updated_utc": now.strftime("%Y-%m-%dT%H:%MZ"),
        "summary": summary,
        "hourly": _records(hourly),
        "daily": _records(daily),
        "runs": _records(runs),
    }
    (DOCS_DIR / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {DOCS_DIR / 'data.json'}")


def main() -> None:
    files = sorted(f for f in FCST_DIR.glob("*.csv") if f.name != "run_log.csv")
    if not files:
        print("No forecasts published yet.")
        return
    fc = pd.concat(
        pd.read_csv(f, parse_dates=["time_utc"], index_col="time_utc").assign(target_date=f.stem)
        for f in files
    )
    # Older files predate these columns: v1 model or baseline, timing unknown
    if "model_version" not in fc:
        fc["model_version"] = pd.NA
    fc["model_version"] = fc["model_version"].fillna(fc["method"].map({"model": "v1", "baseline": "baseline"}))
    if "before_gate_closure" not in fc:
        fc["before_gate_closure"] = pd.NA

    actual = load_hourly()
    fc["actual_mw"] = actual.reindex(fc.index)
    fc["err_model"] = fc["forecast_mw"] - fc["actual_mw"]
    fc["err_baseline"] = fc["baseline_mw"] - fc["actual_mw"]

    SCORE_DIR.mkdir(exist_ok=True)
    hourly = fc.dropna(subset=["actual_mw"])
    hourly.to_csv(SCORE_DIR / "hourly.csv")

    # A day counts only once every one of its hours has an actual value
    complete = fc.groupby("target_date")["actual_mw"].apply(lambda s: s.notna().all())
    days = [d for d, ok in complete.items() if ok]

    rows = []
    for d in days:
        x = fc[fc.target_date == d]
        rows.append({
            "date": d,
            "method": x["method"].iloc[0],
            "model_version": x["model_version"].iloc[0],
            "before_gate_closure": x["before_gate_closure"].iloc[0],
            "hours": len(x),
            "mae_model": x.err_model.abs().mean(),
            "mae_baseline": x.err_baseline.abs().mean(),
            "mape_model": (x.err_model.abs() / x.actual_mw).mean() * 100,
            "mape_baseline": (x.err_baseline.abs() / x.actual_mw).mean() * 100,
            "bias_model": x.err_model.mean(),
        })
    cols = ["date", "method", "model_version", "before_gate_closure", "hours", "mae_model", "mae_baseline",
            "mape_model", "mape_baseline", "bias_model"]
    daily = pd.DataFrame(rows, columns=cols).round(2)
    daily.to_csv(SCORE_DIR / "daily.csv", index=False)

    write_site_data(fc, daily)

    if not days:
        print(f"{len(files)} forecast(s) published; none fully observed yet.")
        return
    h = hourly[hourly.target_date.isin(days)]
    last7 = h[h.target_date.isin(days[-7:])]
    print(f"Scored {len(days)} day(s): {days[0]} -> {days[-1]}")
    print(f"  all days   MAE model {h.err_model.abs().mean():6.1f} MW | baseline {h.err_baseline.abs().mean():6.1f} MW")
    print(f"  last 7     MAE model {last7.err_model.abs().mean():6.1f} MW | baseline {last7.err_baseline.abs().mean():6.1f} MW")
    print(f"  fallbacks  {(daily.method != 'model').sum()} of {len(daily)} day(s)")
    for v, g in daily.groupby("model_version"):
        print(f"  {v:9s}  {len(g)} day(s)  MAE model {g.mae_model.mean():6.1f} | baseline {g.mae_baseline.mean():6.1f}")


if __name__ == "__main__":
    main()
