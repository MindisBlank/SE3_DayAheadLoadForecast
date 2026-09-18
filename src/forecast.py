"""Publish tomorrow's hourly SE3 load forecast to forecasts/<date>.csv.

Run once a day in the morning (UTC) by GitHub Actions, before the outcome is known.
"Tomorrow" is the next calendar day in Swedish local time (23, 24 or 25 hours).

Fallback chain (every run logs which path it took in forecasts/run_log.csv):
  1. model    - LightGBM with the fresh SMHI forecast
  2. model    - LightGBM with the newest ARCHIVED SMHI forecast, if it still covers
                every target hour (weather_source says "archive")
  3. baseline - same hour last week, if the weather is missing or the 48 h lag is stale
  4. skipped  - not even last week's load is available; nothing is published

A forecast file is never overwritten: it is the public record of what was predicted
before the fact. Re-running on the same day logs "exists" and does nothing.

Usage:
    python src/forecast.py
    python src/forecast.py --dry-run              # print, write nothing
    python src/forecast.py --dry-run --now 2026-09-18T08:15Z   # test as of a past moment
"""
import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from features import LOCAL_TZ, load_hourly, make_features, temp_hourly
from fetch_weather import fetch_forecast, parse_forecast
from model import FEATURES, train

ROOT = Path(__file__).resolve().parent.parent
FCST_DIR = ROOT / "forecasts"
WEATHER_ARCHIVE = ROOT / "data" / "smhi_forecast"
LOG = FCST_DIR / "run_log.csv"


def target_hours(now: pd.Timestamp) -> tuple[str, pd.DatetimeIndex]:
    """UTC hours of tomorrow's local calendar day (handles the DST 23/25 h days)."""
    tomorrow = (now.tz_convert(LOCAL_TZ).normalize() + pd.Timedelta(days=1)).date()
    start = pd.Timestamp(tomorrow, tz=LOCAL_TZ)
    end = pd.Timestamp(tomorrow + timedelta(days=1), tz=LOCAL_TZ)
    idx = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    return str(tomorrow), idx


def weather_forecast(hours: pd.DatetimeIndex, now: pd.Timestamp, live: bool):
    """Returns (hourly temperature series, source description) or (None, reason)."""
    def covers(s: pd.Series) -> bool:
        s = s.resample("1h").mean().interpolate(limit=6, limit_area="inside")
        return s.reindex(hours).notna().all()

    if live:
        try:
            _, payload = fetch_forecast()
            s = parse_forecast(payload)
            if covers(s):
                return s, f"smhi {payload.get('referenceTime')}"
            print("Fresh SMHI forecast does not cover all target hours")
        except Exception as e:  # network, HTTP error, changed format ...
            print(f"SMHI forecast fetch failed: {e}")

    # Fall back to the newest archived forecast issued before `now`
    for f in sorted(WEATHER_ARCHIVE.glob("*.json"), reverse=True):
        issued = pd.Timestamp(f.stem.replace("Z", ""), tz="UTC")
        if issued > now:
            continue
        s = parse_forecast(json.loads(f.read_text(encoding="utf-8")))
        if covers(s):
            return s, f"archive {f.stem}"
        break  # older ones won't cover it better
    return None, "no weather forecast covers the target day"


def log_run(row: dict, dry: bool) -> None:
    print("RUN:", row)
    if dry:
        return
    FCST_DIR.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--now", help="pretend it is this UTC time (only with --dry-run)")
    a = p.parse_args()
    if a.now and not a.dry_run:
        raise SystemExit("--now is only allowed with --dry-run: a live forecast must be made live.")

    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    date, hours = target_hours(now)
    out = FCST_DIR / f"{date}.csv"
    run = {"run_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "target_date": date}

    if out.exists() and not a.dry_run:
        log_run({**run, "method": "exists", "note": f"{out.name} already published"}, a.dry_run)
        return

    # Only use data that existed at `now` (matters for --now tests; a no-op live)
    load = load_hourly()
    load = load[load.index < now]
    temp_obs = temp_hourly()
    temp_obs = temp_obs[temp_obs.index < now]

    temp_fc, weather_src = weather_forecast(hours, now, live=not a.now)
    if temp_fc is not None:
        temp_fc = temp_fc.resample("1h").mean().interpolate(limit=6, limit_area="inside")
        temp = temp_obs.combine_first(temp_fc)  # observed where we have it, forecast after
    else:
        temp = temp_obs

    X = make_features(load, temp, hours)
    baseline = X["load_lag_168"]

    if baseline.isna().any():
        log_run({**run, "method": "skipped", "note": f"load data ends {load.index.max()}; "
                 "even last week's values are missing"}, a.dry_run)
        return

    if temp_fc is None:
        method, note, pred = "baseline", "no usable weather forecast", baseline
    elif X["load_lag_48"].isna().any():
        method, note, pred = "baseline", f"load data stale (ends {load.index.max()})", baseline
    else:
        train_df = make_features(load, temp_obs, load.index)
        model = train(train_df)
        pred = pd.Series(model.predict(X[FEATURES]), index=hours)
        method, note = "model", f"trained on {len(train_df.dropna(subset=FEATURES))} h"

    result = pd.DataFrame({
        "forecast_mw": pred.round(1),
        "baseline_mw": baseline.round(1),
        "temp_fc_c": X["temp_c"].round(1),
        "method": method,
        "issued_utc": run["run_utc"],
        "weather": weather_src if temp_fc is not None else "none",
    }, index=hours.rename("time_utc"))

    print(result[["forecast_mw", "baseline_mw", "temp_fc_c"]].to_string())
    print(f"\n{date}: {len(hours)} h, method={method}, weather={weather_src}")
    if not a.dry_run:
        FCST_DIR.mkdir(exist_ok=True)
        result.to_csv(out)
        print(f"Saved {out}")
    log_run({**run, "method": method, "note": f"{note}; weather: {weather_src}"}, a.dry_run)


if __name__ == "__main__":
    main()
