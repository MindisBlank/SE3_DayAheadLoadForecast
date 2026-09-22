"""Publish tomorrow's hourly SE3 load forecast to forecasts/<date>.csv.

Run every morning by GitHub Actions, before the outcome is known.
"Tomorrow" is the next calendar day in Swedish local time (23, 24 or 25 hours).

Fallback chain (every run logs which path it took in forecasts/run_log.csv):
  1. v2 model - LightGBM on population-weighted temperature over 9 SE3 cities,
                using fresh SMHI forecasts (or the newest archived bundle that still
                covers tomorrow). Needs >= 80 % of the regional weight for every hour.
  2. v1 model - LightGBM on Stockholm temperature only (the original model)
  3. baseline - same hour last week, if no usable weather forecast or the 48 h lag is stale
  4. skipped  - not even last week's load is available; nothing is published

Each forecast records `model_version` and `before_gate_closure` (issued before 12:00
Swedish time on the day before, when the day-ahead market closes).

A forecast file is never overwritten: it is the public record of what was predicted
before the fact. Re-running on the same day logs "exists" and does nothing.

Usage:
    python src/forecast.py
    python src/forecast.py --dry-run              # print, write nothing
    python src/forecast.py --dry-run --now 2026-09-22T08:15Z   # test as of a past moment
"""
import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from features import (LOCAL_TZ, MIN_COVERAGE, DATA_DIR, load_hourly, make_features,
                      regional_obs_hourly, temp_hourly, weighted_temperature)
from fetch_weather import (REGIONAL_FCST_DIR, fetch_forecast, fetch_regional_forecast,
                           load_regional_forecast, parse_forecast)
from model import FEATURES, VERSIONS, train

ROOT = Path(__file__).resolve().parent.parent
FCST_DIR = ROOT / "forecasts"
WEATHER_ARCHIVE = DATA_DIR / "smhi_forecast"
LOG = FCST_DIR / "run_log.csv"


# ------------------------------------------------------------------ time helpers
def target_hours(now: pd.Timestamp) -> tuple[str, pd.DatetimeIndex]:
    """UTC hours of tomorrow's local calendar day (handles the DST 23/25 h days)."""
    tomorrow = (now.tz_convert(LOCAL_TZ).normalize() + pd.Timedelta(days=1)).date()
    start = pd.Timestamp(tomorrow, tz=LOCAL_TZ)
    end = pd.Timestamp(tomorrow + timedelta(days=1), tz=LOCAL_TZ)
    idx = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    return str(tomorrow), idx


def gate_closure(target_date: str) -> pd.Timestamp:
    """Day-ahead market gate closure: 12:00 Swedish time (CET/CEST) on the day before."""
    d = pd.Timestamp(target_date).date() - timedelta(days=1)
    return pd.Timestamp(f"{d} 12:00", tz=LOCAL_TZ).tz_convert("UTC")


def _stamp(name: str) -> pd.Timestamp:
    return pd.Timestamp(name.replace("Z", ""), tz="UTC")


def _hourly(df):
    return df.resample("1h").mean().interpolate(limit=6, limit_area="inside")


# ------------------------------------------------------------------ weather
def regional_weather(hours: pd.DatetimeIndex, now: pd.Timestamp, live: bool):
    """(hourly DataFrame per city, source) covering `hours` with enough weight, or (None, reason).
    Also returns the freshest bundle it saw, so v1 can borrow its Stockholm column."""
    def covered(fc: pd.DataFrame) -> bool:
        if fc.empty:
            return False
        cov = weighted_temperature(_hourly(fc))[1].reindex(hours).fillna(0)
        return bool((cov >= MIN_COVERAGE).all())

    seen = None
    if live:
        try:
            _, fc = fetch_regional_forecast(now)
            seen = fc
            if covered(fc):
                return fc, f"smhi regional {len(fc.columns)} cities (fresh)", seen
            print("Fresh regional forecast does not cover enough weight for every target hour")
        except Exception as e:  # network, HTTP error, changed format ...
            print(f"Regional forecast fetch failed: {e}")

    for f in sorted(REGIONAL_FCST_DIR.glob("*.json"), reverse=True):
        if _stamp(f.stem) > now:        # never use a forecast fetched after `now`
            continue
        fc = load_regional_forecast(f)
        seen = fc if seen is None else seen
        if covered(fc):
            return fc, f"smhi regional {len(fc.columns)} cities (archive {f.stem})", seen
        break
    return None, "no regional forecast covers tomorrow", seen


def stockholm_weather(hours, now, live, regional_seen=None):
    """Stockholm-only forecast for v1: fresh, else the regional bundle's Stockholm column, else archive."""
    def covers(s):
        return _hourly(s).reindex(hours).notna().all()

    if live:
        try:
            _, payload = fetch_forecast()
            s = parse_forecast(payload)
            if covers(s):
                return s, f"smhi stockholm {payload.get('referenceTime')}"
        except Exception as e:
            print(f"Stockholm forecast fetch failed: {e}")
    if regional_seen is not None and "Stockholm" in regional_seen and covers(regional_seen["Stockholm"]):
        return regional_seen["Stockholm"], "smhi stockholm (from regional bundle)"
    for f in sorted(WEATHER_ARCHIVE.glob("*.json"), reverse=True):
        if _stamp(f.stem) > now:
            continue
        s = parse_forecast(json.loads(f.read_text(encoding="utf-8")))
        if covers(s):
            return s, f"smhi stockholm (archive {f.stem})"
        break
    return None, "no Stockholm forecast covers tomorrow"


# ------------------------------------------------------------------ run log
def log_run(row: dict, dry: bool) -> None:
    """Append one row. Rewrites the file so that new columns never corrupt older rows."""
    print("RUN:", row)
    if dry:
        return
    FCST_DIR.mkdir(exist_ok=True)
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--now", help="pretend it is this UTC time (only with --dry-run)")
    a = p.parse_args()
    if a.now and not a.dry_run:
        raise SystemExit("--now is only allowed with --dry-run: a live forecast must be made live.")

    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    live = not a.now
    date, hours = target_hours(now)
    out = FCST_DIR / f"{date}.csv"
    on_time = bool(now < gate_closure(date))
    run = {"run_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "target_date": date}

    if out.exists() and not a.dry_run:
        log_run({**run, "method": "exists", "note": f"{out.name} already published"}, a.dry_run)
        return

    # Only use data that existed at `now` (matters for --now tests; a no-op live)
    load = load_hourly()
    load = load[load.index < now]
    baseline = load.shift(168, freq="h").reindex(hours)
    if baseline.isna().any():
        log_run({**run, "method": "skipped", "model_version": "none", "before_gate_closure": on_time,
                 "note": f"load data ends {load.index.max()}; even last week's values are missing"}, a.dry_run)
        return
    lag48_ok = load.shift(48, freq="h").reindex(hours).notna().all()

    notes, pred, version, weather_src, temp_used = [], None, None, "none", None

    if not lag48_ok:
        notes.append(f"load data stale (ends {load.index.max()})")
    else:
        # ---- 1. v2: regional temperature
        regional_seen = None
        try:
            obs = regional_obs_hourly()
            obs = obs[obs.index < now]
            fc, src, regional_seen = regional_weather(hours, now, live)
            if fc is None:
                notes.append(f"v2 skipped: {src}")
            else:
                temps = obs.combine_first(_hourly(fc))          # observed where known, forecast after
                temp, _ = weighted_temperature(temps)
                X = make_features(load, temp, hours)
                if X[["temp_c", "temp_24h_mean"]].isna().any().any():
                    notes.append("v2 skipped: regional temperature has gaps in the target day")
                else:
                    train_df = make_features(load, weighted_temperature(obs)[0], load.index)
                    pred = pd.Series(train(train_df).predict(X[FEATURES]), index=hours)
                    version, weather_src, temp_used = VERSIONS["regional"], src, X["temp_c"]
                    notes.append(f"trained on {len(train_df.dropna(subset=FEATURES))} h")
        except (SystemExit, Exception) as e:   # no station list, bad file, ...: fall back, never crash
            notes.append(f"v2 skipped: {type(e).__name__}: {e}")

        # ---- 2. v1: Stockholm temperature
        if pred is None:
            s, src = stockholm_weather(hours, now, live, regional_seen)
            if s is None:
                notes.append(f"v1 skipped: {src}")
            else:
                t_obs = temp_hourly()
                t_obs = t_obs[t_obs.index < now]
                X = make_features(load, t_obs.combine_first(_hourly(s)), hours)
                if X[["temp_c", "temp_24h_mean"]].isna().any().any():
                    notes.append("v1 skipped: Stockholm temperature has gaps in the target day")
                else:
                    train_df = make_features(load, t_obs, load.index)
                    pred = pd.Series(train(train_df).predict(X[FEATURES]), index=hours)
                    version, weather_src, temp_used = VERSIONS["stockholm"], src, X["temp_c"]
                    notes.append(f"trained on {len(train_df.dropna(subset=FEATURES))} h")

    # ---- 3. baseline
    method = "model" if pred is not None else "baseline"
    if pred is None:
        pred, version = baseline, "baseline"

    result = pd.DataFrame({
        "forecast_mw": pred.round(1),
        "baseline_mw": baseline.round(1),
        "temp_fc_c": (temp_used.round(1) if temp_used is not None else float("nan")),
        "method": method,
        "model_version": version,
        "issued_utc": run["run_utc"],
        "before_gate_closure": on_time,
        "weather": weather_src,
    }, index=hours.rename("time_utc"))

    print(result[["forecast_mw", "baseline_mw", "temp_fc_c"]].to_string())
    print(f"\n{date}: {len(hours)} h, {method} {version}, weather={weather_src}, "
          f"{'before' if on_time else 'AFTER'} gate closure ({gate_closure(date):%H:%M} UTC)")
    if not a.dry_run:
        FCST_DIR.mkdir(exist_ok=True)
        result.to_csv(out)
        print(f"Saved {out}")
    log_run({**run, "method": method, "model_version": version, "before_gate_closure": on_time,
             "note": "; ".join(notes + [f"weather: {weather_src}"])}, a.dry_run)


if __name__ == "__main__":
    main()
