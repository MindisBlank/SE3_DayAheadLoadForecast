"""Turn cached raw pulls into one clean hourly feature table (index = UTC hour).

Data decisions (keep this list in sync with the README):
- All timestamps are UTC. Local Swedish time is derived ONLY for calendar features.
- ENTSO-E SE3 load is hourly until 2025-12-01 22:00 UTC and 15-minute from
  2025-12-01 23:00 UTC onward. Everything is resampled to hourly MEAN MW (= MWh
  per hour). An hour is kept only if fully covered; otherwise it becomes NaN.
- ENTSO-E load timestamps mark the START of the hour. SMHI temperature is an
  instantaneous reading AT the timestamp. Temperature at 12:00 is therefore paired
  with the load averaged over 12:00-13:00.
- Temperature gaps of up to 3 hours are linearly interpolated; longer gaps stay NaN.
- Holidays are computed in holidays_se() below (no external package).

Usage:
    python src/features.py            # build and summarise the table
    python src/features.py --check    # alignment sanity checks
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_TZ = "Europe/Stockholm"


# ---------------------------------------------------------------- load
def load_raw(pattern: str = "load_se3_*.csv") -> pd.Series:
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern} in {DATA_DIR}")
    parts = [pd.read_csv(f, parse_dates=["time_utc"], index_col="time_utc")["load_mw"] for f in files]
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="last")].sort_index()


def to_hourly(s: pd.Series) -> pd.Series:
    step = s.index.to_series().diff().shift(-1).fillna(pd.Timedelta("1h"))
    covered = (step.dt.total_seconds() / 60).clip(upper=60).resample("1h").sum()
    hourly = s.resample("1h").mean()
    hourly[covered < 60] = float("nan")
    hourly.name = "load_mw"
    return hourly


def load_hourly() -> pd.Series:
    return to_hourly(load_raw())


# ---------------------------------------------------------------- temperature
def temp_hourly(station: str = "98230") -> pd.Series:
    f = DATA_DIR / f"temp_obs_{station}.csv"
    if not f.exists():
        raise SystemExit(f"{f} missing - run: python src/fetch_weather.py")
    df = pd.read_csv(f, parse_dates=["time_utc"], index_col="time_utc")
    s = df["temp_c"].resample("1h").mean()  # also puts it on a strict hourly grid
    s = s.interpolate(limit=3, limit_area="inside")
    s.name = "temp_c"
    return s


# ---------------------------------------------------------------- calendar
def _easter(y: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(y, month, day)


def _saturday_from(start: date) -> date:
    return start + timedelta((5 - start.weekday()) % 7)


def holidays_se(years) -> tuple[dict, dict]:
    """Returns (public_holidays, de_facto_eves). Sundays are NOT included on purpose."""
    public, eves = {}, {}
    for y in years:
        e = _easter(y)
        midsummer_day = _saturday_from(date(y, 6, 20))   # Saturday 20-26 June
        all_saints = _saturday_from(date(y, 10, 31))     # Saturday 31 Oct - 6 Nov
        public.update({
            date(y, 1, 1): "Nyårsdagen", date(y, 1, 6): "Trettondedag jul",
            e - timedelta(2): "Långfredagen", e: "Påskdagen", e + timedelta(1): "Annandag påsk",
            date(y, 5, 1): "Första maj", e + timedelta(39): "Kristi himmelsfärd",
            e + timedelta(49): "Pingstdagen", date(y, 6, 6): "Nationaldagen",
            midsummer_day: "Midsommardagen", all_saints: "Alla helgons dag",
            date(y, 12, 25): "Juldagen", date(y, 12, 26): "Annandag jul",
        })
        eves.update({
            e - timedelta(1): "Påskafton", midsummer_day - timedelta(1): "Midsommarafton",
            date(y, 12, 24): "Julafton", date(y, 12, 31): "Nyårsafton",
        })
    return public, eves


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(LOCAL_TZ)
    days = pd.Index(local.date)
    public, eves = holidays_se(range(local.year.min() - 1, local.year.max() + 2))
    off = set(public) | set(eves)

    is_holiday = days.isin(list(public))
    is_eve = days.isin(list(eves))
    # Bridge day ("klämdag"): Friday after a day off, or Monday before one
    prev_off = pd.Index([d - timedelta(1) for d in days]).isin(list(off))
    next_off = pd.Index([d + timedelta(1) for d in days]).isin(list(off))
    dow = local.dayofweek
    is_bridge = ((dow == 4) & prev_off) | ((dow == 0) & next_off)

    md = local.month * 100 + local.day
    return pd.DataFrame({
        "hour": local.hour,
        "dow": dow,
        "month": local.month,
        "doy": local.dayofyear,
        "is_weekend": (dow >= 5).astype(int),
        "is_holiday": is_holiday.astype(int),
        "is_eve": is_eve.astype(int),
        "is_bridge": (is_bridge & ~is_holiday & ~is_eve).astype(int),
        "xmas_period": ((md >= 1224) | (md <= 106)).astype(int),
    }, index=index)


# ---------------------------------------------------------------- the table
def make_features(load: pd.Series, temp: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Feature table for the hours in `index`. Used by BOTH training and live forecasting,
    so the model never sees features computed two different ways.

    load: hourly actuals (may end before `index` ends; lags reach back far enough).
    temp: hourly temperature covering `index` plus the 24 h before it
          (observed history, and in live runs the SMHI forecast appended).
    """
    t = temp.sort_index()
    df = pd.DataFrame(index=index)
    df["load_mw"] = load.reindex(index)
    df["temp_c"] = t.reindex(index)
    df["temp_24h_mean"] = t.rolling(24, min_periods=18).mean().reindex(index)  # buildings respond slowly
    df["hdh"] = (17 - df["temp_c"]).clip(lower=0)                               # heating degree-hours, 17 C base
    # Lags must be known at forecast time (morning of D-1). All of D-2 is known then,
    # so 48h is the shortest safe lag for every hour of D. 24h is NOT safe.
    df["load_lag_48"] = load.shift(48, freq="h").reindex(index)
    df["load_lag_168"] = load.shift(168, freq="h").reindex(index)           # same hour last week
    return df.join(calendar_features(index))


def build_dataset() -> pd.DataFrame:
    y = load_hourly()
    return make_features(y, temp_hourly(), y.index)


def check(df: pd.DataFrame) -> None:
    """If these look wrong, something is shifted. Expected values are for Sweden."""
    print("1) Mean load by LOCAL hour - expect minimum ~03-05, peaks ~08-09 and ~17-19:")
    prof = df.groupby("hour")["load_mw"].mean()
    print(f"   min at {prof.idxmin():02d}h, max at {prof.idxmax():02d}h")
    print("2) Mean temperature by LOCAL hour - expect minimum ~04-06, maximum ~14-16:")
    tp = df.groupby("hour")["temp_c"].mean()
    print(f"   min at {tp.idxmin():02d}h, max at {tp.idxmax():02d}h")
    print("3) Correlation load vs temperature (expect strongly negative, about -0.8 or lower):")
    print(f"   {df['load_mw'].corr(df['temp_c']):.2f}  |  vs 24h mean temp: {df['load_mw'].corr(df['temp_24h_mean']):.2f}")
    print("4) Weekday holidays vs the same hour one week earlier (expect clearly negative, ~-5% or more):")
    h = df[(df.is_holiday == 1) & (df.is_weekend == 0)]
    print(f"   {(h.load_mw / h.load_lag_168 - 1).mean() * 100:+.1f} %  over {h.index.normalize().nunique()} days")
    print("5) Missing values per column:")
    na = df.isna().sum()
    print("   " + (", ".join(f"{k}={v}" for k, v in na[na > 0].items()) or "none"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    df = build_dataset()
    print(f"{df.index.min()} -> {df.index.max()}  |  {len(df)} rows  |  {df.shape[1]} columns")
    if a.check:
        check(df)
