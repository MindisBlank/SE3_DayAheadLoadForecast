"""Fetch temperature from SMHI Open Data (no token needed) and cache it to data/.

Modes:
    python src/fetch_weather.py                       # refresh observed history: Stockholm + regional stations
    python src/fetch_weather.py --forecast            # archive the Stockholm SNOW1g forecast (v1 input)
    python src/fetch_weather.py --regional-forecast   # archive the 9-city SNOW1g forecasts (v2 input)
    python src/fetch_weather.py --relookup            # re-pick the regional stations, then fetch them

Observations: metobs parameter 1 = air temperature, timestamps in UTC. Two periods are stitched:
    corrected-archive  quality-controlled, up to ~3 months ago
    latest-months      the last ~4 months, not yet fully controlled
Output: data/temp_obs_<station>.csv  (time_utc, temp_c, quality)

Regional stations: for each city in REGIONAL, the nearest active station that reports
HOURLY (at least 90 % of hours over the last year) is chosen once and saved to
data/regional_stations.csv. Later runs reuse that list, so the inputs never change
silently. Use --relookup to choose again.

Forecasts: SMHI does not keep old forecasts, so every one we use is archived.
    data/smhi_forecast/<referenceTime>.json            Stockholm, raw response (v1)
    data/smhi_forecast_regional/<fetched-UTC>.json     9 cities, temperature only (v2);
                                                       named by FETCH time, so an archived
                                                       forecast is never dated after it was used
"""
import argparse
import io
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGIONAL_META = DATA_DIR / "regional_stations.csv"
REGIONAL_FCST_DIR = DATA_DIR / "smhi_forecast_regional"

# Stockholm-Observatoriekullen A (v1 model input)
STATIONS = {"98230": "Stockholm"}
LON, LAT = 18.0549, 59.3417  # Observatoriekullen: forecast at the same spot as the observations

# SE3 population centres and rough weights (share of SE3 population, rounded).
REGIONAL = {
    "Stockholm": (59.33, 18.07, 0.38),
    "Göteborg":  (57.71, 11.97, 0.20),
    "Uppsala":   (59.86, 17.64, 0.08),
    "Linköping": (58.41, 15.62, 0.08),   # Östergötland (Linköping + Norrköping)
    "Örebro":    (59.27, 15.21, 0.06),
    "Västerås":  (59.61, 16.55, 0.06),
    "Jönköping": (57.78, 14.16, 0.06),
    "Karlstad":  (59.40, 13.50, 0.05),
    "Gävle":     (60.67, 17.14, 0.03),   # northern edge of SE3, colder
}
MIN_HOURLY_SHARE = 0.90   # a regional station must report at least this share of hours
MAX_CANDIDATES = 6        # stations tried per city, nearest first

STATION_LIST_URL = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/1.json"
OBS_URL = ("https://opendata-download-metobs.smhi.se/api/version/1.0/"
           "parameter/1/station/{station}/period/{period}/data.csv")
FCST_URL = ("https://opendata-download-metfcst.smhi.se/api/category/snow1g/version/1/"
            "geotype/point/lon/{lon}/lat/{lat}/data.json")


def get(url: str, retries: int = 3) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"  attempt {attempt} failed ({e}); retrying")
            time.sleep(5 * attempt)


# ------------------------------------------------------------------ observations
def parse_obs_csv(text: str) -> pd.DataFrame:
    """SMHI CSVs start with station metadata; the table begins at the 'Datum;Tid (UTC)' row."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Datum;Tid"))
    station_line = lines[1] if len(lines) > 1 else ""
    print(f"  station: {station_line.split(';')[0]}")
    table = "\n".join(";".join(l.split(";")[:4]) for l in lines[start + 1:] if l.strip())
    df = pd.read_csv(io.StringIO(table), sep=";", header=None,
                     names=["date", "time", "temp_c", "quality"])
    df["time_utc"] = pd.to_datetime(df["date"] + " " + df["time"], utc=True)
    return df.set_index("time_utc")[["temp_c", "quality"]]


def fetch_obs(station: str, since: str) -> pd.DataFrame:
    parts = []
    for period in ["corrected-archive", "latest-months"]:
        print(f"Fetching station {station}, {period}")
        r = get(OBS_URL.format(station=station, period=period))
        r.encoding = "utf-8"
        parts.append(parse_obs_csv(r.text))
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()   # overlap: keep the newer pull
    df = df[df.index >= pd.Timestamp(since, tz="UTC")]

    steps = df.index.to_series().diff().value_counts().head(3)
    print(f"  {len(df)} rows, {df.index.min()} -> {df.index.max()}")
    print(f"  time steps: {dict(steps)}")
    print(f"  quality flags: {df['quality'].value_counts().to_dict()}")
    print(f"  temp min/mean/max: {df.temp_c.min():.1f} / {df.temp_c.mean():.1f} / {df.temp_c.max():.1f} C")
    return df


def hourly_share(df: pd.DataFrame, days: int = 365) -> float:
    """Share of hours in the last `days` days that have a reading."""
    if df.empty:
        return 0.0
    end = df.index.max()
    recent = df[df.index > end - pd.Timedelta(days=days)]
    return recent.index.floor("h").nunique() / (days * 24)


def candidate_stations(lat: float, lon: float, since: str) -> list[dict]:
    """Active stations whose record reaches back to `since`, nearest first."""
    stations = get(STATION_LIST_URL).json()["station"]
    t_since = pd.Timestamp(since, tz="UTC").value // 10**6   # SMHI uses epoch ms
    out = []
    for st in stations:
        if not st.get("active") or st.get("from", 0) > t_since:
            continue
        dy = (st["latitude"] - lat) * 111
        dx = (st["longitude"] - lon) * 111 * math.cos(math.radians(lat))
        out.append({"id": str(st["id"]), "station": st["name"], "km": round(math.hypot(dx, dy), 1),
                    "lat": st["latitude"], "lon": st["longitude"]})
    return sorted(out, key=lambda s: s["km"])


def lookup_regional(since: str) -> pd.DataFrame:
    """Pick, per city, the nearest station that really reports hourly. Saves the list."""
    rows = []
    for city, (lat, lon, w) in REGIONAL.items():
        chosen = None
        for st in candidate_stations(lat, lon, since)[:MAX_CANDIDATES]:
            print(f"{city}: trying {st['station']} (id {st['id']}, {st['km']} km)")
            try:
                df = fetch_obs(st["id"], since)
            except Exception as e:
                print(f"  failed: {e}")
                continue
            share = hourly_share(df)
            if share < MIN_HOURLY_SHARE:
                print(f"  rejected: only {share:.0%} of hours reported over the last year\n")
                continue
            df.to_csv(DATA_DIR / f"temp_obs_{st['id']}.csv")
            chosen = {"city": city, "weight": w, **st}
            print(f"  accepted ({share:.0%} hourly)\n")
            break
        if chosen:
            rows.append(chosen)
        else:
            print(f"{city}: NO hourly station found within {MAX_CANDIDATES} candidates - left out\n")
    meta = pd.DataFrame(rows)
    meta.to_csv(REGIONAL_META, index=False)
    print(f"Saved {REGIONAL_META}  ({len(meta)} of {len(REGIONAL)} cities)")
    return meta


def refresh_regional(since: str) -> None:
    """Re-download history for the saved regional stations (daily job)."""
    meta = pd.read_csv(REGIONAL_META, dtype={"id": str})
    for _, r in meta.iterrows():
        if r["id"] in STATIONS:      # Stockholm is already refreshed as the v1 station
            continue
        print(f"{r['city']}:")
        try:
            fetch_obs(r["id"], since).to_csv(DATA_DIR / f"temp_obs_{r['id']}.csv")
        except Exception as e:      # keep the old file; the forecast step checks coverage
            print(f"  FAILED ({e}); keeping the cached file")
        print()


# ------------------------------------------------------------------ forecasts
def parse_forecast(payload: dict) -> pd.Series:
    """SNOW1g: timeSeries[i].time and timeSeries[i].data.air_temperature. 9999 = missing."""
    rows = {pd.Timestamp(e["time"]): e["data"].get("air_temperature") for e in payload["timeSeries"]}
    s = pd.Series(rows, name="temp_c", dtype=float).sort_index()
    s.index = s.index.tz_convert("UTC")
    return s.where(s != 9999)


def fetch_forecast() -> tuple[Path, dict]:
    """Stockholm SNOW1g forecast, archived raw (v1 model input)."""
    r = get(FCST_URL.format(lon=LON, lat=LAT))
    payload = r.json()
    issued = payload.get("referenceTime") or payload.get("approvedTime") or pd.Timestamp.now(tz="UTC").isoformat()
    stamp = pd.Timestamp(issued).tz_convert("UTC").strftime("%Y%m%dT%H%MZ")
    out_dir = DATA_DIR / "smhi_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    temps = parse_forecast(payload)
    print(f"Forecast issued {issued}: {len(temps)} steps, {temps.index.min()} -> {temps.index.max()}")
    print(f"Saved {out}")
    return out, payload


def _forecast_point(city: str, meta: pd.DataFrame) -> tuple[float, float]:
    """Forecast at the observing station when its coordinates are known, else at the city."""
    row = meta[meta.city == city]
    if len(row) and {"lat", "lon"} <= set(meta.columns) and pd.notna(row.iloc[0].get("lat")):
        return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
    lat, lon, _ = REGIONAL[city]
    return lat, lon


def fetch_regional_forecast(now: pd.Timestamp | None = None) -> tuple[Path | None, pd.DataFrame]:
    """Fetch SNOW1g for every regional city; archive a compact bundle (temperature only).
    Cities that fail are simply missing; the caller checks weight coverage."""
    now = now or pd.Timestamp.now(tz="UTC")
    meta = pd.read_csv(REGIONAL_META, dtype={"id": str})
    bundle, cols = {"fetched_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "cities": {}}, {}
    for city in meta.city:
        lat, lon = _forecast_point(city, meta)
        try:
            payload = get(FCST_URL.format(lon=round(lon, 4), lat=round(lat, 4))).json()
            s = parse_forecast(payload)
        except Exception as e:
            print(f"  {city}: forecast FAILED ({e})")
            continue
        cols[city] = s
        bundle["cities"][city] = {
            "referenceTime": payload.get("referenceTime"),
            "lat": lat, "lon": lon,
            "series": {t.strftime("%Y-%m-%dT%H:%MZ"): (None if pd.isna(v) else float(v)) for t, v in s.items()},
        }
        print(f"  {city}: {len(s)} steps, issued {payload.get('referenceTime')}")
    if not cols:
        return None, pd.DataFrame()
    REGIONAL_FCST_DIR.mkdir(parents=True, exist_ok=True)
    out = REGIONAL_FCST_DIR / f"{now:%Y%m%dT%H%MZ}.json"
    out.write_text(json.dumps(bundle), encoding="utf-8")
    print(f"Saved {out}  ({len(cols)} of {len(meta)} cities)")
    return out, pd.DataFrame(cols)


def load_regional_forecast(path: Path) -> pd.DataFrame:
    """Read an archived bundle back into a DataFrame (UTC index, one column per city)."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    cols = {}
    for city, c in bundle["cities"].items():
        s = pd.Series(c["series"], dtype=float)
        s.index = pd.to_datetime(s.index, utc=True)
        cols[city] = s.sort_index()
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", action="store_true", help="archive the Stockholm SNOW1g forecast")
    p.add_argument("--regional-forecast", action="store_true", help="archive the 9-city SNOW1g forecasts")
    p.add_argument("--relookup", action="store_true", help="re-pick the regional stations (hourly only)")
    p.add_argument("--regional", action="store_true", help="alias of --relookup when no station list exists yet")
    p.add_argument("--since", default="2023-01-01")
    a = p.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    if a.forecast:
        fetch_forecast()
        return
    if a.regional_forecast:
        fetch_regional_forecast()
        return
    if a.relookup or (a.regional and not REGIONAL_META.exists()):
        lookup_regional(a.since)
        return

    # Default (daily job): Stockholm history, then the saved regional stations
    for station in STATIONS:
        df = fetch_obs(station, a.since)
        out = DATA_DIR / f"temp_obs_{station}.csv"
        df.to_csv(out)
        print(f"Saved {out}\n")
    if REGIONAL_META.exists():
        refresh_regional(a.since)
    else:
        print("No regional station list yet - run: python src/fetch_weather.py --relookup")


if __name__ == "__main__":
    main()
