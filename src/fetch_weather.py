"""Fetch temperature from SMHI Open Data (no token needed) and cache it to data/.

Two modes:
    python src/fetch_weather.py                 # observed hourly temperature history
    python src/fetch_weather.py --forecast      # archive today's SNOW1g point forecast

Observations: metobs parameter 1 = air temperature, instantaneous value once per hour,
timestamps in UTC. Two periods are stitched together:
    corrected-archive  quality-controlled, up to ~3 months ago
    latest-months      the last ~4 months, not yet fully controlled
Output: data/temp_obs_<station>.csv  (time_utc, temp_c, quality)

Forecast: SMHI does not keep old forecasts, so we archive every one we use.
Output: data/smhi_forecast/<issued-UTC>.json  (raw response, untouched)
"""
import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Stockholm-Observatoriekullen A. Check the id/name printed on first run.
STATIONS = {"98230": "Stockholm"}
LON, LAT = 18.0549, 59.3417  # Observatoriekullen: forecast at the same spot as the observations

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
    # Where the periods overlap, keep the newer pull
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= pd.Timestamp(since, tz="UTC")]

    steps = df.index.to_series().diff().value_counts().head(3)
    print(f"  {len(df)} rows, {df.index.min()} -> {df.index.max()}")
    print(f"  time steps: {dict(steps)}")
    print(f"  quality flags: {df['quality'].value_counts().to_dict()}")
    print(f"  temp min/mean/max: {df.temp_c.min():.1f} / {df.temp_c.mean():.1f} / {df.temp_c.max():.1f} C")
    return df


def fetch_forecast() -> tuple[Path, dict]:
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


def parse_forecast(payload: dict) -> pd.Series:
    """SNOW1g: timeSeries[i].time and timeSeries[i].data.air_temperature. 9999 = missing."""
    rows = {pd.Timestamp(e["time"]): e["data"].get("air_temperature") for e in payload["timeSeries"]}
    s = pd.Series(rows, name="temp_c").sort_index()
    s.index = s.index.tz_convert("UTC")
    return s.where(s != 9999)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", action="store_true", help="archive the current SNOW1g forecast")
    p.add_argument("--since", default="2023-01-01")
    a = p.parse_args()

    if a.forecast:
        fetch_forecast()
        return

    DATA_DIR.mkdir(exist_ok=True)
    for station in STATIONS:
        df = fetch_obs(station, a.since)
        out = DATA_DIR / f"temp_obs_{station}.csv"
        df.to_csv(out)
        print(f"Saved {out}\n")


if __name__ == "__main__":
    main()
