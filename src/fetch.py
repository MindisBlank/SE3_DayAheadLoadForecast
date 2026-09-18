"""Fetch actual total load for SE3 from ENTSO-E and cache it to data/.

Usage (from the repo root):
    python src/fetch.py                                   # smoke test: last 7 days
    python src/fetch.py --start 2023-01-01 --end 2026-09-01   # history
    python src/fetch.py --update                          # daily job: extend data/load_se3_all.csv

Output: data/load_se3_<start>_<end>.csv, timestamps in UTC, native resolution
(no resampling here; that is a deliberate, documented step later).
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

ZONE = "SE_3"  # EIC 10Y1001A1001A46L
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_load(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    load_dotenv()
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise SystemExit("ENTSOE_API_TOKEN not set (put it in .env in the repo root)")
    client = EntsoePandasClient(api_key=token)

    # ENTSO-E allows max one year per request, so pull year by year.
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(years=1), end)
        print(f"Fetching {chunk_start:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}")
        df = client.query_load(ZONE, start=chunk_start, end=chunk_end)
        chunks.append(df["Actual Load"])
        chunk_start = chunk_end

    load = pd.concat(chunks)
    load.index = load.index.tz_convert("UTC")  # UTC everywhere, convert at the edges only
    load = load[~load.index.duplicated(keep="first")].sort_index()
    load.name = "load_mw"
    load.index.name = "time_utc"
    return load


def report(load: pd.Series) -> None:
    """Print what the data looks like, so problems are seen, not assumed away."""
    steps = load.index.to_series().diff().value_counts()
    print(f"\nRows: {len(load)}  |  {load.index.min()} -> {load.index.max()}")
    print("Time steps found (count):")
    print(steps.to_string())
    print(f"Missing values: {load.isna().sum()}")
    print(f"Min / mean / max MW: {load.min():.0f} / {load.mean():.0f} / {load.max():.0f}")


def update() -> None:
    """Merge every cached load file, fetch from 3 days before the last value up to now,
    and write the result to one master file. Re-fetching a few days also picks up
    late corrections that ENTSO-E publishes after the first release."""
    from features import load_raw  # local import: features does not import fetch

    try:
        existing = load_raw()
        start = existing.index.max().floor("D") - pd.Timedelta(days=3)
    except SystemExit:
        existing = pd.Series(dtype=float, name="load_mw")
        start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp.now(tz="UTC").ceil("h")

    new = fetch_load(start, end)
    report(new)
    merged = pd.concat([existing, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.name, merged.index.name = "load_mw", "time_utc"

    out = DATA_DIR / "load_se3_all.csv"
    merged.to_csv(out)
    print(f"\nSaved {out}  ({merged.index.min()} -> {merged.index.max()})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="YYYY-MM-DD (UTC)")
    p.add_argument("--update", action="store_true", help="extend data/load_se3_all.csv up to now")
    args = p.parse_args()

    if args.update:
        update()
        return

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("D")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else end - pd.Timedelta(days=7)

    load = fetch_load(start, end)
    report(load)

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"load_se3_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    load.to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
