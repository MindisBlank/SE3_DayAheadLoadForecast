# SE3 Day-Ahead Load Forecast

A day-ahead hourly electricity load forecast for bidding zone SE3 that runs unattended every day, publishes its prediction before the fact, and records its own error against what actually happened.

**Live page:** https://mindisblank.github.io/SE3_DayAheadLoadForecast/

Every morning at 08:15 UTC a GitHub Actions job pulls the latest load and weather data, publishes tomorrow's 24 hourly values to [`forecasts/`](forecasts/), scores every earlier forecast against the actual load in [`scores/`](scores/), and commits the result. The commit timestamp on each forecast file is the proof that it was made before the outcome was known.

## Results

### Live (the honest number)

Running since 18 September 2026. Rolling error is in [`scores/daily.csv`](scores/daily.csv) and on the live page.

| Period | Days | Model MAE | Baseline MAE | Fallback days |
|---|---|---|---|---|
| _to be filled in after the first 7 scored days_ | | | | |

### Backtest

Trained on Jan 2023 – Aug 2025, tested on the held-out year Sep 2025 – Aug 2026 (8,760 hours). A second year is shown as a check that the result isn't a lucky split.

| Test year | Baseline MAE | LightGBM MAE | MAPE | Error reduction |
|---|---|---|---|---|
| Sep 2025 – Aug 2026 | 639 MW (6.6 %) | **306 MW** | 3.2 % | 52 % |
| Sep 2024 – Aug 2025 | 716 MW (7.4 %) | **321 MW** | 3.4 % | 55 % |

The **baseline** is the same hour one week earlier. It is a fair day-ahead baseline because all of that day is known when the forecast is made.

What the backtest shows:

- The model helps most where the baseline breaks: in cold months and around holidays. January MAE drops from 1,184 to 447 MW. On holidays, eves and bridge days it drops from 791 to 450 MW, but those days remain the model's weakest spot.
- The model under-forecasts by about 76–90 MW on average. Load in the test years sits slightly above the training years, and tree models cannot extrapolate a trend.
- **The backtest is optimistic.** It uses *observed* temperature as if it were a perfect forecast. Live runs use SMHI's real forecast, so live error will be higher. Every SMHI forecast the job uses is archived in [`data/smhi_forecast/`](data/smhi_forecast/), so this gap can be measured once enough days have accumulated.

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `src/fetch.py --update` | Pulls SE3 actual total load from ENTSO-E, re-fetching the last 3 days to pick up late corrections |
| 2 | `src/fetch_weather.py` | Pulls observed hourly temperature from SMHI (Stockholm-Observatoriekullen, station 98230) |
| 3 | `src/forecast.py` | Fetches the SMHI SNOW1g point forecast, retrains LightGBM on all history, and writes `forecasts/<date>.csv` |
| 4 | `src/score.py` | Rebuilds `scores/` from all published forecasts and writes `docs/data.json` for the live page |

**Features:** temperature, 24-hour mean temperature, heating degree-hours, load 48 h and 168 h earlier, and Swedish-local-time calendar features (hour, weekday, day of year, public holidays, eves such as Midsommarafton and Julafton, bridge days, the Christmas period).

The shortest load lag is 48 hours on purpose. When the forecast for day D is made on the morning of D-1, the evening of D-1 has not happened yet, so a 24-hour lag would leak information that is not available in real life.

### When a data source is down

The job never stops because an API is down. It takes the best path available and logs which one in [`forecasts/run_log.csv`](forecasts/run_log.csv):

1. **model** with the fresh SMHI forecast
2. **model** with the newest archived SMHI forecast, if it still covers every hour of tomorrow
3. **baseline** if no usable weather forecast exists or the load data is stale
4. **skipped** if even last week's load is missing; nothing is published

A published forecast file is never overwritten. Backtest predictions live in `backtest/` (not committed; reproducible) and never in `forecasts/`.

## Data decisions

- **All timestamps are UTC.** Swedish local time is only used to derive calendar features.
- **Resolution change:** ENTSO-E SE3 load is hourly until 2025-12-01 22:00 UTC and 15-minute from 23:00 UTC. Everything is averaged to hourly MW. An hour is kept only if it is fully covered.
- **Alignment:** ENTSO-E timestamps mark the start of the hour; SMHI temperature is an instantaneous reading at the timestamp. Checked with `python src/features.py --check`: load bottoms out at 04h and peaks at 17h local time, temperature bottoms out at 05h and peaks at 14h, and load correlates at −0.86 with the 24-hour mean temperature.
- **Temperature gaps** of up to 3 hours are interpolated; longer gaps stay missing.
- **Holidays** are computed in code rather than with the `holidays` package, which counts every Sunday as a Swedish holiday by default.
- **One weather station.** SE3 is larger than Stockholm; adding Göteborg is an obvious next step.

## Incidents

_None yet. Any day the job falls back or skips gets a short write-up here: what failed, what the job did, and what changed._

## Run it locally

```bash
conda env create -f environment.yml && conda activate se3
# put ENTSOE_API_TOKEN=... in a .env file in the repo root
python src/fetch.py --start 2023-01-01 --end 2026-09-01   # load history
python src/fetch_weather.py                                # temperature history
python src/features.py --check                             # alignment checks
python src/baseline.py                                     # baseline error
python src/model.py                                        # backtest, model vs baseline
python src/forecast.py --dry-run                           # tomorrow's forecast, writes nothing
```

## Repository layout

```
src/                 fetch, features, baseline, model, forecast, score
forecasts/           one file per day: the public prediction record, plus run_log.csv
scores/              forecast vs actual vs baseline, rebuilt daily
data/                load history, temperature history, archived SMHI forecasts
docs/                the GitHub Pages site (index.html + data.json)
.github/workflows/   the daily job
```

## Data sources

- Load: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), actual total load, bidding zone SE3
- Weather: [SMHI Open Data](https://opendata.smhi.se/), meteorological observations and the SNOW1g point forecast
