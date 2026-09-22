# SE3 Day-Ahead Load Forecast

A day-ahead hourly electricity load forecast for bidding zone SE3 that runs unattended every day, publishes its prediction before the fact, and records its own error against what actually happened.

**Live page:** https://mindisblank.github.io/SE3_DayAheadLoadForecast/

Every morning a GitHub Actions job pulls the latest load and weather data, publishes tomorrow's 24 hourly values to [`forecasts/`](forecasts/), scores every earlier forecast against the actual load in [`scores/`](scores/), and commits the result. The commit timestamp on each forecast file is the proof that it was made before the outcome was known, and each file also records whether it was issued before the day-ahead market's gate closure (12:00 Swedish time the day before).

## Results

### Live

Running since 18 September 2026. Rolling error is in [`scores/daily.csv`](scores/daily.csv) and on the live page.

| Period | Model version | Days | Model MAE | Baseline MAE | In 80 % range | Fallback days |
|---|---|---|---|---|---|---|
| _to be filled in after the first 7 scored days of v2_ | | | | | | |

The first days (19–22 Sep) ran v1; v2 runs from 23 Sep. Scores are kept per version, so the two are never mixed.

### Backtest

Trained on everything before the test year, tested on a held-out year (8,760 hours). Two test years are shown as a check that the result isn't a lucky split.

| Test year | Baseline MAE | v1 MAE (Stockholm temp.) | **v2 MAE (9-city temp.)** | v2 MAPE | v2 vs baseline |
|---|---|---|---|---|---|
| Sep 2025 – Aug 2026 | 639 MW (6.6 %) | 306 MW | **269 MW** | 2.9 % | −58 % |
| Sep 2024 – Aug 2025 | 716 MW (7.4 %) | 321 MW | **290 MW** | 3.1 % | −60 % |

The **baseline** is the same hour one week earlier. It is a fair day-ahead baseline because all of that day is known when the forecast is made.

What the backtest shows:

- The model helps most where the baseline breaks: in cold months and around holidays. In 2025–26, January MAE drops from 1,184 to 350 MW. On holidays, eves and bridge days it drops from 791 to 401 MW, but those days remain the model's weakest spot.
- v1 under-forecast by 76–90 MW on average. Using one Stockholm station missed part of the heating demand elsewhere in SE3; the 9-city temperature in v2 cuts this to 48–67 MW and improves winter (Dec–Feb) MAE by 12–18 %. With regional temperature, the 24-hour mean temperature becomes one of the model's two strongest inputs (28–43 % of total gain, up from 8–15 % with Stockholm alone). The rest of the bias looks like year-to-year level variation: at the same temperature, weekday and hour, SE3 load differs by about ±200 MW between years, with no steady trend.
- Three other ideas were tested and rejected (predicting the change from last week, a recent-level feature, weighting recent data more). All were worse than v1 on both test years. See [Model versions](#model-versions).
- **The backtest is optimistic.** It uses *observed* temperature as if it were a perfect forecast. Live runs use SMHI's real forecast, so live error will be higher. Every SMHI forecast the job uses is archived in [`data/smhi_forecast/`](data/smhi_forecast/), so this gap can be measured once enough days have accumulated.

### Forecast range (P10–P90)

Every model forecast also publishes an **80 % range**: `p10_mw` and `p90_mw`. If the range is honest, the actual load should land inside it in about 8 of 10 hours, below it in 1 and above it in 1. It was added on 22 Sep 2026 without changing the point forecast, so it is still v2.

| Test year | In range (target 80 %) | Below P10 / above P90 (targets 10 / 10) | Mean width | Pinball loss |
|---|---|---|---|---|
| Sep 2025 – Aug 2026 | **81.5 %** | 7.2 % / 11.3 % | 916 MW | 64.4 MW |
| Sep 2024 – Aug 2025 | **83.0 %** | 7.8 % / 9.2 % | 1,040 MW | 65.1 MW |

How it is built: the training history is cut into 4 time blocks; for each block the model is trained on the other three and its errors on the held-out block are recorded. The 10th and 90th percentiles of those out-of-sample errors, sized separately by season, time of day and days off, are added to the point forecast. The first attempt, separate LightGBM quantile models, held only about 52 % of hours in the backtest: errors a model makes on its own training data are smaller than its real future errors, so the band came out far too narrow. It was never published.

Caveats: the range is slightly too wide in winter (84–93 % held) and too narrow in early summer (67–69 % in June), and it is calibrated with *observed* temperature. Live forecasts use forecast weather, so expect live coverage somewhat under 80 % until it is recalibrated on live errors.

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `src/fetch.py --update` | Pulls SE3 actual total load from ENTSO-E, re-fetching the last 3 days to pick up late corrections |
| 2 | `src/fetch_weather.py` | Pulls observed hourly temperature from SMHI for the 9 regional stations in [`data/regional_stations.csv`](data/regional_stations.csv) |
| 3 | `src/forecast.py` | Fetches SMHI SNOW1g point forecasts for the 9 cities, retrains LightGBM on all history, and writes `forecasts/<date>.csv` |
| 4 | `src/score.py` | Rebuilds `scores/` from all published forecasts and writes `docs/data.json` for the live page |

**Features:** population-weighted temperature over 9 SE3 cities (v2), its 24-hour mean, heating degree-hours, load 48 h and 168 h earlier, and Swedish-local-time calendar features (hour, weekday, day of year, public holidays, eves such as Midsommarafton and Julafton, bridge days, the Christmas period).

The shortest load lag is 48 hours on purpose. When the forecast for day D is made on the morning of D-1, the evening of D-1 has not happened yet, so a 24-hour lag would leak information that is not available in real life.

### When a data source is down

The job never stops because an API is down. It takes the best path available and logs which one in [`forecasts/run_log.csv`](forecasts/run_log.csv):

1. **v2 model** with fresh SMHI forecasts for the 9 cities, or the newest archived set that still covers tomorrow. Cities that fail are left out and the weights renormalised, as long as at least 80 % of the weight is present for every hour.
2. **v1 model** (Stockholm temperature only) if the regional forecasts don't cover enough
3. **baseline** if no usable weather forecast exists or the load data is stale
4. **skipped** if even last week's load is missing; nothing is published

A published forecast file is never overwritten. Backtest predictions live in `backtest/` (not committed; reproducible) and never in `forecasts/`.

## Model versions

Every forecast file records the `model_version` that produced it. A version changes only after it beats the live one in the backtest on **both** test years, and the date of the switch is recorded here.

| Version | Live from | Change | Backtest MAE (2025–26 / 2024–25) |
|---|---|---|---|
| v1 | 18 Sep 2026 | LightGBM, Stockholm temperature (one station) | 306 / 321 MW |
| v2 | 23 Sep 2026 | Same model, temperature = population-weighted mean over 9 SE3 cities | 269 / 290 MW |
| v2 + range | 22 Sep 2026 (code) | 80 % range added next to the unchanged point forecast | coverage 81.5 / 83.0 % |

How v2 was chosen: [`src/experiment.py`](src/experiment.py) backtests six variants on the same hours. Only the regional temperature won on both years; everything else was worse than v1. (The table below is the original experiment run, before the Västerås station was replaced with an hourly one; the final v2 numbers above come from `src/model.py` after the fix.)

| Variant (average of both test years) | MAE | Winter MAE | Days-off MAE | Bias |
|---|---|---|---|---|
| v1 | 314 | 389 | 455 | −83 |
| predict change vs last week | 357 | 483 | 501 | −85 |
| + recent level feature | 353 | 476 | 504 | −85 |
| + recent data weighted more | 356 | 475 | 479 | −84 |
| **9-city temperature (v2)** | **280** | **335** | **410** | **−57** |
| 9-city + change + level | 332 | 442 | 465 | −49 |

## Data decisions

- **All timestamps are UTC.** Swedish local time is only used to derive calendar features.
- **Resolution change:** ENTSO-E SE3 load is hourly until 2025-12-01 22:00 UTC and 15-minute from 23:00 UTC. Everything is averaged to hourly MW. An hour is kept only if it is fully covered.
- **Alignment:** ENTSO-E timestamps mark the start of the hour; SMHI temperature is an instantaneous reading at the timestamp. Checked with `python src/features.py --check`: load bottoms out at 04h and peaks at 17h local time, temperature bottoms out at 05h and peaks at 14h, and load correlates at −0.86 with the 24-hour mean temperature.
- **The newest 3 hours of load are ignored.** ENTSO-E publishes the latest values as preliminary: on 22 Sep 2026 the last 45 minutes read about 6,300 MW against 8,800 MW just before, about 28 % too low, and are corrected later. Nothing needs those hours (the shortest lag is 48 h and only finished days are scored), and the daily job re-fetches the last 3 days, so they return once settled. An hour also only counts if all four of its quarter-hours are present.
- **Temperature gaps** of up to 3 hours are interpolated; longer gaps stay missing.
- **Holidays** are computed in code rather than with the `holidays` package, which counts every Sunday as a Swedish holiday by default.
- **Regional temperature (v2):** one SMHI station per city (Stockholm, Göteborg, Uppsala, Linköping, Örebro, Västerås, Jönköping, Karlstad, Gävle), weighted by rough share of SE3 population. A station is only accepted if it reports at least 90 % of hours; the nearest Västerås station turned out to report only twice a day, so Västerås uses the nearest hourly station instead (Eskilstuna A, 26 km away, also in the Mälaren region). Each hour uses the stations that have data, with the weights renormalised; below 80 % of the total weight the hour counts as missing. The station list is fixed in `data/regional_stations.csv`, so the inputs never change silently.

## Incidents

**18–22 Sep 2026: forecasts issued after gate closure.** The job was scheduled for 08:15 UTC, but GitHub started it 4–7 hours late every day (12:28–15:04 UTC). Every forecast was still published before the day it predicts, so the track record is valid, but after the 10:00 UTC day-ahead gate closure, so it could not have been used for bidding. Fix: three trigger times (04:37, 06:07, 07:37 UTC). The first run that gets through publishes and later ones only refresh the scores. Each forecast now records `before_gate_closure`, and the live page shows it per day.

_Any other day the job falls back or skips gets a short write-up here: what failed, what the job did, and what changed._

## Run it locally

```bash
conda env create -f environment.yml && conda activate se3
# put ENTSOE_API_TOKEN=... in a .env file in the repo root
python src/fetch.py --start 2023-01-01 --end 2026-09-01   # load history
python src/fetch_weather.py --relookup                     # pick the 9 regional stations (first time only)
python src/fetch_weather.py                                # temperature history
python src/features.py --check                             # alignment checks
python src/baseline.py                                     # baseline error
python src/model.py                                        # backtest v2 vs baseline (--temp stockholm for v1)
python src/experiment.py                                   # compare model variants
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
