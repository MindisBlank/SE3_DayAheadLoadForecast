# SE3 Day-Ahead Load Forecast

A day-ahead hourly electricity load forecast for bidding zone SE3 that runs unattended every day, publishes its prediction before the fact, and records its own error against what actually happened.

**Live page:** https://mindisblank.github.io/SE3_DayAheadLoadForecast/

Every morning a GitHub Actions job pulls the latest load and weather data, publishes tomorrow's 24 hourly values to [`forecasts/`](forecasts/), scores every earlier forecast against the actual load in [`scores/`](scores/), and commits the result. The commit timestamp on each forecast file is the proof that it was made before the outcome was known, and each file also records whether it was issued before the day-ahead market's gate closure (12:00 Swedish time the day before).

## Results

### Live

Running since 18 September 2026. Rolling error is in [`scores/daily.csv`](scores/daily.csv) and on the live page.

| Period | Model version | Days | Model MAE | Baseline MAE | Beat baseline | In 80 % range | Fallback days |
|---|---|---|---|---|---|---|---|
| 19–22 Sep 2026 | v1 | 4 | 220 MW (2.66 %) | 244 MW | 2 of 4 | – | 0 |
| 23 Sep – 11 Oct 2026 | v2 | 19 | **243 MW (2.74 %)** | 385 MW | **17 of 19** | **84.5 %** | 0 |
| … excluding 5 Oct | v2 | 18 | 218 MW (2.52 %) | 376 MW | 17 of 18 | | |

Scores are kept per version, so versions are never mixed. Zero fallback days and zero skipped days so far: every run has published a model forecast. The range is holding slightly wide – 84.5 % of hours inside an 80 % band – which is the safe direction to be wrong in. One day, 5 October, accounts for 25 MW of the 19-day v2 average on its own; it is written up under Incidents rather than quietly dropped.

**The honest caveat on all of this:** only 4 of those 19 forecasts were issued before the day-ahead gate closure. The rest were published before the day they predict, so the track record is real, but too late to have been bid into the market. That is a limitation of free hosted cron rather than of the model, it could not be fixed by scheduling earlier without a worse trade-off, and it is written up in full under Incidents.

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
| 2 | `src/fetch_weather.py` | Pulls observed hourly temperature from SMHI for the 9 regional stations in [`data/regional_stations.csv`](data/regional_stations.csv), plus wind, cloud cover, humidity and precipitation for the stations in [`data/wx_stations.csv`](data/wx_stations.csv) |
| 3 | `src/forecast.py` | Fetches SMHI SNOW1g point forecasts (all five variables) for the 9 cities, retrains LightGBM on all history, applies the bias correction, and writes `forecasts/<date>.csv` |
| 4 | `src/score.py` | Rebuilds `scores/` from all published forecasts and writes `docs/data.json` for the live page |

**Features:** population-weighted temperature over 9 SE3 cities (v2), its 24-hour mean, heating degree-hours, load 48 h and 168 h earlier, and Swedish-local-time calendar features (hour, weekday, day of year, public holidays, eves such as Midsommarafton and Julafton, bridge days, the Christmas period). v3 adds weighted wind, cloud cover, humidity and precipitation, a wind-chill style interaction and 24-hour means of wind and cloud.

**Bias correction (v3):** the model under-forecasts, so every hour of tomorrow is shifted by the median of the model's last three scored daily mean errors, of which 0.4 is applied, capped at ±150 MW and skipped on holidays, eves and bridge days. Only days that were fully scored before the forecast was made are used, and the shift each forecast received is recorded in `bias_shift_mw` and added back before the next day's median, so the filter always measures the model's bias rather than its own correction. The settings come from a sweep (below), not from a guess.

The shortest load lag is 48 hours on purpose. When the forecast for day D is made on the morning of D-1, the evening of D-1 has not happened yet, so a 24-hour lag would leak information that is not available in real life.

### When a data source is down

The job never stops because an API is down. It takes the best path available and logs which one in [`forecasts/run_log.csv`](forecasts/run_log.csv):

1. **v3 model** with fresh SMHI forecasts for the 9 cities, or the newest archived set that still covers tomorrow, plus wind / cloud / humidity / rain, then the bias correction. Cities that fail are left out and the weights renormalised, as long as at least 80 % of the weight is present for every hour.
2. **v2 model** (temperature only) if the extra weather has a gap in the target day. The same model code trains on whichever columns are present, so this is a feature-set fallback, not a second model to maintain.
3. **v1 model** (Stockholm temperature only) if the regional forecasts don't cover enough
4. **baseline** if no usable weather forecast exists or the load data is stale
5. **skipped** if even last week's load is missing; nothing is published

A published forecast file is never overwritten. Backtest predictions live in `backtest/` (not committed; reproducible) and never in `forecasts/`.

## Model versions

Every forecast file records the `model_version` that produced it. A version changes only after it beats the live one in the backtest on **both** test years, and the date of the switch is recorded here.

| Version | Live from | Change | Backtest MAE (2025–26 / 2024–25) |
|---|---|---|---|
| v1 | 18 Sep 2026 | LightGBM, Stockholm temperature (one station) | 306 / 321 MW |
| v2 | 23 Sep 2026 | Same model, temperature = population-weighted mean over 9 SE3 cities | 269 / 290 MW |
| v2 + range | 22 Sep 2026 (code) | 80 % range added next to the unchanged point forecast | coverage 81.5 / 83.0 % |
| v3 | 14 Oct 2026 | + wind, cloud, humidity, rain, and a self-correcting bias filter | 264 MW single-split, 249 MW walk-forward |

The v3 row is walk-forward (retrain, forecast one day, learn the error, repeat), so it is not comparable with the single-split v1/v2 numbers above it; the two changes are measured separately against the same walk-forward v2 in the two sections below – the weather variables are worth 4–10 MW and the bias filter 1–8 MW on those years (and 19 MW on the held-out live window, which is where the bias actually is).

How v2 was chosen: [`src/experiment.py`](src/experiment.py) backtests six variants on the same hours. Only the regional temperature won on both years; everything else was worse than v1. (The table below is the original experiment run, before the Västerås station was replaced with an hourly one; the final v2 numbers above come from `src/model.py` after the fix.)

| Variant (average of both test years) | MAE | Winter MAE | Days-off MAE | Bias |
|---|---|---|---|---|
| v1 | 314 | 389 | 455 | −83 |
| predict change vs last week | 357 | 483 | 501 | −85 |
| + recent level feature | 353 | 476 | 504 | −85 |
| + recent data weighted more | 356 | 475 | 479 | −84 |
| **9-city temperature (v2)** | **280** | **335** | **410** | **−57** |
| 9-city + change + level | 332 | 442 | 465 | −49 |

### Weather variables (live in v3)

Walk-forward test over both test years, one variable at a time, change in MAE against v2:

| Added | 2024–25 | 2025–26 |
|---|---|---|
| cloud cover | −5.8 MW | −7.6 MW |
| humidity | −11.6 MW | −1.9 MW |
| rain | −3.4 MW | −1.0 MW |
| wind | **+2.1 MW** | **+5.8 MW** |
| all four together | −9.7 MW | −4.4 MW |

Cloud cover is the strongest single addition, which fits lighting demand plus behind-the-meter solar. Rain helps slightly and consistently. Wind on its own makes the model *worse* in both years, and adding it to cloud dilutes cloud's gain, so the wind-chill interaction looks like noise the trees overfit. All four together still win, as boosted trees route around a weak feature. The gaps are 1–4 % of MAE: consistent in sign across two years, but not individually significant.

### Bias correction (how the settings were chosen)

The model under-forecasts in every period tested: its mean error runs from −20 MW on the calmest year to −130 MW on the most recent window. A constant adder would hide that rather than fix it, so instead the forecast is shifted by the model's own measured bias, which lets it follow a drift and unwind if the bias goes away.

That leaves five knobs, so [`src/filter_sweep.py`](src/filter_sweep.py) tries all 1,320 combinations of them: how fast the filter follows a drift (half-life 1–60 days), how much of the measured bias it actually applies (damping 0.1–1.0), mean or median, a cap (none / 150 / 300 MW), and whether to apply it on holidays and eves. The expensive part – the walk-forward predictions – is computed once and cached, so the sweep itself costs seconds.

The protocol was fixed before looking at the numbers:

- settings are **chosen on the two historical years** (Sep 2024 – Aug 2026); the live window (Sep–Oct 2026) is **held out** and only used to check the choice afterwards
- the choice is simply the **lowest error** – MAE pooled over all selection hours at once, not the average of the two years' MAEs, because 2024 has 257 scored days to 2025's 347 and averaging two MAEs would weight a day in the thinner year more
- **one kill switch**, doing the only job a kill switch has: discarding a setting whose error saturates above the baseline *across periods*. A setting is dropped only if it is more than 3 % worse than no filter on **both** selection periods. Two of 1,320 settings were. A check that rejects on a single period, or on one bad day, is a tie-breaker wearing a safety check's clothes – it will throw out the best setting because of one unlucky day, and an earlier version of this file did exactly that

The chosen filter is the lowest-error one: **median of the last 3 scored days, of which 0.4 is applied, capped at ±150 MW, not applied on holidays/eves/bridge days**.

| Period | Days | MAE no filter | MAE with filter | Bias no filter | Bias with filter |
|---|---|---|---|---|---|
| Sep 2024 – Aug 2025 (selection) | 257 | 250.9 MW | **249.7 MW** (−0.5 %) | −20 MW | −13 MW |
| Sep 2025 – Aug 2026 (selection) | 347 | 257.4 MW | **249.0 MW** (−3.3 %) | −25 MW | −14 MW |
| pooled selection | 604 | 254.7 MW | **249.3 MW** (−2.1 %) | | |
| Sep–Oct 2026 (**held out**) | 41 | 203.5 MW | **184.2 MW** (−9.4 %) | −130 MW | −83 MW |

Damping 0.4 is an **interior** optimum rather than the edge of the grid – 0.3 scores 249.6 and 0.5 scores 249.5 against 249.3 – which is worth checking before believing any swept parameter. The best twelve settings sit within 0.4 MW of each other, so this is a plateau and no claim is made that these exact five numbers are special; a cap of 150 MW appears in all twelve, which is the one setting other than damping that clearly earns its place.

The gain on the historical years is small – 2.1 % pooled – because those years are barely biased. Almost all of the filter's value is in the held-out window, for the reason in the next section.

The day counts above are days that have a prediction at all: 2024 has 257 of 365, because SMHI's rain series starts in 2024 and a prediction needs every weather value for the target day.

### What the held-out window actually shows

The bias in the two historical years is −20 to −25 MW, and the model is under-forecasting on roughly half the days – which is what an unbiased model looks like. The live window is different:

| | Sep 2024 – Aug 2025 | Sep 2025 – Aug 2026 | Sep–Oct 2026 |
|---|---|---|---|
| Mean daily error | −23 MW | −28 MW | **−130 MW** |
| Days under-forecast | 139 of 257 | 202 of 347 | **36 of 41** |

−130 MW with a day-to-day standard deviation of 146 over 41 days puts the standard error of that mean at 23 MW, so it is about five standard errors from the historical level. It is not sampling noise. Looking at it week by week, it is also not a seasonal ramp:

| Week | Mean daily error |
|---|---|
| 1–7 Sep | +8 MW |
| 8–15 Sep | −132 MW |
| 16–22 Sep | −132 MW |
| 23–29 Sep | −167 MW |
| 30 Sep – 6 Oct | −235 MW |
| 7–12 Oct | −123 MW |

The first week is clean and then it steps down and stays down. The gap is also flat across the day (−56 to −184 MW at every hour, slightly worse overnight), which is the signature of a level shift rather than a shape error – the model has the daily curve right and the base wrong. Best read: SE3 load stepped up by something over 1 % in early September 2026, and a model trained on 2023–2026 does not know it yet.

That is a case for a bias filter rather than a retune, and it is also why the filter earns 9.4 % here against 2.1 % on the historical years. The honest limitation: at damping 0.4 the filter removes about a third of a step like this (−130 MW to −83 MW), and a stronger setting would remove more – damping 1.0 takes the live bias to −29 MW. It is not chosen because it is worse on the selection periods, where there is no step to chase and a heavy correction only adds noise. If the step turns out to be permanent, the right fix is not a heavier filter but giving the model a recent-level feature, which lost on historical data for the same reason: those years have no step in them. The live scoring will settle it.

## Data decisions

- **All timestamps are UTC.** Swedish local time is only used to derive calendar features.
- **Resolution change:** ENTSO-E SE3 load is hourly until 2025-12-01 22:00 UTC and 15-minute from 23:00 UTC. Everything is averaged to hourly MW. An hour is kept only if it is fully covered.
- **Alignment:** ENTSO-E timestamps mark the start of the hour; SMHI temperature is an instantaneous reading at the timestamp. Checked with `python src/features.py --check`: load bottoms out at 04h and peaks at 17h local time, temperature bottoms out at 05h and peaks at 14h, and load correlates at −0.86 with the 24-hour mean temperature.
- **The newest 3 hours of load are ignored.** ENTSO-E publishes the latest values as preliminary: on 22 Sep 2026 the last 45 minutes read about 6,300 MW against 8,800 MW just before, about 28 % too low, and are corrected later. Nothing needs those hours (the shortest lag is 48 h and only finished days are scored), and the daily job re-fetches the last 3 days, so they return once settled. An hour also only counts if all four of its quarter-hours are present.
- **Temperature gaps** of up to 3 hours are interpolated; longer gaps stay missing.
- **Holidays** are computed in code rather than with the `holidays` package, which counts every Sunday as a Swedish holiday by default.
- **Regional temperature (v2):** one SMHI station per city (Stockholm, Göteborg, Uppsala, Linköping, Örebro, Västerås, Jönköping, Karlstad, Gävle), weighted by rough share of SE3 population. A station is only accepted if it reports at least 90 % of hours; the nearest Västerås station turned out to report only twice a day, so Västerås uses the nearest hourly station instead (Eskilstuna A, 26 km away, also in the Mälaren region). Each hour uses the stations that have data, with the weights renormalised; below 80 % of the total weight the hour counts as missing. The station list is fixed in `data/regional_stations.csv`, so the inputs never change silently.

## Incidents

**18–22 Sep 2026: forecasts issued after gate closure. Attempted fix did not work; limitation accepted.** The job was first scheduled for 08:15 UTC, but GitHub started it 4–7 hours late every day (12:28–15:04 UTC). Every forecast was still published before the day it predicts, so the track record is valid, but after the 10:00 UTC day-ahead gate closure, so it could not have been bid into the market.

The fix tried was three earlier triggers (04:37, 06:07, 07:37 UTC). **It did not work, and the run log says so.** Over the following ten days the first run of each day actually started between 10:01 and 12:22 UTC – a queue delay of 5 h 24 m to 7 h 45 m, consistently – so only **4 of 19** forecasts landed before gate closure. GitHub delays scheduled workflows under load on every plan; this is not something a paid plan fixes.

Beating a 7 h 45 m delay would mean firing the first trigger at 02:07 UTC, and that runs into the data: at that hour ENTSO-E has published through about D−1 01:00, the 3-hour settle rule trims it to D−2 22:00, and the 48-hour lag needs D−2 21:00 – one hour of margin. A slow morning there would publish a *baseline* forecast, and since a forecast file is never overwritten, that would lock the day into a baseline. Trading a gate-closure miss for an occasional baseline day is the worse deal, so the schedule stays where it is.

**The limitation is therefore permanent and stated rather than fixed:** these are genuine before-the-fact forecasts, published and timestamped before the day they predict, but mostly an hour or two too late to have been bid into the day-ahead market. Every forecast records `before_gate_closure` and the live page shows it per day, so the subset that did make the deadline can be read off directly. Nothing here is a claim about market usability.

**5 Oct 2026: a 2.6 GW midday spike the model had no way to see.** The forecast for Monday 5 October was out by 677 MW on daily average – the worst day of the live record by a factor of two – and 17 of 24 hours fell outside the 80 % range. The cause was not the pipeline and not the weather: the temperature forecast for that day was accurate to −0.31 °C on average and 1.12 °C at worst, and the run completed normally on fresh SMHI data.

What happened is that actual load spiked to **12,008 MW at 11:00 local**, against a forecast of 9,394 MW and a typical early-October Monday peak of 9,700–10,700 MW in the morning or evening. The excursion lasted about three hours (09:00–12:00 local, +1.3 to +2.6 GW), then load returned to normal for the rest of the day; the surrounding Monday peaks were 9,742 MW on 28 Sep and 10,738 MW on 12 Oct, both at the usual times. A 25 % three-hour jump at midday on a mild day (14 °C) is not a demand shape any weather-driven model produces, and the value is in ENTSO-E's settled data, so it is not the preliminary-value problem either.

No code change: nothing in the pipeline failed and nothing would have predicted it. It is recorded here because a single day moved the 19-day average by 25 MW, and the headline numbers above show the record with and without it. The 80 % range is doing its job by being visibly wrong on a day like this rather than quietly wide enough to swallow it.

_Any other day the job falls back or skips gets a short write-up here: what failed, what the job did, and what changed._

## Run it locally

```bash
conda env create -f environment.yml && conda activate se3
# put ENTSOE_API_TOKEN=... in a .env file in the repo root
python src/fetch.py --start 2023-01-01 --end 2026-09-01   # load history
python src/fetch_weather.py --relookup                     # pick the 9 regional stations (first time only)
python src/fetch_weather.py                                # temperature history
python src/fetch_weather.py --extra                        # wind / cloud / humidity / rain (first time only)
python src/features.py --check                             # alignment checks
python src/baseline.py                                     # baseline error
python src/model.py --weather                              # backtest v3 vs baseline (--temp stockholm for v1)
python src/experiment.py                                   # compare model variants
python src/experiment_adaptive.py --ablation                # which weather variable earns its place
python src/filter_sweep.py                                 # choose the bias filter (minutes, then cached)
python src/forecast.py --dry-run                           # tomorrow's forecast, writes nothing
```

## Repository layout

```
src/                 fetch, features, baseline, model, forecast, score
                     experiment*, filter_sweep: offline studies, never touch forecasts/
forecasts/           one file per day: the public prediction record, plus run_log.csv
scores/              forecast vs actual vs baseline, rebuilt daily
data/                load history, weather history, archived SMHI forecasts, station lists
docs/                the GitHub Pages site (index.html + data.json)
.github/workflows/   the daily job
```

## Data sources

- Load: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), actual total load, bidding zone SE3
- Weather: [SMHI Open Data](https://opendata.smhi.se/), meteorological observations and the SNOW1g point forecast
