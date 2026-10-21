# trading-ml

Machine-learning trading research for stocks on the Indonesia Stock Exchange (IDX).
It downloads daily prices from Yahoo Finance, backtests several strategies without
look-ahead bias, gives a daily **BUY / HOLD / SELL** suggestion, and forecasts a
20-day price range.

> **Not financial advice.** The project's own hold-out test found no setting whose
> past performance reliably carried forward. Treat every signal as something to
> paper-trade and check, not something to act on blindly.

---

## Contents

- [What's inside](#whats-inside)
- [Installation](#installation)
- [Quick start: the interactive tool](#quick-start-the-interactive-tool)
- [How it works](#how-it-works)
- [The scripts in detail](#the-scripts-in-detail)
- [How to read the results](#how-to-read-the-results)
- [A daily routine](#a-daily-routine)
- [Limitations](#limitations)
- [Project layout](#project-layout)

---

## What's inside

| Script (in `src/`) | What it does | Typical run time |
|---|---|---|
| `stock_cli.py` | **Start here.** Interactive tool, run any time: type a code, get the backtest, signal, open/close forecasts and a BUY / HOLD / SELL / WAIT decision for the next session (or today's pre-closing during the session). | first run after a close ~5–8 min, then seconds |
| `bbca_walkforward.py` | The engine: data download, features, walk-forward models, backtest, HTML report. Other scripts import it. | ~1 min |
| `today_signal.py` | Today's BUY / SELL / HOLD / STAY OUT for one stock, logged for paper-trading. | ~1 min |
| `lstm_forecast.py` | LSTM forecast of the 80% price range for the next 20 trading days, with an honesty check. | ~2 min (GPU) |
| `next_day.py` | *Trial (BBCA).* After 16:15 WIB: next session's open and close as 80% ranges (LSTM, LightGBM, baseline). During the session: today's close updated from the live price (`intraday_close.py`). Every forecast is logged and scored against what happened. | ~1 min / ~30 s |
| `sweep.py` | Tries hundreds of settings (horizon × thresholds × smoothing × model). | 5–15 min |
| `holdout.py` | Checks whether the sweep's best settings keep working on data they weren't picked on. | 5–15 min |
| `intraday_test.py` | Tests whether an *hourly* forecaster would make money after costs (answer so far: no). | ~1 min |

---

## Installation

You need **Python 3.11+** (tested on 3.14) and **git**. A GPU is optional; it only speeds up the LSTM.

**1. Clone the repository**

```bash
git clone git@github.com:Rian021102/trading-ml.git
cd trading-ml
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate          # bash / zsh
source .venv/bin/activate.fish     # fish
.venv\Scripts\activate             # Windows
```

**3. Install PyTorch** (do this first so you get the right build for your machine)

- NVIDIA GPU: pick the CUDA command for your system at <https://pytorch.org/get-started/locally/>
- CPU only:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

**4. Install the rest**

```bash
pip install -r requirements.txt
```

**5. Check it works**

```bash
cd src
python -c "import yfinance, lightgbm, arch, torch; print('ok, GPU:', torch.cuda.is_available())"
```

**6. Run it**

```bash
python stock_cli.py
```

> Always run the scripts **from inside `src/`**. They import each other and write
> results to `src/output/`. Downloaded prices go to `data/` and charts to `images/`
> at the project root; all three folders are created automatically.

---

## Quick start: the interactive tool

```bash
cd src
python stock_cli.py               # asks for a code
python stock_cli.py MTLA BBCA     # analyses these first, then asks
```

1. Type a stock code, e.g. `mtla`. `.JK` is added for you, so `mtla` becomes `MTLA.JK`.
2. Answer **Do you own MTLA.JK now? [y/N]**. The suggestion depends on it (see [below](#the-next-day-suggestion)).
3. The first run after a close takes a few minutes (models are trained and cached); later runs that day take seconds. It prints the backtest table, the last 10 sessions, the LSTM tables, the next-session forecast and scorecard (plus the intraday update during the session) and a **summary** with both calls, the decision and prices, then opens the forecast chart.
4. Type another code, or press **Esc** for the menu: `1` = another company, `2` = exit.
   **Ctrl-C** cancels an analysis that's running.

Useful options:

| Option | Effect |
|---|---|
| `--no-lstm` | Skip the LSTM forecast (~2 min faster; no range or limit prices) |
| `--seeds 2` | Smaller LSTM ensemble, faster but noisier |
| `--no-open` | Don't open the chart automatically |
| `--offline` | Use the CSVs already in `data/` instead of downloading |
| `--refresh` | Retrain instead of reusing the models cached at the last close |
| `--threshold 0.55 --exit-threshold 0.50 --smooth 3` | Signal settings (these are the defaults) |
| `--model LogReg` | Use Logistic Regression or `LightGBM+GARCH` instead of LightGBM |

Each run writes `src/output/<CODE>/report.html` (interactive charts), `images/lstm_forecast_<CODE>.png`,
and one row in `src/output/signals_log.csv`.

---

## How it works

```
Yahoo Finance daily OHLCV (stock + IHSG ^JKSE), plus today's live price when the market is open
        │
        ▼
27 features per day, using only data up to that day's close
(returns, volatility, trend, RSI, candle shape, volume, calendar, IHSG, beta)
        │
        ▼
Walk-forward training: train on the past → predict the next 21 days → move forward → retrain
        │                      (models never see the days they are scored on)
        ▼
P(up) = probability tomorrow's close is higher than today's (LightGBM and Logistic Regression)
        │
        ▼
Position rule: buy when smoothed P(up) > entry (0.55), hold until it drops below exit (0.50)
        │
        ▼
Backtest after IDX costs (0.15% buy, 0.25% sell incl. tax) vs Buy & Hold
```

Key ideas:

- **Walk-forward, no look-ahead.** The first ~3 years are used only for training. After that the model is retrained every 21 trading days on all earlier data, and only predicts days it has never seen. Every backtest number comes from these predictions.
- **Embargo.** Training rows whose target overlaps the test window are dropped, so labels can't leak.
- **It predicts direction, not price.** The target is "is the close *h* days later higher than today's?".
- **Long or flat only.** Short-selling is very restricted for retail investors on IDX.
- **Hysteresis.** A separate entry level (0.55) and exit level (0.50) stop the position from flip-flopping when P(up) hovers around one number.
- **GARCH sizing (optional strategy).** Same signal, but it holds less of the stock when forecast volatility is above 20% a year.
- **Live price.** When the market is open, today's bar is updated with the latest 1-minute Yahoo price. The signal is then marked **PROVISIONAL**; re-run after 16:15 WIB for the final one.

---

## The scripts in detail

### `bbca_walkforward.py`: backtest and report

```bash
python bbca_walkforward.py --ticker BBCA.JK
python bbca_walkforward.py --ticker BBRI.JK --threshold 0.55 --exit-threshold 0.48 --smooth 3
python bbca_walkforward.py --csv ../data/BBRI.JK.csv          # re-use a saved download
```

Compares five strategies on the same out-of-sample days: **Buy & Hold**, **SMA 50/200**
crossover, **Logistic Regression**, **LightGBM**, and **LightGBM + GARCH** volatility sizing.
Writes `src/output/summary.csv`, `yearly_returns.csv`, `equity_and_probs.csv`,
`feature_importance.csv`, `report.html` and `images/backtest.png`.

The default ticker is `MTLA.JK`, so always pass `--ticker`.

### `today_signal.py`: daily signal

```bash
python today_signal.py --ticker BBCA.JK
python today_signal.py --ticker BBRI.JK --model LogReg --threshold 0.58 --exit-threshold 0.48
python today_signal.py --ticker BBCA.JK --no-live      # ignore today's unfinished bar
```

Rebuilds the full walk-forward history (needed because the hold/exit rule depends on the
past), retrains on all data for today's prediction, prints the action and recent sessions,
and logs it to `src/output/signals_log.csv`.

### `lstm_forecast.py`: 20-day price range

```bash
python lstm_forecast.py --ticker BBCA.JK
python lstm_forecast.py --ticker BBRI.JK --days 10 --offline
```

An LSTM reads the last 60 days and predicts the 10th, 50th and 90th percentile of the
price for each of the next 20 days. Five seeds are averaged, and the band is recalibrated
on recent errors. It is scored on the most recent 20% of history against a **random walk**
whose band is just ±recent volatility. Writes `src/output/lstm_forecast_<CODE>.csv` and
`images/lstm_forecast_<CODE>.png`.

### `sweep.py` and `holdout.py`: choosing settings honestly

```bash
python sweep.py --ticker BBCA.JK                      # uses data/ CSVs; add --download for fresh data
python holdout.py --ticker BBCA.JK --split 2021-12-31
```

`sweep.py` trains once per horizon (1, 3, 5, 10, 21 days) and backtests every entry /
exit / smoothing / model combination (→ `src/output/sweep.csv`).
`holdout.py` picks the best settings on the **first** half of the out-of-sample period and
scores them on the **second** half, which wasn't used to choose (→ `src/output/holdout.csv`).

### `intraday_test.py`: would an hourly model pay?

```bash
python intraday_test.py --ticker BBCA.JK
python intraday_test.py --ticker BBRI.JK --hours 1 2 --same-day    # day trades only
```

Uses ~2 years of hourly bars, forecasts 1, 2, 3 and 7 hours ahead with LightGBM and the
LSTM, and simulates trades after fees **plus one price tick of spread**. On BBCA and BBRI
the hourly edge (mostly short reversals of 0.06–0.17%) is far below the ~0.7–0.8%
round-trip cost, so hourly trading loses money.

---

## How to read the results

### Backtest table

| Column | Meaning | What's good |
|---|---|---|
| **CAGR** | Average yearly growth after costs | Higher than Buy & Hold, or similar with less risk |
| **Vol** | Yearly volatility of daily returns | Lower = smoother ride |
| **Sharpe** | Return per unit of risk | > Buy & Hold's; > 1 is strong |
| **MaxDD** | Worst fall from a peak | Closer to 0% |
| **Calmar** | CAGR ÷ \|MaxDD\| | Higher |
| **Exposure** | Share of days in the stock | Context only: 2% means it almost never buys |
| **Trades/yr** | Position changes per year | Fewer = lower costs |
| **AUC** | How well P(up) ranks up days vs down days | 0.50 = coin flip; 0.52–0.55 is typical for real skill |
| **HitRate** | How often P(up) > 0.5 matched an up day | Compare with the share of up days |

**Read it like this:** a strategy is only interesting if it beats the **Buy & Hold** row on
Sharpe, or matches its CAGR with a much smaller MaxDD.

**Caution with illiquid stocks.** Stocks that often close unchanged (e.g. MTLA) show a high
AUC because "no change" counts as "not up", and the model learns to predict that. P(up) then
sits far below 0.55, the strategy almost never buys, and Exposure is tiny. For such stocks
try lower levels, e.g. `--threshold 0.45 --exit-threshold 0.40`.

### The signal

| Value | Meaning |
|---|---|
| **P(up)** | Model's probability that the next close is higher than today's |
| **smoothed** | P(up) averaged over the last 3 days; this is what the rule uses |
| **BUY** | Model was out, now in |
| **HOLD** | Model was in, stays in |
| **SELL** | Model was in, now out |
| **STAY OUT** | Model was out, stays out: don't own it, don't buy it |

These four come from the model's **own paper position**, which may differ from yours.

### The decision

`stock_cli.py` asks whether you own the stock and shows **two calls** for **your** position:

| You | Signal call (smoothed P(up)) | Forecast call (next close P50 vs last price) |
|---|---|---|
| don't own it | **BUY** above entry (0.55), else **WAIT** | **BUY** above +0.40% (round-trip cost), else **WAIT** |
| own it | **SELL** below exit (0.50), else **HOLD** | **SELL** below −0.25% (selling cost), else **HOLD** |

The forecast call uses whichever next-session model (LSTM or LightGBM) has the best test skill
on the close; if neither beats the baseline it has no view and says WAIT / HOLD.

**The decision (`>>>`) only trades when both calls agree.** If they disagree you keep your
current position (WAIT if out, HOLD if in), and both calls are logged so the record shows
which one is worth following.

**When you run it decides what it's for (WIB):**

| Run at | Decision is for | Extra lines |
|---|---|---|
| before 08:45 | today's session | today's expected open and close range |
| 09:00–16:00 | today's pre-closing (PROVISIONAL, on the live price) | today's close range from the live price; tomorrow's ranges (provisional) |
| 16:00–16:15 | nothing new: the closing auction is running | |
| after 16:15 | the next session | next session's open and close range |

**Prices:**
- **Main price = the last price.** The backtest assumes you trade at the close, so the closest real-world equivalent is the **pre-closing session (15:50–16:00 WIB)**.
- **Patient limit**: the next-session close median, rounded to the IDX tick. It's cheaper (BUY) or higher (SELL) than the last price but **may not fill**.
- **Ranges**: 80% ranges (P10–P90) for the open and close. Context, not targets.
- **Last forecast check**: the latest evening forecast against what actually happened.

**Speed:** heavy models train only on finished sessions and are cached once per close in
`src/output/cache/`. The first run after a close takes a few minutes; later runs that day take
seconds (about 15 s during the session, which adds a quick hourly model). `--refresh` retrains.

### LSTM forecast tables

Test-period evaluation (one row per days-ahead):

| Column | Meaning | Target |
|---|---|---|
| **skill calibrated** | Accuracy gain over the random-walk baseline | > 0% means it adds information; ≤ 2% everywhere = no real edge |
| **80% hit calibrated** | Share of actual closes inside P10–P90 | ~80% |
| **below P10 / above P90** | Share of closes outside each edge | ~10% each |
| **below P50** | Share of closes under the median | ~50% (much more or less = biased) |
| **width calibrated vs width RW** | Average band width, LSTM vs baseline | Narrower *and* still ~80% hit is better |

Forecast table: `lstm_p10..lstm_p90` is the 80% range for each future close; `rw_p10..rw_p90`
is the baseline. **If skill is ≤ 0, read the band as "how much this stock usually moves",
not as a prediction of direction.** Dates skip weekends but not IDX holidays.

### Sweep and hold-out

- **CAGR_1st / CAGR_2nd** (sweep): a setting that only works in one half was probably luck.
- **Rank correlation, selection vs test** (hold-out): ~0 means the settings that won in the first half did no better than random picks in the second. That's what this project found, and it's why signals are labelled "not validated".
- **Median test-period rank of the top picks**: compare with "random would be ~N/2".

### Intraday test

| Column | Meaning |
|---|---|
| **skill** | Accuracy gain over a time-of-day volatility random walk |
| **IC** | Rank correlation of the predicted move with the real move |
| **trades / win / avg net / total net** | Simulated trades when the predicted move exceeds the cost hurdle, net of fees and spread |
| **always long (cost drag)** | Buying every *h* hours regardless: what costs alone do to you |
| **base rate** | Share of real moves bigger than the cost hurdle |

The verdict line at the end says whether any model × horizon beats the baseline **and** makes money on ≥ 30 trades.

### Paper-trading log (`src/output/signals_log.csv`)

One row per stock, date and setting; re-running on the same day overwrites that row.
Columns: `bar_date, ticker, model, horizon, entry, exit, smooth, close, p_up, p_smoothed,
position, action, owned, advice, signal_advice, forecast_advice, forecast_model, forecast_move,
provisional, price_at, run_at`. `advice` is the decision; `signal_advice` and `forecast_advice`
are the two calls. Compare them with what the price actually did over the following days
before trusting either with money. Forecasts themselves go to `src/output/forecast_log.csv`.

---

## A daily routine

1. After **16:15 WIB**, run `python stock_cli.py` and enter your stocks. This trains and caches the day's models and logs the evening forecast.
2. Read the **SUMMARY** block: both calls, the decision, the prices.
3. Before the open or during the day, run it again whenever you like (seconds). During the session it re-checks the decision on the live price and logs a close update.
4. For BUY/SELL, place the order in the pre-closing session (15:50–16:00) or at the open.
5. Let the logs build up for a few weeks, then compare the decisions and forecasts with what happened.

To act **on the same day**, run it during **15:30–15:50 WIB**: the decision is provisional,
but its price is close to the closing price the backtest assumes.

---

## Limitations

- **Not validated.** The hold-out test found no setting that reliably carries forward. Backtest results overstate live results.
- **Data from Yahoo Finance**: intraday prices are delayed ~10 minutes; volumes can be incomplete; prices are dividend-adjusted.
- **Costs are assumptions** (0.15% buy / 0.25% sell). Change them with `--buy-cost` / `--sell-cost` (`bbca_walkforward.py`, `today_signal.py`, `sweep.py`, `holdout.py`, `intraday_test.py`), or edit the defaults in `Config` in `bbca_walkforward.py`, to match your broker.
- **Holidays**: forecast dates and "next session" skip weekends only.
- **Short histories**: the walk-forward backtest needs about 4 years of data, so recent IPOs can't be analysed.
- **Intraday trading** doesn't pay at IDX costs, per `intraday_test.py`.

---

## Project layout

```
trading-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # downloaded prices (git-ignored)
├── images/                  # generated charts (git-ignored)
└── src/
    ├── stock_cli.py         # interactive tool
    ├── bbca_walkforward.py  # engine: data, features, walk-forward, backtest, report
    ├── today_signal.py      # daily signal + live price + log
    ├── lstm_forecast.py     # LSTM range forecast
    ├── sweep.py             # parameter grid
    ├── holdout.py           # hold-out check of the grid
    ├── intraday_test.py     # hourly-model test
    └── output/              # results (git-ignored except signals_log.csv)
```
