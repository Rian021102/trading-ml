"""
Intraday forecast test: is an hourly model worth building?
==========================================================

Before turning lstm_forecast.py into an hourly forecaster, this checks whether
there is anything to find. On Yahoo's 60-minute bars (about 2 years) it
forecasts the return over the next 1, 2, 3 and 7 bars (7 bars ~ one IDX day)
with two models, both strictly out-of-sample:

  LightGBM  quantile regression (P10 / P50 / P90), walk-forward, refit every --retrain bars
  LSTM      the QuantileLSTM from lstm_forecast.py, trained once on data before the test

Both are scored against a random walk whose band comes from time-of-day
volatility (the 09:00 bar carries the overnight gap and is ~3x as volatile as
the rest). All three get the same rolling conformal calibration as
lstm_forecast.py, so the comparison is about information, not band width.

Then the question that matters: a long-only rule buys at a bar's close when the
model's median forecast clears the round-trip hurdle (buy fee + sell fee + one
price tick of spread) and sells h bars later. If that loses money after costs,
an hourly model doesn't pay, however good its forecast scores look.

    python intraday_test.py --ticker BBCA.JK
    python intraday_test.py --ticker BBRI.JK --hours 1 2 --same-day    # day trades only, no overnight
    python intraday_test.py --ticker BBCA.JK --no-lstm --offline

Horizons are in bars; bars are hourly but forecasts cross the lunch break and
the overnight gap unless --same-day. Volume is not used: Yahoo reports 0 for
most 09:00 bars.

Outputs: data/<ticker>_60m.csv, output/intraday_test_<ticker>.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import bbca_walkforward as wf
import lstm_forecast as lf

JKT = "Asia/Jakarta"
QUANTILES = lf.QUANTILES
Z90 = lf.Z90


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_hourly(ticker: str, data_dir: str, offline: bool) -> pd.DataFrame:
    """60-minute OHLCV in WIB. Yahoo keeps ~730 days of hourly history."""
    path = Path(data_dir) / f"{ticker.lstrip('^')}_60m.csv"
    if offline:
        if not path.exists():
            raise SystemExit(f"{path} not found - run without --offline once to download it")
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(JKT)
    else:
        import yfinance as yf

        df = yf.download(ticker, period="730d", interval="60m", auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"No hourly data returned for {ticker}.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = df.index.tz_convert(JKT)
        df.index.name = "Datetime"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        print(f"Saved {ticker} hourly data to {path}")
    # 12:00 bars are a few stray minutes after session I closes; dropping them
    # folds their move into the 13:00 bar's close-to-close return
    return df[df.index.hour != 12].sort_index()


def make_features(px: pd.DataFrame, mkt: pd.DataFrame | None) -> pd.DataFrame:
    """Per-bar inputs, all known at the close of bar t."""
    c = px["Close"]
    lc = np.log(c)
    r = lc.diff()
    date = pd.Series(px.index.date, index=px.index)
    f = pd.DataFrame(index=px.index)
    f["ret"] = r
    for k in (2, 3, 7, 14, 35):
        f[f"ret_{k}b"] = lc - lc.shift(k)
    f["vol_7b"] = r.rolling(7).std()
    f["vol_35b"] = r.rolling(35).std()
    rng = np.log(px["High"] / px["Low"])
    f["range"] = rng
    f["range_7b"] = rng.rolling(7).mean()
    f["close_loc"] = ((c - px["Low"]) / (px["High"] - px["Low"]).replace(0, np.nan)).fillna(0.5)
    lo, hi = c.rolling(35).min(), c.rolling(35).max()
    f["pos_35b"] = ((c - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)
    f["ret_today"] = np.log(c / px["Open"].groupby(date).transform("first"))
    f["ret_vs_prev_close"] = np.log(c / date.map(c.groupby(date).last().shift(1)))
    f["hour"] = px.index.hour
    f["dow"] = px.index.dayofweek
    f["bars_left"] = c.groupby(date).cumcount(ascending=False)
    if mkt is not None:
        m = np.log(mkt["Close"].reindex(px.index).ffill())
        mr = m.diff()
        f["mkt_ret"] = mr
        f["mkt_ret_7b"] = m - m.shift(7)
        f["mkt_vol_35b"] = mr.rolling(35).std()
        f["rel_7b"] = f["ret_7b"] - f["mkt_ret_7b"]
    return f.replace([np.inf, -np.inf], np.nan)


def tick_size(price: np.ndarray) -> np.ndarray:
    """IDX price fraction (Rp) for regular-market stocks."""
    return np.select([price < 200, price < 500, price < 2000, price < 5000], [1, 2, 5, 10], 25)


# ----------------------------------------------------------------------------
# Forecasts: (n, H, 3) arrays of P10 / P50 / P90 cumulative log return in %
# ----------------------------------------------------------------------------
def rw_seasonal(r: pd.Series, hour: pd.Series, H: int, n_obs: int = 20) -> np.ndarray:
    """Random walk, zero drift. Variance of each future bar is the mean squared
    return of that time-of-day slot over its last n_obs occurrences up to t."""
    slots = sorted(hour.unique())
    V = np.column_stack([(r[hour == s] ** 2).rolling(n_obs, min_periods=10).mean()
                         .reindex(r.index).ffill().values for s in slots])
    slot = hour.map({s: i for i, s in enumerate(slots)}).values
    n = len(r)
    var, acc = np.full((n, H), np.nan), np.zeros(n)
    for k in range(1, H + 1):
        add = np.full(n, np.nan)
        add[: n - k] = V[np.arange(n - k), slot[k:]]
        acc = acc + add
        var[:, k - 1] = acc
    s = np.sqrt(var) * 100
    return np.stack([-Z90 * s, np.zeros_like(s), Z90 * s], axis=-1)


def lgbm_walk_forward(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, first_oos: int,
                      hours: list[int], retrain: int, seed: int) -> np.ndarray:
    """Expanding-window quantile LightGBM, one model per horizon x quantile.
    A row is used for training only once its target is resolved (t + h <= start)."""
    from lightgbm import LGBMRegressor

    n, H = Y.shape
    out = np.full((n, H, 3), np.nan)
    starts = list(range(first_oos, n, retrain))
    t0 = time.time()
    for j, start in enumerate(starts):
        test = np.arange(start, min(start + retrain, n))
        test = test[valid[test]]
        for h in hours:
            tr = np.arange(start - h + 1)
            tr = tr[valid[tr] & ~np.isnan(Y[tr, h - 1])]
            for k, q in enumerate(QUANTILES):
                m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.03,
                                  num_leaves=15, max_depth=4, min_child_samples=50, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
                m.fit(X[tr], Y[tr, h - 1])
                out[test, h - 1, k] = m.predict(X[test])
        if (j + 1) % 10 == 0 or j + 1 == len(starts):
            print(f"  LightGBM refit {j + 1}/{len(starts)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def lstm_once(feats: pd.DataFrame, Y: np.ndarray, valid: np.ndarray, first_oos: int,
              L: int, seeds: list[int]) -> np.ndarray:
    """QuantileLSTM ensemble trained on targets resolved before first_oos
    (last 20% of that as early-stopping validation), then frozen."""
    import torch

    n, H = Y.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w0 = int(np.argmax(valid))
    rows = np.arange(n)[valid & ~np.isnan(Y).any(axis=1)]
    pre = rows[(rows >= w0 + L - 1) & (rows < first_oos - H)]
    n_va = int(len(pre) * 0.2)
    va, tr = pre[len(pre) - n_va:], pre[: len(pre) - n_va - H]
    mu, sd = feats.iloc[: tr[-1] + 1].mean(), feats.iloc[: tr[-1] + 1].std().replace(0, 1)
    X = ((feats - mu) / sd).fillna(0).values
    win = lf.Windows(X, Y, L, device)
    t0 = time.time()
    models = lf.ensemble(win, tr, va, X.shape[1], H, seeds)
    print(f"  LSTM: {len(seeds)} seeds on {len(tr)} train / {len(va)} val bars, {device} ({time.time() - t0:.0f}s)")
    out = lf.predict_all(models, win, n, L, H)
    out[:first_oos] = np.nan                      # in-sample, never scored
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def pinball(pred: np.ndarray, y: np.ndarray) -> float:
    q = np.array(QUANTILES)
    e = y[:, None] - pred
    return float(np.maximum(q * e, (q - 1) * e).mean())


def simulate(go: np.ndarray, y: np.ndarray, cost: np.ndarray, h: int, rows: np.ndarray) -> np.ndarray:
    """Non-overlapping long trades: buy at close t where go[t], sell at close t+h.
    Returns the net simple return of each trade."""
    nets, nxt = [], -1
    for t in rows:
        if t >= nxt and go[t]:
            nets.append(np.expm1(y[t] / 100) - cost[t])
            nxt = t + h
    return np.array(nets)


def score(name: str, cal: np.ndarray, rw: np.ndarray, Y: np.ndarray, rows: np.ndarray,
          cost: np.ndarray, hurdle: np.ndarray, h: int) -> dict:
    i = h - 1
    p, y = cal[rows, i], Y[rows, i]
    nz = y != 0
    nets = simulate(cal[:, i, 1] > hurdle, Y[:, i], cost, h, rows)
    return {
        "hours": h, "model": name,
        "skill": 1 - pinball(p, y) / pinball(rw[rows, i], y),
        "80% hit": ((y >= p[:, 0]) & (y <= p[:, 2])).mean(),
        "IC": pd.Series(p[:, 1]).corr(pd.Series(y), method="spearman"),
        "dir hit": (np.sign(p[nz, 1]) == np.sign(y[nz])).mean(),
        **trade_stats(nets),
    }


def trade_stats(nets: np.ndarray) -> dict:
    return {"trades": len(nets),
            "win": (nets > 0).mean() if len(nets) else np.nan,
            "avg net": nets.mean() if len(nets) else np.nan,
            "total net": nets.sum() if len(nets) else 0.0}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=wf.Config.ticker)
    ap.add_argument("--market", default=wf.Config.market, help="market index ticker, or 'none'")
    ap.add_argument("--hours", type=int, nargs="+", default=[1, 2, 3, 7], help="horizons in bars")
    ap.add_argument("--retrain", type=int, default=70, help="LightGBM refit every N bars (70 ~ 2 weeks)")
    ap.add_argument("--lookback", type=int, default=35, help="bars the LSTM sees (35 ~ 1 week)")
    ap.add_argument("--seeds", type=int, default=3, help="LSTM ensemble size")
    ap.add_argument("--no-lstm", action="store_true", help="skip the LSTM (LightGBM + baseline only)")
    ap.add_argument("--oos-frac", type=float, default=0.50,
                    help="forecasts are out-of-sample from this fraction of the history on")
    ap.add_argument("--test-frac", type=float, default=0.35,
                    help="score the last this-much of the history (the gap before it warms up calibration)")
    ap.add_argument("--calib-window", type=int, default=350, help="bars of recent errors used to calibrate")
    ap.add_argument("--gamma", type=float, default=0.005, help="adaptive-conformal step size")
    ap.add_argument("--same-day", action="store_true",
                    help="only score/trade forecasts that end the same day (no overnight holding)")
    ap.add_argument("--spread-ticks", type=float, default=1.0, help="ticks of spread paid per round trip")
    ap.add_argument("--buy-cost", type=float, default=wf.Config.buy_cost)
    ap.add_argument("--sell-cost", type=float, default=wf.Config.sell_cost)
    ap.add_argument("--offline", action="store_true", help="use data/<ticker>_60m.csv instead of downloading")
    ap.add_argument("--out", default=wf.Config.out_dir)
    a = ap.parse_args()

    hours = sorted(set(a.hours))
    H = max(hours)
    px = load_hourly(a.ticker, wf.Config.data_dir, a.offline)
    mkt = None if a.market.lower() == "none" else load_hourly(a.market, wf.Config.data_dir, a.offline)
    feats = make_features(px, mkt)
    valid = feats.notna().all(axis=1).values
    close = px["Close"]
    Y = lf.make_targets(close, H)
    n = len(px)
    first_oos, test0 = int(n * a.oos_frac), int(n * (1 - a.test_frac))
    if test0 - first_oos < a.calib_window // 2:
        raise SystemExit("--oos-frac / --test-frac leave too little calibration warm-up before the test")

    price = close.values
    cost = a.buy_cost + a.sell_cost + a.spread_ticks * tick_size(price) / price
    hurdle = np.log1p(cost) * 100                       # in the same units as Y (log %)
    dates = px.index.date
    idx = px.index
    print(f"{a.ticker}: {n} hourly bars {idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d} | "
          f"slots {', '.join(sorted(set(idx.strftime('%H:%M'))))}")
    print(f"out-of-sample from {idx[first_oos]:%Y-%m-%d}, scored {idx[test0]:%Y-%m-%d} -> end "
          f"({n - test0} bars) | horizons {hours} bars{' | same-day only' if a.same_day else ''}")
    print(f"round-trip hurdle: fees {a.buy_cost + a.sell_cost:.2%} + {a.spread_ticks:g} tick "
          f"= {np.median(cost[test0:]):.2%} median over the test period\n")

    # ---- forecasts
    rw = rw_seasonal(feats["ret"], feats["hour"], H)
    raw = {"LightGBM": lgbm_walk_forward(feats.values, Y, valid, first_oos, hours, a.retrain, wf.Config.seed)}
    if not a.no_lstm:
        raw["LSTM"] = lstm_once(feats, Y, valid, first_oos, a.lookback, list(range(a.seeds)))
    raw["RW (time-of-day vol)"] = rw

    sigma = feats["ret"].rolling(35).std().values * 100
    cal = {k: lf.apply_calibration(p, lf.rolling_adjust(p, Y, sigma, first_oos, a.calib_window, a.gamma), sigma)
           for k, p in raw.items()}
    rw_cal = cal["RW (time-of-day vol)"]

    # ---- score
    rows_all = np.arange(test0, n)
    res = []
    for h in hours:
        ok = ~np.isnan(Y[rows_all, h - 1]) & valid[rows_all]
        if a.same_day:
            ok &= dates[np.minimum(rows_all + h, n - 1)] == dates[rows_all]
        rows = rows_all[ok]
        if len(rows) < 30:
            print(f"  skipping {h}-bar horizon: only {len(rows)} scorable bars")
            continue
        for name, c in cal.items():
            res.append(score(name, c, rw_cal, Y, rows, cost, hurdle, h))
        always = simulate(np.ones(n, bool), Y[:, h - 1], cost, h, rows)
        res.append({"hours": h, "model": "always long (cost drag)", **trade_stats(always),
                    "base rate": (np.abs(Y[rows, h - 1]) > hurdle[rows]).mean(),
                    "median |move|": np.median(np.abs(np.expm1(Y[rows, h - 1] / 100)))})
    res = pd.DataFrame(res)

    show = res.copy().astype(object)
    fmts = {"skill": "{:+.1%}", "80% hit": "{:.0%}", "IC": "{:+.3f}", "dir hit": "{:.1%}", "trades": "{:.0f}",
            "win": "{:.0%}", "avg net": "{:+.2%}", "total net": "{:+.1%}", "base rate": "{:.0%}",
            "median |move|": "{:.2%}"}
    for col, f in fmts.items():
        show[col] = res[col].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    print(f"\nTest period, after calibration ({a.ticker}):")
    print(show.to_string(index=False))
    print("""
  skill      pinball-loss gain vs the time-of-day random walk (>0 = adds information)
  80% hit    share of outcomes inside P10..P90 (target 80%)
  IC         rank correlation of the median forecast with the actual move; dir hit = sign right (non-zero moves)
  trades     buy when median forecast > hurdle, sell h bars later, no overlapping trades; net of fees + spread
  always     buy every h bars regardless: the cost drag a model has to overcome
  base rate  share of actual h-bar moves bigger than the hurdle in either direction""")

    # ---- verdict
    models = res[~res["model"].str.startswith(("always", "RW"))]
    good = models[(models["skill"] > 0.01) & (models["trades"] >= 30) & (models["total net"] > 0)]
    print()
    if good.empty:
        print("Verdict: no model x horizon both beats the random walk by >1% and makes money after costs "
              "on >= 30 trades. An hourly forecaster isn't worth building on this data.")
    else:
        print("Verdict: worth a closer look (still one test period - re-check on other tickers / splits):")
        print(show.loc[good.index, ["hours", "model", "skill", "trades", "win", "avg net", "total net"]]
              .to_string(index=False))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"intraday_test_{a.ticker.lstrip('^')}.csv"
    res.to_csv(path, index=False, float_format="%.5f")
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
