"""
Intraday close forecast (phase 2 of next_day.py)
================================================

During the session the open is known and the price has moved, so the evening
forecast for today's close can be sharpened. This module forecasts

    rest = log(today's close / latest price)

as P10 / P50 / P90, from any moment between 09:00 and 16:00 WIB. next_day.py
calls it automatically when run during the session and logs every run, so
the scorecard can show how the error shrinks through the day.

Two predictors:

  Baseline  close = latest price, band = +/- the volatility still to come today:
            the sum of each remaining hourly slot's recent variance (the 09:00
            hour and the 15:50-16:00 closing auction are the noisy ones), with
            the slot in progress pro-rated by the minutes left
  LightGBM  quantile regression on the move still to come in units of that
            volatility, from Yahoo's hourly bars (~2 years): time of day, move
            since the open and since yesterday's close, recent hourly returns
            and ranges, IHSG. Retrained every 5 sessions, then recalibrated on
            the last 60 sessions' misses, same idea as next_day.py

Training rows are "checkpoints": the open, and the end of each hourly bar
(10:00, 11:00, lunch, 14:00, 15:00, 15:50 pre-close). A live run between
checkpoints uses the last one's features with the live price, and the exact
time left for the band.

Yahoo's hourly bars are NOT dividend-adjusted (the daily ones are), so all
intraday arithmetic stays inside the hourly data; only raw prices from today
are compared with the daily log.
"""

from __future__ import annotations

from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

import lstm_forecast as lf
from intraday_test import load_hourly, make_features as hourly_features

# Yahoo hourly bar start hours per session day (False = Mon-Thu, True = Friday); 16 = closing auction
SLOTS = {False: (9, 10, 11, 13, 14, 15, 16), True: (9, 10, 11, 14, 15, 16)}
CHECKS = (None, 9, 10, 11, 13, 14, 15)             # None = at the open, else after that hour's bar
LABEL = {None: "09:00 open", 9: "10:00", 10: "11:00", 11: "lunch", 13: "14:00", 14: "15:00", 15: "15:50 pre-close"}


def slot_window(h: int, fri: bool) -> tuple[dtime, dtime]:
    """Trading time covered by Yahoo's hourly bar that starts at hour h."""
    return {9: (dtime(9), dtime(10)), 10: (dtime(10), dtime(11)),
            11: (dtime(11), dtime(11, 30) if fri else dtime(12)),
            13: (dtime(13, 30), dtime(14)), 14: (dtime(14), dtime(15)),
            15: (dtime(15), dtime(15, 50)), 16: (dtime(15, 50), dtime(16))}[h]


def minutes(t: dtime) -> int:
    return t.hour * 60 + t.minute


# ----------------------------------------------------------------------------
# Volatility still to come
# ----------------------------------------------------------------------------
def slot_variance(h: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Day x slot: mean squared return (%^2) of that slot over the last n days up to
    and including the day (shift by one day before using it for that day)."""
    date = h.index.date
    lc = np.log(h["Close"])
    r = lc.groupby(date).diff()
    first = r.isna().values
    r[first] = (lc - np.log(h["Open"]))[first]           # first bar: open -> its close
    tbl = pd.DataFrame({"d": date, "s": h.index.hour, "r2": (r * 100) ** 2})
    tbl = tbl.pivot_table(index="d", columns="s", values="r2")
    return tbl.rolling(n, min_periods=10).mean()


def remaining_var(v: pd.Series, fri: bool, t: dtime) -> float:
    """Expected variance (%^2) from time t to the close: the slot in progress pro-rated,
    plus every later slot. Lunch break counts as nothing in progress."""
    v = v.fillna(v.mean())
    out = 0.0
    for s in SLOTS[fri]:
        a, b = slot_window(s, fri)
        if t >= b:
            continue
        left = 1.0 if t <= a else (minutes(b) - minutes(t)) / (minutes(b) - minutes(a))
        out += left * v.get(s, v.mean())
    return out


# ----------------------------------------------------------------------------
# Checkpoint rows
# ----------------------------------------------------------------------------
def checkpoint_x(frow: pd.Series, k: int | None, fri: bool, price: float, open_: float,
                 prev_close: float, sig: float) -> pd.Series:
    """Model inputs at a checkpoint: the hourly features of the last finished bar,
    with the move-so-far and time-of-day fields set for this moment."""
    x = frow.copy()
    x["hour"] = 8 if k is None else k
    x["bars_left"] = sum(s > (k or 8) for s in SLOTS[fri])
    x["ret_today"] = np.log(price / open_)
    x["ret_vs_prev_close"] = np.log(price / prev_close)
    x["sig_rem"] = sig
    return x


def checkpoints(h: pd.DataFrame, mkt_h: pd.DataFrame | None) -> pd.DataFrame:
    """One row per finished day x checkpoint, with target y = log(close / price) in %."""
    f = hourly_features(h, mkt_h)
    date = pd.Series(h.index.date, index=h.index)
    days = sorted(date.unique())
    close_day = h["Close"].groupby(date).last()            # closing-auction price
    open_day = h["Open"].groupby(date).first()
    v = slot_variance(h).shift(1)                          # known before the day opens
    last_bar = h.index.to_series().groupby(date).last()
    rows = []
    for d_prev, d in zip(days, days[1:]):
        if d not in v.index or v.loc[d].isna().all():
            continue
        fri = pd.Timestamp(d).dayofweek == 4
        bars = h[date == d]
        prev_c, open_, close = close_day[d_prev], open_day[d], close_day[d]
        stops = [(None, open_, f.loc[last_bar[d_prev]], dtime(9))]
        stops += [(ts.hour, bars.at[ts, "Close"], f.loc[ts], slot_window(ts.hour, fri)[1])
                  for ts in bars.index if ts.hour in CHECKS and ts.hour in SLOTS[fri]]
        for k, price, frow, t in stops:
            sig = np.sqrt(remaining_var(v.loc[d], fri, t))
            if not sig > 0:
                continue
            x = checkpoint_x(frow, k, fri, price, open_, prev_c, sig)
            rows.append({"day": d, "k": -1 if k is None else k, "price": price, "sig": sig,
                         "y": np.log(close / price) * 100, **x.to_dict()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
NOT_X = ("day", "k", "price", "sig", "y")


def fit_models(X_tr: pd.DataFrame, z_tr: np.ndarray, seed: int) -> list:
    from lightgbm import LGBMRegressor

    return [LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.03,
                          num_leaves=15, max_depth=4, min_child_samples=50, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1).fit(X_tr, z_tr)
            for q in lf.QUANTILES]


def predict_z(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.sort(np.column_stack([m.predict(X) for m in models]), axis=1)


def conformal(days: np.ndarray, z: np.ndarray, zp: np.ndarray, oos_days: list, window: int) -> dict:
    """Shift per day (in vol units) that would have put each quantile at its target hit
    rate over the previous `window` out-of-sample days. Key None = for the next day."""
    e = z[:, None] - zp
    shift = {}
    for i, d in enumerate([*oos_days, None]):
        past = np.isin(days, oos_days[max(0, i - window): i])
        shift[d] = (np.array([np.nanquantile(e[past, k], q) for k, q in enumerate(lf.QUANTILES)])
                    if i >= window // 3 else np.zeros(len(lf.QUANTILES)))
    return shift


def evaluate(rows: pd.DataFrame, preds: dict[str, np.ndarray], test: np.ndarray) -> pd.DataFrame:
    y = rows["y"].values[test]
    out = []
    for k in CHECKS:
        sel = rows["k"].values[test] == (-1 if k is None else k)
        if not sel.any():
            continue
        loss = {m: lf.pinball_np(p[test][sel][:, None, :], y[sel][:, None])[0] for m, p in preds.items()}
        for m, p in preds.items():
            p = p[test][sel]
            out.append({
                "checkpoint": LABEL[k], "model": m,
                "skill": 1 - loss[m] / loss["Baseline"],
                "avg error": np.abs(y[sel] - p[:, 1]).mean() / 100,
                "inside 80%": ((y[sel] >= p[:, 0]) & (y[sel] <= p[:, 2])).mean(),
                "80% width": (p[:, 2] - p[:, 0]).mean() / 100,
            })
    return pd.DataFrame(out).set_index(["checkpoint", "model"])


# ----------------------------------------------------------------------------
# Main computation
# ----------------------------------------------------------------------------
def load(ticker: str, market: str | None, data_dir: str, offline: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Hourly bars for the stock and the market index (includes today's bars so far)."""
    return load_hourly(ticker, data_dir, offline), (load_hourly(market, data_dir, offline) if market else None)


def fit(h_all: pd.DataFrame, mkt_h: pd.DataFrame | None, today, label: str, test_days: int = 120,
        calib_days: int = 60, retrain: int = 5, seed: int = 42) -> dict:
    """Walk-forward test and final models on the days before `today`. Cache it per day;
    predict() applies it to the live price."""
    h = h_all[h_all.index.date < today]
    rows = checkpoints(h, mkt_h)
    cols = [c for c in rows.columns if c not in NOT_X]
    X, z = rows[cols], (rows["y"] / rows["sig"]).values
    days = rows["day"].values
    all_days = sorted(rows["day"].unique())
    if len(all_days) < test_days + calib_days + 120:
        raise ValueError(f"{label}: only {len(all_days)} days of hourly data.")
    oos_days = all_days[-test_days - calib_days:]
    zp = np.full((len(rows), len(lf.QUANTILES)), np.nan)
    for i in range(0, len(oos_days), retrain):
        tr, te = days < oos_days[i], np.isin(days, oos_days[i:i + retrain])
        zp[te] = predict_z(fit_models(X[tr], z[tr], seed), X[te])
    shift = conformal(days, z, zp, oos_days, calib_days)
    adj = np.array([shift.get(d, np.zeros(3)) for d in days])
    sig = rows["sig"].values[:, None]
    preds = {"LightGBM": np.sort((zp + adj) * sig, axis=1),
             "Baseline": np.array([-lf.Z90, 0, lf.Z90])[None, :] * sig}
    test = np.isin(days, all_days[-test_days:])
    return {"label": label, "cols": cols, "models": fit_models(X, z, seed), "shift": shift[None],
            "v_today": slot_variance(h).iloc[-1], "ev": evaluate(rows, preds, test),
            "n_test": int(len(all_days[-test_days:]))}


def predict(state: dict, h_all: pd.DataFrame, mkt_h: pd.DataFrame | None, live_at: pd.Timestamp,
            price: float, open_: float, prev_close: float) -> dict:
    """Today's close as P10 / P50 / P90 from `price` at `live_at` (WIB)."""
    today, fri = live_at.date(), live_at.dayofweek == 4
    h = h_all[h_all.index.date < today]
    done = [ts for ts in h_all.index[h_all.index.date == today]
            if ts.hour in SLOTS[fri] and ts.hour != 16 and slot_window(ts.hour, fri)[1] <= live_at.time()]
    sig_now = np.sqrt(remaining_var(state["v_today"], fri, live_at.time()))
    f_now = hourly_features(pd.concat([h, h_all.loc[done]]), mkt_h)
    k = done[-1].hour if done else None
    x_now = checkpoint_x(f_now.iloc[-1], k, fri, price, open_, prev_close, sig_now)
    z_now = predict_z(state["models"], x_now[state["cols"]].to_frame().T.astype(float))[0] + state["shift"]
    q = {"LightGBM": np.sort(z_now) * sig_now, "Baseline": np.array([-lf.Z90, 0, lf.Z90]) * sig_now}
    return {
        "label": state["label"], "live_at": live_at, "price": price, "open": open_, "prev_close": prev_close,
        "checkpoint": LABEL[k], "sig_rem": sig_now / 100,
        "now": {m: price * np.exp(v / 100) for m, v in q.items()},        # close P10 / P50 / P90 prices
        "ev": state["ev"], "n_test": state["n_test"],
    }
