"""
Next-session open & close forecast, with a scorecard
====================================================

Run it after the close (16:15 WIB). For the next session it forecasts:

  gap    = log(next open  / today's close)    -> next OPEN price
  intra  = log(next close / next open)        -> kept for the intraday update (phase 2)
  close  = log(next close / today's close)    -> next CLOSE price

each as a P10 / P50 / P90 range, from three models:

  LSTM      quantile LSTM on the last 60 days (lstm_forecast.QuantileLSTM)
  LightGBM  quantile regression on the 27 bbca_walkforward features
  Baseline  no change, band = +/- recent volatility of that move (the bar to beat)

Every model's band is recalibrated on its own recent misses (adaptive
conformal inference, same as lstm_forecast.py), so "80% range" stays honest.

The forecast goes into output/forecast_log.csv. The next run fills in what
actually happened and prints how far off each model was, plus a scorecard over
the last 20 / 60 scored sessions. That log is the real test: unlike a backtest
it can't peek at the future.

Log rules: one row per close date x model. A row can be overwritten until
08:45 WIB of the next weekday (the pre-opening auction), then it is locked.
Runs on an unfinished bar (market open, or before 16:15) are shown but not
logged. Prices are stored as returns plus the close they start from, so
Yahoo's dividend re-adjustment of old prices doesn't shift the comparison.

    python next_day.py                      # BBCA.JK
    python next_day.py --ticker BBRI.JK --seeds 2
    python next_day.py --offline --no-log   # saved CSVs, don't touch the log

Takes ~2-3 min (LSTM ensemble on a GPU, LightGBM walk-forward on the CPU).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import bbca_walkforward as wf
import lstm_forecast as lf
import today_signal as ts
from intraday_test import tick_size

TARGETS = ("gap", "intra", "close")          # next-session log returns, in %
MODELS = ("LSTM", "LightGBM", "Baseline")
PRE_OPEN = dtime(8, 45)                      # IDX pre-opening auction starts; log rows lock here
LOG = Path(__file__).resolve().parent / "output" / "forecast_log.csv"
LOG_KEYS = ["bar_date", "ticker", "model", "stage"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def realized(px: pd.DataFrame) -> pd.DataFrame:
    """Each day's gap / intraday / close-to-close log return in %, known at its close."""
    lo, lc = np.log(px["Open"]), np.log(px["Close"])
    return pd.DataFrame({"gap": lo - lc.shift(1), "intra": lc - lo, "close": lc - lc.shift(1)}) * 100


def prepare(px: pd.DataFrame, mkt: pd.DataFrame | None, since: str) -> dict:
    """Features for both models, next-session targets and trailing volatility,
    on a common index that starts once every feature is available."""
    px = px[px["Volume"] > 0]                   # zero-volume rows are holidays / stale Yahoo bars
    tree = wf.build_features(px, mkt, 1).drop(columns=["ret", "target", "fwd_ret"])
    seq = lf.make_features(px, mkt)
    real = realized(px)
    keep = (px.index >= pd.Timestamp(since)) & tree.notna().all(axis=1).values & seq.notna().all(axis=1).values
    idx = px.index[int(np.argmax(keep)):]
    return {
        "px": px.loc[idx],
        "tree": tree.loc[idx],
        "seq": seq.loc[idx].ffill().fillna(0),
        "Y": real.shift(-1).loc[idx, list(TARGETS)].values,                    # (n, 3), NaN on the last row
        "sigma": real.rolling(63).std().loc[idx, list(TARGETS)].values,        # (n, 3) daily vol in %
    }


# ----------------------------------------------------------------------------
# Models: every predictor returns (n, 3 targets, 3 quantiles) of log returns in %
# ----------------------------------------------------------------------------
def lstm_predict(seq: pd.DataFrame, Y: np.ndarray, L: int, tr, va, tr_f, va_f, seeds) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble trained before the evaluation period -> predictions for every day;
    a second ensemble refit on all data -> today's prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu, sd = seq.iloc[: tr[-1] + 1].mean(), seq.iloc[: tr[-1] + 1].std().replace(0, 1)
    X = ((seq - mu) / sd).values
    win = lf.Windows(X, Y, L, device)
    k = len(TARGETS)                            # the LSTM's "days ahead" axis carries the 3 targets
    models = lf.ensemble(win, tr, va, X.shape[1], k, seeds)
    oos = lf.predict_all(models, win, len(seq), L, k)
    final = lf.ensemble(win, tr_f, va_f, X.shape[1], k, seeds)
    return oos, lf.predict(final, win, [len(seq) - 1])[0]


def lgbm_fit_predict(F: pd.DataFrame, Y: np.ndarray, train_idx, pred_idx, seed: int) -> np.ndarray:
    from lightgbm import LGBMRegressor

    out = np.empty((len(pred_idx), len(TARGETS), len(lf.QUANTILES)))
    for j in range(len(TARGETS)):
        for k, q in enumerate(lf.QUANTILES):
            m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.03,
                              num_leaves=15, max_depth=4, min_child_samples=50, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1)
            m.fit(F.iloc[train_idx], Y[train_idx, j])
            out[:, j, k] = m.predict(F.iloc[pred_idx])
    return np.sort(out, axis=-1)


def lgbm_walk_forward(F: pd.DataFrame, Y: np.ndarray, valid: np.ndarray, first: int,
                      retrain: int, seed: int) -> np.ndarray:
    """Expanding window from day `first`, retrained every `retrain` days on the days
    whose outcome was known; the last day (today) gets a model fit on everything."""
    n = len(F)
    pred = np.full((n, len(TARGETS), len(lf.QUANTILES)), np.nan)
    for s in range(first, n, retrain):
        pred[s:min(s + retrain, n)] = lgbm_fit_predict(F, Y, valid[valid < s], np.arange(s, min(s + retrain, n)), seed)
    pred[n - 1] = lgbm_fit_predict(F, Y, valid, [n - 1], seed)[0]
    return pred


def baseline(sigma: np.ndarray) -> np.ndarray:
    """No change expected; 80% band from trailing volatility of each move."""
    s = lf.Z90 * sigma
    return np.stack([-s, np.zeros_like(s), s], axis=-1)


def conformal_shift(pred: np.ndarray, Y: np.ndarray, sigma: np.ndarray, first: int,
                    window: int, gamma: float) -> np.ndarray:
    """Per-target shift (in vol units) learned from resolved misses only; see lf.rolling_adjust."""
    adj = np.zeros_like(pred)
    for j in range(len(TARGETS)):
        adj[:, [j]] = lf.rolling_adjust(pred[:, [j]], Y[:, [j]], sigma[:, j], first, window, gamma)
    return adj


def shifted(pred: np.ndarray, adj: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.sort(pred + np.nan_to_num(adj) * sigma[..., None], axis=-1)


# ----------------------------------------------------------------------------
# Evaluation on the held-out period
# ----------------------------------------------------------------------------
def evaluate(preds: dict[str, np.ndarray], Y: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    y = Y[rows]
    loss = {m: lf.pinball_np(p[rows], y) for m, p in preds.items()}
    out = []
    for t, label in (("gap", "open"), ("close", "close")):
        j = TARGETS.index(t)
        for m, p in preds.items():
            p = p[rows, j]
            moved = (y[:, j] != 0) & (p[:, 1] != 0)
            out.append({
                "price": label, "model": m,
                "skill": 1 - loss[m][j] / loss["Baseline"][j],
                "avg error": np.abs(y[:, j] - p[:, 1]).mean() / 100,
                "direction hit": (np.sign(y[moved, j]) == np.sign(p[moved, 1])).mean() if moved.any() else np.nan,
                "inside 80%": ((y[:, j] >= p[:, 0]) & (y[:, j] <= p[:, 2])).mean(),
                "below P50": (y[:, j] < p[:, 1]).mean(),
            })
    return pd.DataFrame(out).set_index(["price", "model"])


# ----------------------------------------------------------------------------
# Main computation
# ----------------------------------------------------------------------------
def run(px: pd.DataFrame, mkt: pd.DataFrame | None, label: str, L: int = 60, n_seeds: int = 3,
        test_days: int = 500, calib_window: int = 250, gamma: float = 0.005, retrain: int = 21,
        since: str = "2013-01-01", seed: int = 42) -> dict:
    """Evaluate all models on the last `test_days` closes, then forecast the next session."""
    d = prepare(px, mkt, since)
    Y, sigma, n = d["Y"], d["sigma"], len(d["Y"])
    valid = np.arange(L - 1, n)[~np.isnan(Y[L - 1:]).any(axis=1)]
    if len(valid) < test_days + calib_window + 500:
        raise ValueError(f"{label}: {len(valid)} usable days; need at least {test_days + calib_window + 500}.")
    te = valid[-test_days:]
    first = valid[-test_days - calib_window]         # first out-of-sample day: calibration warm-up starts
    pre = valid[valid < first]
    n_va = int(len(pre) * 0.15)
    n_va_f = int(len(valid) * 0.15)
    idx = d["px"].index
    print(f"{label}: {n} days from {idx[0]:%Y-%m-%d} | train ..{idx[pre[-1]]:%Y-%m-%d} | "
          f"calibration warm-up {idx[first]:%Y-%m-%d} | test {idx[te[0]]:%Y-%m-%d}..{idx[te[-1]]:%Y-%m-%d}")

    t0 = time.time()
    print(f"  LSTM ({n_seeds} seeds x 2 fits) ...", flush=True)
    lstm_oos, lstm_now = lstm_predict(d["seq"], Y, L, pre[:-n_va], pre[-n_va:],
                                      valid[: len(valid) - n_va_f], valid[-n_va_f:], list(range(n_seeds)))
    print(f"  LightGBM walk-forward ({len(TARGETS) * len(lf.QUANTILES)} quantile models, "
          f"retrained every {retrain} days) ...", flush=True)
    lgbm = lgbm_walk_forward(d["tree"], Y, valid, first, retrain, seed)
    t_fit = time.time() - t0

    # calibrate on resolved misses; today's LSTM forecast (refit model) borrows the
    # shift learned from the first ensemble's out-of-sample errors
    adj_lstm = conformal_shift(lstm_oos, Y, sigma, first, calib_window, gamma)
    adj_lgbm = conformal_shift(lgbm, Y, sigma, first, calib_window, gamma)
    preds = {
        "LSTM": shifted(lstm_oos, adj_lstm, sigma),
        "LightGBM": shifted(lgbm, adj_lgbm, sigma),
        "Baseline": baseline(sigma),
    }
    now = {
        "LSTM": shifted(lstm_now[None], adj_lstm[[n - 1]], sigma[[n - 1]])[0],
        "LightGBM": preds["LightGBM"][n - 1],
        "Baseline": preds["Baseline"][n - 1],
    }
    return {"label": label, "px": d["px"], "ev": evaluate(preds, Y, te), "now": now,
            "n_test": len(te), "t_fit": t_fit}


# ----------------------------------------------------------------------------
# Forecast rows, log, scorecard
# ----------------------------------------------------------------------------
def forecast_rows(r: dict, made_at: datetime) -> pd.DataFrame:
    px = r["px"]
    ref = float(px["Close"].iloc[-1])
    rows = []
    for m in MODELS:
        q = r["now"][m]                                   # (3 targets, 3 quantiles)
        row = {"bar_date": f"{px.index[-1]:%Y-%m-%d}", "ticker": r["label"], "model": m,
               "stage": "overnight", "ref_close": round(ref, 2)}
        for j, t in enumerate(TARGETS):
            for k, p in enumerate((10, 50, 90)):
                row[f"r_{t}_p{p}"] = round(float(q[j, k]), 4)
        for name, t in (("open", "gap"), ("close", "close")):
            for k, p in enumerate((10, 50, 90)):
                row[f"{name}_p{p}"] = round(ref * np.exp(q[TARGETS.index(t), k] / 100), 2)
        row["made_at"] = made_at.strftime("%Y-%m-%d %H:%M WIB")
        rows.append(row)
    return pd.DataFrame(rows)


def lock_time(bar_date: str) -> datetime:
    """08:45 WIB of the next weekday. IDX holidays make it lock early, never late."""
    nxt = pd.Timestamp(bar_date) + pd.offsets.BDay(1)
    return datetime.combine(nxt.date(), PRE_OPEN, tzinfo=ts.JKT)


def fill_actuals(log: pd.DataFrame, px: pd.DataFrame, ticker: str, provisional: bool) -> pd.DataFrame:
    """Add what happened in the session after each logged close. Uses return ratios
    from the current download, so dividend re-adjustment of old prices cancels out."""
    real = realized(px)
    log = log.copy()
    todo = log.index[(log["ticker"] == ticker) & log["next_date"].isna()]
    for i in todo:
        d = pd.Timestamp(log.at[i, "bar_date"])
        later = px.index[px.index > d]
        if d not in px.index or not len(later) or (later[0] == px.index[-1] and provisional):
            continue
        nxt, ref = later[0], log.at[i, "ref_close"]
        log.at[i, "next_date"] = f"{nxt:%Y-%m-%d}"
        for t in TARGETS:
            log.at[i, f"r_{t}_actual"] = round(float(real.at[nxt, t]), 4)
        log.at[i, "actual_open"] = round(ref * np.exp(real.at[nxt, "gap"] / 100), 2)
        log.at[i, "actual_close"] = round(ref * np.exp(real.at[nxt, "close"] / 100), 2)
    return log


def read_log(path: Path) -> pd.DataFrame:
    cols = ["next_date", "actual_open", "actual_close"] + [f"r_{t}_actual" for t in TARGETS]
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_KEYS)
    for c in cols:
        if c not in log:
            log[c] = np.nan
    return log.astype({"next_date": object})


def merge_rows(log: pd.DataFrame, new: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, bool]:
    """Insert or replace today's rows unless the forecast window has closed."""
    if now >= lock_time(new["bar_date"].iloc[0]):
        return log, False
    if log.empty:
        return new.reindex(columns=list(dict.fromkeys([*new.columns, *log.columns]))), True
    key =lambda df: df[LOG_KEYS].astype(str).agg("|".join, axis=1)
    log = pd.concat([log[~key(log).isin(key(new))], new], ignore_index=True)
    return log.sort_values(["ticker", "bar_date", "model"], kind="stable").reset_index(drop=True), True


def scored(log: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return log[(log["ticker"] == ticker) & (log["stage"] == "overnight") & log["next_date"].notna()]


def scorecard(log: pd.DataFrame, ticker: str, windows=(20, 60)) -> pd.DataFrame | None:
    s = scored(log, ticker)
    if s.empty:
        return None
    rows = []
    for w in windows:
        recent = s[s["bar_date"].isin(sorted(s["bar_date"].unique())[-w:])]
        for m in MODELS:
            g = recent[recent["model"] == m]
            if g.empty:
                continue
            up_p, up_a = np.sign(g["close_p50"] - g["ref_close"]), np.sign(g["actual_close"] - g["ref_close"])
            moved = (up_p != 0) & (up_a != 0)
            rows.append({
                "last": f"{w} ({g['bar_date'].nunique()} scored)", "model": m,
                "open error": (g["actual_open"] / g["open_p50"] - 1).abs().mean(),
                "close error": (g["actual_close"] / g["close_p50"] - 1).abs().mean(),
                "close direction": (up_p[moved] == up_a[moved]).mean() if moved.any() else np.nan,
                "open inside 80%": g["actual_open"].between(g["open_p10"], g["open_p90"]).mean(),
                "close inside 80%": g["actual_close"].between(g["close_p10"], g["close_p90"]).mean(),
            })
    return pd.DataFrame(rows).set_index(["last", "model"])


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def to_tick(p: float, how: str) -> float:
    t = float(tick_size(np.array([p]))[0])
    return {"down": np.floor, "up": np.ceil, "near": np.round}[how](p / t) * t


def band_txt(p10: float, p50: float, p90: float) -> str:
    return f"{to_tick(p10, 'down'):>7,.0f} {to_tick(p50, 'near'):>7,.0f} {to_tick(p90, 'up'):>7,.0f}"


def pct_table(df: pd.DataFrame, signed: tuple[str, ...] = ()) -> str:
    show = df.copy().astype(object)
    for c in df.columns:
        show[c] = df[c].map(lambda v, c=c: "-" if pd.isna(v) else (f"{v:+.1%}" if c in signed else f"{v:.1%}"))
    return show.to_string()


def section(title: str) -> None:
    print(f"\n{'-' * 72}\n {title}\n{'-' * 72}")


def report_last_check(log: pd.DataFrame, ticker: str) -> None:
    s = scored(log, ticker)
    section("Last forecast vs what happened")
    if s.empty:
        print("  Nothing to compare yet: the first forecast is scored after the next session closes.")
        return
    day = s["bar_date"].max()
    g = s[s["bar_date"] == day].set_index("model").reindex(list(MODELS)).dropna(subset=["ref_close"])
    a = g.iloc[0]
    print(f"  Forecast made after the {day} close ({a.ref_close:,.0f}) for {a.next_date}:")
    print(f"  actual open {a.actual_open:,.0f} ({a.actual_open / a.ref_close - 1:+.1%}), "
          f"actual close {a.actual_close:,.0f} ({a.actual_close / a.ref_close - 1:+.1%})\n")
    print(f"  {'':<9}{'open P10   P50   P90':>24}  {'off by':>7}  {'in 80%':>6}   "
          f"{'close P10   P50   P90':>24}  {'off by':>7}  {'in 80%':>6}")
    for m, x in g.iterrows():
        cells = ""
        for name in ("open", "close"):
            act, p10, p50, p90 = x[f"actual_{name}"], x[f"{name}_p10"], x[f"{name}_p50"], x[f"{name}_p90"]
            cells += (f"{band_txt(p10, p50, p90):>24}  {act / p50 - 1:>+7.1%}  "
                      f"{'yes' if p10 <= act <= p90 else 'no':>6}   ")
        print(f"  {m:<9}{cells}")


def report(r: dict, log: pd.DataFrame | None, provisional: bool) -> None:
    ticker, px = r["label"], r["px"]
    if log is not None:
        report_last_check(log, ticker)
        section("Live scorecard (forecast_log.csv)")
        sc = scorecard(log, ticker)
        if sc is None:
            print("  No scored sessions yet. Give it ~20 sessions for a first impression, 60+ before trusting it.")
        else:
            print(pct_table(sc))
            print("  error = average |actual / P50 - 1|; direction = up/down call vs ref close "
                  "(Baseline has none); inside 80% should be ~80%.")

    section(f"Test-period check ({r['n_test']} sessions the models never trained on)")
    print(pct_table(r["ev"], signed=("skill",)))
    print("  skill > 0 = beats the Baseline (no change +/- volatility); avg error is on the P50 move;"
          "\n  inside 80% should be ~80%, below P50 ~50%.")

    ref, day = float(px["Close"].iloc[-1]), px.index[-1]
    nxt = day + pd.offsets.BDay(1)
    state = "PROVISIONAL - built on an unfinished bar, not logged" if provisional else "close"
    section(f"{ticker} next session: {nxt:%a %Y-%m-%d} (if IDX is open)")
    print(f"  From {day:%a %Y-%m-%d} {state}: {ref:,.0f}\n")
    print(f"  {'':<9}{'open P10   P50   P90':>24}  {'P50 vs close':>12}   {'close P10   P50   P90':>24}  "
          f"{'P50 vs close':>12}   test skill (open / close)")
    ev = r["ev"]["skill"]
    for m in MODELS:
        q = r["now"][m]
        o = ref * np.exp(q[TARGETS.index("gap")] / 100)
        c = ref * np.exp(q[TARGETS.index("close")] / 100)
        print(f"  {m:<9}{band_txt(*o):>24}  {o[1] / ref - 1:>+12.1%}   {band_txt(*c):>24}  {c[1] / ref - 1:>+12.1%}   "
              + ("" if m == "Baseline" else f"{ev['open', m]:+.1%} / {ev['close', m]:+.1%}"))
    print("  P10-P90 = 80% range, rounded to the IDX tick. Not financial advice.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="BBCA.JK")
    ap.add_argument("--market", default=wf.Config.market, help="market index ticker, or 'none'")
    ap.add_argument("--start", default=wf.Config.start, help="download start date")
    ap.add_argument("--since", default="2013-01-01", help="first day used (Yahoo's 2012 IDX opens are unreliable)")
    ap.add_argument("--seeds", type=int, default=3, help="LSTM ensemble size")
    ap.add_argument("--lookback", type=int, default=60, help="days of history the LSTM sees")
    ap.add_argument("--test-days", type=int, default=500, help="held-out sessions for the test-period check")
    ap.add_argument("--calib-window", type=int, default=250, help="recent misses used to calibrate the bands")
    ap.add_argument("--gamma", type=float, default=0.005, help="adaptive-conformal step size")
    ap.add_argument("--offline", action="store_true", help="use the CSVs in data/ instead of downloading")
    ap.add_argument("--no-log", action="store_true", help="don't read or write the forecast log")
    ap.add_argument("--log", default=str(LOG))
    a = ap.parse_args()

    t0 = time.time()
    cfg = wf.Config(ticker=a.ticker, market=a.market, start=a.start)
    px = ts.load(cfg.ticker, cfg, a.offline)
    mkt = None if cfg.market.lower() == "none" else ts.load(cfg.market, cfg, a.offline)
    provisional = bool(px.attrs.get("provisional", False))

    r = run(px, mkt, cfg.ticker, a.lookback, a.seeds, a.test_days, a.calib_window, a.gamma, since=a.since)

    log, status = None, ""
    if not a.no_log:
        path, now = Path(a.log), datetime.now(ts.JKT)
        log = fill_actuals(read_log(path), r["px"], cfg.ticker, provisional)
        if provisional:
            status = "Not logged: the session isn't finished. Re-run after 16:15 WIB."
        else:
            log, ok = merge_rows(log, forecast_rows(r, now), now)
            day = f"{r['px'].index[-1]:%Y-%m-%d}"
            status = (f"Logged the forecast from the {day} close." if ok else
                      f"Not logged: the session after {day} has already opened (locked at 08:45 WIB).")
        path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(path, index=False)

    report(r, log, provisional)
    if status:
        print(f"\n  {status}\n  Log: {Path(a.log).resolve()}")
    print(f"  ({time.time() - t0:.0f}s, models {r['t_fit']:.0f}s)")


if __name__ == "__main__":
    main()
