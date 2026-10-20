"""
Today's signal: BUY / SELL / HOLD / STAY OUT
============================================

Downloads fresh data, rebuilds the walk-forward history for one setting (needed
because the dead zone makes today's action depend on whether you're already
long), retrains the model on everything up to today, and prints the action for
the next session. Every run is appended to output/signals_log.csv so you can
paper-trade it and compare against what actually happened.

    python today_signal.py --ticker BBCA.JK
    python today_signal.py --ticker BBRI.JK --model LogReg --threshold 0.58 --exit-threshold 0.48
    python today_signal.py --ticker BBCA.JK --offline        # use data/ CSVs, no download

Each online run patches today's bar with the latest 1-minute price from Yahoo,
so you can run it during the session (e.g. in the 15:50-16:00 pre-closing) and
get a signal on the current price. Before 16:15 WIB the signal is marked
PROVISIONAL: High/Low are so-far and Volume is pro-rated by elapsed session time.
Re-run after the close to get the final signal (the log row for today is
overwritten). --no-live restores the old behaviour of dropping today's bar until
the close. The backtest assumes you trade at the closing price.

Default setting (LightGBM, h=1, enter 0.55, exit 0.50, smooth 3) is a middle-of-
the-grid choice, NOT a validated one: the hold-out test found no setting whose
first-half performance predicted its second-half performance. Paper-trade first.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bbca_walkforward as wf

JKT = ZoneInfo("Asia/Jakarta")
IDX_CLOSE = dtime(16, 15)          # 16:00 close + a margin for Yahoo to settle the bar
MODEL_COL = {"LogReg": "p_logit", "LightGBM": "p_lgbm", "LightGBM+GARCH": "p_lgbm"}


def drop_partial_bar(df: pd.DataFrame, name: str) -> pd.DataFrame:
    now = datetime.now(JKT)
    if len(df) and df.index[-1].date() == now.date() and now.time() < IDX_CLOSE:
        print(f"  note: dropping today's unfinished {name} bar (IDX still open, {now:%H:%M} WIB)")
        return df.iloc[:-1]
    return df


def session_fraction(now: datetime) -> float:
    """Share of today's IDX continuous-trading minutes already elapsed (1.0 after the close)."""
    if now.weekday() == 4:                                   # Friday has a longer lunch break
        sessions = [(dtime(9, 0), dtime(11, 30)), (dtime(14, 0), dtime(16, 0))]
    else:
        sessions = [(dtime(9, 0), dtime(12, 0)), (dtime(13, 30), dtime(16, 0))]
    mins = lambda t: t.hour * 60 + t.minute
    total = sum(mins(b) - mins(a) for a, b in sessions)
    t = mins(now.time())
    done = sum(min(max(t - mins(a), 0), mins(b) - mins(a)) for a, b in sessions)
    return done / total


def patch_live_bar(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Overwrite today's daily bar with the latest 1-minute price, so every run
    uses the current price instead of Yahoo's (lagging) daily bar or yesterday's close.

    Before the close the bar is provisional: High/Low are so-far, and Volume is
    scaled up by the elapsed share of the session so the volume z-score isn't
    dragged down by a half-finished day. Sets df.attrs["live_at"] when patched."""
    import yfinance as yf

    now = datetime.now(JKT)
    m = yf.download(ticker, period="5d", interval="1m", auto_adjust=True, progress=False)
    if isinstance(m.columns, pd.MultiIndex):
        m.columns = m.columns.get_level_values(0)
    if m.empty:
        return df
    m = m.tz_convert(JKT) if m.index.tz else m.tz_localize(JKT)
    m = m[m.index.date == now.date()].dropna(subset=["Close"])
    if m.empty:                                              # weekend / holiday / pre-open
        return df

    day = pd.Timestamp(now.date())
    bar = {"Open": m["Open"].iloc[0], "High": m["High"].max(), "Low": m["Low"].min(),
           "Close": m["Close"].iloc[-1], "Volume": m["Volume"].sum()}
    if day in df.index:                                      # daily bar's volume is more complete
        d = df.loc[day]
        bar.update(Open=d["Open"], High=max(bar["High"], d["High"]), Low=min(bar["Low"], d["Low"]),
                   Volume=max(bar["Volume"], d["Volume"]))
    frac = session_fraction(now)
    if 0 < frac < 1 and bar["Volume"] > 0:
        bar["Volume"] = round(bar["Volume"] / frac)

    df = df.copy()
    df.loc[day, list(bar)] = list(bar.values())
    df = df.sort_index()
    df.attrs["live_at"] = m.index[-1]
    df.attrs["provisional"] = now.time() < IDX_CLOSE
    state = "provisional, session still open" if df.attrs["provisional"] else "after close"
    print(f"  {ticker}: live price {bar['Close']:,.2f} at {m.index[-1]:%H:%M} WIB ({state})")
    return df


def load(ticker: str, cfg: wf.Config, offline: bool, live: bool = True) -> pd.DataFrame:
    if offline:
        path = Path(cfg.data_dir) / f"{ticker.lstrip('^')}.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found - run without --offline once to download it")
        return drop_partial_bar(wf.load_csv(path), ticker)
    df = wf.load_yahoo(ticker, cfg.start, cfg.data_dir)
    return patch_live_bar(df, ticker) if live else drop_partial_bar(df, ticker)


def refit_last(df: pd.DataFrame, feature_cols: list[str], probs: pd.DataFrame, cfg: wf.Config) -> pd.DataFrame:
    """Walk-forward retrains every `retrain` days, so the latest prediction can come
    from a model up to a month old. Refit on all labelled data (minus the embargo)
    and replace the last row's probabilities with the fresh model's."""
    data = df.dropna(subset=feature_cols)
    train = data.iloc[: len(data) - cfg.horizon]
    train = train[train["target"].notna()]
    last = data.iloc[[-1]]
    probs = probs.copy()
    for col, model in wf.make_models(cfg).items():
        model.fit(train[feature_cols], train["target"].astype(int))
        probs.loc[last.index, col] = model.predict_proba(last[feature_cols])[:, 1]
    return probs


def action(prev: float, cur: float) -> str:
    if prev == 0 and cur > 0:
        return "BUY"
    if prev > 0 and cur == 0:
        return "SELL"
    if prev > 0:
        return "HOLD" if abs(cur - prev) < 1e-9 else "HOLD (resize)"
    return "STAY OUT"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=wf.Config.ticker)
    ap.add_argument("--market", default=wf.Config.market, help="market index ticker, or 'none'")
    ap.add_argument("--start", default=wf.Config.start)
    ap.add_argument("--model", default="LightGBM", choices=list(MODEL_COL))
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.55, help="enter long when P(up) > this")
    ap.add_argument("--exit-threshold", type=float, default=0.50, help="exit when P(up) < this")
    ap.add_argument("--smooth", type=int, default=3, help="average P(up) over N days first")
    ap.add_argument("--buy-cost", type=float, default=wf.Config.buy_cost)
    ap.add_argument("--sell-cost", type=float, default=wf.Config.sell_cost)
    ap.add_argument("--offline", action="store_true", help="use the CSVs in data/ instead of downloading")
    ap.add_argument("--no-live", action="store_true",
                    help="don't patch in the latest intraday price; drop today's bar until the close instead")
    ap.add_argument("--history", type=int, default=10, help="recent days to show")
    ap.add_argument("--log", default=str(Path(wf.Config.out_dir) / "signals_log.csv"))
    a = ap.parse_args()

    cfg = wf.Config(ticker=a.ticker, market=a.market, start=a.start, horizon=a.horizon,
                    prob_threshold=a.threshold, exit_threshold=a.exit_threshold, smooth=a.smooth,
                    buy_cost=a.buy_cost, sell_cost=a.sell_cost)

    live = not a.no_live
    px = load(cfg.ticker, cfg, a.offline, live)
    mkt = None if cfg.market.lower() == "none" else load(cfg.market, cfg, a.offline, live)

    df = wf.build_features(px, mkt, cfg.horizon)
    feature_cols = [c for c in df.columns if c not in ("ret", "target", "fwd_ret")]
    print("Rebuilding walk-forward history ...", flush=True)
    probs, _ = wf.walk_forward_predict(df, feature_cols, cfg)
    probs = refit_last(df, feature_cols, probs, cfg)
    oos = probs.index

    vol_fc = pd.Series(np.nan, index=oos)
    if a.model == "LightGBM+GARCH":
        print("GARCH volatility forecast ...", flush=True)
        vol_fc = wf.garch_vol_forecast(df["ret"], cfg)
    positions = wf.make_positions(df, probs, vol_fc, cfg)

    ret = df["ret"].reindex(oos)
    bt = wf.backtest(positions[a.model], ret, cfg)            # applies the rebalance band too
    bh = wf.backtest(positions["Buy & Hold"], ret, cfg)
    p = probs[MODEL_COL[a.model]]
    p_s = p.rolling(cfg.smooth, min_periods=1).mean()
    held = bt["pos"]

    day, prev_pos, cur_pos = oos[-1], held.iloc[-2], held.iloc[-1]
    act = action(prev_pos, cur_pos)
    close = px["Close"].reindex(oos)
    live_at = px.attrs.get("live_at")
    provisional = px.attrs.get("provisional", False)
    price_txt = (f"close {day:%a %Y-%m-%d}" if live_at is None or not provisional
                 else f"live {day:%a %Y-%m-%d} {live_at:%H:%M} WIB")

    # ---- headline
    print(f"\n{'=' * 64}")
    print(f"  {cfg.ticker}  |  {price_txt}: {close.iloc[-1]:,.0f}")
    print(f"  {a.model}, h={cfg.horizon}d, enter > {cfg.prob_threshold}, exit < {cfg.exit_threshold}, "
          f"smooth {cfg.smooth}d")
    print(f"  P(up) today {p.iloc[-1]:.3f}  |  smoothed {p_s.iloc[-1]:.3f}")
    print(f"\n  >>> {act}" + (f"  (target position {cur_pos:.0%})" if cur_pos > 0 else ""))
    if act == "STAY OUT":
        gap = cfg.prob_threshold - p_s.iloc[-1]
        print(f"      smoothed P(up) needs to rise {gap:.3f} more to trigger a buy")
    elif act.startswith("HOLD"):
        gap = p_s.iloc[-1] - cfg.exit_threshold
        print(f"      smoothed P(up) is {gap:.3f} above the exit level")
    if provisional:
        print("  PROVISIONAL: built on an unfinished bar. Re-run after 16:15 WIB to confirm.")
    print(f"{'=' * 64}")

    # ---- recent history
    k = a.history
    recent = pd.DataFrame({
        "close": close.iloc[-k:].map("{:,.0f}".format),
        "P(up)": p.iloc[-k:].map("{:.3f}".format),
        "smoothed": p_s.iloc[-k:].map("{:.3f}".format),
        "position": held.iloc[-k:].map("{:.0%}".format),
        "action": [action(a_, b_) for a_, b_ in zip(held.shift(1).fillna(0).iloc[-k:], held.iloc[-k:])],
    })
    recent.index = recent.index.strftime("%Y-%m-%d")
    print(f"\nLast {k} sessions:")
    print(recent.to_string())

    # ---- how this exact setting has done (walk-forward, after costs)
    def row(b: pd.DataFrame, n: int | None = None) -> str:
        part = b.iloc[-n:].copy() if n else b.copy()
        part["equity"] = (1 + part["ret"]).cumprod()
        m = wf.metrics(part)
        return f"CAGR {m['CAGR']:6.1%}  Sharpe {m['Sharpe']:5.2f}  MaxDD {m['MaxDD']:6.1%}  trades/yr {m['Trades/yr']:3.0f}"

    print(f"\nThis setting, walk-forward after costs:")
    print(f"  since {oos[0]:%Y-%m}   strategy  {row(bt)}")
    print(f"                    buy&hold  {row(bh)}")
    print(f"  last 12 months    strategy  {row(bt, wf.TRADING_DAYS)}")
    print(f"                    buy&hold  {row(bh, wf.TRADING_DAYS)}")
    print("\nNot validated: the hold-out test found no setting that reliably carries forward. Paper-trade it.")

    # ---- log (one row per bar date + setting; re-running the same day overwrites)
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = pd.DataFrame([{
        "bar_date": f"{day:%Y-%m-%d}", "ticker": cfg.ticker, "model": a.model, "horizon": cfg.horizon,
        "entry": cfg.prob_threshold, "exit": cfg.exit_threshold, "smooth": cfg.smooth,
        "close": round(float(close.iloc[-1]), 2), "p_up": round(float(p.iloc[-1]), 4),
        "p_smoothed": round(float(p_s.iloc[-1]), 4), "position": round(float(cur_pos), 4),
        "action": act, "provisional": provisional,
        "price_at": "" if live_at is None else f"{live_at:%Y-%m-%d %H:%M}", "run_at": datetime.now(JKT).strftime("%Y-%m-%d %H:%M WIB"),
    }])
    append_log(log, entry)
    print(f"Logged to {log.resolve()}")


def append_log(log: Path, entry: pd.DataFrame) -> None:
    """Add a signal row; a row with the same bar date + setting is replaced."""
    keys = ["bar_date", "ticker", "model", "horizon", "entry", "exit", "smooth"]
    if log.exists():
        old = pd.read_csv(log)
        same = (old[keys].astype(str) == entry[keys].astype(str).iloc[0]).all(axis=1)
        entry = pd.concat([old[~same], entry], ignore_index=True)
    entry.to_csv(log, index=False)


if __name__ == "__main__":
    main()
