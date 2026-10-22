"""
Backtest everything stock_cli.py shows
======================================

Replays, day by day on the sessions the models never trained on, what
stock_cli.py would have said, and checks it against what happened:

  1. Next-session open / close forecasts (LSTM, LightGBM, Baseline), last 500 sessions:
     accuracy overall, per half-year, and the last 10 sessions one by one
  2. The decision: signal call, forecast call and the combined BUY / SELL / HOLD,
     run as paper strategies on the same 500 sessions (after costs), vs buy & hold;
     then the same decision with take-profit / stop-loss exits and the LSTM veto
  3. Today's-close forecast made from the price at the lunch break (and at every
     other checkpoint), last 120 sessions of hourly data
  4. Today so far: every forecast made for today vs the latest price

Uses the same cached models as stock_cli.py when they hold the test predictions;
otherwise refits them (~5-8 min) and refreshes the cache for stock_cli too.

    python backtest_all.py MTLA.JK
    python backtest_all.py BBCA --offline
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import bbca_walkforward as wf
import intraday_close as ic
import lstm_forecast as lf
import model_cache as mc
import next_day as nd
import stock_cli as sc
import today_signal as ts

pd.set_option("display.width", 200)


def cached_with_oos(kind: str, ticker: str, day: pd.Timestamp, params: dict, build, refresh: bool) -> dict:
    """mc.cached, refitting once if the cached object predates the stored test predictions."""
    obj, hit = mc.cached(kind, ticker, day, params, build, refresh)
    if "oos" not in obj:
        print(f"  cached {kind} model has no test predictions; refitting ...", flush=True)
        obj, hit = mc.cached(kind, ticker, day, params, build, True)
    if hit:
        print(f"  using the {kind} models fitted after the {day:%Y-%m-%d} close (cached)")
    return obj


# ----------------------------------------------------------------------------
# 1. Next-session forecasts
# ----------------------------------------------------------------------------
def forecast_by_period(o: dict) -> pd.DataFrame:
    """Skill and coverage of the close forecast per half-year of the test period."""
    te, idx = o["test"], o["index"]
    half = pd.Series(idx[te]).dt.year.astype(str) + np.where(pd.Series(idx[te]).dt.month <= 6, " H1", " H2")
    out = {}
    for h in half.unique():
        rows = te[(half == h).values]
        ev = nd.evaluate(o["preds"], o["Y"], rows).xs("close", level="price")
        out[f"{h} ({len(rows)}d)"] = ev.drop(index="Baseline", columns="below P50")
    return pd.concat(out, names=["period", "model"])


def last_sessions(o: dict, model: str, k: int = 10) -> pd.DataFrame:
    """The last k test sessions: forecast range for the next open / close vs actual."""
    te, idx, close, Y, p = o["test"][-k:], o["index"], o["close"], o["Y"], o["preds"][model]
    rows = []
    for i in te:
        ref = close[i]
        o_q = ref * np.exp(p[i, nd.TARGETS.index("gap")] / 100)
        c_q = ref * np.exp(p[i, nd.TARGETS.index("close")] / 100)
        a_o = ref * np.exp(Y[i, nd.TARGETS.index("gap")] / 100)
        a_c = ref * np.exp(Y[i, nd.TARGETS.index("close")] / 100)
        rows.append({
            "from close": f"{idx[i]:%Y-%m-%d}", "close": f"{ref:,.0f}",
            "open 80%": nd.band_txt(*o_q), "actual open": f"{a_o:,.0f}",
            "ok ": "yes" if o_q[0] <= a_o <= o_q[2] else "NO",
            "close 80%": nd.band_txt(*c_q), "actual close": f"{a_c:,.0f}",
            "ok": "yes" if c_q[0] <= a_c <= c_q[2] else "NO",
            "dir": "-" if a_c == ref or round(c_q[1]) == round(ref) else
                   ("right" if (a_c > ref) == (c_q[1] > ref) else "wrong"),
        })
    return pd.DataFrame(rows).set_index("from close")


# ----------------------------------------------------------------------------
# 2. The decision, replayed
# ----------------------------------------------------------------------------
def replay(dates: pd.DatetimeIndex, p_s: pd.Series, move: pd.Series, cfg: wf.Config) -> pd.DataFrame:
    """Walk through the test days with stock_cli's rules, starting out of the stock.
    Decision at close t, traded at that close. Returns position per rule."""
    entry, ex = cfg.prob_threshold, min(cfg.exit_threshold, cfg.prob_threshold)
    hurdle = cfg.buy_cost + cfg.sell_cost
    own = {"Signal only": False, "Forecast only": False, "Both agree": False}
    out = {k: [] for k in own}
    for d in dates:
        for rule, owned in own.items():
            s = ("SELL" if p_s[d] < ex else "HOLD") if owned else ("BUY" if p_s[d] > entry else "WAIT")
            f = ("SELL" if move[d] < -cfg.sell_cost else "HOLD") if owned else ("BUY" if move[d] > hurdle else "WAIT")
            call = {"Signal only": s, "Forecast only": f}.get(rule, s if s == f else None)
            if call == "BUY":
                own[rule] = True
            elif call == "SELL":
                own[rule] = False
            out[rule].append(float(own[rule]))
    return pd.DataFrame(out, index=dates)


def trade_stats(bt: pd.DataFrame, close: pd.Series, cfg: wf.Config) -> dict:
    """Round trips (bought at a close, sold at a later close), net of costs."""
    trips, entry, prev = [], None, 0.0
    for p, c in zip(bt["pos"].values, close.values):
        if p > 0 and prev == 0:
            entry = c
        elif p == 0 and prev > 0:
            trips.append(c / entry * (1 - cfg.sell_cost) / (1 + cfg.buy_cost) - 1)
        prev = p
    t = np.array(trips)                         # a trip still open at the end isn't counted
    return {"round trips": len(t), "won": f"{(t > 0).mean():.0%}" if len(t) else "-",
            "avg trip": f"{t.mean():+.2%}" if len(t) else "-"}


def replay_exits(px: pd.DataFrame, dates: pd.DatetimeIndex, p_s: pd.Series, move: pd.Series,
                 sd: pd.Series, cfg: wf.Config, k_tp: float, k_sl: float,
                 veto: pd.Series | None = None) -> dict:
    """'Both agree' with stock_cli's exits: bought at the close, a sell limit k_tp x the
    daily vol above the buy price and a stop k_sl x below (0 = off), checked on each later
    session's high / low (stop first when both trade; a gap through fills at the open).
    `veto`: no BUY when the LSTM 20-day median move is below zero."""
    entry, ex, hurdle = cfg.prob_threshold, min(cfg.exit_threshold, cfg.prob_threshold), cfg.buy_cost + cfg.sell_cost
    owned, ref, paid, tp, sl = False, 0.0, 0.0, np.inf, 0.0
    rets, trips, n_tp, n_sl = [], [], 0, 0
    for d in dates:
        o, h, lo, c = px.loc[d, ["Open", "High", "Low", "Close"]]
        g = 1.0
        if owned:
            out = min(o, sl) if lo <= sl else (max(o, tp) if h >= tp else None)
            if out is None:
                g, ref = c / ref, c
            else:
                n_sl, n_tp = n_sl + (lo <= sl), n_tp + (lo > sl)
                g, owned = out / ref * (1 - cfg.sell_cost), False
                trips.append(out / paid * (1 - cfg.sell_cost) / (1 + cfg.buy_cost) - 1)
        s = ("SELL" if p_s[d] < ex else "HOLD") if owned else ("BUY" if p_s[d] > entry else "WAIT")
        f = ("SELL" if move[d] < -cfg.sell_cost else "HOLD") if owned else ("BUY" if move[d] > hurdle else "WAIT")
        call = s if s == f else None
        if call == "BUY" and veto is not None and veto.get(d, 0.0) < 0:
            call = None
        if owned and call == "SELL":
            g, owned = g * (1 - cfg.sell_cost), False
            trips.append(c / paid * (1 - cfg.sell_cost) / (1 + cfg.buy_cost) - 1)
        elif not owned and call == "BUY":
            g, owned, ref, paid = g / (1 + cfg.buy_cost), True, c, c
            tp = c * np.exp(k_tp * sd[d]) if k_tp > 0 else np.inf
            sl = c * np.exp(-k_sl * sd[d]) if k_sl > 0 else 0.0
        rets.append(g - 1)
    r = pd.Series(rets, index=dates)
    t = np.array(trips)
    return {**wf.metrics(pd.DataFrame({"ret": r, "equity": (1 + r).cumprod(), "pos": 0.0, "turnover": 0.0})),
            "round trips": len(t), "won": f"{(t > 0).mean():.0%}" if len(t) else "-",
            "avg trip": f"{t.mean():+.2%}" if len(t) else "-", "take-profits": n_tp, "stops": n_sl}


def lstm_test_moves(r: dict) -> pd.Series:
    """LSTM 20-day median move on its own test days, indexed by the day it was made."""
    close = r["close"]
    i = close.index.get_indexer(r["test_dates"]) - r["H"]
    m = pd.Series(r["test_band"][:, 1] / close.values[i] - 1, index=close.index[i])
    return m[~m.index.duplicated()]


# ----------------------------------------------------------------------------
# 3. Intraday forecast by checkpoint
# ----------------------------------------------------------------------------
def checkpoint_sessions(o: dict, k: int, n: int = 10) -> pd.DataFrame:
    """Last n test sessions at checkpoint k (11 = price at the lunch break)."""
    r = o["rows"].reset_index(drop=True)
    sel = np.where(o["test"] & (r["k"].values == k))[0][-n:]
    rows = []
    for i in sel:
        price = r.at[i, "price"]
        actual = price * np.exp(r.at[i, "y"] / 100)
        lg = price * np.exp(o["preds"]["LightGBM"][i] / 100)
        bl = price * np.exp(o["preds"]["Baseline"][i] / 100)
        rows.append({
            "day": f"{pd.Timestamp(r.at[i, 'day']):%a %Y-%m-%d}", "price at break": f"{price:,.0f}",
            "LightGBM 80%": nd.band_txt(*lg), "ok": "yes" if lg[0] <= actual <= lg[2] else "NO",
            "Baseline 80%": nd.band_txt(*bl), "ok ": "yes" if bl[0] <= actual <= bl[2] else "NO",
            "actual close": f"{actual:,.0f}", "move": f"{actual / price - 1:+.1%}",
        })
    return pd.DataFrame(rows).set_index("day")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code", help="stock code, e.g. MTLA or MTLA.JK")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--exit-threshold", type=float, default=0.50)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-lstm", action="store_true", help="skip the LSTM veto rows in 2c")
    a = ap.parse_args()
    ticker = sc.normalize(a.code) or a.code
    cfg = wf.Config(ticker=ticker, horizon=1, prob_threshold=a.threshold, exit_threshold=a.exit_threshold,
                    smooth=a.smooth, data_dir=wf.Config.data_dir)
    now = datetime.now(ts.JKT)

    print(f"Downloading {ticker} ...", flush=True)
    px = ts.load(ticker, cfg, a.offline)
    mkt = ts.load(cfg.market, cfg, a.offline)
    px_f, mkt_f, day_f = nd.finished(px, mkt)
    sig, _ = mc.cached("signal", ticker, day_f, {"retrain": cfg.retrain, "min_train": cfg.min_train},
                       lambda: sc.signal_fit(px_f, mkt_f, cfg), a.refresh)
    st = cached_with_oos("next_day", ticker, day_f, {"n_seeds": 3},
                         lambda: nd.fit(px_f, mkt_f, ticker, n_seeds=3), a.refresh)
    o = st["oos"]
    dates = o["index"][o["test"]]

    # ================= 1. next-session forecasts =================
    nd.section(f"1. Next-session forecasts, {len(dates)} test sessions {dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}")
    print(nd.pct_table(st["ev"], signed=("skill",)))
    print("  skill > 0 = beats the Baseline (no change +/- volatility). inside 80% should be ~80%.")
    print("\n  Close forecast by half-year (does the edge hold up, or is it one lucky stretch?):")
    print(nd.pct_table(forecast_by_period(o), signed=("skill",)))
    m_live = sc.best_model(st["ev"], "close")
    print(f"\n  Last 10 test sessions ({m_live}, the model stock_cli uses for the close):")
    print(last_sessions(o, m_live).to_string())

    # ================= 2. the decision =================
    p = sig["probs"]["p_lgbm"]
    p_s = p.rolling(cfg.smooth, min_periods=1).mean().reindex(dates).ffill()
    close = pd.Series(o["close"], index=o["index"])
    ret = np.log(close).diff()
    fwd_up = (ret.shift(-1) > 0).astype(float).reindex(dates)
    nd.section(f"2. BUY / SELL / HOLD decision replayed on the same {len(dates)} sessions (after costs)")
    print(f"  Signal: smoothed P(up) entry > {cfg.prob_threshold}, exit < {cfg.exit_threshold} | "
          f"Forecast: next close P50 > +{cfg.buy_cost + cfg.sell_cost:.2%} to buy, < -{cfg.sell_cost:.2%} to sell\n")
    rows, curves = {}, {}
    bh = wf.backtest(pd.Series(1.0, index=dates), ret, cfg)
    rows["Buy & Hold"] = {**wf.metrics(bh), "round trips": 0, "won": "-", "avg trip": "-"}
    curves["Buy & Hold"] = bh
    ci = nd.TARGETS.index("close")
    for m in ("LightGBM", "LSTM"):
        move = pd.Series(np.exp(o["preds"][m][o["test"], ci, 1] / 100) - 1, index=dates)
        pos = replay(dates, p_s, move, cfg)
        for rule in pos:
            if rule == "Signal only" and m != "LightGBM":
                continue                                    # doesn't depend on the forecast model
            name = rule if rule == "Signal only" else f"{rule} ({m})"
            bt = wf.backtest(pos[rule], ret, cfg)
            rows[name] = {**wf.metrics(bt), **trade_stats(bt, close.reindex(dates), cfg)}
            curves[name] = bt
        if m == "LightGBM":
            fc_up = move > 0
            moved = (move.abs() > 1e-9) & (ret.shift(-1).reindex(dates) != 0)
            hit_fc = (fc_up[moved] == (fwd_up[moved] == 1)).mean()
    tbl = pd.DataFrame(rows).T[["CAGR", "Vol", "Sharpe", "MaxDD", "Exposure", "Trades/yr",
                                "round trips", "won", "avg trip"]]
    print(sc.fmt_table(tbl))
    print(f"\n  Direction hit (days where both moved): signal P(up)>0.5 "
          f"{((p.reindex(dates) > 0.5) == (fwd_up == 1)).mean():.0%} | LightGBM forecast {hit_fc:.0%}")
    print(f"  stock_cli's decision = 'Both agree ({m_live})' row. Starts out of the stock on {dates[0]:%Y-%m-%d}.")

    m_open = sc.best_model(st["ev"], "open")
    nd.section(f"2b. Limit prices for the next session ({m_open} open range), same {len(dates)} sessions")
    lim = pd.concat({m: nd.limit_check(o, px_f, m) for m in dict.fromkeys((m_open, "Baseline"))},
                    names=["range from", "order"])
    print(nd.pct_table(lim, signed=("better than open", "better than close")))
    print("  Order placed after the close for the whole next session. fills = share of sessions it filled;")
    print("  better than open / close = average price advantage on those days vs trading at that open / close.")
    print("  stock_cli uses 'buy at open P10' as its BUY price and 'sell at open P50' as its SELL price.")

    nd.section(f"2c. Exits on the 'Both agree ({m_live})' decision: take-profit / stop-loss, LSTM veto")
    sd = (sig["vol_fc"] / np.sqrt(wf.TRADING_DAYS)).reindex(dates).ffill()
    move = pd.Series(np.exp(o["preds"][m_live][o["test"], ci, 1] / 100) - 1, index=dates)
    veto = None
    if not a.no_lstm:
        print("  LSTM 20-day range (cached by stock_cli, else ~2 min) ...", flush=True)
        r, _ = mc.cached("lstm_range", ticker, day_f, {"days": 20, "seeds": 5},
                         lambda: lf.run(px_f, mkt_f, ticker, 20, 60, 5), a.refresh)
        veto = lstm_test_moves(r)
        on = dates[dates >= veto.index[0]]
        print(f"  LSTM test days cover {len(on)} of the {len(dates)} sessions; "
              f"its median pointed down on {(veto.reindex(on) < 0).mean():.0%} of them.")
    rows = {}
    for k_tp, k_sl in ((0, 0), (1.5, 0), (2, 0), (3, 0), (2, 2), (2, 3)):
        name = f"TP {k_tp:g}x / stop {k_sl:g}x" if k_tp or k_sl else "no exits"
        rows[name] = replay_exits(px_f, dates, p_s, move, sd, cfg, k_tp, k_sl)
        if veto is not None and k_sl == 0 and k_tp in (0, 2):
            rows[name + " + LSTM veto"] = replay_exits(px_f, dates, p_s, move, sd, cfg, k_tp, k_sl, veto)
    print(sc.fmt_table(pd.DataFrame(rows).T[["CAGR", "Sharpe", "MaxDD", "round trips", "won", "avg trip",
                                            "take-profits", "stops"]]))
    print("  TP / stop = k x the GARCH daily volatility from the buy price. Bought at the close, exits on")
    print("  later highs / lows. stock_cli defaults: TP 2x, no stop, no LSTM veto. Few trips = weak evidence.")

    # ================= 3. intraday =================
    live_today = px.attrs.get("provisional") and px.index[-1].date() == now.date()
    try:
        h_all, mkt_h = ic.load(ticker, cfg.market, cfg.data_dir, a.offline)
        # same cache key as nd.intraday_now during the session: the previous close
        key, today = (px.index[-2], px.index[-1].date()) if live_today else (px.index[-1], now.date() + pd.Timedelta(days=1))
        ist = cached_with_oos("intraday", ticker, key, {}, lambda: ic.fit(h_all, mkt_h, today, ticker), a.refresh)
        nd.section(f"3. Today's-close forecast during the session, last {ist['n_test']} sessions (hourly data)")
        print(nd.pct_table(ist["ev"], signed=("skill",)))
        print("  skill > 0 = beats the Baseline (close = latest price +/- volatility left today).")
        print("\n  From the price at the lunch break (12:00, Fri 11:30), last 10 test sessions:")
        print(checkpoint_sessions(ist["oos"], 11).to_string())
    except Exception as e:
        ist = None
        print(f"\n  intraday backtest skipped: {e}")

    # ================= 4. today so far =================
    if live_today:
        live_at, price = px.attrs.get("live_at"), float(px["Close"].iloc[-1])
        prev = float(px_f["Close"].iloc[-1])
        nd.section(f"4. Today so far: {price:,.0f} at {live_at:%H:%M} WIB ({price / prev - 1:+.1%} vs "
                   f"{prev:,.0f} prev close) vs every forecast made for today")
        ev = nd.forecast(st, px_f, mkt_f)
        made = [(f"{day_f:%a %d %b} close (evening)", m, prev * np.exp(ev["now"][m][ci] / 100)) for m in nd.MODELS]
        log = nd.read_log(nd.LOG)
        g = log[(log["ticker"] == ticker) & (log["stage"] == "intraday")
                & (log["bar_date"] == f"{day_f:%Y-%m-%d}")].sort_values("as_of")
        made += [(f"{x.as_of[-5:]} run (price {float(x.price_at):,.0f})", x.model,
                  np.array([x.close_p10, x.close_p50, x.close_p90], dtype=float)) for _, x in g.iterrows()]
        t = pd.DataFrame([{"made": w, "model": m, "close P10 / P50 / P90": nd.band_txt(*q),
                           f"{price:,.0f} now is": "below range" if price < q[0] else
                           ("ABOVE range" if price > q[2] else "inside")} for w, m, q in made])
        print(t.set_index(["made", "model"]).to_string())
        print("  These forecast today's CLOSE; the price now is only where it stands so far. They are")
        print("  scored properly after 16:15 WIB, in stock_cli's 'Last forecast vs what happened'.")
        if ist is not None:
            try:
                ir = nd.intraday_now(px, cfg, a.offline)
                for m, c in ir["now"].items():
                    print(f"  Now, from {price:,.0f} ({ir['checkpoint']}): {m:<9} close 80% {nd.band_txt(*c)}")
            except Exception as e:
                print(f"  live intraday forecast skipped: {e}")

    out = sc.OUT_DIR / ticker.lstrip("^")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v["pos"] for k, v in curves.items()}).assign(close=close.reindex(dates), p_smoothed=p_s) \
        .to_csv(out / "backtest_replay.csv")
    print(f"\n  Day-by-day positions: {(out / 'backtest_replay.csv').resolve()}")


if __name__ == "__main__":
    main()
