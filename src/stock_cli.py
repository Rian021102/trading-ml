"""
Interactive stock analyser
==========================

Type an IDX code (MTLA, bbca, BBRI.JK ...) and it runs the whole pipeline on
fresh data, using the latest intraday price when the market is open:

  1. walk-forward backtest of every strategy           (bbca_walkforward.py)
  2. today's BUY / SELL / HOLD / STAY OUT signal        (today_signal.py)
  3. next-session open & close forecast + scorecard     (next_day.py)
  4. LSTM 20-day price range forecast + chart           (lstm_forecast.py)
  5. during the session: today's close updated live     (intraday_close.py)
  6. summary: BUY / HOLD / SELL / WAIT, with order prices

Run it any time; the answer fits the clock (WIB):

  before 08:45     decision for today's session, with the expected open range
  09:00-16:00      the same decision on the live price (PROVISIONAL), plus
                   today's close range, for acting in the pre-closing
  after 16:15      final decision for the next session; the day is scored

It asks whether you own the stock: if you do the answer is HOLD or SELL, if
not BUY or WAIT. Two calls are shown side by side and the decision only
trades when they agree:

  signal call    smoothed P(up) vs the entry / exit levels (the backtested rule)
  forecast call  next close P50 of the best next-session model vs trading costs:
                 BUY above the round trip (0.40%), SELL below -0.25%

Every BUY and HOLD comes with a take-profit: a sell limit at the buy price
+ 2x the GARCH daily volatility (--take-profit; backtest_all.py section 2c
replays the multiples). If you own it you can give your average buy price, so
the target is measured from what you paid; once the price is at or above it
the answer is SELL. --stop-loss adds a stop the same way (off by default: in
the replay it helped some stocks and hurt others; the exit signal already
acts as a stop).

The LSTM 20-day median is shown next to the two calls but has no vote unless
you pass --lstm-veto: it has no edge over a random walk, and in the replay
vetoing BUYs on it cut returns.

Heavy models train only on finished sessions and are cached once per close
(output/cache/), so the first run after a close takes a few minutes and later
runs that day take seconds. --refresh retrains.

Press Esc at the prompt for the menu (1 = another company, 2 = exit).
Ctrl-C cancels a running analysis.

    python stock_cli.py
    python stock_cli.py MTLA BBCA                  # analyse these first, then prompt
    python stock_cli.py --no-lstm --no-open        # faster, don't open the chart

sweep.py, holdout.py and intraday_test.py are research tools and are not run
here. Every decision is appended to output/signals_log.csv, every forecast to
output/forecast_log.csv.
"""

from __future__ import annotations

import argparse
import os
import re
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import bbca_walkforward as wf
import holdout
import lstm_forecast as lf
import model_cache as mc
import next_day as nd
import today_signal as ts
from intraday_test import tick_size

OUT_DIR = Path(__file__).resolve().parent / "output"
LINE = "=" * 72


# ----------------------------------------------------------------------------
# Terminal input: Esc is detected as a key, not typed as text
# ----------------------------------------------------------------------------
def read_input(prompt: str, single: bool = False) -> str | None:
    """Read a line (or one key if `single`) with echo. Returns None on Esc or Ctrl-D.
    Off a terminal (pipes, IDEs without a TTY) falls back to input(), where typing
    'esc' counts as Esc and the end of the input raises EOFError (main() exits)."""
    try:
        import termios
    except ImportError:
        termios = None
    if termios is None or not sys.stdin.isatty():
        s = input(prompt)
        return None if s.strip().lower() == "esc" else s

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~(termios.ICANON | termios.ECHO)
    new[6][termios.VMIN], new[6][termios.VTIME] = 1, 0
    termios.tcflush(fd, termios.TCIFLUSH)          # drop keys pressed while the last analysis ran
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    out, buf = sys.stdout, []
    out.write(prompt)
    out.flush()
    try:
        while True:
            ch = os.read(fd, 1)
            if ch == b"\x1b":
                if select.select([fd], [], [], 0.05)[0]:   # arrow / function key: escape sequence, ignore
                    os.read(fd, 16)
                    continue
                out.write("\n")
                return None
            if ch == b"\x04":
                out.write("\n")
                return None
            if ch in (b"\r", b"\n"):
                out.write("\n")
                return "".join(buf)
            if ch in (b"\x7f", b"\x08"):
                if buf:
                    buf.pop()
                    out.write("\b \b")
            elif 32 <= ch[0] < 127:
                buf.append(ch.decode())
                out.write(ch.decode())
                if single:
                    out.write("\n")
                    return buf[0]
            out.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def menu() -> str:
    print("\n  1. Enter another company\n  2. Exit")
    while True:
        k = read_input("  Choose 1 or 2: ", single=True)
        if k is None or k == "1":                  # Esc again = back to the prompt
            return "1"
        if k == "2":
            return "2"


def normalize(code: str) -> str | None:
    """'mtla' -> 'MTLA.JK'; codes with a suffix ('BBCA.JK', 'AAPL.US') or '^' are kept."""
    t = code.strip().upper()
    if not t:
        return None
    if "." not in t and not t.startswith("^"):
        t += ".JK"
    return t if re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.\-]*", t) else None


def open_file(path: Path) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if sys.platform.startswith("win"):
        os.startfile(path)                          # noqa: S606 - local file we just wrote
    elif shutil.which(opener):
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
FMT = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:.1%}", "Calmar": "{:.2f}",
       "Exposure": "{:.0%}", "Trades/yr": "{:.0f}", "AUC": "{:.3f}", "HitRate": "{:.1%}"}


def fmt_table(tbl: pd.DataFrame) -> str:
    show = tbl.copy().astype(object)
    for c, f in FMT.items():
        if c in show:
            show[c] = tbl[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    return show.to_string()


def section(title: str) -> None:
    print(f"\n{'-' * 72}\n {title}\n{'-' * 72}")


def to_tick(p: float, up: bool) -> float:
    """Round a price to the IDX tick: down for buy limits, up for sell limits."""
    t = float(tick_size(np.array([p]))[0])
    return (np.ceil if up else np.floor)(p / t) * t


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def next_day_call(owned: bool, p_s: float, cfg: wf.Config) -> tuple[str, str]:
    """BUY / HOLD / SELL / WAIT for your actual position, with the backtest's entry/exit rule:
    buy when smoothed P(up) > entry, then hold until it drops below exit."""
    entry = cfg.prob_threshold
    ex = entry if cfg.exit_threshold is None else min(cfg.exit_threshold, entry)
    if owned:
        if p_s < ex:
            return "SELL", f"smoothed P(up) {p_s:.3f} is below the exit level {ex}"
        return "HOLD", f"smoothed P(up) {p_s:.3f} is still above the exit level {ex} (margin {p_s - ex:.3f})"
    if p_s > entry:
        return "BUY", f"smoothed P(up) {p_s:.3f} is above the entry level {entry}"
    return "WAIT - don't buy", f"smoothed P(up) {p_s:.3f} is below the entry level {entry} (needs +{entry - p_s:.3f})"


def exit_levels(base: float, sd: float, a: argparse.Namespace) -> tuple[float | None, float | None]:
    """Take-profit / stop-loss prices for a position bought at `base`: k x the daily
    volatility away, rounded to the tick (k = 0 turns a level off)."""
    tp = to_tick(base * np.exp(a.take_profit * sd), up=True) if a.take_profit > 0 else None
    sl = to_tick(base * np.exp(-a.stop_loss * sd), up=False) if a.stop_loss > 0 else None
    return tp, sl


def best_model(ev: pd.DataFrame, price: str) -> str:
    """Next-session model with the best test skill for 'open' or 'close'; Baseline if none beats it."""
    s = ev.xs(price, level="price")["skill"].drop("Baseline")
    return s.idxmax() if s.max() > 0 else "Baseline"


def forecast_call(owned: bool, fc: dict, cfg: wf.Config) -> tuple[str, str, str, float]:
    """BUY / HOLD / SELL / WAIT from the next-session close forecast: act only when the
    expected move beats what the trade costs. Returns (call, why, model, expected move)."""
    m = best_model(fc["ev"], "close")
    ref = float(fc["px"]["Close"].iloc[-1])
    p50 = ref * np.exp(fc["now"][m][nd.TARGETS.index("close"), 1] / 100)
    move = p50 / ref - 1
    hurdle = cfg.buy_cost + cfg.sell_cost
    if m == "Baseline":
        return ("HOLD" if owned else "WAIT - don't buy"), "no model beats the baseline on the close: no view", m, 0.0
    txt = f"{m} next close P50 {p50:,.0f} ({move:+.1%})"
    if owned:
        if move < -cfg.sell_cost:
            return "SELL", f"{txt} is below -{cfg.sell_cost:.2%}, the cost of selling", m, move
        return "HOLD", f"{txt}; it says SELL below -{cfg.sell_cost:.2%}", m, move
    if move > hurdle:
        return "BUY", f"{txt} clears the {hurdle:.2%} round-trip cost", m, move
    return "WAIT - don't buy", f"{txt}; needs more than +{hurdle:.2%} to cover costs", m, move


def combine(owned: bool, a: str, b: str) -> tuple[str, str]:
    """Trade only when both calls agree; otherwise keep the current position."""
    if a == b:
        return a, "both calls agree"
    return ("HOLD" if owned else "WAIT - don't buy"), "the calls disagree, so no new trade"


def band(q: np.ndarray) -> str:
    """P10 - P90 and the median, rounded to the IDX tick."""
    return f"{to_tick(q[0], up=False):,.0f} - {to_tick(q[2], up=True):,.0f}, median {nd.to_tick(q[1], 'near'):,.0f}"


def signal_fit(px: pd.DataFrame, mkt: pd.DataFrame, cfg: wf.Config) -> dict:
    """Walk-forward P(up) history, final classifiers (for today's and live rows) and GARCH
    volatility, all on finished sessions. Cached per close."""
    df = wf.build_features(px, mkt, cfg.horizon)
    cols = [c for c in df.columns if c not in ("ret", "target", "fwd_ret")]
    probs, importance = wf.walk_forward_predict(df, cols, cfg)
    data = df.dropna(subset=cols)
    train = data.iloc[: len(data) - cfg.horizon]
    train = train[train["target"].notna()]
    models = {c: m.fit(train[cols], train["target"].astype(int)) for c, m in wf.make_models(cfg).items()}
    return {"cols": cols, "probs": probs, "importance": importance, "models": models,
            "vol_fc": wf.garch_vol_forecast(df["ret"], cfg)}


def signal_probs(sig: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward history, with the last finished day and any live day predicted
    by the final classifiers (same as today_signal.refit_last)."""
    probs = sig["probs"].copy()
    # a live bar can lack a feature (e.g. close_loc when only one price has traded, so
    # High == Low); carry the previous day's value instead of silently dropping today
    rows = df[sig["cols"]].ffill()
    new = rows.index[rows.index >= probs.index[-1]]
    for c, m in sig["models"].items():
        p = pd.Series(m.predict_proba(rows.loc[new])[:, 1], index=new)
        probs = probs.reindex(probs.index.union(new))
        probs.loc[new, c] = p
    return probs


def last_check_line(log: pd.DataFrame, ticker: str) -> str | None:
    """One line: the latest scored evening forecast of the best model vs what happened."""
    s = nd.scored(log, ticker)
    if s.empty:
        return None
    g = s[s["bar_date"] == s["bar_date"].max()].set_index("model")
    m = "LightGBM" if "LightGBM" in g.index else g.index[0]
    x = g.loc[m]
    hit = "inside" if x.close_p10 <= x.actual_close <= x.close_p90 else "outside"
    return (f"{x.next_date} close forecast ({m}) {x.close_p50:,.0f} vs actual {x.actual_close:,.0f} "
            f"({x.actual_close / x.close_p50 - 1:+.1%}, {hit} its 80% range)")


def analyze(ticker: str, a: argparse.Namespace, owned: bool, cost: float | None = None) -> None:
    t_start = time.time()
    name = ticker.lstrip("^")
    cfg = wf.Config(ticker=ticker, horizon=1, prob_threshold=a.threshold, exit_threshold=a.exit_threshold,
                    smooth=a.smooth, data_dir=wf.Config.data_dir)
    now = datetime.now(ts.JKT)

    print(f"\n[1/5] Downloading {ticker} and {cfg.market} ...", flush=True)
    px = ts.load(ticker, cfg, a.offline)
    mkt = ts.load(cfg.market, cfg, a.offline)
    need = cfg.min_train + wf.TRADING_DAYS
    if len(px) < need:
        raise ValueError(f"{ticker} has only {len(px)} trading days; the walk-forward backtest needs "
                         f"at least {need} (~{need / wf.TRADING_DAYS:.0f} years).")
    print(f"  {len(px)} bars, {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}")
    live_at, provisional = px.attrs.get("live_at"), px.attrs.get("provisional", False)
    in_session = live_at is not None and provisional and nd.SESSION[0] <= now.time() < nd.SESSION[1]
    px_f, mkt_f, day_f = nd.finished(px, mkt)

    # ---- walk-forward backtest (shared by the table and the signal)
    print("[2/5] Signal: walk-forward models + GARCH (~1 min, once per close) ...", flush=True)
    sig, hit = mc.cached("signal", ticker, day_f, {"retrain": cfg.retrain, "min_train": cfg.min_train},
                         lambda: signal_fit(px_f, mkt_f, cfg), a.refresh)
    if hit:
        print(f"  using the models fitted after the {day_f:%Y-%m-%d} close (cached)")
    df = wf.build_features(px, mkt, cfg.horizon)
    probs, importance = signal_probs(sig, df), sig["importance"]
    oos = probs.index
    vol_fc = sig["vol_fc"].reindex(oos).ffill()
    positions = wf.make_positions(df, probs, vol_fc, cfg)
    ret, fwd_up = df["ret"].reindex(oos), df["target"].reindex(oos)
    curves, results = {}, {}
    for strat, pos in positions.items():
        curves[strat] = wf.backtest(pos, ret, cfg)
        pr = probs["p_logit"] if strat == "LogReg" else (probs["p_lgbm"] if strat.startswith("LightGBM") else None)
        results[strat] = wf.metrics(curves[strat], fwd_up, pr)
    tbl = pd.DataFrame(results).T
    last_year = {k: holdout.period_metrics(curves[k], start=oos[-wf.TRADING_DAYS])
                 for k in ("Buy & Hold", a.model)}

    # ---- today's signal
    held = curves[a.model]["pos"]
    p = probs[ts.MODEL_COL[a.model]]
    p_s = p.rolling(cfg.smooth, min_periods=1).mean()
    day, prev_pos, cur_pos = oos[-1], held.iloc[-2], held.iloc[-1]
    act = ts.action(prev_pos, cur_pos)
    close = px["Close"].reindex(oos)
    last = float(close.iloc[-1])

    # ---- next-session forecast, and today's close while the session runs
    print("[3/5] Next-session open/close forecast (~1 min, once per close) ...", flush=True)
    fc = nd.next_session(px, mkt, ticker, a.refresh, n_seeds=3)
    r = None
    if not a.no_lstm:
        print(f"[4/5] LSTM {a.days}-day range forecast ({a.seeds} seeds x 2 fits, once per close) ...", flush=True)
        try:
            r, hit = mc.cached("lstm_range", ticker, day_f, {"days": a.days, "seeds": a.seeds},
                               lambda: lf.run(px_f, mkt_f, ticker, a.days, 60, a.seeds), a.refresh)
            if hit:
                print("  using the range fitted after the last close (cached)")
        except Exception as e:                      # short history etc.: keep the rest of the report
            print(f"  LSTM skipped: {e}")
    ir = None
    print("[5/5] " + ("Intraday close forecast + report ..." if in_session else "Writing report ..."), flush=True)
    if in_session:
        try:
            ir = nd.intraday_now(px, cfg, a.offline, a.refresh)
        except Exception as e:
            print(f"  intraday forecast skipped: {e}")
    out_t = OUT_DIR / name
    out_t.mkdir(parents=True, exist_ok=True)
    realized = df["ret"].rolling(21).std().shift(-20) * np.sqrt(wf.TRADING_DAYS)
    wf.build_report(out_t / "report.html", label=ticker, cfg=cfg, tbl=tbl, curves=curves, close=close,
                    probs=probs, fwd_up=fwd_up, vol_fc=vol_fc, realized_vol=realized.reindex(oos),
                    importance=importance)
    png = None
    if r is not None:
        _, png = lf.save(r, OUT_DIR, wf.Config.img_dir)

    # ---- forecast log: evening rows after the close, intraday rows during the session
    new = None
    if not provisional:
        new = nd.forecast_rows(fc, now)
    elif ir is not None:
        new = nd.intraday_rows(ir, px.index[-2], now)
    flog, logged = nd.update_log(nd.LOG, px, ticker, provisional, new, lock=not provisional)

    # ================= output =================
    section(f"{ticker} walk-forward backtest, {oos[0]:%Y-%m-%d} -> {oos[-1]:%Y-%m-%d} (after costs)")
    print(f"Entry > {cfg.prob_threshold}, exit < {cfg.exit_threshold}, P(up) smoothed {cfg.smooth}d | "
          f"costs buy {cfg.buy_cost:.2%} / sell {cfg.sell_cost:.2%}\n")
    print(fmt_table(tbl))

    section(f"Last {a.history} sessions ({a.model})")
    k = a.history
    recent = pd.DataFrame({
        "close": close.iloc[-k:].map("{:,.0f}".format),
        "P(up)": p.iloc[-k:].map("{:.3f}".format),
        "smoothed": p_s.iloc[-k:].map("{:.3f}".format),
        "position": held.iloc[-k:].map("{:.0%}".format),
        "action": [ts.action(x, y) for x, y in zip(held.shift(1).fillna(0).iloc[-k:], held.iloc[-k:])],
    })
    recent.index = recent.index.strftime("%Y-%m-%d")
    print(recent.to_string())

    if r is not None:
        section(f"LSTM {a.days}-day price range (from the {day_f:%Y-%m-%d} close)")
        lf.report(r)

    nd.report(fc, flog, provisional)
    if ir is not None:
        nd.report_intraday(ir, px.index[-2], flog)

    # ---- summary
    price_txt = (f"live {live_at:%H:%M} WIB, PROVISIONAL" if provisional and live_at is not None
                 else "close")
    nxt = day + pd.offsets.BDay(1)
    if in_session:
        when = "today's pre-closing (15:50-16:00)"
    elif provisional:
        when = "the closing auction is running; re-run after 16:15 WIB for tomorrow"
    else:
        when = (f"{'today, ' if nxt.date() == now.date() else ''}{nxt:%a %d %b}, "
                f"at the open or the pre-closing (15:50-16:00)")
    s_call, s_why = next_day_call(owned, float(p_s.iloc[-1]), cfg)
    f_call, f_why, f_model, f_move = forecast_call(owned, fc, cfg)
    call, why = combine(owned, s_call, f_call)
    size = 1.0
    if a.model == "LightGBM+GARCH" and pd.notna(vol_fc.iloc[-1]):
        size = min(cfg.vol_target / vol_fc.iloc[-1], cfg.max_leverage)

    print(f"\n{LINE}")
    print(f"  {ticker}  SUMMARY   {now:%a %Y-%m-%d %H:%M} WIB   last price {last:,.0f} ({price_txt})")
    print(LINE)
    print(f"  You {'own' if owned else 'do not own'} it. Decision for {when}:\n")
    print(f"  {'Signal call':<15}{s_call:<18}{s_why}")
    print(f"  {'Forecast call':<15}{f_call:<18}{f_why}")
    # one price per action. Today: the expected close, as a limit in the closing auction.
    # Tomorrow: the limits that held up in backtest_all.py - buy at the low end of the
    # open range, sell at its median - each placed for the whole session
    ci = nd.TARGETS.index("close")
    m_close, m_open = best_model(fc["ev"], "close"), best_model(fc["ev"], "open")
    c_q = last * np.exp(fc["now"][m_close][ci] / 100)
    o_q = last * np.exp(fc["now"][m_open][nd.TARGETS.index("gap")] / 100)
    lim = nd.limit_check(fc["oos"], fc["px"], m_open) if fc.get("oos") is not None else None
    fills = {"BUY": "buy at open P10", "SELL": "sell at open P50"}
    tom = {"BUY": to_tick(o_q[0], up=False), "SELL": nd.to_tick(o_q[1], "near")}
    tom_txt = {k: f"{k} at {v:,.0f}" + ("" if lim is None else f" (filled {lim.at[fills[k], 'fills']:.0%} of test days)")
               for k, v in tom.items()}
    t = None
    if ir is not None:
        m_now = "LightGBM" if ir["ev"]["skill"].get((ir["checkpoint"], "LightGBM"), 0.0) > 0 else "Baseline"
        t = ir["now"][m_now]
        order = {k: f"{k} at {nd.to_tick(t[1], 'near'):,.0f}  - limit in today's closing auction, 15:50-16:00"
                 for k in ("BUY", "SELL")}
    elif not provisional:
        order = {k: f"{tom_txt[k]}  - limit order for {nxt:%a %d %b}" for k in ("BUY", "SELL")}
    else:
        order = {k: f"{k}  - price after the 16:15 run" for k in ("BUY", "SELL")}

    # ---- exits, measured from your buy price (or today's price), or from the BUY order's price
    sd = float(vol_fc.iloc[-1]) / np.sqrt(wf.TRADING_DAYS) if pd.notna(vol_fc.iloc[-1]) \
        else float(df["ret"].iloc[-63:].std())
    buy_at = nd.to_tick(t[1], "near") if t is not None else (last if provisional else tom["BUY"])
    base = (cost or last) if owned else buy_at
    base_txt = ("your buy price" if cost else "today's price") if owned else "the buy price"
    tp, sl = exit_levels(base, sd, a)
    if owned and call != "SELL":
        if tp is not None and last >= tp:
            call, why = "SELL", f"price {last:,.0f} is at or above the take-profit {tp:,.0f}"
            order["SELL"] = f"SELL at {tp:,.0f} or better  - take-profit limit, fills at once while the price is above it"
        elif sl is not None and last <= sl:
            call, why = "SELL", f"price {last:,.0f} is at or below the stop-loss {sl:,.0f}"

    # ---- LSTM 20-day median: shown next to the calls, blocks a BUY only with --lstm-veto
    lstm_move = None
    if r is not None:
        med = float(r["fc"].iloc[-1].lstm_p50)
        lstm_move = med / last - 1
        ref = last if owned else base
        if a.lstm_veto and call == "BUY" and med < ref:
            call, why = "WAIT - don't buy", f"LSTM {a.days}-day median {med:,.0f} is below the buy price {ref:,.0f}"
        bull = call == "BUY" or (owned and call == "HOLD")
        print(f"  {'LSTM veto' if a.lstm_veto else 'LSTM (no vote)':<15}"
              f"{'agrees' if (med >= ref) == bull else 'DISAGREES':<18}"
              f"{a.days}-day median {med:,.0f} ({lstm_move:+.1%}) vs {'price now' if owned else 'buy price'} {ref:,.0f}")

    exits = []
    if tp is not None:
        exits.append(f"Take profit: SELL limit at {tp:,.0f}  ({tp / base - 1:+.1%} from {base_txt} {base:,.0f}, "
                     f"{a.take_profit:g} x daily vol {sd:.1%})")
    if sl is not None:
        exits.append(f"Stop loss  : SELL if it trades at {sl:,.0f}  ({sl / base - 1:+.1%}, {a.stop_loss:g} x daily vol)")

    if call in order:
        print(f"\n  >>> {order[call]}"
              + (f"  ({size:.0%} of the money you'd put in this stock)" if call == "BUY" else ""))
        print(f"      ({why})")
        if call == "BUY":
            for x in exits:
                print(f"      {x}")
            print("      " + ("Place the take-profit as soon as the buy fills and leave it in; otherwise h"
                             if tp is not None else "H") + "old until this says SELL.")
    else:
        turn = "SELL" if owned else "BUY"
        print(f"\n  >>> {call}  ({why})")
        if owned:
            for x in exits:
                print(f"      {x}")
        print(f"      If it turns {turn}: {order[turn]}"
              + ("" if owned or tp is None else f", then take profit at {tp:,.0f} ({tp / base - 1:+.1%})"))
        if s_call != f_call:
            print("      Re-run later: the decision trades once both calls point the same way.")

    print()
    if t is not None:
        print(f"  {'Today close 80% range':<26}: {band(t)} ({m_now}, from {ir['price']:,.0f} at {ir['live_at']:%H:%M})")
    lbl = "Tomorrow (prov.)" if provisional else f"{nxt:%a %d %b}"
    print(f"  {lbl + ' open 80%':<26}: {band(o_q)} ({m_open})")
    print(f"  {lbl + ' close 80%':<26}: {band(c_q)} ({m_close})")

    model_owns = cur_pos > 0
    print(f"\n  Model's own paper position: {act} (P(up) today {p.iloc[-1]:.3f})"
          + ("" if model_owns == owned else
             f"\n  Note: the model is {'in' if model_owns else 'out of'} the stock but you are "
             f"{'in' if owned else 'out'}; the calls above use YOUR position."))
    if r is not None:
        e = r["fc"].iloc[-1]
        print(f"  {f'{a.days}-day 80% range':<26}: {e.lstm_p10:,.0f} - {e.lstm_p90:,.0f}, "
              f"median {e.lstm_p50:,.0f} ({e.lstm_p50 / last - 1:+.1%})"
              + ("  (volatility range, no edge: it has no vote)" if (r["skill"] <= 0.02).all() else ""))
    chk = last_check_line(flog, ticker)
    if chk:
        print(f"  {'Last forecast check':<26}: {chk}")

    m, b = last_year[a.model], last_year["Buy & Hold"]
    s_all, b_all = tbl.loc[a.model], tbl.loc["Buy & Hold"]
    print(f"\n  Backtest since {oos[0]:%Y}: strategy CAGR {s_all.CAGR:.1%} / Sharpe {s_all.Sharpe:.2f} / "
          f"MaxDD {s_all.MaxDD:.1%}   buy&hold {b_all.CAGR:.1%} / {b_all.Sharpe:.2f} / {b_all.MaxDD:.1%}")
    print(f"  Last 12 months  : strategy CAGR {m['CAGR']:.1%} / Sharpe {m['Sharpe']:.2f}   "
          f"buy&hold {b['CAGR']:.1%} / {b['Sharpe']:.2f}")
    if provisional:
        print("\n  PROVISIONAL: built on an unfinished bar. Re-run after 16:15 WIB to confirm.")
    print("  Not validated: neither call has a proven live record yet. Paper-trade it and watch the logs.")
    print(LINE)

    log = OUT_DIR / "signals_log.csv"
    ts.append_log(log, pd.DataFrame([{
        "bar_date": f"{day:%Y-%m-%d}", "ticker": ticker, "model": a.model, "horizon": cfg.horizon,
        "entry": cfg.prob_threshold, "exit": cfg.exit_threshold, "smooth": cfg.smooth,
        "close": round(last, 2), "p_up": round(float(p.iloc[-1]), 4),
        "p_smoothed": round(float(p_s.iloc[-1]), 4), "position": round(float(cur_pos), 4),
        "action": act, "owned": owned, "advice": call, "signal_advice": s_call,
        "forecast_advice": f_call, "forecast_model": f_model, "forecast_move": round(f_move, 4),
        "lstm_20d_move": None if lstm_move is None else round(lstm_move, 4),
        "buy_price": cost, "take_profit": tp, "stop_loss": sl,
        "provisional": provisional,
        "price_at": "" if live_at is None else f"{live_at:%Y-%m-%d %H:%M}",
        "run_at": now.strftime("%Y-%m-%d %H:%M WIB"),
    }]))
    print(f"  Report : {(out_t / 'report.html').resolve()}")
    if png is not None:
        print(f"  Chart  : {png.resolve()}")
    print(f"  Logged : {log.resolve()}" + (f"\n           {nd.LOG.resolve()}" if logged else "")
          + f"   ({time.time() - t_start:.0f}s)")
    if png is not None and not a.no_open:
        open_file(png)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("codes", nargs="*", help="stock codes to analyse before prompting")
    ap.add_argument("--model", default="LightGBM", choices=list(ts.MODEL_COL))
    ap.add_argument("--threshold", type=float, default=0.55, help="enter long when P(up) > this")
    ap.add_argument("--exit-threshold", type=float, default=0.50, help="exit when P(up) < this")
    ap.add_argument("--smooth", type=int, default=3, help="average P(up) over N days first")
    ap.add_argument("--days", type=int, default=20, help="LSTM forecast horizon in trading days")
    ap.add_argument("--seeds", type=int, default=5, help="LSTM ensemble size (fewer = faster)")
    ap.add_argument("--history", type=int, default=10, help="recent sessions to show")
    ap.add_argument("--take-profit", type=float, default=2.0,
                    help="take-profit at the buy price + this x daily GARCH volatility (0 = off)")
    ap.add_argument("--stop-loss", type=float, default=0.0,
                    help="stop-loss at the buy price - this x daily GARCH volatility (0 = off)")
    ap.add_argument("--lstm-veto", action="store_true",
                    help="don't BUY when the LSTM 20-day median is below the buy price")
    ap.add_argument("--no-lstm", action="store_true", help="skip the LSTM forecast (~2 min faster)")
    ap.add_argument("--no-open", action="store_true", help="don't open the forecast chart")
    ap.add_argument("--offline", action="store_true", help="use the CSVs in data/ instead of downloading")
    ap.add_argument("--refresh", action="store_true", help="retrain instead of using the models cached at the last close")
    a = ap.parse_args()

    print(f"{LINE}\n  IDX stock analyser - type a code (e.g. MTLA), Enter to run. "
          f"Esc = menu, Ctrl-C = cancel.\n{LINE}")
    queue = list(a.codes)
    try:
        while True:
            raw = queue.pop(0) if queue else read_input("\nStock code: ")
            if raw is None:
                if menu() == "2":
                    break
                continue
            ticker = normalize(raw)
            if ticker is None:
                if raw.strip():
                    print(f"  '{raw.strip()}' doesn't look like a stock code.")
                continue
            ans = read_input(f"  Do you own {ticker} now? [y/N] (Esc = cancel): ", single=True)
            if ans is None:
                continue
            owned, cost = ans.strip().lower() == "y", None
            if owned:
                c = read_input("  Your average buy price? (Enter = skip, Esc = cancel): ")
                if c is None:
                    continue
                c = c.strip().replace(",", "").replace(".", "")    # 5,075 / 5.075 -> 5075
                cost = float(c) if c.isdigit() and int(c) > 0 else None
            try:
                analyze(ticker, a, owned=owned, cost=cost)
            except KeyboardInterrupt:
                print("\n  Cancelled.")
            except (Exception, SystemExit) as e:
                print(f"\n  Could not analyse {ticker}: {e}")
            print("\n  (type another code, or Esc for the menu)")
    except (KeyboardInterrupt, EOFError):
        print()
    print("Bye.")


if __name__ == "__main__":
    main()
