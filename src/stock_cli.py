"""
Interactive stock analyser
==========================

Type an IDX code (MTLA, bbca, BBRI.JK ...) and it runs the whole pipeline on
fresh data, using the latest intraday price when the market is open:

  1. walk-forward backtest of every strategy        (bbca_walkforward.py)
  2. today's BUY / SELL / HOLD / STAY OUT signal     (today_signal.py)
  3. LSTM 20-day price range forecast + chart        (lstm_forecast.py)
  4. summary: BUY / HOLD / SELL / WAIT for the next session, with order prices

It asks whether you own the stock, because the entry/exit rule depends on it:
if you own it the answer is HOLD or SELL, if not it is BUY or WAIT.

Press Esc at the prompt for the menu (1 = another company, 2 = exit).
Ctrl-C cancels a running analysis.

    python stock_cli.py
    python stock_cli.py MTLA BBCA                  # analyse these first, then prompt
    python stock_cli.py --no-lstm --no-open        # faster, don't open the chart

One stock takes ~3-4 minutes (walk-forward + GARCH ~1-2 min, LSTM ~2 min on a
GPU; --no-lstm or --seeds 2 for speed). sweep.py,
holdout.py and intraday_test.py are research tools and are not run here.
Every signal is appended to output/signals_log.csv, same as today_signal.py.
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
    'esc' counts as Esc."""
    try:
        import termios
    except ImportError:
        termios = None
    if termios is None or not sys.stdin.isatty():
        try:
            s = input(prompt)
        except EOFError:
            return None
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


def analyze(ticker: str, a: argparse.Namespace, owned: bool) -> None:
    t_start = time.time()
    name = ticker.lstrip("^")
    cfg = wf.Config(ticker=ticker, horizon=1, prob_threshold=a.threshold, exit_threshold=a.exit_threshold,
                    smooth=a.smooth, data_dir=wf.Config.data_dir)

    print(f"\n[1/4] Downloading {ticker} and {cfg.market} ...", flush=True)
    px = ts.load(ticker, cfg, a.offline)
    mkt = ts.load(cfg.market, cfg, a.offline)
    need = cfg.min_train + wf.TRADING_DAYS
    if len(px) < need:
        raise ValueError(f"{ticker} has only {len(px)} trading days; the walk-forward backtest needs "
                         f"at least {need} (~{need / wf.TRADING_DAYS:.0f} years).")
    print(f"  {len(px)} bars, {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}")

    # ---- walk-forward backtest (shared by the table and the signal)
    print("[2/4] Walk-forward models + GARCH (~1 min) ...", flush=True)
    df = wf.build_features(px, mkt, cfg.horizon)
    feature_cols = [c for c in df.columns if c not in ("ret", "target", "fwd_ret")]
    probs, importance = wf.walk_forward_predict(df, feature_cols, cfg)
    probs = ts.refit_last(df, feature_cols, probs, cfg)
    oos = probs.index
    vol_fc = wf.garch_vol_forecast(df["ret"], cfg).reindex(oos)
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
    live_at, provisional = px.attrs.get("live_at"), px.attrs.get("provisional", False)

    # ---- LSTM range forecast
    r = None
    if not a.no_lstm:
        print(f"[3/4] LSTM {a.days}-day range forecast ({a.seeds} seeds x 2 fits) ...", flush=True)
        try:
            r = lf.run(px, mkt, ticker, a.days, 60, a.seeds)
        except Exception as e:                      # short history etc.: keep the rest of the report
            print(f"  LSTM skipped: {e}")
    print("[4/4] Writing report ...", flush=True)
    out_t = OUT_DIR / name
    out_t.mkdir(parents=True, exist_ok=True)
    realized = df["ret"].rolling(21).std().shift(-20) * np.sqrt(wf.TRADING_DAYS)
    wf.build_report(out_t / "report.html", label=ticker, cfg=cfg, tbl=tbl, curves=curves, close=close,
                    probs=probs, fwd_up=fwd_up, vol_fc=vol_fc, realized_vol=realized.reindex(oos),
                    importance=importance)
    png = None
    if r is not None:
        _, png = lf.save(r, OUT_DIR, wf.Config.img_dir)

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
        section(f"LSTM {a.days}-day price range")
        lf.report(r)

    # ---- summary
    price_txt = (f"live {live_at:%H:%M} WIB, PROVISIONAL" if provisional and live_at is not None
                 else "close")
    call, why = next_day_call(owned, float(p_s.iloc[-1]), cfg)
    size = 1.0
    if a.model == "LightGBM+GARCH" and pd.notna(vol_fc.iloc[-1]):
        size = min(cfg.vol_target / vol_fc.iloc[-1], cfg.max_leverage)
    when = ("today's pre-closing (15:50-16:00)" if provisional
            else f"{day + pd.offsets.BDay(1):%a %d %b}, at the open or the pre-closing (15:50-16:00)")
    print(f"\n{LINE}")
    print(f"  {ticker}  SUMMARY   {day:%a %Y-%m-%d}  last price {last:,.0f} ({price_txt})")
    print(LINE)
    print(f"  You {'own' if owned else 'do not own'} it. Suggestion for the next session ({when}):\n")
    print(f"  >>> {call}" + (f"  ({size:.0%} of the money you'd put in this stock)" if call == "BUY" else ""))
    print(f"      {why}")

    d1 = r["fc"].iloc[0] if r is not None else None
    if call == "BUY":
        print(f"\n  Buy price : around {last:,.0f} - the price the backtest assumes.")
        if d1 is not None:
            print(f"              patient limit {to_tick(min(d1.lstm_p50, last), up=False):,.0f} "
                  f"(next-session median; fills only on a dip), range low {to_tick(d1.lstm_p10, up=False):,.0f}")
    elif call == "SELL":
        print(f"\n  Sell price: around {last:,.0f} - the price the backtest assumes.")
        if d1 is not None:
            print(f"              patient limit {to_tick(max(d1.lstm_p50, last), up=True):,.0f} "
                  f"(next-session median; fills only on a bounce), range high {to_tick(d1.lstm_p90, up=True):,.0f}")
    elif call == "HOLD":
        print(f"\n  No order. It turns SELL once smoothed P(up) drops below {cfg.exit_threshold}; re-run daily.")
    else:
        print(f"\n  No order. It turns BUY once smoothed P(up) rises above {cfg.prob_threshold}; re-run daily.")
    model_owns = cur_pos > 0
    print(f"\n  Model's own paper position: {act} (P(up) today {p.iloc[-1]:.3f})"
          + ("" if model_owns == owned else
             f"\n  Note: the model is {'in' if model_owns else 'out of'} the stock but you are "
             f"{'in' if owned else 'out'}; the call above uses YOUR position."))
    if r is not None:
        e = r["fc"].iloc[-1]
        print(f"  {'Next session 80% range':<24}: {d1.lstm_p10:,.0f} - {d1.lstm_p90:,.0f}")
        print(f"  {f'{a.days}-day 80% range':<24}: {e.lstm_p10:,.0f} - {e.lstm_p90:,.0f}, "
              f"median {e.lstm_p50:,.0f} ({e.lstm_p50 / last - 1:+.1%})")
        skill = r["skill"]
        print(f"  {'LSTM vs random walk':<24}: skill {skill.min():+.1%} .. {skill.max():+.1%}"
              + ("  -> no edge, read it as a volatility range" if (skill <= 0.02).all() else ""))

    m, b = last_year[a.model], last_year["Buy & Hold"]
    s_all, b_all = tbl.loc[a.model], tbl.loc["Buy & Hold"]
    print(f"\n  Backtest since {oos[0]:%Y}: strategy CAGR {s_all.CAGR:.1%} / Sharpe {s_all.Sharpe:.2f} / "
          f"MaxDD {s_all.MaxDD:.1%}   buy&hold {b_all.CAGR:.1%} / {b_all.Sharpe:.2f} / {b_all.MaxDD:.1%}")
    print(f"  Last 12 months  : strategy CAGR {m['CAGR']:.1%} / Sharpe {m['Sharpe']:.2f}   "
          f"buy&hold {b['CAGR']:.1%} / {b['Sharpe']:.2f}")
    if provisional:
        print("\n  PROVISIONAL: built on an unfinished bar. Re-run after 16:15 WIB to confirm.")
    print("  Not validated: the hold-out test found no setting that reliably carries forward. Paper-trade it.")
    print(LINE)

    log = OUT_DIR / "signals_log.csv"
    ts.append_log(log, pd.DataFrame([{
        "bar_date": f"{day:%Y-%m-%d}", "ticker": ticker, "model": a.model, "horizon": cfg.horizon,
        "entry": cfg.prob_threshold, "exit": cfg.exit_threshold, "smooth": cfg.smooth,
        "close": round(last, 2), "p_up": round(float(p.iloc[-1]), 4),
        "p_smoothed": round(float(p_s.iloc[-1]), 4), "position": round(float(cur_pos), 4),
        "action": act, "owned": owned, "advice": call, "provisional": provisional,
        "price_at": "" if live_at is None else f"{live_at:%Y-%m-%d %H:%M}",
        "run_at": datetime.now(ts.JKT).strftime("%Y-%m-%d %H:%M WIB"),
    }]))
    print(f"  Report : {(out_t / 'report.html').resolve()}")
    if png is not None:
        print(f"  Chart  : {png.resolve()}")
    print(f"  Logged : {log.resolve()}   ({time.time() - t_start:.0f}s)")
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
    ap.add_argument("--no-lstm", action="store_true", help="skip the LSTM forecast (~2 min faster)")
    ap.add_argument("--no-open", action="store_true", help="don't open the forecast chart")
    ap.add_argument("--offline", action="store_true", help="use the CSVs in data/ instead of downloading")
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
            try:
                analyze(ticker, a, owned=ans.strip().lower() == "y")
            except KeyboardInterrupt:
                print("\n  Cancelled.")
            except (Exception, SystemExit) as e:
                print(f"\n  Could not analyse {ticker}: {e}")
            print("\n  (type another code, or Esc for the menu)")
    except KeyboardInterrupt:
        print()
    print("Bye.")


if __name__ == "__main__":
    main()
