"""
Parameter sweep: horizon x entry threshold x exit threshold x smoothing
=======================================================================

Walk-forward models are trained once per horizon (the expensive part); every
threshold / smoothing combination is then backtested on those same
out-of-sample probabilities, after costs.

Uses the CSVs saved in ../data by bbca_walkforward.py (no re-download), so run
that once first, or pass --download.

    python sweep.py
    python sweep.py --horizons 1 5 10 --models LogReg LightGBM+GARCH

Results go to output/sweep.csv. Each row also reports CAGR on the first and
second half of the out-of-sample period: a setting that only works in one half
was probably picked by luck. Picking the best row out of hundreds overstates
what you'd get live, so treat the top of the table as a shortlist, not a result.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import time
from pathlib import Path

import pandas as pd

import bbca_walkforward as wf

MODELS = ("LogReg", "LightGBM", "LightGBM+GARCH")


def load(ticker: str, cfg: wf.Config, download: bool) -> pd.DataFrame:
    if download:
        return wf.load_yahoo(ticker, cfg.start, cfg.data_dir)
    path = Path(cfg.data_dir) / f"{ticker.lstrip('^')}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run bbca_walkforward.py first or pass --download")
    return wf.load_csv(path)


def cagr(ret: pd.Series) -> float:
    return (1 + ret).prod() ** (wf.TRADING_DAYS / len(ret)) - 1


def run_grid(px, mkt, base: wf.Config, horizons, entries, exits, smooths, models):
    """Yield (params, backtest) for Buy & Hold once, then every model x setting.
    Models are trained once per horizon; all horizons share the same OOS index."""
    df1 = wf.build_features(px, mkt, 1)
    print("Walk-forward GARCH volatility ...", flush=True)
    vol_fc = wf.garch_vol_forecast(df1["ret"], base)

    first = True
    for h in horizons:
        t0 = time.time()
        cfg_h = dataclasses.replace(base, horizon=h)
        df = wf.build_features(px, mkt, h)
        feature_cols = [c for c in df.columns if c not in ("ret", "target", "fwd_ret")]
        probs, _ = wf.walk_forward_predict(df, feature_cols, cfg_h)
        oos = probs.index
        ret = df["ret"].reindex(oos)

        if first:   # benchmark once, same OOS window for every horizon
            first = False
            yield ({"model": "Buy & Hold", "horizon": None, "entry": None, "exit": None, "smooth": None},
                   wf.backtest(pd.Series(1.0, index=oos), ret, base))

        for entry, ex, sm in itertools.product(entries, [None, *exits], smooths):
            if ex is not None and ex >= entry:
                continue   # no hysteresis -> already covered by ex=None
            cfg = dataclasses.replace(cfg_h, prob_threshold=entry, exit_threshold=ex, smooth=sm)
            positions = wf.make_positions(df, probs, vol_fc, cfg)
            for name in models:
                yield ({"model": name, "horizon": h, "entry": entry,
                        "exit": entry if ex is None else ex, "smooth": sm},
                       wf.backtest(positions[name], ret, cfg))
        print(f"  horizon {h:>2}d done in {time.time() - t0:.0f}s", flush=True)


def add_grid_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ticker", default=wf.Config.ticker)
    ap.add_argument("--market", default=wf.Config.market, help="market index ticker, or 'none'")
    ap.add_argument("--download", action="store_true", help="fetch fresh data from Yahoo instead of ../data")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 21])
    ap.add_argument("--entries", type=float, nargs="+", default=[0.50, 0.52, 0.55, 0.58, 0.60])
    ap.add_argument("--exits", type=float, nargs="+", default=[0.50, 0.48, 0.45],
                    help="exit thresholds; the entry itself (no hysteresis) is always included")
    ap.add_argument("--smooths", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--buy-cost", type=float, default=wf.Config.buy_cost)
    ap.add_argument("--sell-cost", type=float, default=wf.Config.sell_cost)
    ap.add_argument("--out", default=wf.Config.out_dir)


def setup(a: argparse.Namespace):
    """Config + price data from parsed grid args."""
    base = wf.Config(ticker=a.ticker, market=a.market, buy_cost=a.buy_cost, sell_cost=a.sell_cost)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    px = load(base.ticker, base, a.download)
    mkt = None if base.market.lower() == "none" else load(base.market, base, a.download)
    print(f"{base.ticker}: {len(px)} bars | costs buy {base.buy_cost:.2%} / sell {base.sell_cost:.2%}")
    return base, px, mkt


def grid(a: argparse.Namespace, px, mkt, base: wf.Config):
    return run_grid(px, mkt, base, a.horizons, a.entries, a.exits, a.smooths, a.models)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_grid_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    base, px, mkt = setup(a)

    rows = []
    for params, bt in grid(a, px, mkt, base):
        oos = bt.index
        half = oos[len(oos) // 2]
        rows.append({**params, **wf.metrics(bt),
                     "CAGR_1st": cagr(bt["ret"][:half]), "CAGR_2nd": cagr(bt["ret"][half:])})

    res = pd.DataFrame(rows)
    res.to_csv(out / "sweep.csv", index=False)

    bh = res.iloc[0]
    print(f"\nOOS {oos[0]:%Y-%m-%d} -> {oos[-1]:%Y-%m-%d}, halves split at {half:%Y-%m-%d}")
    print(f"Buy & Hold: CAGR {bh.CAGR:.1%}  Sharpe {bh.Sharpe:.2f}  MaxDD {bh.MaxDD:.1%}  "
          f"(1st half {bh.CAGR_1st:.1%}, 2nd half {bh.CAGR_2nd:.1%})")

    strat = res.iloc[1:].copy()
    cols = ["model", "horizon", "entry", "exit", "smooth", "CAGR", "Sharpe", "MaxDD",
            "Exposure", "Trades/yr", "CAGR_1st", "CAGR_2nd"]
    pct = {"CAGR", "MaxDD", "Exposure", "CAGR_1st", "CAGR_2nd"}

    def show(t: pd.DataFrame) -> str:
        t = t[cols].copy().astype(object)
        for c in cols:
            if c in pct:
                t[c] = t[c].map(lambda v: f"{v:.1%}")
            elif c in ("Sharpe",):
                t[c] = t[c].map(lambda v: f"{v:.2f}")
            elif c == "Trades/yr":
                t[c] = t[c].map(lambda v: f"{v:.0f}")
            elif c == "horizon" or c == "smooth":
                t[c] = t[c].map(lambda v: f"{int(v)}")
        return t.to_string(index=False)

    print(f"\nTop 15 of {len(strat)} by Sharpe (after costs):")
    print(show(strat.sort_values("Sharpe", ascending=False).head(15)))

    print("\nBest Sharpe per model x horizon:")
    best = strat.loc[strat.groupby(["model", "horizon"])["Sharpe"].idxmax()]
    print(show(best.sort_values(["model", "horizon"])))

    robust = strat[(strat.CAGR_1st > 0) & (strat.CAGR_2nd > 0)]
    beat = robust[robust.Sharpe > bh.Sharpe]
    print(f"\n{len(robust)} of {len(strat)} settings are profitable in both halves; "
          f"{len(beat)} of those also beat Buy & Hold's Sharpe.")
    print(f"Saved {out.resolve() / 'sweep.csv'}")


if __name__ == "__main__":
    main()
