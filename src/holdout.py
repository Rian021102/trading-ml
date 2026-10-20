"""
Hold-out test for the sweep
===========================

sweep.py ranks settings on the whole out-of-sample period, so the "best" rows
are partly luck. This script splits that period in two:

  selection period  OOS start -> --split   pick settings here only
  test period       --split  -> OOS end    score the picked settings, untouched

The models themselves are walk-forward in both periods; what is held out here
is the *choice of horizon / thresholds / smoothing*. If settings that won in the
selection period keep winning in the test period, the edge is more likely real.

    python holdout.py                      # split at the middle of the OOS period
    python holdout.py --split 2021-12-31
    python holdout.py --top 20             # also average the top-20 selected settings

Results go to output/holdout.csv (one row per setting, metrics for both periods).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import bbca_walkforward as wf
import sweep


def period_metrics(bt: pd.DataFrame, start=None, end=None) -> dict:
    """Metrics on a slice of a backtest, with equity rebased to 1 at the slice start.
    Positions carry over from before the slice, exactly as they would live."""
    part = bt.loc[start:end].copy()
    part["equity"] = (1 + part["ret"]).cumprod()
    return wf.metrics(part)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sweep.add_grid_args(ap)
    ap.add_argument("--split", help="first day of the test period (default: middle of the OOS period)")
    ap.add_argument("--select-by", default="Sharpe", choices=["Sharpe", "CAGR", "Calmar"])
    ap.add_argument("--top", type=int, default=10, help="also report the average of the top-N selected settings")
    a = ap.parse_args()
    out = Path(a.out)
    base, px, mkt = sweep.setup(a)

    rows, split = [], None
    for params, bt in sweep.grid(a, px, mkt, base):
        if split is None:
            split = pd.Timestamp(a.split) if a.split else bt.index[len(bt) // 2]
            sel_end = bt.index[bt.index < split][-1]
        sel = period_metrics(bt, end=sel_end)
        test = period_metrics(bt, start=split)
        rows.append({**params,
                     **{f"sel_{k}": v for k, v in sel.items()},
                     **{f"test_{k}": v for k, v in test.items()}})

    res = pd.DataFrame(rows)
    res.to_csv(out / "holdout.csv", index=False)

    key = a.select_by
    bh = res.iloc[0]
    strat = res.iloc[1:].copy()
    strat["sel_rank"] = strat[f"sel_{key}"].rank(ascending=False, method="first").astype(int)
    strat["test_rank"] = strat[f"test_{key}"].rank(ascending=False, method="first").astype(int)

    print(f"\nSelection period: OOS start -> {sel_end:%Y-%m-%d} | test period: {split:%Y-%m-%d} -> end")
    print(f"Settings chosen by selection-period {key}; test-period columns were never used to choose.\n")

    cols = ["model", "horizon", "entry", "exit", "smooth",
            f"sel_{key}", "sel_CAGR", "test_Sharpe", "test_CAGR", "test_MaxDD", "test_Trades/yr", "test_rank"]
    pct = {"sel_CAGR", "test_CAGR", "test_MaxDD", "sel_MaxDD"}

    def show(t: pd.DataFrame) -> str:
        t = t[cols].copy().astype(object)
        for c in cols:
            if c in pct:
                t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
            elif c in ("sel_Sharpe", "test_Sharpe", "sel_Calmar"):
                t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
            elif c == "test_Trades/yr":
                t[c] = t[c].map(lambda v: f"{v:.0f}")
            elif c in ("horizon", "smooth"):
                t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        return t.to_string(index=False)

    print(f"Buy & Hold   selection: Sharpe {bh.sel_Sharpe:.2f}, CAGR {bh.sel_CAGR:.1%} | "
          f"test: Sharpe {bh.test_Sharpe:.2f}, CAGR {bh.test_CAGR:.1%}, MaxDD {bh.test_MaxDD:.1%}")
    default = strat[(strat.horizon == 1) & (strat.entry == wf.Config.prob_threshold)
                    & (strat.exit == wf.Config.prob_threshold) & (strat.smooth == 1)]
    if len(default):
        print("\nOriginal default settings (h=1, 0.52, no dead zone, no smoothing), for reference:")
        print(show(default))

    print(f"\nPick per model = best selection-period {key}:")
    picks = strat.loc[strat.groupby("model")[f"sel_{key}"].idxmax()]
    print(show(picks))

    top = strat.nsmallest(a.top, "sel_rank")
    print(f"\nTop {a.top} by selection-period {key} (all models):")
    print(show(top))
    print(f"  average test Sharpe {top.test_Sharpe.mean():.2f}, test CAGR {top.test_CAGR.mean():.1%} "
          f"| {(top.test_Sharpe > bh.test_Sharpe).sum()} of {a.top} beat Buy & Hold in the test period")

    rho = strat[f"sel_{key}"].corr(strat[f"test_{key}"], method="spearman")
    med_rank = top.test_rank.median()
    print(f"\nAll {len(strat)} settings: median test Sharpe {strat.test_Sharpe.median():.2f}, "
          f"{(strat.test_Sharpe > bh.test_Sharpe).mean():.0%} beat Buy & Hold in the test period")
    print(f"Rank correlation, selection vs test {key}: {rho:.2f}  "
          f"(~0 = selection-period winners are no better than random picks)")
    print(f"Median test-period rank of the top {a.top} picks: {med_rank:.0f} of {len(strat)} "
          f"(random would be ~{len(strat) / 2:.0f})")
    print(f"Saved {out.resolve() / 'holdout.csv'}")


if __name__ == "__main__":
    main()
