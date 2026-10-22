"""
Walk-forward backtesting framework — BBCA.JK starter
=====================================================

Compares several strategies on the same out-of-sample walk-forward periods:

  1. Buy & Hold                 (benchmark you must beat)
  2. SMA crossover (50/200)     (naive rule-based baseline)
  3. Logistic Regression        (simple ML baseline)
  4. LightGBM                   (main ML signal)
  5. LightGBM + GARCH vol-target (same signal, position sized by forecast volatility)

Design choices
--------------
- Target: direction of the next H-day log return (default H=1), not price.
- Walk-forward: expanding training window, retrain every `--retrain` days,
  with an embargo gap of H days between train and test so labels never leak.
- Long/flat only (retail short-selling on IDX is very restricted).
- Costs modelled as IDX-style asymmetric fees: buy ~0.15%, sell ~0.25%
  (sell side includes 0.1% final income tax). Adjust for your broker.
- All features use only information available at the close of day t;
  the position decided at close t earns the return from t to t+1.

Usage
-----
    pip install yfinance lightgbm arch scikit-learn pandas numpy matplotlib plotly
    python bbca_walkforward.py                      # downloads BBCA.JK + ^JKSE
    python bbca_walkforward.py --ticker BBRI.JK     # any other ticker
    python bbca_walkforward.py --csv my_data.csv    # own OHLCV csv (Date,Open,High,Low,Close,Volume)
    python bbca_walkforward.py --csv ../data/BBRI.JK.csv   # re-run on a previous Yahoo download
    python bbca_walkforward.py --horizon 5 --threshold 0.55 --exit-threshold 0.48 --smooth 3
    python sweep.py                                 # grid over horizon x thresholds (see sweep.py)

Yahoo downloads are saved to <project>/data/<ticker>.csv and the PNG chart
to <project>/images/backtest.png (project = parent of this src/ folder).

Not financial advice. A good backtest is necessary but not sufficient —
paper-trade before risking money.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # trading-ml/


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    ticker: str = "MTLA.JK"       # stock to trade
    market: str = "^JKSE"          # IHSG composite, used as a market feature
    start: str = "2012-01-01"
    horizon: int = 1               # predict direction of next H-day return
    min_train: int = 750           # ~3 years before the first prediction
    retrain: int = 21              # retrain every ~1 month
    buy_cost: float = 0.0015       # 0.15% buy fee
    sell_cost: float = 0.0025      # 0.25% sell fee (incl. 0.1% tax)
    prob_threshold: float = 0.52   # go long only if P(up) > threshold
    exit_threshold: float | None = None  # once long, stay until P(up) < this (None = same as entry)
    smooth: int = 1                # average P(up) over this many days before acting
    vol_target: float = 0.20       # annualised vol target for GARCH sizing
    max_leverage: float = 1.0      # cap position at 100% (no margin)
    band: float = 0.10             # skip position resizes smaller than 10%
    out_dir: str = "output"
    data_dir: str = str(PROJECT_ROOT / "data")      # raw Yahoo downloads
    img_dir: str = str(PROJECT_ROOT / "images")     # static PNG charts
    seed: int = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_yahoo(ticker: str, start: str, data_dir: str | Path) -> pd.DataFrame:
    """Download OHLCV from Yahoo and save a copy to data_dir/<ticker>.csv."""
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker or your connection.")
    if isinstance(df.columns, pd.MultiIndex):        # newer yfinance returns MultiIndex
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{ticker.lstrip('^')}.csv"           # '^JKSE' -> 'JKSE.csv'
    df.to_csv(path)                                        # re-loadable with --csv
    print(f"Saved {ticker} data to {path}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ----------------------------------------------------------------------------
# Features — everything uses data up to and including day t only
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def build_features(px: pd.DataFrame, mkt: pd.DataFrame | None, horizon: int) -> pd.DataFrame:
    f = pd.DataFrame(index=px.index)
    c = px["Close"]
    r = np.log(c).diff()
    f["ret"] = r                                                     # today's return (used for P&L, not a feature)

    # Momentum / reversal
    for k in (1, 2, 3, 5, 10, 21, 63):
        f[f"ret_{k}d"] = np.log(c / c.shift(k))
    # Volatility
    for k in (5, 21, 63):
        f[f"vol_{k}d"] = r.rolling(k).std() * np.sqrt(TRADING_DAYS)
    f["vol_ratio"] = f["vol_5d"] / f["vol_63d"]
    # Range-based (Parkinson) volatility
    hl = np.log(px["High"] / px["Low"])
    f["park_vol_21d"] = np.sqrt((hl**2).rolling(21).mean() / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)
    # Trend
    for k in (10, 50, 200):
        f[f"px_sma{k}"] = c / c.rolling(k).mean() - 1
    f["sma50_sma200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    f["rsi_14"] = rsi(c, 14)
    # Candle shape
    f["close_loc"] = (c - px["Low"]) / (px["High"] - px["Low"]).replace(0, np.nan)
    f["gap"] = np.log(px["Open"] / c.shift(1))
    # Volume
    lv = np.log(px["Volume"].replace(0, np.nan))
    f["vol_z_21"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
    # Calendar
    f["dow"] = px.index.dayofweek
    f["month"] = px.index.month

    # Market (IHSG) features
    if mkt is not None:
        m = np.log(mkt["Close"].reindex(px.index).ffill())
        mr = m.diff()
        for k in (1, 5, 21):
            f[f"mkt_ret_{k}d"] = m - m.shift(k)
        f["mkt_vol_21d"] = mr.rolling(21).std() * np.sqrt(TRADING_DAYS)
        f["beta_63d"] = r.rolling(63).cov(mr) / mr.rolling(63).var()
        f["rel_strength_21d"] = f["ret_21d"] - f["mkt_ret_21d"]

    # Target: sign of the forward H-day return (t -> t+H). NaN at the tail.
    fwd = np.log(c.shift(-horizon) / c)
    f["target"] = (fwd > 0).astype(float).where(fwd.notna())
    f["fwd_ret"] = fwd
    return f


# ----------------------------------------------------------------------------
# Walk-forward engine
# ----------------------------------------------------------------------------
def make_models(cfg: Config) -> dict:
    """Fresh, unfitted classifiers: {"p_logit": ..., "p_lgbm": ...}."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "p_logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "p_lgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0, random_state=cfg.seed, verbose=-1,
        ),
    }


def walk_forward_predict(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> pd.DataFrame:
    """Expanding-window walk-forward. Returns out-of-sample P(up) per model."""
    data = df.dropna(subset=feature_cols).copy()
    n = len(data)
    out = pd.DataFrame(index=data.index, columns=["p_logit", "p_lgbm"], dtype=float)
    importances = []

    for start in range(cfg.min_train, n, cfg.retrain):
        end = min(start + cfg.retrain, n)
        # Embargo: the last `horizon` training rows have labels that overlap the
        # test window, so drop them.
        train = data.iloc[: start - cfg.horizon]
        train = train[train["target"].notna()]
        test = data.iloc[start:end]
        X_tr, y_tr = train[feature_cols], train["target"].astype(int)

        for col, model in make_models(cfg).items():
            model.fit(X_tr, y_tr)
            out.loc[test.index, col] = model.predict_proba(test[feature_cols])[:, 1]
        importances.append(pd.Series(model.feature_importances_, index=feature_cols))   # LightGBM

    out = out.iloc[cfg.min_train:]
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return out, imp


def garch_vol_forecast(returns: pd.Series, cfg: Config, horizon: int = 1) -> pd.Series:
    """Walk-forward 1-day-ahead GARCH(1,1) vol forecast (annualised), refit every
    `retrain` days on all data up to t. Forecast made at t applies to t+1. With
    horizon > 1: the average over days t+1..t+horizon, so a volatility spike is
    expected to fade the way GARCH says it will (x sqrt(horizon / 252) = the
    horizon's vol)."""
    from arch import arch_model

    r = returns.dropna() * 100                     # arch prefers % returns
    fc = pd.Series(index=r.index, dtype=float)
    params = None
    for start in range(cfg.min_train, len(r), cfg.retrain):
        end = min(start + cfg.retrain, len(r))
        am = arch_model(r.iloc[:start], vol="GARCH", p=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off", starting_values=params)
        params = res.params
        # Filter forward through the test window with fixed params (no refit leak)
        full = arch_model(r.iloc[:end], vol="GARCH", p=1, q=1, dist="t", mean="Constant")
        f = full.fix(params).forecast(horizon=horizon, start=r.index[start - 1], reindex=True)
        fc.iloc[start - 1:end] = np.sqrt(f.variance.iloc[start - 1:end, :horizon].mean(axis=1).values)
    return fc / 100 * np.sqrt(TRADING_DAYS)


# ----------------------------------------------------------------------------
# Positions
# ----------------------------------------------------------------------------
def prob_to_position(p: pd.Series, entry: float, exit: float | None = None, smooth: int = 1) -> pd.Series:
    """Long/flat signal from P(up) with optional smoothing and hysteresis:
    go long when P > entry, stay long until P < exit. exit=None -> exit=entry."""
    exit = entry if exit is None else min(exit, entry)
    p = p.rolling(smooth, min_periods=1).mean() if smooth > 1 else p
    out = np.zeros(len(p))
    cur = 0.0
    for i, v in enumerate(p.values):
        if np.isnan(v):
            pass
        elif cur == 0 and v > entry:
            cur = 1.0
        elif cur == 1 and v < exit:
            cur = 0.0
        out[i] = cur
    return pd.Series(out, index=p.index)


def make_positions(df: pd.DataFrame, probs: pd.DataFrame, vol_fc: pd.Series, cfg: Config) -> dict[str, pd.Series]:
    """All strategy positions on the out-of-sample index (decided at close t)."""
    oos = probs.index
    sig = lambda p: prob_to_position(p, cfg.prob_threshold, cfg.exit_threshold, cfg.smooth)
    lgbm_pos = sig(probs["p_lgbm"])
    vol_scale = (cfg.vol_target / vol_fc.reindex(oos)).clip(upper=cfg.max_leverage)
    return {
        "Buy & Hold": pd.Series(1.0, index=oos),
        "SMA 50/200": (df["sma50_sma200"] > 0).astype(float).reindex(oos),
        "LogReg": sig(probs["p_logit"]),
        "LightGBM": lgbm_pos,
        "LightGBM+GARCH": lgbm_pos * vol_scale,
    }


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def backtest(position: pd.Series, ret: pd.Series, cfg: Config) -> pd.DataFrame:
    """position[t] is decided at close t and earns ret[t+1]. Costs charged on
    changes in position, asymmetric for buys vs sells."""
    raw = position.fillna(0).clip(0, cfg.max_leverage).values
    # Rebalance band: ignore resizes smaller than `cfg.band` (avoids paying fees
    # for tiny vol-targeting adjustments); moves to/from flat always execute.
    held = np.empty_like(raw)
    cur = 0.0
    for i, p in enumerate(raw):
        if p == 0 or cur == 0 or abs(p - cur) >= cfg.band:
            cur = p
        held[i] = cur
    pos = pd.Series(held, index=position.index)
    nxt = ret.shift(-1).reindex(pos.index).fillna(0)
    dpos = pos.diff().fillna(pos.iloc[0])
    cost = dpos.clip(lower=0) * cfg.buy_cost + (-dpos).clip(lower=0) * cfg.sell_cost
    # log returns -> simple for P&L on a fractional position
    strat = pos * (np.exp(nxt) - 1) - cost
    eq = (1 + strat).cumprod()
    return pd.DataFrame({"pos": pos, "ret": strat, "equity": eq, "turnover": dpos.abs()})


def metrics(bt: pd.DataFrame, fwd_up: pd.Series | None = None, proba: pd.Series | None = None) -> dict:
    r = bt["ret"]
    years = len(r) / TRADING_DAYS
    eq = bt["equity"]
    dd = eq / eq.cummax() - 1
    m = {
        "CAGR": eq.iloc[-1] ** (1 / years) - 1,
        "Vol": r.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan,
        "MaxDD": dd.min(),
        "Calmar": (eq.iloc[-1] ** (1 / years) - 1) / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Exposure": (bt["pos"] > 0).mean(),
        "Trades/yr": (bt["turnover"] > 0).sum() / years,
    }
    if proba is not None and fwd_up is not None:
        from sklearn.metrics import roc_auc_score
        ok = proba.notna() & fwd_up.notna()
        m["AUC"] = roc_auc_score(fwd_up[ok], proba[ok])
        m["HitRate"] = ((proba[ok] > 0.5) == (fwd_up[ok] == 1)).mean()
    return m


def ann_index(s: pd.Series, freq: str = "YE") -> pd.Series:
    return (1 + s).resample(freq).prod() - 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=Config.ticker)
    ap.add_argument("--market", default=Config.market, help="market index ticker, or 'none'")
    ap.add_argument("--start", default=Config.start)
    ap.add_argument("--csv", help="use a local OHLCV csv instead of Yahoo")
    ap.add_argument("--horizon", type=int, default=Config.horizon)
    ap.add_argument("--retrain", type=int, default=Config.retrain)
    ap.add_argument("--threshold", type=float, default=Config.prob_threshold, help="enter long when P(up) > this")
    ap.add_argument("--exit-threshold", type=float, default=Config.exit_threshold,
                    help="exit when P(up) < this (default: same as --threshold)")
    ap.add_argument("--smooth", type=int, default=Config.smooth, help="average P(up) over N days first")
    ap.add_argument("--buy-cost", type=float, default=Config.buy_cost, help="buy fee as a fraction, e.g. 0.0015")
    ap.add_argument("--sell-cost", type=float, default=Config.sell_cost, help="sell fee as a fraction, e.g. 0.0025")
    ap.add_argument("--out", default=Config.out_dir)
    ap.add_argument("--data-dir", default=Config.data_dir, help="where Yahoo downloads are saved")
    ap.add_argument("--img-dir", default=Config.img_dir, help="where PNG charts are saved")
    a = ap.parse_args()

    cfg = Config(ticker=a.ticker, market=a.market, start=a.start, horizon=a.horizon,
                 retrain=a.retrain, prob_threshold=a.threshold, exit_threshold=a.exit_threshold,
                 smooth=a.smooth, buy_cost=a.buy_cost, sell_cost=a.sell_cost, out_dir=a.out,
                 data_dir=a.data_dir, img_dir=a.img_dir)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = Path(cfg.img_dir)
    img.mkdir(parents=True, exist_ok=True)

    # ---- data
    if a.csv:
        px, mkt, label = load_csv(a.csv), None, Path(a.csv).stem
    else:
        px = load_yahoo(cfg.ticker, cfg.start, cfg.data_dir)
        mkt = None if cfg.market.lower() == "none" else load_yahoo(cfg.market, cfg.start, cfg.data_dir)
        label = cfg.ticker
    print(f"{label}: {len(px)} bars, {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}")

    # ---- features
    df = build_features(px, mkt, cfg.horizon)
    feature_cols = [c for c in df.columns if c not in ("ret", "target", "fwd_ret")]
    print(f"{len(feature_cols)} features | horizon={cfg.horizon}d | retrain every {cfg.retrain}d")

    # ---- walk-forward models
    print("Walk-forward training ...")
    probs, importance = walk_forward_predict(df, feature_cols, cfg)
    oos = probs.index
    ret = df["ret"]

    print("Walk-forward GARCH volatility ...")
    vol_fc = garch_vol_forecast(ret, cfg).reindex(oos)

    # ---- positions (decided at close t)
    p_lgbm = probs["p_lgbm"]
    positions = make_positions(df, probs, vol_fc, cfg)

    fwd_up = df["target"].reindex(oos)
    results, curves = {}, {}
    for name, pos in positions.items():
        bt = backtest(pos, ret.reindex(oos), cfg)
        pr = probs["p_logit"] if name == "LogReg" else (p_lgbm if name.startswith("LightGBM") else None)
        results[name] = metrics(bt, fwd_up, pr)
        curves[name] = bt

    # ---- report
    tbl = pd.DataFrame(results).T
    fmt = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:.1%}",
           "Calmar": "{:.2f}", "Exposure": "{:.0%}", "Trades/yr": "{:.0f}",
           "AUC": "{:.3f}", "HitRate": "{:.1%}"}
    show = tbl.copy().astype(object)
    for c, f in fmt.items():
        if c in show:
            show[c] = tbl[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))

    print(f"\nOut-of-sample period: {oos[0]:%Y-%m-%d} -> {oos[-1]:%Y-%m-%d} "
          f"({len(oos) / TRADING_DAYS:.1f} yrs)")
    exit_txt = "" if cfg.exit_threshold is None else f", exit < {cfg.exit_threshold}"
    smooth_txt = "" if cfg.smooth <= 1 else f" | P(up) smoothed over {cfg.smooth}d"
    print(f"Costs: buy {cfg.buy_cost:.2%}, sell {cfg.sell_cost:.2%} | long when P(up) > {cfg.prob_threshold}"
          f"{exit_txt}{smooth_txt}\n")
    print(show.to_string())

    yearly = pd.DataFrame({k: ann_index(v["ret"]) for k, v in curves.items()})
    yearly.index = yearly.index.year
    print("\nCalendar-year returns:")
    print(yearly.map(lambda v: f"{v:.1%}").to_string())

    print("\nTop 10 LightGBM features (avg split importance):")
    print(importance.head(10).round(1).to_string())

    tbl.to_csv(out / "summary.csv")
    yearly.to_csv(out / "yearly_returns.csv")
    pd.DataFrame({k: v["equity"] for k, v in curves.items()}).join(probs).to_csv(out / "equity_and_probs.csv")
    importance.to_csv(out / "feature_importance.csv", header=["importance"])

    plot(curves, p_lgbm, vol_fc, importance, label, img / "backtest.png")
    realized = ret.rolling(21).std().shift(-20) * np.sqrt(TRADING_DAYS)   # vol actually realised over next 21d
    build_report(
        out / "report.html", label=label, cfg=cfg, tbl=tbl, curves=curves,
        close=px["Close"].reindex(oos), probs=probs, fwd_up=fwd_up,
        vol_fc=vol_fc, realized_vol=realized.reindex(oos), importance=importance,
    )
    print(f"\nSaved results to {out.resolve()}/  (open report.html in a browser)")
    print(f"Saved chart to {(img / 'backtest.png').resolve()}")


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
# Fixed colour per strategy (colour follows the entity, never its rank).
# Categorical slots from a colour-blind-validated palette; Buy & Hold is the
# neutral benchmark so it is drawn in ink grey.
COLORS = {
    "Buy & Hold": "#52514e",
    "LightGBM": "#2a78d6",
    "LightGBM+GARCH": "#eb6834",
    "LogReg": "#1baf7a",
    "SMA 50/200": "#eda100",
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"     # diverging blue <-> red, grey midpoint


def plot(curves, p_lgbm, vol_fc, importance, label, path):
    """Static PNG summary (quick look). The full interactive version is report.html."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1.3, 1.3, 2]})
    for name, bt in curves.items():
        ax[0].plot(bt.index, bt["equity"], lw=2 if name == "Buy & Hold" else 1.4,
                   color=COLORS.get(name), label=name)
    ax[0].set_yscale("log")
    ax[0].set_title(f"{label} — walk-forward out-of-sample equity (after costs)")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].plot(p_lgbm.index, p_lgbm.rolling(5).mean(), lw=0.9, color=COLORS["LightGBM"])
    ax[1].axhline(0.5, color=INK2, ls="--", lw=0.8)
    ax[1].set_ylabel("P(up), 5d avg")
    ax[1].grid(alpha=0.3)

    dd = {k: v["equity"] / v["equity"].cummax() - 1 for k, v in curves.items()}
    for k in ("Buy & Hold", "LightGBM", "LightGBM+GARCH"):
        ax[2].plot(dd[k].index, dd[k], lw=1, label=k, color=COLORS[k])
    ax[2].set_ylabel("Drawdown")
    ax[2].legend(fontsize=8, loc="lower left")
    ax[2].grid(alpha=0.3)

    importance.head(15).iloc[::-1].plot.barh(ax=ax[3], color=COLORS["LightGBM"])
    ax[3].set_title("LightGBM feature importance (top 15)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(path, *, label, cfg, tbl, curves, close, probs, fwd_up,
                 vol_fc, realized_vol, importance):
    """Self-contained interactive HTML report (plotly, works offline)."""
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    def style(fig, title, height=380, ytitle=None, yfmt=None, legend=True):
        fig.update_layout(
            title=dict(text=title, x=0, font=dict(size=15, color=INK)),
            height=height, margin=dict(l=60, r=20, t=50, b=40),
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK2),
            hovermode="x unified", showlegend=legend,
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
            hoverlabel=dict(bgcolor="white", font_color=INK),
        )
        fig.update_xaxes(showgrid=False, linecolor=GRID, showspikes=True,
                         spikecolor=INK2, spikethickness=1, spikedash="dot")
        fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle, tickformat=yfmt)
        return fig

    figs = []
    order = ["Buy & Hold", "SMA 50/200", "LogReg", "LightGBM", "LightGBM+GARCH"]

    # 1. Equity curves -----------------------------------------------------
    f = go.Figure()
    for k in order:
        eq = curves[k]["equity"]
        f.add_trace(go.Scatter(x=eq.index, y=eq, name=k, line=dict(color=COLORS[k],
                    width=2.5 if k in ("Buy & Hold", "LightGBM") else 1.6),
                    hovertemplate="%{y:.2f}x"))
    style(f, "Growth of 1 rupiah, after costs (log scale)", 440, "Equity multiple")
    f.update_yaxes(type="log")
    figs.append(("Equity", "Every strategy traded on the same out-of-sample days. "
                 "The ML lines only matter if they beat the grey Buy & Hold line, or match it with smaller drawdowns.", f))

    # 2. Drawdowns ----------------------------------------------------------
    f = go.Figure()
    for k in ("Buy & Hold", "LightGBM", "LightGBM+GARCH"):
        eq = curves[k]["equity"]
        f.add_trace(go.Scatter(x=eq.index, y=eq / eq.cummax() - 1, name=k,
                    line=dict(color=COLORS[k], width=1.6), hovertemplate="%{y:.1%}"))
    style(f, "Drawdown from previous peak", 300, None, ".0%")
    figs.append(("Risk", "How deep and how long each strategy stayed underwater. "
                 "This is usually where a timing model earns its keep.", f))

    # 3. Price with LightGBM positions ---------------------------------------
    pos = curves["LightGBM"]["pos"]
    chg = pos.diff().fillna(pos.iloc[0])
    buys, sells = chg[chg > 0].index, chg[chg < 0].index
    f = go.Figure()
    f.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color=INK2, width=1.2),
                           hovertemplate="%{y:,.0f}"))
    f.add_trace(go.Scatter(x=close.index, y=close.where(pos > 0), name="Held by LightGBM",
                           line=dict(color=COLORS["LightGBM"], width=2), hoverinfo="skip"))
    f.add_trace(go.Scatter(x=buys, y=close.loc[buys], mode="markers", name="Buy",
                           marker=dict(symbol="triangle-up", size=9, color=COLORS["LightGBM"],
                                       line=dict(color="white", width=1)), hovertemplate="Buy %{y:,.0f}"))
    f.add_trace(go.Scatter(x=sells, y=close.loc[sells], mode="markers", name="Sell",
                           marker=dict(symbol="triangle-down", size=9, color=NEG,
                                       line=dict(color="white", width=1)), hovertemplate="Sell %{y:,.0f}"))
    style(f, f"{label} close with LightGBM entries and exits", 500, "Price")
    last_year = close.index[-1] - pd.Timedelta(days=365)
    f.update_layout(margin=dict(t=95), title=dict(y=0.97, yanchor="top"))
    f.update_xaxes(range=[last_year, close.index[-1]],          # open on the last year; slider shows all
                   rangeslider=dict(visible=True, thickness=0.06),
                   rangeselector=dict(x=0, y=1.02, yanchor="bottom", bgcolor=MID, activecolor="#cde2fb",
                                      buttons=[dict(count=3, label="3M", step="month", stepmode="backward"),
                                               dict(count=1, label="1Y", step="year", stepmode="backward"),
                                               dict(step="all", label="All")]))
    vis = close.loc[last_year:]
    pad = (vis.max() - vis.min()) * 0.08
    f.update_yaxes(range=[vis.min() - pad, vis.max() + pad])   # double-click the chart to reset
    figs.append(("Signals", "Blue segments are days the model was in the stock. Opens on the last 12 months; use the buttons "
                 "or drag the slider below to move around. Double-click the chart to reset the zoom.", f))

    # 4. Probability through time ------------------------------------------
    p = probs["p_lgbm"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=p.index, y=p, name="Daily P(up)", line=dict(color="#86b6ef", width=0.8),
                           hovertemplate="%{y:.3f}"))
    f.add_trace(go.Scatter(x=p.index, y=p.rolling(21).mean(), name="21-day average",
                           line=dict(color=COLORS["LightGBM"], width=2), hovertemplate="%{y:.3f}"))
    f.add_hline(y=cfg.prob_threshold, line=dict(color=INK2, dash="dash", width=1),
                annotation_text=f"buy threshold {cfg.prob_threshold}", annotation_position="top left")
    style(f, "LightGBM predicted probability of an up day", 300, "P(up)")
    figs.append(("Model", "A model with real skill spreads its probabilities away from 0.50. "
                 "A flat line hugging 0.50 means it has little to say.", f))

    # 5. Calibration ---------------------------------------------------------
    ok = p.notna() & fwd_up.notna()
    bins = pd.qcut(p[ok], 10, duplicates="drop")
    cal = pd.DataFrame({"pred": p[ok], "actual": fwd_up[ok]}).groupby(bins, observed=True).agg(
        pred=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    f = go.Figure()
    lo, hi = min(cal.pred.min(), cal.actual.min()) - 0.02, max(cal.pred.max(), cal.actual.max()) + 0.02
    f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], name="Perfect calibration", mode="lines",
                           line=dict(color=INK2, dash="dot", width=1), hoverinfo="skip"))
    f.add_trace(go.Scatter(x=cal.pred, y=cal.actual, name="LightGBM (deciles)", mode="lines+markers",
                           line=dict(color=COLORS["LightGBM"], width=2),
                           marker=dict(size=9, line=dict(color="white", width=2)),
                           customdata=cal.n, hovertemplate="predicted %{x:.3f}<br>actual %{y:.3f}<br>%{customdata} days"))
    style(f, "Calibration: predicted vs actual up-day frequency", 340, "Actual share of up days", ".0%")
    f.update_xaxes(title="Predicted P(up)", tickformat=".0%", showspikes=False)
    f.update_layout(hovermode="closest")
    figs.append(("Model", "If the blue line rises left to right, higher predictions really do "
                 "come with more up days. A flat line means the ranking is noise.", f))

    # 6. Calendar-year returns ----------------------------------------------
    yearly = pd.DataFrame({k: ann_index(curves[k]["ret"]) for k in order})
    yearly.index = yearly.index.year
    f = go.Figure()
    for k in order:
        f.add_trace(go.Bar(x=yearly.index.astype(str), y=yearly[k], name=k, marker_color=COLORS[k],
                           marker_line=dict(color=SURFACE, width=1), hovertemplate="%{y:.1%}"))
    style(f, "Calendar-year returns", 360, None, ".0%")
    f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    figs.append(("Returns", "Consistency matters more than one great year.", f))

    # 7. Monthly heatmap (LightGBM) ------------------------------------------
    m = (1 + curves["LightGBM"]["ret"]).resample("ME").prod() - 1
    hm = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.values}).pivot(index="y", columns="m", values="r")
    hm = hm.reindex(columns=range(1, 13))
    lim = float(np.nanmax(np.abs(hm.values))) or 0.01
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    f = go.Figure(go.Heatmap(
        z=hm.values, x=months, y=hm.index.astype(str), zmin=-lim, zmax=lim, zmid=0,
        colorscale=[[0, NEG], [0.5, MID], [1, POS]], xgap=2, ygap=2,
        text=np.where(np.isnan(hm.values), "", np.vectorize(lambda v: f"{v:.1%}")(np.nan_to_num(hm.values))),
        texttemplate="%{text}", textfont=dict(size=10, color=INK),
        colorbar=dict(tickformat=".0%", thickness=10), hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
    style(f, "LightGBM monthly returns", 60 + 34 * len(hm), legend=False)
    f.update_layout(hovermode="closest")
    f.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    figs.append(("Returns", "Blue months made money, red months lost it.", f))

    # 8. Rolling 1-year Sharpe ----------------------------------------------
    f = go.Figure()
    for k in ("Buy & Hold", "LightGBM", "LightGBM+GARCH"):
        r = curves[k]["ret"]
        rs = r.rolling(TRADING_DAYS).mean() / r.rolling(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
        f.add_trace(go.Scatter(x=rs.index, y=rs, name=k, line=dict(color=COLORS[k], width=1.8),
                               hovertemplate="%{y:.2f}"))
    f.add_hline(y=0, line=dict(color=INK2, width=1))
    style(f, "Rolling 1-year Sharpe ratio", 320, "Sharpe")
    figs.append(("Risk", "Shows whether the edge is stable or came from one lucky stretch.", f))

    # 9. GARCH volatility -----------------------------------------------------
    f = go.Figure()
    f.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol, name="Realised (next 21 days)",
                           line=dict(color=INK2, width=1.2), hovertemplate="%{y:.1%}"))
    f.add_trace(go.Scatter(x=vol_fc.index, y=vol_fc, name="GARCH forecast",
                           line=dict(color=COLORS["LightGBM+GARCH"], width=1.8), hovertemplate="%{y:.1%}"))
    f.add_hline(y=cfg.vol_target, line=dict(color=INK2, dash="dash", width=1),
                annotation_text=f"vol target {cfg.vol_target:.0%}", annotation_position="top left")
    style(f, "Annualised volatility: GARCH forecast vs what actually happened", 320, None, ".0%")
    figs.append(("Risk", "When the orange line is above the target, the GARCH strategy holds less than 100% of the position.", f))

    # 10. Feature importance -------------------------------------------------
    top = importance.head(15).iloc[::-1]
    f = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=COLORS["LightGBM"],
                         hovertemplate="%{y}: %{x:.0f}<extra></extra>"))
    style(f, "LightGBM feature importance (average splits, top 15)", 460, legend=False)
    f.update_layout(hovermode="closest", bargap=0.5)
    f.update_xaxes(showspikes=False, gridcolor=GRID, showgrid=True)
    f.update_yaxes(showgrid=False)
    figs.append(("Model", "Split counts show what the trees used, not what made money. "
                 "Treat it as a sanity check, not proof.", f))

    # ---- summary tiles + table --------------------------------------------
    best = tbl.drop(index="Buy & Hold")["Sharpe"].idxmax()
    bh, lg = tbl.loc["Buy & Hold"], tbl.loc["LightGBM"]
    tiles = [
        ("Buy & Hold CAGR", f"{bh.CAGR:.1%}", f"max drawdown {bh.MaxDD:.1%}"),
        ("LightGBM CAGR", f"{lg.CAGR:.1%}", f"max drawdown {lg.MaxDD:.1%}"),
        ("LightGBM AUC", f"{lg.AUC:.3f}", "0.500 = coin flip"),
        ("Best Sharpe (ex B&H)", f"{tbl.loc[best, 'Sharpe']:.2f}", best),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="tl">{a}</div><div class="tv">{b}</div>'
                         f'<div class="ts">{c}</div></div>' for a, b, c in tiles)

    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Exposure", "Trades/yr", "AUC", "HitRate"]
    fmts = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:.1%}", "Calmar": "{:.2f}",
            "Exposure": "{:.0%}", "Trades/yr": "{:.0f}", "AUC": "{:.3f}", "HitRate": "{:.1%}"}
    rows = ""
    for k in order:
        cells = "".join(f"<td>{'' if pd.isna(tbl.loc[k, c]) else fmts[c].format(tbl.loc[k, c])}</td>" for c in cols)
        rows += f'<tr><th><span class="sw" style="background:{COLORS[k]}"></span>{k}</th>{cells}</tr>'
    table_html = f'<table><thead><tr><th>Strategy</th>{"".join(f"<th>{c}</th>" for c in cols)}</tr></thead><tbody>{rows}</tbody></table>'

    oos = curves["Buy & Hold"].index
    sections, cfg_js = "", {"displaylogo": False, "responsive": True}
    for i, (tag, note, fig) in enumerate(figs):
        div = fig.to_html(full_html=False, include_plotlyjs=False, config=cfg_js, div_id=f"c{i}")
        sections += f'<section class="card"><div class="tag">{tag}</div>{div}<p class="note">{note}</p></section>'

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{label} Backtest Report</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{margin:0;background:#f4f3ef;color:{INK};font-family:Inter,system-ui,sans-serif}}
 main{{max-width:1180px;margin:0 auto;padding:28px 16px 60px}}
 h1{{font-size:26px;margin:0 0 4px}} .sub{{color:{INK2};margin:0 0 22px;font-size:14px}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}}
 .tile,.card{{background:{SURFACE};border:1px solid {GRID};border-radius:10px}}
 .tile{{padding:14px 16px}} .tl{{font-size:12px;color:{INK2};text-transform:uppercase;letter-spacing:.04em}}
 .tv{{font-size:28px;font-weight:650;margin:4px 0 2px}} .ts{{font-size:12px;color:{INK2}}}
 .card{{padding:14px 16px 10px;margin-bottom:16px;overflow-x:auto}}
 .tag{{font-size:11px;font-weight:600;color:{INK2};text-transform:uppercase;letter-spacing:.06em}}
 .note{{font-size:13px;color:{INK2};margin:4px 0 6px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
 th,td{{padding:8px 10px;border-bottom:1px solid {GRID};text-align:right;white-space:nowrap}}
 thead th{{color:{INK2};font-weight:600}} tbody th{{text-align:left;font-weight:500}}
 th:first-child{{text-align:left}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:-1px}}
 .warn{{font-size:12px;color:{INK2};margin-top:24px}}
</style></head><body><main>
<h1>{label} walk-forward backtest</h1>
<p class="sub">Out-of-sample {oos[0]:%d %b %Y} to {oos[-1]:%d %b %Y} ({len(oos) / TRADING_DAYS:.1f} years) ·
predicts {cfg.horizon}-day direction · retrained every {cfg.retrain} days ·
costs {cfg.buy_cost:.2%} buy / {cfg.sell_cost:.2%} sell · long when P(up) &gt; {cfg.prob_threshold}</p>
<div class="tiles">{tiles_html}</div>
<section class="card"><div class="tag">Summary</div>{table_html}</section>
{sections}
<p class="warn">Backtest results, not financial advice. Past performance in a walk-forward test does not guarantee future returns.</p>
</main></body></html>"""
    Path(path).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
