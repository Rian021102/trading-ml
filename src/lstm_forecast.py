"""
LSTM price-range forecast
=========================

Forecasts a *range* for the next H trading days (default 20), not a single
price path: for each day ahead it predicts the 10th, 50th and 90th percentile
of the cumulative return, i.e. "80% chance the close lands between P10 and P90".
A single-line price forecast would hide how uncertain it is.

Trained with quantile (pinball) loss on the last 60 days of returns, range,
volume and market (IHSG) features. An ensemble of 5 seeds is averaged.

Calibration: a raw LSTM range tends to be too narrow and biased towards the
period it was trained on. Each of P10 / P50 / P90 is therefore shifted, per day
ahead, by the amount that would have made it hit its target rate (10% / 50% /
90% of closes below it) over the last --calib-window days of forecasts whose
outcome was already known. On top of that, adaptive conformal inference keeps
score of misses and nudges each line until its long-run hit rate matches the
target. Shifts are in units of trailing volatility, so the range also widens
when the market is turbulent. A fixed calibration period does not work here: BBCA's 2021-23 errors
said nothing about 2023-26.

Honesty check: the model is scored on the most recent 20% of history (never
seen in training) against a no-skill baseline, a random walk whose band width
comes from the last 63 days' volatility. If the LSTM doesn't beat that baseline
(skill <= 0%), its forecast adds nothing beyond "volatility says +/- this much".

    python lstm_forecast.py --ticker BBCA.JK
    python lstm_forecast.py --ticker BBRI.JK --days 10 --offline

Outputs: images/lstm_forecast_<ticker>.png, output/lstm_forecast_<ticker>.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import bbca_walkforward as wf
from today_signal import load

QUANTILES = (0.1, 0.5, 0.9)
Z90 = 1.2815515655446004            # standard-normal 90th percentile, for the random-walk band


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_features(px: pd.DataFrame, mkt: pd.DataFrame | None) -> pd.DataFrame:
    """Per-day inputs, all known at the close of day t."""
    c = px["Close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)
    f["ret"] = r
    f["range"] = np.log(px["High"] / px["Low"])
    f["gap"] = np.log(px["Open"] / c.shift(1))
    f["vol_5d"] = r.rolling(5).std()
    f["vol_21d"] = r.rolling(21).std()
    lv = np.log(px["Volume"].replace(0, np.nan))
    f["volume_z"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
    f["px_sma50"] = c / c.rolling(50).mean() - 1
    if mkt is not None:
        m = np.log(mkt["Close"].reindex(px.index).ffill())
        f["mkt_ret"] = m.diff()
        f["mkt_vol_21d"] = f["mkt_ret"].rolling(21).std()
    return f.replace([np.inf, -np.inf], np.nan)


def make_targets(close: pd.Series, H: int) -> np.ndarray:
    """Y[t, h-1] = log(close[t+h] / close[t]) in percent, NaN past the end."""
    lc = np.log(close.values)
    Y = np.full((len(lc), H), np.nan)
    for h in range(1, H + 1):
        Y[:-h, h - 1] = (lc[h:] - lc[:-h]) * 100
    return Y


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class QuantileLSTM(nn.Module):
    """LSTM -> (P10, P50, P90) of the cumulative return for each of H days ahead.
    P10 <= P50 <= P90 by construction (softplus offsets around the median)."""

    def __init__(self, n_in: int, H: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.H = H
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 3 * H))

    def forward(self, x):
        o = self.head(self.lstm(x)[0][:, -1]).view(-1, self.H, 3)
        mid = o[..., 0]
        return torch.stack([mid - F.softplus(o[..., 1]), mid, mid + F.softplus(o[..., 2])], dim=-1)


def pinball(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = torch.tensor(QUANTILES, device=pred.device)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


class Windows:
    """Builds (L-day window, H-day target) batches on the device from day indices."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, L: int, device):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.Y = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=device)
        self.offs = torch.arange(-L + 1, 1, device=device)
        self.device = device

    def __call__(self, idx):
        idx = torch.as_tensor(idx, device=self.device)
        return self.X[idx[:, None] + self.offs], self.Y[idx]


def train_one(win: Windows, tr_idx, va_idx, n_in, H, seed, epochs=200, patience=20, batch=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = QuantileLSTM(n_in, H).to(win.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xv, yv = win(va_idx)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for b in np.array_split(np.random.permutation(tr_idx), max(1, len(tr_idx) // batch)):
            xb, yb = win(b)
            opt.zero_grad()
            loss = pinball(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = pinball(model(xv), yv).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def ensemble(win, tr_idx, va_idx, n_in, H, seeds):
    return [train_one(win, tr_idx, va_idx, n_in, H, s) for s in seeds]


def predict(models, win: Windows, idx) -> np.ndarray:
    x, _ = win(np.asarray(idx))
    with torch.no_grad():
        return np.mean([m(x).cpu().numpy() for m in models], axis=0)      # (n, H, 3)


def split(valid: np.ndarray, frac_val: float, frac_test: float, H: int):
    """Chronological train / val / test with an H-day embargo between blocks."""
    n = len(valid)
    n_te, n_va = int(n * frac_test), int(n * frac_val)
    te = valid[n - n_te:]
    va = valid[n - n_te - n_va: n - n_te - H]
    tr = valid[: n - n_te - n_va - H]
    return tr, va, te


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def rw_quantiles(sigma_daily_pct: np.ndarray, H: int) -> np.ndarray:
    """Random-walk baseline: zero drift, band from trailing volatility."""
    s = sigma_daily_pct[:, None] * np.sqrt(np.arange(1, H + 1))[None, :]
    return np.stack([-Z90 * s, np.zeros_like(s), Z90 * s], axis=-1)


def rolling_adjust(pred: np.ndarray, Y: np.ndarray, sigma: np.ndarray, first: int,
                   window: int, gamma: float = 0.005, min_periods: int = 60) -> np.ndarray:
    """Per-day shift (in units of trailing daily vol) for each horizon and quantile,
    via adaptive conformal inference (Gibbs & Candes, 2021).

    pred, Y: (n, H, 3) / (n, H) for every day; forecasts before day `first` are
    in-sample and ignored. For day t and horizon h the shift is a quantile of the
    normalised residuals of forecasts made on days t-h-window+1 .. t-h (only
    outcomes known at the close of t). The quantile level starts at the target
    (0.1 / 0.5 / 0.9) and is nudged by `gamma` every time a resolved forecast
    lands on the wrong side, so the long-run hit rate converges to the target
    even when the recent window understates the tails."""
    n, H, _ = pred.shape
    resid = (Y[..., None] - pred) / sigma[:, None, None]
    resid[:first] = np.nan
    adj = np.zeros_like(resid)
    for i in range(H):
        lag = i + 1                                    # outcome of day s is known at s + lag
        for k, q in enumerate(QUANTILES):
            r, level = resid[:, i, k], q
            for t in range(first, n):
                s_ = t - lag
                if s_ >= first and not np.isnan(r[s_]):   # feedback from the forecast resolved today
                    below = Y[s_, i] < pred[s_, i, k] + adj[s_, i, k] * sigma[s_]
                    level = float(np.clip(level + gamma * (q - below), 0.001, 0.999))
                w = r[max(first, s_ - window + 1): s_ + 1] if s_ >= first else r[:0]
                w = w[~np.isnan(w)]
                if len(w) >= min_periods:
                    adj[t, i, k] = np.quantile(w, level)
    return adj


def apply_calibration(pred: np.ndarray, adj: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    out = pred + np.nan_to_num(adj) * sigma[:, None, None]
    return np.sort(out, axis=-1)                     # keep P10 <= P50 <= P90


def predict_all(models, win: Windows, n: int, L: int, H: int) -> np.ndarray:
    """Predictions for every day that has a full lookback window; NaN before."""
    out = np.full((n, H, 3), np.nan)
    idx = np.arange(L - 1, n)
    for chunk in np.array_split(idx, max(1, len(idx) // 1024)):
        out[chunk] = predict(models, win, chunk)
    return out


def pinball_np(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean pinball loss per horizon -> shape (H,)."""
    q = np.array(QUANTILES)
    e = y[..., None] - pred
    return np.maximum(q * e, (q - 1) * e).mean(axis=(0, 2))


def evaluate(raw, cal, rw, y, horizons) -> pd.DataFrame:
    pl_raw, pl_cal, pl_rw = pinball_np(raw, y), pinball_np(cal, y), pinball_np(rw, y)
    band = lambda p, i: ((y[:, i] >= p[:, i, 0]) & (y[:, i] <= p[:, i, 2])).mean()
    rows = []
    for h in horizons:
        i = h - 1
        rows.append({
            "days ahead": h,
            "skill raw": 1 - pl_raw[i] / pl_rw[i],
            "skill calibrated": 1 - pl_cal[i] / pl_rw[i],
            "80% hit raw": band(raw, i),
            "80% hit calibrated": band(cal, i),
            "80% hit RW": band(rw, i),
            "below P10": (y[:, i] < cal[:, i, 0]).mean(),
            "below P50": (y[:, i] < cal[:, i, 1]).mean(),
            "above P90": (y[:, i] > cal[:, i, 2]).mean(),
            "width calibrated": (cal[:, i, 2] - cal[:, i, 0]).mean() / 100,
            "width RW": (rw[:, i, 2] - rw[:, i, 0]).mean() / 100,
        })
    return pd.DataFrame(rows).set_index("days ahead")


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def plot(label, close, fc, test_dates, test_actual, test_band, H, path):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    blue, orange = wf.COLORS["LightGBM"], wf.COLORS["LightGBM+GARCH"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})

    hist = close.iloc[-120:]
    a = ax[0]
    a.plot(hist.index, hist.values, color=wf.INK, lw=1.6, label="Close")
    x = [hist.index[-1], *fc.index]
    join = lambda col: [hist.iloc[-1], *fc[col].values]
    a.fill_between(x, join("lstm_p10"), join("lstm_p90"), color=blue, alpha=0.18, lw=0,
                   label="LSTM 80% range, calibrated (P10-P90)")
    a.plot(x, join("lstm_p50"), color=blue, lw=2, label="LSTM median, calibrated (P50)")
    a.plot(x, join("rw_p10"), color=orange, lw=1.2, ls="--", label="Random-walk 80% range (baseline)")
    a.plot(x, join("rw_p90"), color=orange, lw=1.2, ls="--")
    for col, va in (("lstm_p90", "bottom"), ("lstm_p50", "center"), ("lstm_p10", "top")):
        a.annotate(f"{fc[col].iloc[-1]:,.0f}", (fc.index[-1], fc[col].iloc[-1]), xytext=(6, 0),
                   textcoords="offset points", va=va, fontsize=9, color=wf.INK2)
    a.set_title(f"{label} - next {H} trading days: forecast range vs random-walk baseline")
    a.legend(loc="upper left", fontsize=9)
    a.grid(alpha=0.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    b = ax[1]
    b.fill_between(test_dates, test_band[:, 0], test_band[:, 2], color=blue, alpha=0.18, lw=0,
                   label=f"Calibrated LSTM 80% range, made {H}d earlier")
    b.plot(test_dates, test_actual, color=wf.INK, lw=1, label="Actual close")
    hit = ((test_actual >= test_band[:, 0]) & (test_actual <= test_band[:, 2])).mean()
    b.set_title(f"Test period check: did the {H}-day-ahead range contain the actual close? ({hit:.0%} of days; 80% expected)")
    b.legend(loc="upper left", fontsize=9)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(px: pd.DataFrame, mkt: pd.DataFrame | None, label: str, H: int = 20, L: int = 60,
        n_seeds: int = 5, calib_window: int = 250, gamma: float = 0.005) -> dict:
    """Honesty check on the held-out test period, then refit on all data and
    forecast the next H days from the last close. Returns what report() and save() need."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = make_features(px, mkt)
    ok = feats.notna().all(axis=1).values
    first = int(np.argmax(ok))
    feats, px = feats.iloc[first:], px.iloc[first:]
    if feats.isna().any().any():
        feats = feats.ffill().fillna(0)
    close = px["Close"]
    Y = make_targets(close, H)
    n = len(feats)

    full_y = ~np.isnan(Y).any(axis=1)
    valid = np.arange(L - 1, n)[full_y[L - 1:]]
    tr, va, te = split(valid, 0.15, 0.20, H)

    # standardise inputs with training-period statistics only
    mu, sd = feats.iloc[: tr[-1] + 1].mean(), feats.iloc[: tr[-1] + 1].std().replace(0, 1)
    X = ((feats - mu) / sd).values
    win = Windows(X, Y, L, device)
    seeds = list(range(n_seeds))
    print(f"{label}: {n} days | lookback {L}d -> forecast {H}d | {feats.shape[1]} inputs | {device}")
    print(f"train {feats.index[tr[0]]:%Y-%m-%d}..{feats.index[tr[-1]]:%Y-%m-%d} | "
          f"val ..{feats.index[va[-1]]:%Y-%m-%d} | test {feats.index[te[0]]:%Y-%m-%d}..{feats.index[te[-1]]:%Y-%m-%d}")

    # ---- 1. honesty check on the held-out test period
    t0 = time.time()
    models = ensemble(win, tr, va, X.shape[1], H, seeds)
    sigma = feats["ret"].rolling(63).std().values * 100
    # rolling calibration from the first post-training day; each test day only
    # uses forecasts whose outcome was already known on that day
    allp = predict_all(models, win, n, L, H)
    adj = rolling_adjust(allp, Y, sigma, va[0], calib_window, gamma)
    raw = allp[te]
    pred = apply_calibration(raw, adj[te], sigma[te])
    rw = rw_quantiles(sigma[te], H)
    horizons = sorted({h for h in (1, 5, 10, 20) if h <= H} | {H})
    ev = evaluate(raw, pred, rw, Y[te], horizons)
    t_test = time.time() - t0
    skill = ev["skill calibrated"]

    # ---- 2. refit on all data and forecast from today
    t0 = time.time()
    n_va = int(len(valid) * 0.15)
    va_final = valid[len(valid) - n_va:]
    final = ensemble(win, valid[: len(valid) - n_va - H], va_final, X.shape[1], H, seeds)
    allp_final = predict_all(final, win, n, L, H)
    adj_final = rolling_adjust(allp_final, Y, sigma, va_final[0], calib_window, gamma)
    q = apply_calibration(allp_final[[n - 1]], adj_final[[n - 1]], sigma[[n - 1]])[0] / 100   # (H, 3)
    last = close.iloc[-1]
    dates = pd.bdate_range(close.index[-1] + pd.offsets.BDay(1), periods=H)   # ignores IDX holidays
    rw_now = rw_quantiles(sigma[[n - 1]], H)[0] / 100
    fc = pd.DataFrame({
        "lstm_p10": last * np.exp(q[:, 0]), "lstm_p50": last * np.exp(q[:, 1]), "lstm_p90": last * np.exp(q[:, 2]),
        "rw_p10": last * np.exp(rw_now[:, 0]), "rw_p90": last * np.exp(rw_now[:, 2]),
    }, index=dates)
    fc.index.name = "date"
    fc.insert(0, "days_ahead", np.arange(1, H + 1))

    verdict = (f"LSTM beats the random-walk baseline at {(skill > 0).sum()} of {len(skill)} horizons tested "
               f"(skill {skill.min():+.1%} .. {skill.max():+.1%}).")
    if (skill <= 0.02).all():
        verdict += " That is no meaningful edge: read the range as a volatility estimate, not a prediction."
    return {
        "label": label, "H": H, "close": close, "fc": fc, "ev": ev, "skill": skill, "verdict": verdict,
        "horizons": horizons, "n_test": len(te), "calib_window": calib_window,
        "t_test": t_test, "t_refit": time.time() - t0,
        "test_dates": feats.index[np.minimum(te + H, n - 1)],
        "test_actual": close.values[np.minimum(te + H, n - 1)],
        "test_band": close.values[te][:, None] * np.exp(pred[:, H - 1, :] / 100),
    }


def report(r: dict) -> None:
    """Print the test-period evaluation, the forecast table and the verdict."""
    print(f"Test-period evaluation ({r['n_test']} days, ensemble trained in {r['t_test']:.0f}s, "
          f"rolling {r['calib_window']}-day calibration):")
    show = r["ev"].copy()
    for c in show.columns:
        show[c] = show[c].map("{:+.1%}".format if c.startswith("skill") else "{:.1%}".format)
    print(show.to_string())
    print("  skill > 0: beats the volatility-only random walk (RW). Targets: 80% hit 80%, "
          "below P10 10%, below P50 50%, above P90 10%.")

    close, fc = r["close"], r["fc"]
    last = close.iloc[-1]
    print(f"\nForecast from close {close.index[-1]:%Y-%m-%d} = {last:,.0f}  (refit in {r['t_refit']:.0f}s)")
    tbl = fc.iloc[[h - 1 for h in r["horizons"]]].copy()
    tbl["median chg"] = (tbl["lstm_p50"] / last - 1).map("{:+.1%}".format)
    for c in ("lstm_p10", "lstm_p50", "lstm_p90", "rw_p10", "rw_p90"):
        tbl[c] = tbl[c].map("{:,.0f}".format)
    tbl.index = tbl.index.strftime("%Y-%m-%d")
    print(tbl.to_string())
    print("  lstm_p10..p90 = calibrated 80% range for that day's close; rw_* = random-walk baseline range."
          "\n  Dates skip weekends only, not IDX holidays.")
    print(f"\n{r['verdict']}")


def save(r: dict, out: str | Path, img: str | Path) -> tuple[Path, Path]:
    """Write the forecast CSV and chart; returns their paths."""
    out, img = Path(out), Path(img)
    out.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    name = r["label"].lstrip("^")
    csv, png = out / f"lstm_forecast_{name}.csv", img / f"lstm_forecast_{name}.png"
    r["fc"].to_csv(csv, float_format="%.2f")
    plot(r["label"], r["close"], r["fc"], r["test_dates"], r["test_actual"], r["test_band"], r["H"], png)
    return csv, png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=wf.Config.ticker)
    ap.add_argument("--market", default=wf.Config.market, help="market index ticker, or 'none'")
    ap.add_argument("--start", default=wf.Config.start)
    ap.add_argument("--days", type=int, default=20, help="forecast horizon in trading days")
    ap.add_argument("--lookback", type=int, default=60, help="days of history the LSTM sees")
    ap.add_argument("--seeds", type=int, default=5, help="ensemble size")
    ap.add_argument("--calib-window", type=int, default=250,
                    help="days of recent forecast errors used to calibrate the range")
    ap.add_argument("--gamma", type=float, default=0.005,
                    help="adaptive-conformal step size (0 = plain rolling calibration)")
    ap.add_argument("--offline", action="store_true", help="use the CSVs in data/ instead of downloading")
    ap.add_argument("--out", default=wf.Config.out_dir)
    ap.add_argument("--img-dir", default=wf.Config.img_dir)
    a = ap.parse_args()

    cfg = wf.Config(ticker=a.ticker, market=a.market, start=a.start)
    px = load(cfg.ticker, cfg, a.offline)
    mkt = None if cfg.market.lower() == "none" else load(cfg.market, cfg, a.offline)
    r = run(px, mkt, cfg.ticker, a.days, a.lookback, a.seeds, a.calib_window, a.gamma)
    report(r)
    csv, png = save(r, a.out, a.img_dir)
    print(f"Saved {csv.resolve()}")
    print(f"Saved {png.resolve()}")


if __name__ == "__main__":
    main()
