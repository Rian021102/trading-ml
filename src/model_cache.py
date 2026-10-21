"""
Per-close model cache
=====================

Heavy models (walk-forward signal, next-session forecast, LSTM range, hourly
model) only ever train on finished sessions, so their result can't change until
the next close. They are stored once per close date in output/cache/<ticker>/
and reused by every later run that day; the live price is then only used for
fast predictions. One file per kind is kept: a newer close replaces it.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

import pandas as pd

CACHE = Path(__file__).resolve().parent / "output" / "cache"


def cached(kind: str, ticker: str, day: pd.Timestamp, params: dict, build: Callable[[], Any],
           refresh: bool = False) -> tuple[Any, bool]:
    """Return (object, came_from_cache). `params` are the settings that change the result."""
    tag = hashlib.md5(repr(sorted(params.items())).encode()).hexdigest()[:8]
    folder = CACHE / ticker.lstrip("^")
    path = folder / f"{kind}-{day:%Y-%m-%d}-{tag}.pkl"
    if path.exists() and not refresh:
        try:
            with path.open("rb") as f:
                return pickle.load(f), True
        except Exception:                           # stale format after a code change: rebuild
            pass
    obj = build()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{kind}-*.pkl"):
        old.unlink()
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return obj, False
