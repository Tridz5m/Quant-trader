"""Synthetic gold-like M5 bars for tests and offline demos.

This is NOT market data. It has realistic structure (sessions, volatility
clustering, trending regimes, spread widening at rollover) so the full
pipeline can be exercised without an MT5 terminal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_DAY = 276  # 01:00 - 23:55 server time


def synthetic_gold_bars(
    n_bars: int = 40_000,
    seed: int = 7,
    start: str = "2025-01-06",
    start_price: float = 2650.0,
    trend_persistence: float = 0.99,
    trend_strength: float = 0.12,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = int(np.ceil(n_bars / BARS_PER_DAY)) + 1
    days = pd.bdate_range(start, periods=n_days)
    offsets = pd.timedelta_range("01:00:00", "23:55:00", freq="5min")
    idx = pd.DatetimeIndex((days.values[:, None] + offsets.values[None, :]).ravel()[:n_bars])

    hours = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    # Intraday volatility profile: quiet Asia, busy London and New York.
    profile = 0.55 + 0.6 * np.exp(-(((hours - 10.5) / 1.6) ** 2)) + 1.0 * np.exp(-(((hours - 16.0) / 1.8) ** 2))

    # Volatility clustering (log-AR(1)).
    hvol = np.zeros(n_bars)
    shocks = rng.normal(0, 0.06, n_bars)
    for t in range(1, n_bars):
        hvol[t] = 0.985 * hvol[t - 1] + shocks[t]
    base_vol = 0.0009
    vol = base_vol * profile * np.exp(hvol)

    # Persistent latent drift -> trending regimes a model can learn from.
    drift = np.zeros(n_bars)
    d_shocks = rng.normal(0, 1, n_bars) * np.sqrt(1 - trend_persistence**2)
    for t in range(1, n_bars):
        drift[t] = trend_persistence * drift[t - 1] + d_shocks[t]
    mu = trend_strength * vol * drift

    # Five one-minute steps per bar to form realistic OHLC.
    steps = mu[:, None] / 5 + (vol[:, None] / np.sqrt(5)) * rng.standard_t(5, size=(n_bars, 5)) / np.sqrt(5 / 3)
    log_path = np.log(start_price) + np.cumsum(steps.ravel()).reshape(n_bars, 5)
    close = np.exp(log_path[:, -1])
    open_ = np.empty(n_bars)
    open_[0] = start_price
    open_[1:] = close[:-1]
    intrabar = np.exp(log_path)
    high = np.maximum(intrabar.max(axis=1), open_)
    low = np.minimum(intrabar.min(axis=1), open_)

    tick_volume = np.round(80 * profile * np.exp(hvol) * (1 + np.abs(rng.normal(0, 0.5, n_bars)))) + 1
    spread = 18 + rng.integers(0, 12, n_bars).astype(float)
    rollover = (hours < 1.5) | (hours >= 23.5)
    spread[rollover] += 35

    return pd.DataFrame(
        {
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "tick_volume": tick_volume,
            "spread": spread,
        },
        index=idx,
    ).rename_axis("time")
