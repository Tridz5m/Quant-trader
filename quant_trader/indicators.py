"""Causal technical indicators (each value uses only current and past bars)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0.0), n)
    dn = wilder((-d).clip(lower=0.0), n)
    out = 100.0 - 100.0 / (1.0 + up / dn.replace(0.0, np.nan))
    out = out.mask((dn == 0) & (up > 0), 100.0)
    return out.mask((dn == 0) & (up == 0), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = wilder(true_range(high, low, close), n).replace(0.0, np.nan)
    plus_di = 100.0 * wilder(plus_dm, n) / tr
    minus_di = 100.0 * wilder(minus_dm, n) / tr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder(dx, n), plus_di, minus_di


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return (close - ll) / (hh - ll).replace(0.0, np.nan)
