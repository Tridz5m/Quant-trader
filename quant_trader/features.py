"""Feature engineering for the XAUUSD M5 model.

Every feature is scale-free (normalised by ATR or bounded) so the model keeps
working as the gold price moves from $2,000 to $4,000+, and every feature is
causal: the value on a bar uses only that bar and earlier ones. Higher
timeframe (H1/H4) features are only used once their candle has fully closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, atr, ema, macd, rsi, stochastic

FEATURE_VERSION = 1
# M5 bars needed before all indicators (incl. H4 EMA20) have converged.
WARMUP_BARS = 3000

REQUIRED_COLUMNS = ("open", "high", "low", "close")


def prepare_bars(bars: pd.DataFrame, default_spread_points: float) -> pd.DataFrame:
    """Validate and normalise an OHLC frame (sorted, unique, float)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    df = bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by a DatetimeIndex")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(float)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 1.0
    if "spread" not in df.columns:
        df["spread"] = default_spread_points
    df["tick_volume"] = df["tick_volume"].astype(float)
    df["spread"] = df["spread"].astype(float).fillna(default_spread_points)
    return df


def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    return atr(bars["high"], bars["low"], bars["close"], period)


def _htf_features(bars: pd.DataFrame, minutes: int, prefix: str, base_minutes: int, spans: tuple[int, ...]) -> pd.DataFrame:
    agg = (
        bars[["open", "high", "low", "close"]]
        .resample(f"{minutes}min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    h, l, c = agg["high"], agg["low"], agg["close"]
    a = atr(h, l, c, 14).replace(0.0, np.nan)
    cols = {}
    for span in spans:
        cols[f"{prefix}_ema{span}_dist"] = (c - ema(c, span)) / a
    e = ema(c, spans[0])
    cols[f"{prefix}_ema_slope"] = (e - e.shift(3)) / a
    cols[f"{prefix}_rsi"] = (rsi(c, 14) - 50.0) / 50.0
    cols[f"{prefix}_ret3"] = (c - c.shift(3)) / a
    f = pd.DataFrame(cols, index=agg.index)
    # A higher-timeframe candle opening at T is complete once the M5 bar
    # opening at T + minutes - base_minutes has closed.
    f.index = f.index + pd.Timedelta(minutes=minutes - base_minutes)
    return f.reindex(bars.index, method="ffill")


def build_features(bars: pd.DataFrame, point: float, atr_period: int = 14, base_minutes: int = 5) -> pd.DataFrame:
    """Build the model feature matrix, one row per closed M5 bar."""
    idx = bars.index
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    a = atr(h, l, c, atr_period)
    a_safe = a.replace(0.0, np.nan)
    f: dict[str, pd.Series] = {}

    for k in (1, 3, 6, 12, 24, 48):
        f[f"ret_{k}"] = (c - c.shift(k)) / a_safe
    f["atr_pct"] = a / c * 1000.0
    f["atr_ratio"] = a / atr(h, l, c, 100).replace(0.0, np.nan)

    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    f["ema20_dist"] = (c - e20) / a_safe
    f["ema50_dist"] = (c - e50) / a_safe
    f["ema200_dist"] = (c - e200) / a_safe
    f["ema20_slope"] = (e20 - e20.shift(5)) / a_safe
    f["ema50_slope"] = (e50 - e50.shift(10)) / a_safe
    f["ema20_50"] = (e20 - e50) / a_safe
    f["ema50_200"] = (e50 - e200) / a_safe

    f["rsi14"] = (rsi(c, 14) - 50.0) / 50.0
    f["rsi7"] = (rsi(c, 7) - 50.0) / 50.0
    m_line, m_signal = macd(c)
    f["macd"] = m_line / a_safe
    f["macd_hist"] = (m_line - m_signal) / a_safe

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    f["bb_pos"] = (c - mid) / (2.0 * sd).replace(0.0, np.nan)
    f["bb_width"] = 4.0 * sd / a_safe
    f["stoch14"] = stochastic(h, l, c, 14)
    adx_v, plus_di, minus_di = adx(h, l, c, 14)
    f["adx"] = adx_v / 100.0
    f["di_diff"] = (plus_di - minus_di) / 100.0

    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    f["body"] = (c - o) / a_safe
    f["upper_wick"] = (h - top) / a_safe
    f["lower_wick"] = (bottom - l) / a_safe
    f["bar_range"] = (h - l) / a_safe
    for n in (20, 100):
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        f[f"range_pos_{n}"] = (c - ll) / (hh - ll).replace(0.0, np.nan)
    f["dist_high_50"] = (h.rolling(50).max() - c) / a_safe
    f["dist_low_50"] = (c - l.rolling(50).min()) / a_safe

    logret = np.log(c).diff()
    f["rv_ratio"] = logret.rolling(12).std() / logret.rolling(96).std().replace(0.0, np.nan)
    vol = bars["tick_volume"]
    f["vol_ratio"] = np.log((vol + 1.0) / (vol.rolling(50).mean() + 1.0))
    f["spread_atr"] = bars["spread"] * point / a_safe

    # Time of day / week in server time (sessions are fixed in server time).
    minutes = idx.hour * 60 + idx.minute
    f["tod_sin"] = pd.Series(np.sin(2 * np.pi * minutes / 1440.0), index=idx)
    f["tod_cos"] = pd.Series(np.cos(2 * np.pi * minutes / 1440.0), index=idx)
    f["dow"] = pd.Series(idx.dayofweek.astype(float), index=idx)

    # Daily reference levels: today's open, previous day's high/low/close.
    day = idx.normalize()
    day_open = o.groupby(day).transform("first")
    f["day_open_dist"] = (c - day_open) / a_safe
    daily = bars.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev = daily.shift(1)
    for col in ("high", "low", "close"):
        f[f"prev_{col}_dist"] = (c - pd.Series(prev[col].reindex(day).to_numpy(), index=idx)) / a_safe

    out = pd.DataFrame(f, index=idx)
    out = out.join(_htf_features(bars, 60, "h1", base_minutes, (20, 50)))
    out = out.join(_htf_features(bars, 240, "h4", base_minutes, (20,)))
    return out.replace([np.inf, -np.inf], np.nan).astype(float)
