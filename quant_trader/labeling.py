"""Triple-barrier labels: what would a trade opened at this bar's close do?

For every bar we simulate a long and a short trade with the exact stop,
target and maximum holding time the live bot uses, including spread and
slippage. The model learns P(target hit before stop) for each side, and the
realised R multiples are used to choose confidence thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Geometry, StrategyConfig
from .features import compute_atr

LABEL_COLUMNS = ("long_win", "short_win", "long_r", "short_r", "long_exit", "short_exit")


def triple_barrier(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    spread: np.ndarray,
    sl_mult: float,
    tp_mult: float,
    horizon: int,
    cost: float,
) -> dict[str, np.ndarray]:
    n = len(close)
    risk = sl_mult * atr
    valid = np.isfinite(atr) & (atr > 0)

    entry_l = close + spread + cost  # buy at ask
    sl_l, tp_l = entry_l - risk, entry_l + tp_mult * atr
    entry_s = close - cost  # sell at bid
    sl_s, tp_s = entry_s + risk, entry_s - tp_mult * atr

    r_l = np.full(n, np.nan)
    r_s = np.full(n, np.nan)
    ex_l = np.zeros(n, dtype=np.int32)
    ex_s = np.zeros(n, dtype=np.int32)
    win_l = np.zeros(n)
    win_s = np.zeros(n)
    base = np.arange(n)
    tp_r = tp_mult / sl_mult

    for k in range(1, horizon + 1):
        j = base + k
        ok = valid & (j < n)
        jj = np.where(ok, j, 0)
        o, hi, lo, sp = open_[jj], high[jj], low[jj], spread[jj]

        # Long: stop/target trigger on the bid. Gaps through the stop fill at the open.
        live = ok & np.isnan(r_l)
        sl_hit = live & (lo <= sl_l)
        tp_hit = live & ~sl_hit & (hi >= tp_l)
        exit_px = np.minimum(sl_l, o)
        r_l[sl_hit] = (exit_px[sl_hit] - entry_l[sl_hit]) / risk[sl_hit]
        r_l[tp_hit] = tp_r
        win_l[tp_hit] = 1.0
        ex_l[sl_hit | tp_hit] = k

        # Short: stop/target trigger on the ask (bid + spread).
        live = ok & np.isnan(r_s)
        sl_hit = live & (hi + sp >= sl_s)
        tp_hit = live & ~sl_hit & (lo + sp <= tp_s)
        exit_px = np.maximum(sl_s, o + sp)
        r_s[sl_hit] = (entry_s[sl_hit] - exit_px[sl_hit]) / risk[sl_hit]
        r_s[tp_hit] = tp_r
        win_s[tp_hit] = 1.0
        ex_s[sl_hit | tp_hit] = k

    # Time exit at the close of the horizon bar. The cost allowance is
    # charged once, on entry, exactly as the backtester does.
    j = base + horizon
    ok = valid & (j < n)
    jj = np.where(ok, j, 0)
    t_l = ok & np.isnan(r_l)
    r_l[t_l] = (close[jj][t_l] - entry_l[t_l]) / risk[t_l]
    ex_l[t_l] = horizon
    t_s = ok & np.isnan(r_s)
    r_s[t_s] = (entry_s[t_s] - (close[jj][t_s] + spread[jj][t_s])) / risk[t_s]
    ex_s[t_s] = horizon

    # Rows whose outcome is not yet known stay NaN.
    unknown_l = np.isnan(r_l)
    unknown_s = np.isnan(r_s)
    win_l[unknown_l] = np.nan
    win_s[unknown_s] = np.nan
    return {
        "long_win": win_l,
        "short_win": win_s,
        "long_r": r_l,
        "short_r": r_s,
        "long_exit": ex_l,
        "short_exit": ex_s,
    }


def make_labels(bars: pd.DataFrame, point: float, cfg: StrategyConfig, geometry: Geometry | None = None) -> pd.DataFrame:
    """Labels for ``geometry`` (default: the configured base stop/target/horizon)."""
    g = geometry or cfg.base_geometry
    a = compute_atr(bars, cfg.atr_period).to_numpy(dtype=float)
    out = triple_barrier(
        bars["open"].to_numpy(dtype=float),
        bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        bars["close"].to_numpy(dtype=float),
        a,
        bars["spread"].to_numpy(dtype=float) * point,
        g.sl_atr_mult,
        g.tp_atr_mult,
        g.horizon_bars,
        cfg.slippage,
    )
    return pd.DataFrame(out, index=bars.index)
