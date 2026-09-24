"""Trading rules shared by the live bot, the simulator and the backtester.

Keeping these in one place guarantees that what the backtest measures is what
the live bot actually does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig

LONG = 1
SHORT = -1


def decide(p_long: float, p_short: float, thr_long: float, thr_short: float) -> int:
    """Pick a direction: the side whose probability clears its threshold by more."""
    long_ok = math.isfinite(p_long) and p_long >= thr_long
    short_ok = math.isfinite(p_short) and p_short >= thr_short
    if long_ok and short_ok:
        return LONG if (p_long - thr_long) >= (p_short - thr_short) else SHORT
    if long_ok:
        return LONG
    if short_ok:
        return SHORT
    return 0


def decide_vec(p_long: np.ndarray, p_short: np.ndarray, thr_long, thr_short) -> np.ndarray:
    p_long = np.nan_to_num(np.asarray(p_long, dtype=float), nan=-np.inf)
    p_short = np.nan_to_num(np.asarray(p_short, dtype=float), nan=-np.inf)
    thr_long = np.broadcast_to(np.asarray(thr_long, dtype=float), p_long.shape)
    thr_short = np.broadcast_to(np.asarray(thr_short, dtype=float), p_short.shape)
    long_ok = p_long >= thr_long
    short_ok = p_short >= thr_short
    long_better = (p_long - thr_long) >= (p_short - thr_short)
    out = np.zeros(p_long.shape, dtype=np.int8)
    out[long_ok & (~short_ok | long_better)] = LONG
    out[short_ok & (~long_ok | ~long_better)] = SHORT
    return out


def _hour(ts: pd.Timestamp) -> float:
    return ts.hour + ts.minute / 60.0


def in_session(ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    """May a new trade be opened at server time ``ts``?"""
    if ts.dayofweek >= 5:
        return False
    h = _hour(ts)
    if not (cfg.session_start_hour <= h < cfg.session_end_hour):
        return False
    if ts.dayofweek == 4 and h >= cfg.friday_last_entry_hour:
        return False
    return True


def weekend_close_due(ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    return bool(cfg.close_before_weekend and ts.dayofweek == 4 and _hour(ts) >= cfg.friday_close_hour)


def entry_block_reason(now: pd.Timestamp, spread: float, atr: float, bar_range: float, cfg: StrategyConfig) -> str | None:
    """Market-condition filters applied before any new entry."""
    if not in_session(now, cfg):
        return "outside trading session"
    if not np.isfinite(atr) or atr < cfg.min_atr:
        return "volatility too low"
    if spread > cfg.max_spread:
        return f"spread {spread:.2f} > max {cfg.max_spread:.2f}"
    if spread > cfg.max_spread_atr_frac * atr:
        return f"spread {spread:.2f} too wide vs ATR {atr:.2f}"
    if bar_range > cfg.max_bar_range_atr * atr:
        return "abnormal candle (news spike)"
    return None


def check_bar_exit(direction: int, sl: float, tp: float, o: float, h: float, l: float, spread: float):
    """Did a stop or target trigger during a bar?

    Bars are bid prices. Longs exit on the bid; shorts exit on the ask
    (bid + spread). If both levels are inside one bar we assume the stop was
    hit first (conservative). Returns ``(price, reason)`` or ``None``.
    """
    if direction == LONG:
        if sl > 0 and o <= sl:
            return o, "sl"
        if sl > 0 and l <= sl:
            return sl, "sl"
        if tp > 0 and h >= tp:
            return tp, "tp"
    else:
        ao, ah, al = o + spread, h + spread, l + spread
        if sl > 0 and ao >= sl:
            return ao, "sl"
        if sl > 0 and ah >= sl:
            return sl, "sl"
        if tp > 0 and al <= tp:
            return tp, "tp"
    return None


@dataclass
class ManageAction:
    kind: str  # "close" or "modify"
    reason: str
    sl: float | None = None


def manage_position(
    direction: int,
    entry: float,
    sl: float,
    initial_risk: float,
    bid: float,
    ask: float,
    atr: float,
    bars_held: int,
    now: pd.Timestamp,
    cfg: StrategyConfig,
    horizon_bars: int | None = None,
) -> ManageAction | None:
    """Decide what to do with an open position at a bar close.

    ``horizon_bars`` is the trade's own maximum holding time (it depends on
    the geometry the model chose); defaults to the configured one.
    """
    if bars_held >= (horizon_bars or cfg.horizon_bars):
        return ManageAction("close", "time_exit")
    if weekend_close_due(now, cfg):
        return ManageAction("close", "weekend")
    if initial_risk <= 0:
        return None
    price = bid if direction == LONG else ask
    gained = (price - entry) * direction
    new_sl = sl
    if cfg.breakeven_at_r is not None and gained >= cfg.breakeven_at_r * initial_risk:
        be = entry + direction * cfg.breakeven_lock_r * initial_risk
        new_sl = _tighter(direction, new_sl, be)
    if cfg.trailing_start_r is not None and gained >= cfg.trailing_start_r * initial_risk and atr > 0:
        trail = price - direction * cfg.trailing_atr_mult * atr
        new_sl = _tighter(direction, new_sl, trail)
    # The new stop must stay on the losing side of the current price.
    if direction == LONG and new_sl >= bid:
        return None
    if direction == SHORT and new_sl <= ask:
        return None
    if new_sl != sl and abs(new_sl - sl) > 1e-9:
        return ManageAction("modify", "breakeven/trail", sl=new_sl)
    return None


def _tighter(direction: int, current: float, candidate: float) -> float:
    if current <= 0:
        return candidate
    return max(current, candidate) if direction == LONG else min(current, candidate)


def simulate_policy(
    direction: np.ndarray,
    r_long: np.ndarray,
    r_short: np.ndarray,
    exit_long: np.ndarray,
    exit_short: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Take signals one at a time (no overlapping positions).

    Returns the R multiple of each trade and the bar index it was opened on.
    """
    rs: list[float] = []
    idx: list[int] = []
    next_free = 0
    for i in np.flatnonzero(direction != 0):
        if i < next_free:
            continue
        if direction[i] == LONG:
            r, ex = r_long[i], exit_long[i]
        else:
            r, ex = r_short[i], exit_short[i]
        if not np.isfinite(r) or ex <= 0:
            continue
        rs.append(float(r))
        idx.append(int(i))
        next_free = i + int(ex)
    return np.asarray(rs, dtype=float), np.asarray(idx, dtype=int)


def trade_stats(r: np.ndarray) -> dict:
    """Summary statistics of a sequence of R multiples."""
    r = np.asarray(r, dtype=float)
    n = int(r.size)
    if n == 0:
        return {
            "trades": 0, "expectancy_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "total_r": 0.0, "max_dd_r": 0.0, "t_stat": 0.0,
        }
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    equity = np.cumsum(r)
    dd = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    return {
        "trades": n,
        "expectancy_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "total_r": float(r.sum()),
        "max_dd_r": float(dd.max()),
        "t_stat": float(r.mean() / std * np.sqrt(n)) if std > 0 else 0.0,
    }
