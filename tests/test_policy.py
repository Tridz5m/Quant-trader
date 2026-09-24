import math

import numpy as np
import pandas as pd

from quant_trader.config import StrategyConfig
from quant_trader.policy import (
    LONG,
    SHORT,
    check_bar_exit,
    decide,
    decide_vec,
    entry_block_reason,
    in_session,
    manage_position,
    simulate_policy,
    trade_stats,
)

WED_NOON = pd.Timestamp("2025-03-05 12:00")


def test_decide_picks_side_with_larger_margin():
    assert decide(0.7, 0.2, 0.6, 0.6) == LONG
    assert decide(0.2, 0.7, 0.6, 0.6) == SHORT
    assert decide(0.65, 0.72, 0.6, 0.6) == SHORT
    assert decide(0.5, 0.5, 0.6, 0.6) == 0
    assert decide(0.99, 0.1, math.inf, 0.6) == 0


def test_decide_vec_matches_scalar():
    rng = np.random.default_rng(0)
    pl, ps = rng.random(500), rng.random(500)
    tl, ts = 0.55, 0.6
    vec = decide_vec(pl, ps, tl, ts)
    assert all(vec[i] == decide(pl[i], ps[i], tl, ts) for i in range(500))


def test_simulate_policy_never_overlaps():
    direction = np.array([1, 1, 1, 0, -1, 0, 1], dtype=np.int8)
    r = np.array([1.0, -1.0, 2.0, 0.0, 1.5, 0.0, -1.0])
    exits = np.array([3, 1, 1, 1, 2, 1, 1])
    rs, idx = simulate_policy(direction, r, r, exits, exits)
    assert list(idx) == [0, 4, 6]
    assert list(rs) == [1.0, 1.5, -1.0]


def test_trade_stats():
    st = trade_stats(np.array([1.5, -1.0, 1.5, -1.0]))
    assert st["trades"] == 4
    assert st["expectancy_r"] == 0.25
    assert st["profit_factor"] == 1.5
    assert st["win_rate"] == 0.5


def test_check_bar_exit():
    assert check_bar_exit(LONG, 99, 102, 100, 102.5, 98.5, 0.0) == (99, "sl")
    assert check_bar_exit(LONG, 99, 102, 100, 102.5, 99.5, 0.0) == (102, "tp")
    assert check_bar_exit(LONG, 99, 102, 98, 98.5, 97, 0.0) == (98, "sl")
    assert check_bar_exit(SHORT, 101, 98, 100, 100.8, 99, 0.3) == (101, "sl")
    assert check_bar_exit(SHORT, 101, 98, 100, 100.5, 97.5, 0.3) == (98, "tp")
    assert check_bar_exit(LONG, 99, 102, 100, 101, 99.5, 0.0) is None


def test_sessions_and_filters():
    cfg = StrategyConfig()
    assert in_session(WED_NOON, cfg)
    assert not in_session(pd.Timestamp("2025-03-05 01:30"), cfg)  # rollover hour
    assert not in_session(pd.Timestamp("2025-03-07 21:30"), cfg)  # Friday evening
    assert not in_session(pd.Timestamp("2025-03-08 12:00"), cfg)  # Saturday
    assert entry_block_reason(WED_NOON, 0.3, 2.0, 2.0, cfg) is None
    assert "spread" in entry_block_reason(WED_NOON, 1.5, 2.0, 2.0, cfg)
    assert "news" in entry_block_reason(WED_NOON, 0.3, 2.0, 9.0, cfg)
    assert "volatility" in entry_block_reason(WED_NOON, 0.1, 0.1, 0.1, cfg)


def test_manage_position_time_exit_breakeven_and_weekend():
    cfg = StrategyConfig()
    act = manage_position(LONG, 100, 97, 3, 100.5, 100.8, 2, cfg.horizon_bars, WED_NOON, cfg)
    assert act.kind == "close" and act.reason == "time_exit"
    act = manage_position(LONG, 100, 97, 3, 103.2, 103.5, 2, 5, WED_NOON, cfg)
    assert act.kind == "modify" and math.isclose(act.sl, 100.3)
    act = manage_position(SHORT, 100, 103, 3, 96.5, 96.8, 2, 5, WED_NOON, cfg)
    assert act.kind == "modify" and math.isclose(act.sl, 99.7)
    assert manage_position(LONG, 100, 97, 3, 101, 101.3, 2, 5, WED_NOON, cfg) is None
    act = manage_position(LONG, 100, 97, 3, 101, 101.3, 2, 5, pd.Timestamp("2025-03-07 23:10"), cfg)
    assert act.kind == "close" and act.reason == "weekend"


def test_trailing_stop():
    cfg = StrategyConfig(breakeven_at_r=None, trailing_start_r=1.0, trailing_atr_mult=1.0)
    act = manage_position(LONG, 100, 97, 3, 106, 106.3, 2, 5, WED_NOON, cfg)
    assert act.kind == "modify" and math.isclose(act.sl, 104)
    # Never loosen: a lower trail than the current stop is ignored.
    assert manage_position(LONG, 100, 105, 3, 106, 106.3, 2, 5, WED_NOON, cfg) is None
