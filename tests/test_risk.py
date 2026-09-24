import pandas as pd

from quant_trader.broker.base import default_gold_spec
from quant_trader.config import RiskConfig
from quant_trader.risk import RiskManager, RiskState

T0 = pd.Timestamp("2025-03-05 10:00")


def test_position_size_risks_configured_percent():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5))
    vol, risk = rm.position_size(10_000, 5.0, default_gold_spec())
    # 0.5% of 10k = $50; $5 stop * $100 per point per lot -> 0.10 lots.
    assert vol == 0.10
    assert abs(risk - 50.0) < 1e-9


def test_position_size_rounds_down_and_respects_min_lot():
    spec = default_gold_spec()
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5))
    vol, _ = rm.position_size(10_000, 6.0, spec)
    assert vol == 0.08  # 0.0833 rounded down
    vol, _ = rm.position_size(500, 6.0, spec)  # min lot would risk $6 > $2.5
    assert vol == 0.0
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5, min_lot_risk_tolerance=3.0))
    vol, _ = rm.position_size(500, 6.0, spec)
    assert vol == 0.01


def test_round_volume():
    spec = default_gold_spec()
    assert spec.round_volume(0.129) == 0.12
    assert spec.round_volume(0.004) == 0.0
    assert spec.round_volume(500) == spec.volume_max


def test_daily_loss_limit_resets_next_day():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=2.0))
    st = RiskState()
    rm.update_equity(st, T0, 10_000)
    rm.update_equity(st, T0 + pd.Timedelta(hours=2), 9_790)
    ok, why = rm.can_open(st, T0 + pd.Timedelta(hours=2), 0)
    assert not ok and "daily" in why
    rm.update_equity(st, T0 + pd.Timedelta(days=1), 9_790)
    assert rm.can_open(st, T0 + pd.Timedelta(days=1), 0)[0]


def test_drawdown_kill_switch():
    rm = RiskManager(RiskConfig(max_drawdown_pct=10.0, max_daily_loss_pct=50))
    st = RiskState()
    rm.update_equity(st, T0, 10_000)
    rm.update_equity(st, T0 + pd.Timedelta(days=1), 12_000)
    rm.update_equity(st, T0 + pd.Timedelta(days=2), 10_700)
    assert st.halted
    ok, why = rm.can_open(st, T0 + pd.Timedelta(days=3), 0)
    assert not ok and "halted" in why


def test_cooldown_after_consecutive_losses():
    rm = RiskManager(RiskConfig(max_consecutive_losses=3, cooldown_minutes=60))
    st = RiskState()
    rm.register_exit(st, T0, [1.0, -1.0, -1.0])
    assert rm.can_open(st, T0, 0)[0]
    rm.register_exit(st, T0, [-1.0, -1.0, -1.0])
    assert not rm.can_open(st, T0 + pd.Timedelta(minutes=30), 0)[0]
    assert rm.can_open(st, T0 + pd.Timedelta(minutes=61), 0)[0]


def test_limits_on_positions_and_trades_per_day():
    rm = RiskManager(RiskConfig(max_open_positions=1, max_trades_per_day=2))
    st = RiskState()
    rm.update_equity(st, T0, 10_000)
    assert not rm.can_open(st, T0, 1)[0]
    rm.register_entry(st)
    rm.register_entry(st)
    assert not rm.can_open(st, T0, 0)[0]


def test_adaptive_risk_after_losing_streak():
    rm = RiskManager(RiskConfig(adaptive_min_trades=5))
    st = RiskState(peak_equity=10_000)
    assert rm.adaptive([1, -1, 1, -1, 1], st, 10_000) == (1.0, 0.0)
    mult, bump = rm.adaptive([-1, -1, 1.5, -1, -1], st, 10_000)
    assert mult == 0.5 and bump > 0


def test_margin_cap():
    rm = RiskManager(RiskConfig(max_margin_usage_pct=30))
    spec = default_gold_spec()
    assert rm.cap_by_margin(1.0, 2000, 10_000, spec) == 1.0
    assert rm.cap_by_margin(1.0, 6000, 10_000, spec) == 0.5
