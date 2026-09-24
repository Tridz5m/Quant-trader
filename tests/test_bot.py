from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quant_trader.bot import SafetyError, TradingBot
from quant_trader.broker.sim import SimBroker
from quant_trader.features import compute_atr
from quant_trader.journal import Journal
from quant_trader.learner import SelfLearner


class StubModel:
    """Always wants to go long (or short) with fixed confidence."""

    version = "stub"
    tradeable = True
    suspended = False
    thr_long = thr_short = 0.6
    metrics: dict = {}

    def __init__(self, p_long=0.9, p_short=0.1):
        self.p = (p_long, p_short)

    def age_hours(self):
        return 0.0

    def is_compatible(self, scfg):
        return True

    def thresholds(self, bump=0.0):
        return 0.6 + bump, 0.6 + bump

    def predict(self, features):
        n = len(features)
        return np.full(n, self.p[0]), np.full(n, self.p[1])


def _start_index(bars, hour=11):
    """A Wednesday late-morning bar after plenty of history."""
    idx = bars.index
    cands = np.flatnonzero((idx.dayofweek == 2) & (idx.hour == hour) & (idx.minute == 0))
    return int(cands[cands > 6500][0])


def make_bot(cfg, bars, model=None, is_real=False, start=None):
    cfg.schedule.history_bars = 6000
    broker = SimBroker(bars, start_index=start or _start_index(bars), magic=cfg.symbol.magic, is_real=is_real)
    journal = Journal(":memory:")
    learner = SelfLearner(cfg, journal)
    learner.champion = model
    learner.last_attempt = datetime.now(timezone.utc)  # no retraining during the test
    bot = TradingBot(cfg, broker, journal=journal, learner=learner, sleep=lambda s: None)
    bot.start()
    return bot, broker, journal


def test_opens_long_with_correct_risk_and_stops(cfg, noise_bars):
    bot, broker, journal = make_bot(cfg, noise_bars, StubModel())
    report = bot.run_cycle()
    assert report["action"] == "open", report
    (pos,) = broker.positions()
    atr = compute_atr(noise_bars).iloc[broker.i]
    tick = broker.tick()
    assert pos.direction == 1
    assert pos.sl == pytest.approx(tick.ask - cfg.strategy.sl_atr_mult * atr, abs=0.011)
    assert pos.tp == pytest.approx(tick.ask + cfg.strategy.tp_atr_mult * atr, abs=0.011)
    risk_money = pos.volume * (pos.price_open - pos.sl) * broker.spec.value_per_price_unit
    assert risk_money <= 10_000 * cfg.risk.risk_per_trade_pct / 100 + 1e-6
    rec = journal.get_trade(pos.ticket)
    assert rec["status"] == "open" and rec["bar_time"] and rec["features"]


def test_trade_lifecycle_is_journaled_and_one_position_max(cfg, noise_bars):
    bot, broker, journal = make_bot(cfg, noise_bars, StubModel())
    opened = 0
    for _ in range(300):
        report = bot.run_cycle()
        opened += report["action"] == "open"
        assert len(broker.positions()) <= cfg.risk.max_open_positions
        broker.advance()
    closed = journal.closed_trades()
    assert opened >= 2 and len(closed) >= 1
    for t in closed:
        assert t["r_multiple"] is not None
        assert t["close_reason"] in {"sl", "tp", "time_exit", "weekend"}
    assert all(t["r_multiple"] > -1.3 for t in closed)


def test_time_exit_after_horizon(cfg, noise_bars):
    cfg.strategy.sl_atr_mult = 50  # stops far away so only the time exit can close
    cfg.strategy.tp_atr_mult = 50
    cfg.strategy.breakeven_at_r = None
    cfg.risk.min_lot_risk_tolerance = 100
    bot, broker, journal = make_bot(cfg, noise_bars, StubModel())
    assert bot.run_cycle()["action"] == "open"
    for _ in range(cfg.strategy.horizon_bars):
        broker.advance()
        bot.run_cycle()
    (t,) = journal.closed_trades()
    assert t["close_reason"] == "time_exit"


def test_dry_run_sends_no_orders(cfg, noise_bars):
    cfg.dry_run = True
    bot, broker, _ = make_bot(cfg, noise_bars, StubModel())
    report = bot.run_cycle()
    assert report["action"] == "dry_run"
    assert broker.sent == [] and broker.positions() == []


def test_no_model_means_no_trading(cfg, noise_bars):
    bot, broker, _ = make_bot(cfg, noise_bars, None)
    report = bot.run_cycle()
    assert report["action"] == "none" and "model" in report["reason"]
    assert broker.sent == []


def test_real_account_is_refused_by_default(cfg, noise_bars):
    with pytest.raises(SafetyError):
        make_bot(cfg, noise_bars, StubModel(), is_real=True)
    cfg.mt5.allow_real_account = True
    make_bot(cfg, noise_bars, StubModel(), is_real=True)


def test_same_bar_is_not_traded_twice(cfg, noise_bars):
    cfg.risk.max_open_positions = 3
    bot, broker, _ = make_bot(cfg, noise_bars, StubModel())
    assert bot.run_cycle()["action"] == "open"
    assert bot.run_cycle()["reason"] == "bar already processed"
    assert len(broker.positions()) == 1


def test_outside_session_no_entry(cfg, noise_bars):
    start = _start_index(noise_bars, hour=1)  # 01:00 bar closes 01:05, before the 02:00 session start
    bot, broker, _ = make_bot(cfg, noise_bars, StubModel(), start=start)
    report = bot.run_cycle()
    assert report["action"] == "none" and "session" in report["reason"]


def test_scan_schedule_aligns_to_five_minute_closes(cfg, noise_bars):
    bot, _, _ = make_bot(cfg, noise_bars, StubModel())
    base = 1_700_000_100.0  # a multiple of 300 seconds
    bot.wall_clock = lambda: base + 12
    assert bot.seconds_to_next_scan() == pytest.approx(300 - 12 + 5)
    bot.wall_clock = lambda: base + 2
    assert bot.seconds_to_next_scan() == pytest.approx(3)


def test_learner_trains_and_bot_uses_model(cfg, trend_bars):
    journal = Journal(":memory:")
    learner = SelfLearner(cfg, journal)
    assert learner.retrain_reason() == "no model yet"
    model = learner.retrain(trend_bars.iloc[:-500], 0.01, "test")
    assert model is not None and model.tradeable, model.notes if model else None
    assert (cfg.model_path / "champion.joblib").exists()
    assert learner.retrain_reason() is None
    reloaded = SelfLearner(cfg, journal)
    assert reloaded.champion.version == model.version
    assert journal.models()[0]["promoted"] == 1


def test_live_feedback_overrides_labels(cfg, trend_bars):
    from quant_trader.learner import apply_live_feedback, build_dataset

    _, labels = build_dataset(trend_bars, cfg, 0.01)
    base, wide = cfg.strategy.geometries()[:2]
    t1, t2 = labels[base].index[100], labels[base].index[200]
    trades = [
        # A trade from an older journal (no geometry columns) used the base geometry.
        {"bar_time": str(t1), "direction": 1, "r_multiple": -0.8},
        {"bar_time": str(t2), "direction": -1, "r_multiple": 1.2, "sl_atr_mult": wide.sl_atr_mult,
         "tp_atr_mult": wide.tp_atr_mult, "horizon_bars": wide.horizon_bars},
    ]
    new, w = apply_live_feedback(labels, trades, 3.0, cfg)
    assert new[base].loc[t1, "long_r"] == -0.8 and new[base].loc[t1, "long_win"] == 0.0
    assert new[wide].loc[t2, "short_r"] == 1.2 and new[wide].loc[t2, "short_win"] == 1.0
    # Each outcome only rewrites the geometry it was traded with.
    assert new[wide].loc[t1, "long_r"] == labels[wide].loc[t1, "long_r"]
    assert new[base].loc[t2, "short_r"] == labels[base].loc[t2, "short_r"]
    assert w[100] == 3.0 and w[200] == 3.0 and w.sum() == len(w) + 4.0
    assert labels[base].loc[t1, "long_r"] != -0.8  # original frames untouched


def test_snapshot_published_for_the_app(cfg, noise_bars):
    bot, broker, _ = make_bot(cfg, noise_bars, StubModel())
    bot.run_cycle()
    snap = bot.snapshot
    assert snap["report"]["action"] == "open"
    assert snap["account"]["balance"] == 10_000 and len(snap["positions"]) == 1
    assert snap["model"]["model"] == "stub" and "trades_today" in snap["risk"]


def test_run_forever_stops_promptly_and_cleans_up(cfg, noise_bars, tmp_path):
    import threading

    stop = threading.Event()
    broker = SimBroker(noise_bars, start_index=_start_index(noise_bars), magic=cfg.symbol.magic)
    holder = {}

    def run():
        # Like the desktop app: the bot (and its SQLite journal) lives on the worker thread.
        journal = Journal(tmp_path / "j.sqlite")
        learner = SelfLearner(cfg, journal)
        learner.champion = StubModel()
        learner.last_attempt = datetime.now(timezone.utc)
        holder["bot"] = TradingBot(cfg, broker, journal=journal, learner=learner, sleep=stop.wait)
        holder["bot"].run_forever()

    t = threading.Thread(target=run)
    t.start()
    for _ in range(200):
        if holder.get("bot") is not None and holder["bot"].snapshot:
            break
        stop.wait(0.05)
    bot = holder["bot"]
    assert bot.snapshot["report"]["action"] == "started"
    assert bot.next_scan_at is not None
    bot.stop()
    stop.set()
    t.join(timeout=10)
    assert not t.is_alive() and not broker.connected


def test_second_bot_on_same_folder_is_refused(cfg, noise_bars):
    from quant_trader.bot import SafetyError
    from quant_trader.instance import InstanceLock

    lock = InstanceLock(cfg.path(cfg.data_dir) / "bot.lock")
    assert lock.acquire()
    try:
        broker = SimBroker(noise_bars, magic=cfg.symbol.magic)
        bot = TradingBot(cfg, broker, journal=Journal(":memory:"), sleep=lambda s: None)
        with pytest.raises(SafetyError, match="already running"):
            bot.run_forever()
    finally:
        lock.release()
    assert lock.acquire()
    lock.release()


def test_run_once_scans_and_cleans_up(cfg, noise_bars):
    broker = SimBroker(noise_bars, start_index=_start_index(noise_bars), magic=cfg.symbol.magic)
    journal = Journal(":memory:")
    learner = SelfLearner(cfg, journal)
    learner.champion = StubModel()
    learner.last_attempt = datetime.now(timezone.utc)
    bot = TradingBot(cfg, broker, journal=journal, learner=learner, sleep=lambda s: None)
    assert bot.run_once()["action"] == "open"
    assert not broker.connected
    from quant_trader.instance import InstanceLock

    lock = InstanceLock(cfg.path(cfg.data_dir) / "bot.lock")
    assert lock.acquire()  # released again
    lock.release()


def test_entry_uses_the_models_geometry_and_journals_it(cfg, noise_bars):
    from quant_trader.config import Geometry

    model = StubModel()
    model.geometry = Geometry(4.0, 6.0, 12)
    cfg.risk.min_lot_risk_tolerance = 10
    bot, broker, journal = make_bot(cfg, noise_bars, model)
    assert bot.run_cycle()["action"] == "open"
    (pos,) = broker.positions()
    atr = compute_atr(noise_bars).iloc[broker.i]
    assert pos.sl == pytest.approx(broker.tick().ask - 4.0 * atr, abs=0.011)
    assert pos.tp == pytest.approx(broker.tick().ask + 6.0 * atr, abs=0.011)
    rec = journal.get_trade(pos.ticket)
    assert (rec["sl_atr_mult"], rec["tp_atr_mult"], rec["horizon_bars"]) == (4.0, 6.0, 12)
    # The trade is closed after its own 12-bar horizon, not the configured 36.
    for _ in range(12):
        broker.advance()
        bot.run_cycle()
        if journal.closed_trades():
            break
    (t,) = journal.closed_trades()
    assert t["close_reason"] in ("time_exit", "sl", "tp")
    assert broker.i - noise_bars.index.get_loc(pd.Timestamp(rec["bar_time"])) <= 12


def test_stop_is_not_modified_again_when_it_would_not_move(cfg, noise_bars, monkeypatch):
    import quant_trader.bot as botmod
    from quant_trader.policy import ManageAction

    bot, broker, _ = make_bot(cfg, noise_bars, StubModel())
    assert bot.run_cycle()["action"] == "open"
    (pos,) = broker.positions()
    # The rule asks for a stop a hair above the current one; after rounding it is the same price.
    monkeypatch.setattr(botmod, "manage_position", lambda *a, **k: ManageAction("modify", "breakeven", sl=pos.sl + 0.001))
    bot._manage_positions(broker.positions(), broker.tick(), 1.0, broker.now(), noise_bars.index[broker.i], broker.spec)
    assert not [x for x in broker.sent if x["action"] == "modify"]


def test_frozen_tick_feed_is_treated_as_market_closed(cfg, noise_bars):
    """At the weekend MT5 keeps returning Friday's last tick; don't trade on it."""
    clock = [1_000_000.0]
    bot, broker, _ = make_bot(cfg, noise_bars, StubModel())
    bot.wall_clock = lambda: clock[0]
    cfg.risk.max_open_positions = 5
    bot.run_cycle()
    clock[0] += 20 * 60  # 20 minutes pass, no new ticks
    assert "market closed" in bot.run_cycle()["reason"]
    broker.advance()  # quotes move again
    assert "market closed" not in bot.run_cycle()["reason"]


def test_early_retrain_only_counts_current_models_trades(cfg, trend_bars):
    journal = Journal(":memory:")
    learner = SelfLearner(cfg, journal)
    learner.retrain(trend_bars.iloc[:-500], 0.01, "test")
    assert learner.retrain_reason() is None
    n = cfg.learning.early_retrain_trades
    for i in range(n):
        journal.record_entry(ticket=i, direction=1, entry_time=f"2025-03-0{1 + i % 5}T10:00:00",
                             risk_money=50.0, model_version="old-model", status="open")
        journal.close_trade(i, pd.Timestamp("2025-03-06"), 0.0, -50.0, "sl")
    assert learner.retrain_reason() is None  # losses belong to an older model
    for i in range(n, 2 * n):
        journal.record_entry(ticket=i, direction=1, entry_time="2025-03-07T10:00:00",
                             risk_money=50.0, model_version=learner.champion.version, status="open")
        journal.close_trade(i, pd.Timestamp("2025-03-07"), 0.0, -50.0, "sl")
    learner.last_attempt -= pd.Timedelta(hours=cfg.learning.min_hours_between_retrains + 1)
    assert "deteriorated" in learner.retrain_reason()
