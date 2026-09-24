"""The live trading loop: scan XAUUSD every 5 minutes and act on its own."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from .broker.base import LONG, Broker, BrokerError, Position, SymbolSpec, Tick
from .config import BotConfig
from .features import LIVE_HISTORY_BARS, TREND_COLUMN, WARMUP_BARS, build_features, compute_atr, prepare_bars
from .instance import InstanceLock
from .journal import Journal
from .learner import SelfLearner
from .news import NewsCalendar
from .policy import ManageAction, decide, entry_block_reason, manage_position
from .risk import RiskManager, RiskState

log = logging.getLogger(__name__)


class SafetyError(RuntimeError):
    pass


class TradingBot:
    def __init__(
        self,
        cfg: BotConfig,
        broker: Broker,
        journal: Journal | None = None,
        learner: SelfLearner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
        news: NewsCalendar | None = None,
    ):
        self.cfg = cfg
        self.broker = broker
        self.journal = journal or Journal(cfg.db_path)
        self.learner = learner or SelfLearner(cfg, self.journal)
        self.news = news or NewsCalendar(cfg.news, cfg.path(cfg.data_dir) / "news_calendar.json")
        self.risk = RiskManager(cfg.risk)
        self.state = RiskState.from_dict(self.journal.get_state("risk_state"))
        self.sleep = sleep
        self.wall_clock = wall_clock
        self.last_bar_time: pd.Timestamp | None = None
        self.bar_delta = pd.Timedelta(minutes=cfg.strategy.timeframe_minutes)
        self._stop = False
        self._last_tick_time: pd.Timestamp | None = None
        self._tick_changed_at = 0.0
        # Read by the desktop app from its own thread; replaced, never mutated.
        self.snapshot: dict = {}
        self.next_scan_at: float | None = None

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self.broker.connect()
        self.safety_checks()

    def safety_checks(self) -> None:
        acc = self.broker.account()
        mode = "REAL" if acc.is_real else "DEMO"
        log.info(
            "Account %s (%s) balance=%.2f equity=%.2f %s leverage 1:%s",
            acc.login, mode, acc.balance, acc.equity, acc.currency, acc.leverage,
        )
        if acc.is_real and not self.cfg.mt5.allow_real_account:
            raise SafetyError(
                "This is a REAL money account. Test on a demo account first; to trade real money "
                "set mt5.allow_real_account: true in config.yaml."
            )
        if not self.cfg.dry_run and not self.broker.terminal_trade_allowed():
            log.warning("Algo Trading is disabled in the MT5 terminal/account - orders will be rejected. "
                        "Enable the 'Algo Trading' button in MT5.")
        if self.cfg.dry_run:
            log.warning("DRY RUN: signals are logged but no orders are sent.")

    def stop(self) -> None:
        self._stop = True

    def _acquire_lock(self) -> InstanceLock:
        lock = InstanceLock(self.cfg.path(self.cfg.data_dir) / "bot.lock")
        if not lock.acquire():
            raise SafetyError("Another Quant Trader bot is already running from this folder. Stop it first.")
        return lock

    def run_once(self) -> dict:
        """Connect, run a single scan and disconnect."""
        lock = self._acquire_lock()
        try:
            self.start()
            self.maybe_retrain()
            return self.run_cycle()
        finally:
            self.broker.shutdown()
            self.journal.close()
            lock.release()

    def run_forever(self) -> None:
        lock = None
        started = False
        try:
            lock = self._acquire_lock()
            self.start()
            started = True
            if not self._stop:
                self.maybe_retrain()
            self.publish_snapshot({"action": "started", "reason": ""})
            log.info("Bot running: scanning %s every %ds", self.broker.symbol, self.cfg.schedule.scan_interval_seconds)
            while not self._stop:
                self.sleep_until_next_scan()
                if self._stop:
                    break
                try:
                    self.run_cycle()
                except BrokerError as exc:
                    log.error("Broker error: %s - reconnecting", exc)
                    self._reconnect()
                except SafetyError:
                    raise
                except Exception:
                    log.exception("Unexpected error in cycle; continuing")
        except KeyboardInterrupt:
            pass
        finally:
            self.next_scan_at = None
            self.broker.shutdown()
            self.journal.close()
            if lock is not None:
                lock.release()
            if started:
                log.info("Bot stopped. Open positions keep their stop-loss and take-profit on the server.")

    def seconds_to_next_scan(self) -> float:
        interval = self.cfg.schedule.scan_interval_seconds
        delay = self.cfg.schedule.bar_close_delay_seconds
        now = self.wall_clock()
        boundary = math.floor(now / interval) * interval
        target = boundary + delay if now < boundary + delay else boundary + interval + delay
        return max(0.0, target - now)

    def sleep_until_next_scan(self) -> None:
        seconds = self.seconds_to_next_scan()
        self.next_scan_at = self.wall_clock() + seconds
        self.sleep(seconds)

    def _reconnect(self) -> None:
        for attempt in range(5):
            if self._stop:
                return
            try:
                self.broker.shutdown()
                self.broker.connect()
                log.info("Reconnected")
                return
            except BrokerError as exc:
                wait = min(60, 5 * 2**attempt)
                log.error("Reconnect failed (%s); retrying in %ds", exc, wait)
                self.sleep(wait)

    # --- learning ---------------------------------------------------------
    def maybe_retrain(self) -> None:
        reason = self.learner.retrain_reason()
        if reason is None:
            return
        try:
            spec = self.broker.symbol_spec()
            lc = self.cfg.learning
            longest = max(g.horizon_bars for g in self.cfg.strategy.geometries())
            needed = lc.train_bars + WARMUP_BARS + longest + 100
            bars = prepare_bars(self.broker.rates(needed), self._default_spread_points(spec))
            self.learner.retrain(bars, spec.point, reason)
        except BrokerError as exc:
            log.error("Could not load history for training: %s", exc)
        except Exception:
            log.exception("Training failed; the bot keeps running and will retry")

    # --- one scan ---------------------------------------------------------
    def _default_spread_points(self, spec: SymbolSpec) -> float:
        return self.cfg.symbol.default_spread / spec.point

    def run_cycle(self) -> dict:
        """One scan: sync, manage open trades, look for a new entry, learn."""
        report: dict = {"action": "none", "reason": ""}
        if not self.broker.is_connected():
            self._reconnect()
        spec = self.broker.symbol_spec()
        tick = self.broker.tick()
        now = tick.time
        # Enough history for the daily trend (30+ days), whatever the config says.
        n_bars = max(self.cfg.schedule.history_bars, LIVE_HISTORY_BARS)
        bars = prepare_bars(self.broker.rates(n_bars), self._default_spread_points(spec))
        last_bar = bars.index[-1]
        report["bar_time"] = last_bar

        self._sync_closed(now)
        try:
            self.news.refresh()
        except Exception:
            log.warning("News calendar refresh failed", exc_info=True)
        account = self.broker.account()
        self.risk.update_equity(self.state, now, account.equity)
        self.journal.log_equity(now, account.balance, account.equity)

        if not self._market_live(now, last_bar):
            report["reason"] = "market closed / no fresh data"
            self._finish_cycle(report)
            return report
        if len(bars) < WARMUP_BARS + 1:
            report["reason"] = f"only {len(bars)} bars of history; need {WARMUP_BARS + 1} (load more history in MT5)"
            self._finish_cycle(report)
            return report

        feats = build_features(bars, spec.point, self.cfg.strategy.atr_period, self.cfg.strategy.timeframe_minutes)
        report["trend"] = float(feats[TREND_COLUMN].iloc[-1])
        atr_now = float(compute_atr(bars, self.cfg.strategy.atr_period).iloc[-1])
        positions = self.broker.positions()
        self._manage_positions(positions, tick, atr_now, now, last_bar, spec)

        if self.last_bar_time is not None and last_bar <= self.last_bar_time:
            report["reason"] = "bar already processed"
            self._finish_cycle(report)
            return report
        self.last_bar_time = last_bar

        self._look_for_entry(report, bars, feats, tick, atr_now, now, last_bar, spec)
        self._finish_cycle(report)
        return report

    def _market_live(self, now: pd.Timestamp, last_bar: pd.Timestamp) -> bool:
        """Fresh bars and a tick feed that is still moving (not a weekend/holiday)."""
        stale_after = self.cfg.schedule.stale_data_minutes
        if now != self._last_tick_time:
            self._last_tick_time = now
            self._tick_changed_at = self.wall_clock()
        frozen = self.wall_clock() - self._tick_changed_at > stale_after * 60
        stale = now - (last_bar + self.bar_delta) > pd.Timedelta(minutes=stale_after)
        return not frozen and not stale and self.broker.market_open()

    def _finish_cycle(self, report: dict) -> None:
        self.journal.set_state("risk_state", self.state.to_dict())
        probs = ""
        if "p_long" in report:
            probs = f" p_long={report['p_long']:.2f} p_short={report['p_short']:.2f}"
        log.info("Scan %s |%s %s%s", report.get("bar_time"), probs, report["action"],
                 f" ({report['reason']})" if report["reason"] else "")
        self.maybe_retrain()
        self.publish_snapshot(report)

    def publish_snapshot(self, report: dict) -> None:
        """Status for the desktop app: account, positions, model and risk.

        Purely informational, so it can never interrupt trading.
        """
        try:
            try:
                acc = self.broker.account()
                positions = self.broker.positions()
            except Exception:
                acc, positions = None, []
            self.snapshot = {
                "updated": self.wall_clock(),
                "symbol": self.broker.symbol,
                "dry_run": self.cfg.dry_run,
                "report": dict(report),
                "account": asdict(acc) if acc is not None else None,
                "positions": [asdict(p) for p in positions],
                "risk": self.state.to_dict(),
                "model": self.learner.status(),
                "news": self.news.status(self._utc_now()),
            }
        except Exception:
            log.debug("Could not build status snapshot", exc_info=True)

    def _look_for_entry(self, report, bars, feats, tick: Tick, atr_now, now, last_bar, spec: SymbolSpec) -> None:
        scfg = self.cfg.strategy
        model = self.learner.champion
        if model is None or not model.tradeable or not model.is_compatible(scfg):
            report["reason"] = "no tradeable model (waiting for the learner to find an edge)"
            return

        row = feats.iloc[[-1]]
        pl, ps = model.predict(row)
        p_long, p_short = float(pl[0]), float(ps[0])
        fl, fs = model.signals(row)  # minus the side against the daily trend
        account = self.broker.account()
        recent = self.journal.recent_r(self.cfg.risk.adaptive_window)
        risk_mult, bump = self.risk.adaptive(recent, self.state, account.equity)
        thr_l, thr_s = model.thresholds(bump)
        direction = decide(float(fl[0]), float(fs[0]), thr_l, thr_s)
        report.update(p_long=p_long, p_short=p_short, thr_long=thr_l, thr_short=thr_s, direction=direction)

        spread = tick.ask - tick.bid
        bar = bars.iloc[-1]
        signal_row = dict(
            time=now, bar_time=last_bar, price=tick.bid, spread=spread, atr=atr_now,
            p_long=p_long, p_short=p_short, thr_long=thr_l, thr_short=thr_s, decision=direction,
        )

        reason = None
        news = self.news.blackout(self._utc_now()) if direction != 0 else None
        if direction == 0:
            reason = "no signal"
            if decide(p_long, p_short, thr_l, thr_s) != 0:
                trend = report.get("trend", math.nan)
                reason = (
                    "signal against the daily trend" if math.isfinite(trend)
                    else "daily trend not known yet (needs 30+ days of M5 history)"
                )
        elif news is not None:
            reason = f"high-impact news: {news.describe()}"
        else:
            reason = entry_block_reason(now, spread, atr_now, float(bar["high"] - bar["low"]), scfg)
            if reason is None:
                ok, why = self.risk.can_open(self.state, now, len(self.broker.positions()))
                reason = None if ok else why
        if reason is not None:
            report["reason"] = reason
            self.journal.log_signal(**signal_row, action="skip", reason=reason)
            return

        # The stop/target/holding time the model was trained for.
        geo = getattr(model, "geometry", None) or scfg.base_geometry
        entry_ref = tick.ask if direction == LONG else tick.bid
        min_dist = (spec.stops_level + 10) * spec.point
        sl_dist = max(geo.sl_atr_mult * atr_now, min_dist)
        tp_dist = max(geo.tp_atr_mult * atr_now, min_dist)
        sl = spec.round_price(entry_ref - direction * sl_dist)
        tp = spec.round_price(entry_ref + direction * tp_dist)
        volume, _ = self.risk.position_size(account.equity, sl_dist, spec, risk_mult)
        volume = self.risk.cap_by_margin(
            volume, self.broker.margin_required(direction, volume, entry_ref), account.margin_free, spec
        )
        if volume <= 0:
            report["reason"] = "position size is zero (risk/margin limits)"
            self.journal.log_signal(**signal_row, action="skip", reason=report["reason"])
            return

        side = "BUY" if direction == LONG else "SELL"
        desc = f"{side} {volume:.2f} {spec.name} @ {entry_ref:.2f} SL {sl:.2f} TP {tp:.2f} (p={p_long if direction == LONG else p_short:.2f})"
        if self.cfg.dry_run:
            report.update(action="dry_run", reason=desc)
            self.journal.log_signal(**signal_row, action="dry_run", reason=desc)
            return

        res = self.broker.open_market(direction, volume, sl, tp, self.cfg.symbol.comment)
        if not res.ok:
            report.update(action="order_failed", reason=f"retcode {res.retcode} {res.comment}")
            log.error("Order rejected: %s -> retcode %s %s", desc, res.retcode, res.comment)
            self.journal.log_signal(**signal_row, action="order_failed", reason=report["reason"])
            return

        fill = res.price or entry_ref
        initial_risk = abs(fill - sl)
        self.journal.record_entry(
            ticket=res.ticket,
            symbol=spec.name,
            direction=direction,
            volume=res.volume or volume,
            entry_time=now,
            bar_time=last_bar,
            entry_price=fill,
            sl=sl,
            tp=tp,
            initial_risk=initial_risk,
            risk_money=(res.volume or volume) * initial_risk * spec.value_per_price_unit,
            atr=atr_now,
            p_long=p_long,
            p_short=p_short,
            model_version=model.version,
            features=row.iloc[0].to_dict(),
            status="open",
            sl_atr_mult=geo.sl_atr_mult,
            tp_atr_mult=geo.tp_atr_mult,
            horizon_bars=geo.horizon_bars,
        )
        self.risk.register_entry(self.state)
        self.journal.log_signal(**signal_row, action="open", reason=desc)
        report.update(action="open", reason=desc, ticket=res.ticket)
        log.info("OPENED #%s %s", res.ticket, desc)

    # --- open positions ---------------------------------------------------
    def _utc_now(self) -> datetime:
        return datetime.fromtimestamp(self.wall_clock(), timezone.utc)

    def _bars_held(self, pos: Position, last_bar: pd.Timestamp) -> int:
        entry_close = pos.time.floor(self.bar_delta)
        return int((last_bar + self.bar_delta - entry_close) / self.bar_delta)

    def _manage_positions(self, positions: list[Position], tick: Tick, atr_now: float, now, last_bar, spec: SymbolSpec) -> None:
        closed_any = False
        utc = self._utc_now()
        upcoming = self.news.blackout(utc) if self.cfg.news.close_positions_before else None
        close_for_news = upcoming is not None and utc < upcoming.time
        for pos in positions:
            rec = self.journal.get_trade(pos.ticket) or self._adopt(pos, atr_now, spec)
            if close_for_news:
                action = ManageAction("close", "news")
            else:
                action = manage_position(
                    pos.direction, pos.price_open, pos.sl, rec["initial_risk"] or 0.0,
                    tick.bid, tick.ask, atr_now, self._bars_held(pos, last_bar), now, self.cfg.strategy,
                    horizon_bars=rec.get("horizon_bars") or self.cfg.strategy.horizon_bars,
                )
            if action is None:
                continue
            if self.cfg.dry_run:
                log.info("DRY RUN: would %s #%s (%s)", action.kind, pos.ticket, action.reason)
                continue
            if action.kind == "close":
                res = self.broker.close_position(pos, action.reason)
                if res.ok:
                    closed_any = True
                    log.info("CLOSED #%s (%s)", pos.ticket, action.reason)
                else:
                    log.error("Close #%s failed: retcode %s %s", pos.ticket, res.retcode, res.comment)
            else:
                new_sl = spec.round_price(action.sl)
                # After rounding to the broker's price step the stop may not move at all.
                improves = new_sl > pos.sl if pos.direction == LONG else (pos.sl <= 0 or new_sl < pos.sl)
                if not improves:
                    continue
                gap = (tick.bid - new_sl) if pos.direction == LONG else (new_sl - tick.ask)
                if gap < (spec.stops_level + spec.freeze_level + 1) * spec.point:
                    continue
                res = self.broker.modify_sl_tp(pos, new_sl, pos.tp)
                if res.ok:
                    self.journal.update_sl(pos.ticket, new_sl)
                    log.info("Moved SL of #%s to %.2f (%s)", pos.ticket, new_sl, action.reason)
                else:
                    log.warning("Modify #%s failed: retcode %s %s", pos.ticket, res.retcode, res.comment)
        if closed_any:
            self._sync_closed(now)

    def _adopt(self, pos: Position, atr_now: float, spec: SymbolSpec) -> dict:
        """Journal a position we did not record (e.g. after a crash)."""
        initial_risk = abs(pos.price_open - pos.sl) if pos.sl > 0 else self.cfg.strategy.sl_atr_mult * atr_now
        rec = dict(
            ticket=pos.ticket, symbol=pos.symbol, direction=pos.direction, volume=pos.volume,
            entry_time=pos.time, entry_price=pos.price_open, sl=pos.sl, tp=pos.tp,
            initial_risk=initial_risk, risk_money=pos.volume * initial_risk * spec.value_per_price_unit,
            status="open",
        )
        self.journal.record_entry(**rec)
        log.info("Adopted untracked position #%s", pos.ticket)
        return self.journal.get_trade(pos.ticket)

    def _sync_closed(self, now: pd.Timestamp) -> None:
        """Record trades closed on the server (SL/TP hits, time exits)."""
        open_recs = self.journal.open_trades()
        if not open_recs:
            return
        live = {p.ticket for p in self.broker.positions()}
        gone = [r for r in open_recs if r["ticket"] not in live]
        if not gone:
            return
        since = min(pd.Timestamp(r["entry_time"]) for r in gone) - pd.Timedelta(days=1)
        closed = {c.ticket: c for c in self.broker.closed_trades(since)}
        for r in gone:
            c = closed.get(r["ticket"])
            if c is None:
                continue  # history not synced yet; retry next scan
            r_mult = self.journal.close_trade(r["ticket"], c.exit_time, c.exit_price, c.profit, c.reason)
            log.info(
                "Trade #%s closed (%s): profit %.2f, %s",
                r["ticket"], c.reason, c.profit, f"{r_mult:+.2f}R" if r_mult is not None else "R n/a",
            )
            self.risk.register_exit(self.state, now, self.journal.recent_r(self.cfg.risk.max_consecutive_losses))
