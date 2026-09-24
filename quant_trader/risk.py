"""Account-level risk management shared by the live bot and the backtester."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .broker.base import SymbolSpec
from .config import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    trades_today: int = 0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    cooldown_until: str = ""  # server time ISO
    daily_limit_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskState":
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class RiskManager:
    def __init__(self, cfg: RiskConfig, quiet: bool = False):
        self.cfg = cfg
        # Backtests hit these limits thousands of times; log them at debug level there.
        self.quiet = quiet

    def _log(self, level: int, msg: str, *args) -> None:
        log.log(logging.DEBUG if self.quiet else level, msg, *args)

    def update_equity(self, state: RiskState, now: pd.Timestamp, equity: float) -> None:
        """Roll the trading day, track the equity peak and enforce loss limits."""
        day = now.strftime("%Y-%m-%d")
        if state.day != day:
            state.day = day
            state.day_start_equity = equity
            state.trades_today = 0
            state.daily_limit_hit = False
        state.peak_equity = max(state.peak_equity, equity)

        if state.day_start_equity > 0:
            day_pnl_pct = (equity - state.day_start_equity) / state.day_start_equity * 100.0
            if day_pnl_pct <= -self.cfg.max_daily_loss_pct and not state.daily_limit_hit:
                state.daily_limit_hit = True
                self._log(logging.WARNING, "Daily loss limit hit (%.2f%%). No new trades until tomorrow.", day_pnl_pct)
        if state.peak_equity > 0 and not state.halted:
            dd_pct = (state.peak_equity - equity) / state.peak_equity * 100.0
            if dd_pct >= self.cfg.max_drawdown_pct:
                state.halted = True
                state.halt_reason = f"max drawdown {dd_pct:.2f}% >= {self.cfg.max_drawdown_pct}%"
                self._log(logging.ERROR, "KILL SWITCH: %s. Trading halted; run `reset-halt` to resume.", state.halt_reason)

    def drawdown_pct(self, state: RiskState, equity: float) -> float:
        if state.peak_equity <= 0:
            return 0.0
        return max(0.0, (state.peak_equity - equity) / state.peak_equity * 100.0)

    def can_open(self, state: RiskState, now: pd.Timestamp, open_positions: int) -> tuple[bool, str]:
        if state.halted:
            return False, f"halted: {state.halt_reason}"
        if state.daily_limit_hit:
            return False, "daily loss limit reached"
        if open_positions >= self.cfg.max_open_positions:
            return False, "max open positions"
        if state.trades_today >= self.cfg.max_trades_per_day:
            return False, "max trades per day"
        if state.cooldown_until and now < pd.Timestamp(state.cooldown_until):
            return False, f"cooling down after losses until {state.cooldown_until}"
        return True, ""

    def register_entry(self, state: RiskState) -> None:
        state.trades_today += 1

    def register_exit(self, state: RiskState, now: pd.Timestamp, recent_r: list[float]) -> None:
        """``recent_r`` are closed-trade R multiples, oldest first, incl. this one."""
        k = self.cfg.max_consecutive_losses
        if k > 0 and len(recent_r) >= k and all(r < 0 for r in recent_r[-k:]):
            until = now + pd.Timedelta(minutes=self.cfg.cooldown_minutes)
            state.cooldown_until = until.isoformat()
            self._log(logging.WARNING, "%d consecutive losses: cooling down until %s", k, state.cooldown_until)

    def adaptive(self, recent_r: list[float], state: RiskState, equity: float) -> tuple[float, float]:
        """Risk multiplier and probability-threshold bump from recent results.

        After a losing streak the bot trades smaller and demands more
        confidence; it returns to normal once results recover.
        """
        if not self.cfg.adaptive:
            return 1.0, 0.0
        mult, bump = 1.0, 0.0
        window = recent_r[-self.cfg.adaptive_window :]
        if len(window) >= self.cfg.adaptive_min_trades:
            exp = float(np.mean(window))
            if exp < 0:
                mult *= 0.5
                bump += 0.03
            elif exp > 0.3:
                bump -= 0.01
        if self.drawdown_pct(state, equity) >= self.cfg.max_drawdown_pct / 2:
            mult *= 0.5
            bump += 0.02
        return max(mult, 0.25), bump

    def position_size(self, equity: float, sl_distance: float, spec: SymbolSpec, risk_mult: float = 1.0) -> tuple[float, float]:
        """Lots so that hitting the stop loses ``risk_per_trade_pct`` of equity.

        Returns ``(volume, money_at_risk)``; volume 0 means the trade is skipped.
        """
        if sl_distance <= 0 or equity <= 0:
            return 0.0, 0.0
        risk_money = equity * self.cfg.risk_per_trade_pct / 100.0 * risk_mult
        loss_per_lot = sl_distance * spec.value_per_price_unit
        volume = spec.round_volume(risk_money / loss_per_lot)
        if volume == 0.0:
            min_risk = spec.volume_min * loss_per_lot
            if min_risk <= risk_money * self.cfg.min_lot_risk_tolerance:
                volume = spec.volume_min
            else:
                self._log(
                    logging.INFO,
                    "Minimum lot %.2f would risk %.2f > allowed %.2f; skipping (account too small for settings)",
                    spec.volume_min, min_risk, risk_money * self.cfg.min_lot_risk_tolerance,
                )
                return 0.0, 0.0
        return volume, volume * loss_per_lot

    def cap_by_margin(self, volume: float, margin_needed: float | None, free_margin: float, spec: SymbolSpec) -> float:
        """Shrink the volume so it uses at most ``max_margin_usage_pct`` of free margin."""
        if margin_needed is None or margin_needed <= 0 or volume <= 0:
            return volume
        allowed = free_margin * self.cfg.max_margin_usage_pct / 100.0
        if margin_needed <= allowed:
            return volume
        return spec.round_volume(volume * allowed / margin_needed)
