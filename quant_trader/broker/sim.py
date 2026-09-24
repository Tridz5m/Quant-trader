"""In-memory broker that replays historical M5 bars.

Used by the test-suite and for offline dry runs of the full bot loop. Stops
and targets are evaluated with the same rule as the backtester.
"""

from __future__ import annotations

import pandas as pd

from ..policy import LONG, check_bar_exit
from .base import (
    AccountInfo,
    Broker,
    ClosedTrade,
    OrderResult,
    Position,
    SymbolSpec,
    Tick,
    default_gold_spec,
)


class SimBroker(Broker):
    def __init__(
        self,
        bars: pd.DataFrame,
        spec: SymbolSpec | None = None,
        start_index: int | None = None,
        balance: float = 10_000.0,
        magic: int = 0,
        is_real: bool = False,
        commission_per_lot: float = 0.0,
        bar_minutes: int = 5,
        leverage: int = 100,
    ):
        self.bars = bars
        self.spec = spec or default_gold_spec()
        self.symbol = self.spec.name
        self.i = len(bars) - 1 if start_index is None else start_index
        self.balance = balance
        self.magic = magic
        self.is_real = is_real
        self.commission_per_lot = commission_per_lot
        self.bar_delta = pd.Timedelta(minutes=bar_minutes)
        self.leverage = leverage
        self.connected = False
        self._positions: dict[int, Position] = {}
        self._closed: list[ClosedTrade] = []
        self._next_ticket = 1000
        self.sent: list[dict] = []

    # --- connection -------------------------------------------------------
    def connect(self) -> None:
        self.connected = True

    def shutdown(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    # --- market data ------------------------------------------------------
    def symbol_spec(self) -> SymbolSpec:
        return self.spec

    def _spread_at(self, i: int) -> float:
        return float(self.bars["spread"].iat[i]) * self.spec.point

    def now(self) -> pd.Timestamp:
        """Server time: the moment the latest bar closed."""
        return self.bars.index[self.i] + self.bar_delta

    def tick(self) -> Tick:
        bid = float(self.bars["close"].iat[self.i])
        return Tick(time=self.now(), bid=bid, ask=bid + self._spread_at(self.i))

    def rates(self, count: int) -> pd.DataFrame:
        lo = max(0, self.i - count + 1)
        return self.bars.iloc[lo : self.i + 1]

    def advance(self, steps: int = 1) -> bool:
        """Move forward bar by bar, triggering stops/targets on the way."""
        for _ in range(steps):
            if self.i + 1 >= len(self.bars):
                return False
            self.i += 1
            row = self.bars.iloc[self.i]
            spread = self._spread_at(self.i)
            when = self.bars.index[self.i] + self.bar_delta / 2
            for pos in list(self._positions.values()):
                hit = check_bar_exit(pos.direction, pos.sl, pos.tp, row["open"], row["high"], row["low"], spread)
                if hit:
                    self._close(pos, hit[0], hit[1], when)
        return True

    # --- account ----------------------------------------------------------
    def _floating(self, pos: Position) -> float:
        t = self.tick()
        exit_px = t.bid if pos.direction == LONG else t.ask
        return (exit_px - pos.price_open) * pos.direction * pos.volume * self.spec.value_per_price_unit

    def account(self) -> AccountInfo:
        floating = sum(self._floating(p) for p in self._positions.values())
        equity = self.balance + floating
        used = sum(self.margin_required(p.direction, p.volume, p.price_open) or 0 for p in self._positions.values())
        return AccountInfo(
            login=1,
            balance=self.balance,
            equity=equity,
            margin_free=equity - used,
            currency="USD",
            leverage=self.leverage,
            is_real=self.is_real,
        )

    def positions(self) -> list[Position]:
        out = []
        for p in self._positions.values():
            p.profit = self._floating(p)
            out.append(p)
        return out

    def margin_required(self, direction: int, volume: float, price: float) -> float | None:
        return volume * self.spec.contract_size * price / self.leverage

    # --- trading ----------------------------------------------------------
    def open_market(self, direction: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult:
        t = self.tick()
        price = t.ask if direction == LONG else t.bid
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = Position(
            ticket=ticket,
            symbol=self.symbol,
            direction=direction,
            volume=volume,
            price_open=price,
            sl=sl,
            tp=tp,
            time=self.now(),
            profit=0.0,
            magic=self.magic,
            comment=comment,
        )
        self.sent.append({"action": "open", "direction": direction, "volume": volume, "sl": sl, "tp": tp, "price": price})
        return OrderResult(ok=True, retcode=10009, ticket=ticket, price=price, volume=volume)

    def modify_sl_tp(self, position: Position, sl: float, tp: float) -> OrderResult:
        pos = self._positions.get(position.ticket)
        if pos is None:
            return OrderResult(ok=False, retcode=10036, comment="position not found")
        pos.sl, pos.tp = sl, tp
        self.sent.append({"action": "modify", "ticket": pos.ticket, "sl": sl, "tp": tp})
        return OrderResult(ok=True, retcode=10009, ticket=pos.ticket)

    def close_position(self, position: Position, comment: str = "") -> OrderResult:
        pos = self._positions.get(position.ticket)
        if pos is None:
            return OrderResult(ok=False, retcode=10036, comment="position not found")
        t = self.tick()
        price = t.bid if pos.direction == LONG else t.ask
        self._close(pos, price, comment or "close", self.now())
        self.sent.append({"action": "close", "ticket": pos.ticket, "price": price})
        return OrderResult(ok=True, retcode=10009, ticket=pos.ticket, price=price, volume=pos.volume)

    def _close(self, pos: Position, price: float, reason: str, when: pd.Timestamp) -> None:
        gross = (price - pos.price_open) * pos.direction * pos.volume * self.spec.value_per_price_unit
        net = gross - self.commission_per_lot * pos.volume
        self.balance += net
        del self._positions[pos.ticket]
        self._closed.append(
            ClosedTrade(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction=pos.direction,
                volume=pos.volume,
                entry_time=pos.time,
                exit_time=when,
                entry_price=pos.price_open,
                exit_price=price,
                profit=net,
                reason=reason,
            )
        )

    def closed_trades(self, since: pd.Timestamp) -> list[ClosedTrade]:
        return [t for t in self._closed if t.exit_time >= since]
