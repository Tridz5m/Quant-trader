"""Broker abstraction shared by the live MT5 connector and the simulator."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

LONG = 1
SHORT = -1


class BrokerError(RuntimeError):
    pass


@dataclass
class SymbolSpec:
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float  # account currency per tick per 1.0 lot (loss side)
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int = 0  # points
    freeze_level: int = 0  # points

    @property
    def value_per_price_unit(self) -> float:
        """Money gained/lost per 1.0 price move for 1.0 lot."""
        return self.tick_value / self.tick_size

    def round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, self.digits)

    def round_volume(self, volume: float) -> float:
        """Round down to the volume step and clamp to the maximum.

        Returns 0.0 when the volume is below the broker minimum.
        """
        if volume <= 0 or not math.isfinite(volume):
            return 0.0
        steps = math.floor(volume / self.volume_step + 1e-9)
        v = round(min(steps * self.volume_step, self.volume_max), 8)
        return v if v + 1e-12 >= self.volume_min else 0.0


def default_gold_spec(name: str = "XAUUSD") -> SymbolSpec:
    """Typical XAUUSD contract: 100 oz, 2 digits, $1 per 0.01 per lot."""
    return SymbolSpec(
        name=name,
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=1.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=0,
        freeze_level=0,
    )


@dataclass
class Tick:
    time: pd.Timestamp  # server time
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin_free: float
    currency: str
    leverage: int
    is_real: bool
    trade_allowed: bool = True


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: int
    volume: float
    price_open: float
    sl: float
    tp: float
    time: pd.Timestamp  # server time
    profit: float
    magic: int
    comment: str = ""


@dataclass
class OrderResult:
    ok: bool
    retcode: int
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""


@dataclass
class ClosedTrade:
    ticket: int  # position id
    symbol: str
    direction: int
    volume: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    profit: float  # net of commission, swap and fees
    reason: str = ""


class Broker(ABC):
    """Minimal interface the bot needs from a trading venue."""

    symbol: str

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def symbol_spec(self) -> SymbolSpec: ...

    @abstractmethod
    def tick(self) -> Tick: ...

    @abstractmethod
    def rates(self, count: int) -> pd.DataFrame:
        """Last ``count`` *closed* M5 bars indexed by server open time.

        Columns: open, high, low, close, tick_volume, spread (points).
        """

    @abstractmethod
    def account(self) -> AccountInfo: ...

    @abstractmethod
    def positions(self) -> list[Position]:
        """Open positions on this symbol opened by this bot (magic)."""

    @abstractmethod
    def open_market(self, direction: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult: ...

    @abstractmethod
    def modify_sl_tp(self, position: Position, sl: float, tp: float) -> OrderResult: ...

    @abstractmethod
    def close_position(self, position: Position, comment: str = "") -> OrderResult: ...

    @abstractmethod
    def closed_trades(self, since: pd.Timestamp) -> list[ClosedTrade]: ...

    @abstractmethod
    def margin_required(self, direction: int, volume: float, price: float) -> float | None: ...

    def terminal_trade_allowed(self) -> bool:
        return True

    def market_open(self) -> bool:
        return True
