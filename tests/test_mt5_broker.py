"""MT5Broker against a fake MetaTrader5 module (the real one is Windows-only)."""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from quant_trader.broker.base import BrokerError
from quant_trader.config import MT5Config, SymbolConfig

MAGIC = 4242
T0 = int(pd.Timestamp("2025-03-05 10:00").timestamp())


def gold(name="XAUUSDm", filling_mode=2):
    return NS(
        name=name, digits=3, point=0.001, trade_tick_size=0.001, trade_tick_value=0.1,
        trade_tick_value_loss=0.1, trade_contract_size=100.0, volume_min=0.01, volume_max=200.0,
        volume_step=0.01, trade_stops_level=0, trade_freeze_level=0, filling_mode=filling_mode, trade_mode=4,
    )


class FakeMT5:
    """Just enough of the MetaTrader5 API; constants come from the broker's fallbacks."""

    def __init__(self, symbols):
        self.symbols = {s.name: s for s in symbols}
        self.requests = []
        self.rejected_fillings = set()
        self.available_bars = 1000
        self.maxbars = 0
        self.positions = []
        self.deals = []

    def initialize(self, **kw):
        self.init_kwargs = kw
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (1, "Success")

    def symbol_info(self, name):
        return self.symbols.get(name)

    def symbols_get(self):
        return list(self.symbols.values())

    def symbol_select(self, name, enable):
        return name in self.symbols

    def terminal_info(self):
        return NS(name="Fake MT5", connected=True, trade_allowed=True, maxbars=self.maxbars)

    def account_info(self):
        return NS(login=7, server="Demo", balance=1000.0, equity=1010.0, margin_free=900.0, currency="USD",
                  leverage=500, trade_mode=0, trade_allowed=True, trade_expert=True)

    def symbol_info_tick(self, name):
        return NS(time=T0, bid=2650.0, ask=2650.25)

    def copy_rates_from_pos(self, symbol, tf, start, count):
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                 ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
        if self.maxbars and start + count > self.maxbars:
            return None  # the real terminal refuses requests beyond "Max bars in chart"
        n = min(count, self.available_bars)
        arr = np.zeros(n, dtype=dtype)
        arr["time"] = T0 - 300 * np.arange(n, 0, -1)
        arr["open"] = arr["close"] = 2650.0
        arr["high"], arr["low"] = 2651.0, 2649.0
        arr["tick_volume"], arr["spread"] = 100, 250
        return arr

    def order_send(self, req):
        self.requests.append(dict(req))
        if req["action"] == 1 and req.get("type_filling") in self.rejected_fillings:
            return NS(retcode=10030, order=0, deal=0, price=0.0, volume=0.0, comment="Unsupported filling mode")
        return NS(retcode=10009, order=555, deal=777, price=req.get("price", 0.0), volume=req.get("volume", 0.0),
                  comment="Request executed")

    def positions_get(self, symbol=None):
        return tuple(p for p in self.positions if symbol is None or p.symbol == symbol)

    def history_deals_get(self, frm, to):
        return tuple(self.deals)

    def order_calc_margin(self, otype, symbol, volume, price):
        return volume * 100 * price / 500


def make_broker(monkeypatch, symbols):
    fake = FakeMT5(symbols)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    from quant_trader.broker.mt5_broker import MT5Broker

    broker = MT5Broker(MT5Config(login=123, password="pw", server="Demo"), SymbolConfig(name="XAUUSD", magic=MAGIC))
    return broker, fake


def test_connect_resolves_broker_gold_symbol(monkeypatch):
    broker, fake = make_broker(monkeypatch, [NS(name="EURUSDm"), gold("XAUUSDm"), gold("XAUUSDm.pro")])
    broker.connect()
    assert broker.symbol == "XAUUSDm"
    assert fake.init_kwargs["login"] == 123 and fake.init_kwargs["server"] == "Demo"
    spec = broker.symbol_spec()
    assert spec.value_per_price_unit == pytest.approx(100.0)


def test_refuses_non_gold_symbols(monkeypatch):
    broker, _ = make_broker(monkeypatch, [NS(name="EURUSD"), NS(name="BTCUSD"), gold("GOLDEUR"), gold("XAUEUR")])
    with pytest.raises(BrokerError, match="gold"):
        broker.connect()


def test_gold_alias_is_accepted(monkeypatch):
    broker, _ = make_broker(monkeypatch, [gold("GOLDEUR"), gold("GOLD"), gold("XAGUSD")])
    broker.connect()
    assert broker.symbol == "GOLD"


def test_is_xauusd():
    from quant_trader.broker.mt5_broker import is_xauusd

    assert all(is_xauusd(n) for n in ("XAUUSD", "XAUUSDm", "xauusd.a", "GOLD", "GOLD.pro", "XAUUSD+"))
    assert not any(is_xauusd(n) for n in ("XAUEUR", "GOLDEUR", "XAGUSD", "EURUSD", "SILVER"))


def test_rates_frame(monkeypatch):
    broker, _ = make_broker(monkeypatch, [gold("XAUUSD")])
    broker.connect()
    df = broker.rates(500)
    assert list(df.columns) == ["open", "high", "low", "close", "tick_volume", "spread"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert df.index[-1] == pd.Timestamp(T0 - 300, unit="s")


def test_rates_request_is_capped_at_terminal_max_bars(monkeypatch):
    broker, fake = make_broker(monkeypatch, [gold("XAUUSD")])
    broker.connect()
    fake.available_bars, fake.maxbars = 50_000, 3000
    assert len(broker.rates(100_000)) == 2999


def test_market_order_falls_back_to_supported_filling(monkeypatch):
    broker, fake = make_broker(monkeypatch, [gold("XAUUSD", filling_mode=2)])
    broker.connect()
    fake.rejected_fillings = {1}  # IOC advertised but rejected -> fall back
    res = broker.open_market(1, 0.05, 2640.0, 2665.0, "QT-XAU")
    assert res.ok and res.ticket == 555
    sent = [r for r in fake.requests if r["action"] == 1]
    assert [r["type_filling"] for r in sent] == [1, 2]
    last = sent[-1]
    assert last["type"] == 0 and last["price"] == 2650.25 and last["magic"] == MAGIC
    assert last["sl"] == 2640.0 and last["tp"] == 2665.0


def test_positions_are_filtered_by_magic(monkeypatch):
    broker, fake = make_broker(monkeypatch, [gold("XAUUSD")])
    broker.connect()
    common = dict(symbol="XAUUSD", time=T0, volume=0.1, price_open=2650.0, sl=2640.0, tp=2665.0, profit=5.0, comment="")
    fake.positions = [NS(ticket=1, type=1, magic=MAGIC, **common), NS(ticket=2, type=0, magic=999, **common)]
    (pos,) = broker.positions()
    assert pos.ticket == 1 and pos.direction == -1


def test_closed_trades_rebuilt_from_deals(monkeypatch):
    broker, fake = make_broker(monkeypatch, [gold("XAUUSD")])
    broker.connect()

    def deal(pid, entry, dtype, magic, vol, price, profit=0.0, t=T0, reason=3):
        return NS(symbol="XAUUSD", position_id=pid, entry=entry, type=dtype, magic=magic, volume=vol, price=price,
                  profit=profit, commission=-0.35, swap=0.0, fee=0.0, time=t, time_msc=t * 1000, reason=reason,
                  comment="")

    fake.deals = [
        deal(555, 0, 0, MAGIC, 0.1, 2650.25),
        deal(555, 1, 1, 0, 0.1, 2665.0, profit=147.5, t=T0 + 900, reason=5),  # TP by server
        deal(556, 0, 0, 999, 0.1, 2650.0),  # someone else's trade
        deal(556, 1, 1, 999, 0.1, 2660.0, profit=100.0, t=T0 + 600),
        deal(557, 0, 1, MAGIC, 0.2, 2650.0),  # partially closed, still open
        deal(557, 1, 0, MAGIC, 0.1, 2640.0, profit=100.0, t=T0 + 600),
    ]
    (t,) = broker.closed_trades(pd.Timestamp(T0, unit="s"))
    assert t.ticket == 555 and t.direction == 1 and t.reason == "tp"
    assert t.profit == pytest.approx(147.5 - 0.7)
    assert t.exit_price == 2665.0 and t.entry_price == 2650.25
