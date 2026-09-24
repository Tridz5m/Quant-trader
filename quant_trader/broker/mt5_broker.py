"""MetaTrader 5 connector (Windows, uses the official ``MetaTrader5`` package).

The terminal must be installed, logged in (or credentials supplied in the
config) and the "Algo Trading" button must be enabled.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from ..config import MT5Config, SymbolConfig
from .base import (
    LONG,
    AccountInfo,
    Broker,
    BrokerError,
    ClosedTrade,
    OrderResult,
    Position,
    SymbolSpec,
    Tick,
)

log = logging.getLogger(__name__)

# Numeric fallbacks for constants, in case an older package lacks a name.
_C = {
    "TIMEFRAME_M5": 5,
    "ORDER_TYPE_BUY": 0,
    "ORDER_TYPE_SELL": 1,
    "TRADE_ACTION_DEAL": 1,
    "TRADE_ACTION_SLTP": 6,
    "ORDER_TIME_GTC": 0,
    "ORDER_FILLING_FOK": 0,
    "ORDER_FILLING_IOC": 1,
    "ORDER_FILLING_RETURN": 2,
    "POSITION_TYPE_BUY": 0,
    "DEAL_TYPE_BUY": 0,
    "DEAL_ENTRY_IN": 0,
    "DEAL_ENTRY_OUT": 1,
    "DEAL_ENTRY_INOUT": 2,
    "DEAL_ENTRY_OUT_BY": 3,
    "DEAL_REASON_SL": 4,
    "DEAL_REASON_TP": 5,
    "DEAL_REASON_SO": 6,
    "ACCOUNT_TRADE_MODE_REAL": 2,
    "SYMBOL_TRADE_MODE_FULL": 4,
    "TRADE_RETCODE_PLACED": 10008,
    "TRADE_RETCODE_DONE": 10009,
    "TRADE_RETCODE_DONE_PARTIAL": 10010,
    "TRADE_RETCODE_REQUOTE": 10004,
    "TRADE_RETCODE_PRICE_CHANGED": 10020,
    "TRADE_RETCODE_PRICE_OFF": 10021,
    "TRADE_RETCODE_INVALID_FILL": 10030,
    "TRADE_RETCODE_NO_CHANGES": 10025,
    "TRADE_RETCODE_CONNECTION": 10031,
    "TRADE_RETCODE_TIMEOUT": 10012,
}

OTHER_CURRENCIES = ("EUR", "GBP", "AUD", "JPY", "CHF", "CAD", "CNH", "SGD", "HKD", "TRY", "ZAR", "XAG")


def is_xauusd(name: str) -> bool:
    """Gold quoted in USD: XAUUSD variants, or GOLD without another currency."""
    n = name.upper()
    if "XAUUSD" in n:
        return True
    return n.startswith("GOLD") and not any(c in n for c in OTHER_CURRENCIES)


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:  # pragma: no cover - platform specific
        raise BrokerError(
            "The MetaTrader5 package is not installed. It only works on Windows "
            "with the MT5 terminal installed: pip install MetaTrader5"
        ) from exc
    return mt5


class MT5Broker(Broker):
    def __init__(self, mt5_cfg: MT5Config, sym_cfg: SymbolConfig):
        self.cfg = mt5_cfg
        self.sym_cfg = sym_cfg
        self.mt5 = _import_mt5()
        self.symbol = sym_cfg.name
        self.magic = sym_cfg.magic
        self._spec: SymbolSpec | None = None
        self._filling: int | None = None

    def c(self, name: str) -> int:
        return getattr(self.mt5, name, _C[name])

    def _err(self, what: str) -> BrokerError:
        return BrokerError(f"{what} failed: {self.mt5.last_error()}")

    # --- connection -------------------------------------------------------
    def connect(self) -> None:
        kwargs = {"timeout": self.cfg.timeout_ms}
        if self.cfg.terminal_path:
            kwargs["path"] = self.cfg.terminal_path
        if self.cfg.login:
            kwargs["login"] = int(self.cfg.login)
            if self.cfg.password:
                kwargs["password"] = self.cfg.password
            if self.cfg.server:
                kwargs["server"] = self.cfg.server
        if not self.mt5.initialize(**kwargs):
            raise self._err("mt5.initialize")
        self.symbol = self._resolve_symbol(self.sym_cfg.name)
        if not self.mt5.symbol_select(self.symbol, True):
            raise self._err(f"symbol_select({self.symbol})")
        self._spec = None
        self._filling = None
        info = self.mt5.terminal_info()
        acc = self.mt5.account_info()
        log.info(
            "Connected to MT5: %s | account %s @ %s | symbol %s",
            getattr(info, "name", "?"), getattr(acc, "login", "?"), getattr(acc, "server", "?"), self.symbol,
        )

    def shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:  # pragma: no cover
            pass

    def is_connected(self) -> bool:
        info = self.mt5.terminal_info()
        return bool(info is not None and getattr(info, "connected", False))

    def terminal_trade_allowed(self) -> bool:
        info = self.mt5.terminal_info()
        acc = self.mt5.account_info()
        return bool(
            info is not None and info.trade_allowed and acc is not None and acc.trade_allowed and acc.trade_expert
        )

    def _resolve_symbol(self, base: str) -> str:
        """Find the broker's gold symbol (XAUUSD, XAUUSDm, XAUUSD.a, GOLD ...)."""
        if self.mt5.symbol_info(base) is not None:
            name = base
        else:
            symbols = self.mt5.symbols_get() or ()
            names = [s.name for s in symbols]
            cands = [n for n in names if base.upper() in n.upper() and is_xauusd(n)]
            if not cands:
                cands = [n for n in names if is_xauusd(n)]
            # Prefer the plainest name, e.g. "XAUUSD" over "XAUUSD_ECN.pro".
            cands.sort(key=lambda n: (len(n), n))
            if not cands:
                raise BrokerError(f"No gold symbol matching '{base}' found on this broker")
            name = cands[0]
        if not is_xauusd(name):
            raise BrokerError(f"Symbol '{name}' is not gold (XAUUSD); this bot trades XAUUSD only")
        return name

    # --- market data ------------------------------------------------------
    def symbol_spec(self) -> SymbolSpec:
        if self._spec is None:
            s = self.mt5.symbol_info(self.symbol)
            if s is None:
                raise self._err("symbol_info")
            tick_value = getattr(s, "trade_tick_value_loss", 0.0) or s.trade_tick_value
            self._spec = SymbolSpec(
                name=s.name,
                digits=s.digits,
                point=s.point,
                tick_size=s.trade_tick_size or s.point,
                tick_value=tick_value,
                contract_size=s.trade_contract_size,
                volume_min=s.volume_min,
                volume_max=s.volume_max,
                volume_step=s.volume_step,
                stops_level=s.trade_stops_level,
                freeze_level=s.trade_freeze_level,
            )
        return self._spec

    def market_open(self) -> bool:
        s = self.mt5.symbol_info(self.symbol)
        return bool(s is not None and s.trade_mode == self.c("SYMBOL_TRADE_MODE_FULL"))

    def tick(self) -> Tick:
        t = self.mt5.symbol_info_tick(self.symbol)
        if t is None:
            raise self._err("symbol_info_tick")
        return Tick(time=pd.Timestamp(t.time, unit="s"), bid=t.bid, ask=t.ask)

    def rates(self, count: int) -> pd.DataFrame:
        tf = self.c("TIMEFRAME_M5")
        arr = None
        n = count
        while n >= 500:
            # start_pos=1 skips the candle that is still forming.
            arr = self.mt5.copy_rates_from_pos(self.symbol, tf, 1, n)
            if arr is not None and len(arr) > 0:
                break
            n //= 2
        if arr is None or len(arr) == 0:
            raise self._err("copy_rates_from_pos")
        df = pd.DataFrame(arr)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        return df[["open", "high", "low", "close", "tick_volume", "spread"]].astype(float)

    # --- account ----------------------------------------------------------
    def account(self) -> AccountInfo:
        a = self.mt5.account_info()
        if a is None:
            raise self._err("account_info")
        return AccountInfo(
            login=a.login,
            balance=a.balance,
            equity=a.equity,
            margin_free=a.margin_free,
            currency=a.currency,
            leverage=a.leverage,
            is_real=a.trade_mode == self.c("ACCOUNT_TRADE_MODE_REAL"),
            trade_allowed=bool(a.trade_allowed and a.trade_expert),
        )

    def positions(self) -> list[Position]:
        raw = self.mt5.positions_get(symbol=self.symbol)
        if raw is None:
            raise self._err("positions_get")
        out = []
        for p in raw:
            if p.magic != self.magic:
                continue
            out.append(
                Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    direction=LONG if p.type == self.c("POSITION_TYPE_BUY") else -LONG,
                    volume=p.volume,
                    price_open=p.price_open,
                    sl=p.sl,
                    tp=p.tp,
                    time=pd.Timestamp(p.time, unit="s"),
                    profit=p.profit,
                    magic=p.magic,
                    comment=p.comment,
                )
            )
        return out

    def margin_required(self, direction: int, volume: float, price: float) -> float | None:
        otype = self.c("ORDER_TYPE_BUY") if direction == LONG else self.c("ORDER_TYPE_SELL")
        return self.mt5.order_calc_margin(otype, self.symbol, volume, price)

    # --- trading ----------------------------------------------------------
    def _filling_candidates(self) -> list[int]:
        s = self.mt5.symbol_info(self.symbol)
        mode = getattr(s, "filling_mode", 0) if s is not None else 0
        cands = []
        if mode & 1:
            cands.append(self.c("ORDER_FILLING_FOK"))
        if mode & 2:
            cands.append(self.c("ORDER_FILLING_IOC"))
        cands.append(self.c("ORDER_FILLING_RETURN"))
        for f in (self.c("ORDER_FILLING_FOK"), self.c("ORDER_FILLING_IOC")):
            if f not in cands:
                cands.append(f)
        if self._filling in cands:
            cands.remove(self._filling)
            cands.insert(0, self._filling)
        return cands

    def _send(self, request: dict, price_fn=None, attempts: int = 3) -> OrderResult:
        retry_codes = {
            self.c("TRADE_RETCODE_REQUOTE"),
            self.c("TRADE_RETCODE_PRICE_CHANGED"),
            self.c("TRADE_RETCODE_PRICE_OFF"),
            self.c("TRADE_RETCODE_CONNECTION"),
            self.c("TRADE_RETCODE_TIMEOUT"),
        }
        ok_codes = {
            self.c("TRADE_RETCODE_DONE"),
            self.c("TRADE_RETCODE_PLACED"),
            self.c("TRADE_RETCODE_DONE_PARTIAL"),
        }
        fillings = self._filling_candidates() if request["action"] == self.c("TRADE_ACTION_DEAL") else [None]
        last = OrderResult(ok=False, retcode=-1, comment="not sent")
        for filling in fillings:
            if filling is not None:
                request["type_filling"] = filling
            for attempt in range(attempts):
                if price_fn is not None:
                    request["price"] = price_fn()
                res = self.mt5.order_send(request)
                if res is None:
                    last = OrderResult(ok=False, retcode=-1, comment=str(self.mt5.last_error()))
                    time.sleep(0.5)
                    continue
                last = OrderResult(
                    ok=res.retcode in ok_codes,
                    retcode=res.retcode,
                    ticket=res.order or getattr(res, "deal", 0),
                    price=res.price,
                    volume=res.volume,
                    comment=res.comment,
                )
                if last.ok:
                    if filling is not None:
                        self._filling = filling
                    return last
                if res.retcode == self.c("TRADE_RETCODE_INVALID_FILL"):
                    break  # try the next filling mode
                if res.retcode not in retry_codes:
                    return last
                time.sleep(0.3 * (attempt + 1))
            else:
                return last
        return last

    def open_market(self, direction: int, volume: float, sl: float, tp: float, comment: str) -> OrderResult:
        is_buy = direction == LONG

        def price() -> float:
            t = self.mt5.symbol_info_tick(self.symbol)
            return t.ask if is_buy else t.bid

        request = {
            "action": self.c("TRADE_ACTION_DEAL"),
            "symbol": self.symbol,
            "volume": float(volume),
            "type": self.c("ORDER_TYPE_BUY") if is_buy else self.c("ORDER_TYPE_SELL"),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.sym_cfg.deviation_points,
            "magic": self.magic,
            "comment": comment[:31],
            "type_time": self.c("ORDER_TIME_GTC"),
        }
        return self._send(request, price)

    def modify_sl_tp(self, position: Position, sl: float, tp: float) -> OrderResult:
        request = {
            "action": self.c("TRADE_ACTION_SLTP"),
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.magic,
        }
        res = self._send(request)
        if res.retcode == self.c("TRADE_RETCODE_NO_CHANGES"):
            res.ok = True
        return res

    def close_position(self, position: Position, comment: str = "") -> OrderResult:
        closing_buy = position.direction != LONG

        def price() -> float:
            t = self.mt5.symbol_info_tick(self.symbol)
            return t.ask if closing_buy else t.bid

        request = {
            "action": self.c("TRADE_ACTION_DEAL"),
            "symbol": self.symbol,
            "volume": float(position.volume),
            "type": self.c("ORDER_TYPE_BUY") if closing_buy else self.c("ORDER_TYPE_SELL"),
            "position": position.ticket,
            "deviation": self.sym_cfg.deviation_points,
            "magic": self.magic,
            "comment": (comment or "close")[:31],
            "type_time": self.c("ORDER_TIME_GTC"),
        }
        return self._send(request, price)

    def closed_trades(self, since: pd.Timestamp) -> list[ClosedTrade]:
        # Deal times are server time; pad the window generously for the offset.
        frm = since.to_pydatetime() - timedelta(days=2)
        to = datetime.now() + timedelta(days=2)
        deals = self.mt5.history_deals_get(frm, to)
        if deals is None:
            return []
        by_pos: dict[int, list] = defaultdict(list)
        for d in deals:
            if d.symbol == self.symbol and d.position_id:
                by_pos[d.position_id].append(d)
        entry_in = self.c("DEAL_ENTRY_IN")
        entry_out = {self.c("DEAL_ENTRY_OUT"), self.c("DEAL_ENTRY_INOUT"), self.c("DEAL_ENTRY_OUT_BY")}
        reasons = {self.c("DEAL_REASON_SL"): "sl", self.c("DEAL_REASON_TP"): "tp", self.c("DEAL_REASON_SO"): "stop_out"}
        out = []
        for pid, ds in by_pos.items():
            ins = [d for d in ds if d.entry == entry_in]
            outs = [d for d in ds if d.entry in entry_out]
            if not ins or not outs or ins[0].magic != self.magic:
                continue
            vol_in = sum(d.volume for d in ins)
            vol_out = sum(d.volume for d in outs)
            if vol_out + 1e-9 < vol_in:
                continue  # partially closed, still open
            exit_time = pd.Timestamp(max(d.time for d in outs), unit="s")
            if exit_time < since:
                continue
            profit = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in ds)
            exit_price = sum(d.price * d.volume for d in outs) / vol_out
            entry_price = sum(d.price * d.volume for d in ins) / vol_in
            last_out = max(outs, key=lambda d: d.time_msc)
            out.append(
                ClosedTrade(
                    ticket=pid,
                    symbol=self.symbol,
                    direction=LONG if ins[0].type == self.c("DEAL_TYPE_BUY") else -LONG,
                    volume=vol_in,
                    entry_time=pd.Timestamp(ins[0].time, unit="s"),
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit=profit,
                    reason=reasons.get(last_out.reason, last_out.comment or "closed"),
                )
            )
        return out
