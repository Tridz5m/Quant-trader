"""Operations shared by the command line and the desktop app."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .broker.base import SymbolSpec, default_gold_spec
from .config import BotConfig
from .features import WARMUP_BARS, prepare_bars
from .journal import Journal
from .learner import CHAMPION_FILE, SelfLearner, describe_model
from .model import TrainedModel
from .risk import RiskState

log = logging.getLogger(__name__)

BARS_PER_DAY = 276  # M5 bars in one gold trading day

MORE_HISTORY_HELP = (
    "In MT5 set Tools > Options > Charts > Max bars in chart to Unlimited, open an XAUUSD M5 chart "
    "and hold the Home key until no more history loads, then try again."
)


def mt5_broker(cfg: BotConfig):
    from .broker.mt5_broker import MT5Broker

    return MT5Broker(cfg.mt5, cfg.symbol)


def training_bars_needed(cfg: BotConfig) -> int:
    return cfg.learning.train_bars + WARMUP_BARS + cfg.strategy.horizon_bars + 100


def train_from_mt5(cfg: BotConfig) -> TrainedModel | None:
    """Download history from MT5 and (re)train the model now."""
    broker = mt5_broker(cfg)
    broker.connect()
    journal = Journal(cfg.db_path)
    try:
        spec = broker.symbol_spec()
        bars = prepare_bars(broker.rates(training_bars_needed(cfg)), cfg.symbol.default_spread / spec.point)
        return SelfLearner(cfg, journal).retrain(bars, spec.point, "manual")
    finally:
        journal.close()
        broker.shutdown()


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read OHLC bars from a CSV (MT5 export or `download` output)."""
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip("<>"): c for c in df.columns}
    if "date" in cols and "time" in cols:  # MT5 "Export bars" format
        idx = pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str))
    elif "time" in cols:
        idx = pd.to_datetime(df[cols["time"]])
    elif "date" in cols:
        idx = pd.to_datetime(df[cols["date"]])
    else:
        raise ValueError("CSV needs a 'time' (or 'date') column")
    rename = {cols[k]: k for k in ("open", "high", "low", "close", "tick_volume", "spread") if k in cols}
    if "tickvol" in cols:
        rename[cols["tickvol"]] = "tick_volume"
    out = df.rename(columns=rename)
    out.index = pd.DatetimeIndex(idx, name="time")
    keep = [c for c in ("open", "high", "low", "close", "tick_volume", "spread") if c in out.columns]
    return out[keep]


def load_backtest_bars(
    cfg: BotConfig,
    source: str = "mt5",
    csv_path: str | Path | None = None,
    n_bars: int | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, SymbolSpec]:
    """Bars for a backtest from ``source``: "mt5", "csv" or "synthetic"."""
    spec = default_gold_spec()
    if source == "csv":
        raw = load_csv(csv_path)
    elif source == "synthetic":
        from .synthetic import synthetic_gold_bars

        raw = synthetic_gold_bars(n_bars or 40_000, seed=seed)
    elif source == "mt5":
        broker = mt5_broker(cfg)
        broker.connect()
        try:
            spec = broker.symbol_spec()
            raw = broker.rates(n_bars or 100_000)
        finally:
            broker.shutdown()
    else:
        raise ValueError(f"unknown data source {source!r}")
    if n_bars:
        raw = raw.iloc[-n_bars:]
    return prepare_bars(raw, cfg.symbol.default_spread / spec.point), spec


def backtest(
    cfg: BotConfig,
    bars: pd.DataFrame,
    spec: SymbolSpec,
    retrain_days: float = 5.0,
    equity: float = 10_000.0,
    commission: float = 0.0,
) -> BacktestResult:
    return run_backtest(
        bars,
        cfg,
        spec,
        retrain_every_bars=max(1, int(retrain_days * BARS_PER_DAY)),
        start_equity=equity,
        commission_per_lot=commission,
    )


def save_backtest(res: BacktestResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = res.trades
    if len(trades):
        trades = trades.round({"entry": 3, "exit": 3, "sl_initial": 3, "tp": 3, "profit": 2, "r_multiple": 3, "p": 3})
    trades.to_csv(out_dir / "trades.csv", index=False)
    res.equity.to_csv(out_dir / "equity.csv")
    pd.DataFrame(res.models).to_csv(out_dir / "models.csv", index=False)
    return out_dir


def reset_halt(cfg: BotConfig) -> None:
    """Clear the drawdown kill switch and cooldown; drawdown re-anchors to current equity."""
    journal = Journal(cfg.db_path)
    try:
        state = RiskState.from_dict(journal.get_state("risk_state"))
        state.halted = False
        state.halt_reason = ""
        state.peak_equity = 0.0
        state.cooldown_until = ""
        journal.set_state("risk_state", state.to_dict())
    finally:
        journal.close()


def load_champion(cfg: BotConfig) -> TrainedModel | None:
    path = cfg.model_path / CHAMPION_FILE
    if not path.exists():
        return None
    return TrainedModel.load(path)


def model_status(cfg: BotConfig) -> dict:
    return describe_model(load_champion(cfg))
