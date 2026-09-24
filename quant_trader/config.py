"""Configuration loading.

All settings live in a YAML file (see ``config.example.yaml``). Every key is
optional and falls back to the defaults below. Unknown keys are rejected so a
typo can never silently disable a safety limit.

Hours are always in *broker server time* (the clock shown in MT5 Market Watch).
Most gold brokers run their server at UTC+2 in winter / UTC+3 in summer so that
midnight lines up with the New York close. Using server time keeps session
filters aligned with the London / New York sessions all year round.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class MT5Config:
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 60_000
    # Refuse to trade a REAL money account unless explicitly enabled.
    allow_real_account: bool = False


@dataclass
class SymbolConfig:
    # The bot trades gold only. Broker-specific names (XAUUSDm, XAUUSD.a,
    # GOLD, ...) are resolved automatically from this base name.
    name: str = "XAUUSD"
    magic: int = 26092401
    deviation_points: int = 30
    comment: str = "QT-XAU"
    # Used only when historical data has no spread column (CSV backtests).
    default_spread: float = 0.25
    point: float = 0.01


@dataclass
class ScheduleConfig:
    scan_interval_seconds: int = 300
    # Wait a few seconds after the M5 candle closes so the bar is final.
    bar_close_delay_seconds: int = 5
    history_bars: int = 6000
    stale_data_minutes: int = 15


@dataclass
class StrategyConfig:
    timeframe_minutes: int = 5
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.25
    horizon_bars: int = 36
    # Round-trip slippage/commission allowance in price units, used in labels
    # and backtests so the model learns from realistic outcomes.
    slippage: float = 0.05
    # Entry window in server hours: start <= hour < end.
    session_start_hour: float = 2.0
    session_end_hour: float = 23.0
    friday_last_entry_hour: float = 21.0
    close_before_weekend: bool = True
    friday_close_hour: float = 23.0
    max_spread: float = 0.80
    max_spread_atr_frac: float = 0.35
    # Skip entries right after an abnormal (news spike) candle.
    max_bar_range_atr: float = 4.0
    min_atr: float = 0.30
    # Position management. Set to null to disable.
    breakeven_at_r: float | None = 1.0
    breakeven_lock_r: float = 0.1
    trailing_start_r: float | None = None
    trailing_atr_mult: float = 1.0


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 1
    max_trades_per_day: int = 8
    max_daily_loss_pct: float = 2.0
    # Hard kill switch: stop trading until `reset-halt` is run.
    max_drawdown_pct: float = 10.0
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 120
    max_margin_usage_pct: float = 30.0
    # Allow the broker's minimum lot when it risks at most this multiple of
    # the configured risk. 1.0 = never exceed the configured risk.
    min_lot_risk_tolerance: float = 1.0
    # Scale risk down / demand higher confidence after a poor run of trades.
    adaptive: bool = True
    adaptive_window: int = 20
    adaptive_min_trades: int = 8


@dataclass
class LearningConfig:
    model_dir: str = "models"
    train_bars: int = 40_000
    min_train_bars: int = 8_000
    validation_fraction: float = 0.25
    retrain_every_hours: float = 24.0
    # When no tradeable model exists, retry training this often.
    retry_hours: float = 6.0
    # Retrain early when live results deteriorate.
    early_retrain_trades: int = 8
    early_retrain_expectancy_r: float = -0.3
    min_hours_between_retrains: float = 4.0
    # Quality gates a model must pass on out-of-sample data before it trades.
    min_validation_trades: int = 30
    min_validation_expectancy_r: float = 0.05
    min_profit_factor: float = 1.10
    # Expectancy t-statistic (mean / std * sqrt(trades)) on validation trades.
    min_t_stat: float = 2.0
    # A side is only enabled if its model ranks outcomes better than chance.
    min_auc: float = 0.52
    threshold_min: float = 0.30
    threshold_max: float = 0.80
    threshold_step: float = 0.01
    # A challenger replaces the champion unless the champion is doing better
    # on data it has never seen by more than this many R per trade.
    champion_tolerance_r: float = 0.02
    recency_half_life_bars: int = 15_000
    live_trade_weight: float = 3.0
    refit_on_full_data: bool = True
    random_state: int = 7
    # Gradient boosting hyper-parameters.
    max_iter: int = 250
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    keep_models: int = 10


@dataclass
class BotConfig:
    dry_run: bool = False
    data_dir: str = "data"
    log_dir: str = "logs"
    db_file: str = "journal.sqlite"
    log_level: str = "INFO"
    mt5: MT5Config = field(default_factory=MT5Config)
    symbol: SymbolConfig = field(default_factory=SymbolConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    # Directory relative paths are resolved against (set by the loader).
    base_dir: str = "."

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else Path(self.base_dir) / p

    @property
    def db_path(self) -> Path:
        return self.path(self.data_dir) / self.db_file

    @property
    def model_path(self) -> Path:
        return self.path(self.learning.model_dir)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mt5"]["password"] = "***" if self.mt5.password else None
        return d


def _from_dict(cls: type, data: Any, path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown config key(s) in '{path}': {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def validate(cfg: BotConfig) -> None:
    s, r, lc = cfg.strategy, cfg.risk, cfg.learning
    checks = [
        (0 < r.risk_per_trade_pct <= 5, "risk.risk_per_trade_pct must be in (0, 5]"),
        (r.max_open_positions >= 1, "risk.max_open_positions must be >= 1"),
        (0 < r.max_daily_loss_pct <= 50, "risk.max_daily_loss_pct must be in (0, 50]"),
        (0 < r.max_drawdown_pct <= 90, "risk.max_drawdown_pct must be in (0, 90]"),
        (r.min_lot_risk_tolerance >= 1.0, "risk.min_lot_risk_tolerance must be >= 1"),
        (s.sl_atr_mult > 0 and s.tp_atr_mult > 0, "strategy SL/TP multipliers must be > 0"),
        (s.horizon_bars >= 2, "strategy.horizon_bars must be >= 2"),
        (s.timeframe_minutes == 5, "strategy.timeframe_minutes must be 5 (M5 scanning)"),
        (0 <= s.session_start_hour < s.session_end_hour <= 24, "invalid session hours"),
        (0.05 <= lc.validation_fraction <= 0.5, "learning.validation_fraction must be in [0.05, 0.5]"),
        (lc.threshold_min < lc.threshold_max, "learning.threshold_min must be < threshold_max"),
        (lc.min_train_bars >= 1000, "learning.min_train_bars must be >= 1000"),
        (cfg.schedule.scan_interval_seconds >= 60, "schedule.scan_interval_seconds must be >= 60"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ConfigError("; ".join(errors))


def _apply_env(cfg: BotConfig) -> None:
    env = os.environ
    if env.get("MT5_LOGIN"):
        cfg.mt5.login = int(env["MT5_LOGIN"])
    if env.get("MT5_PASSWORD"):
        cfg.mt5.password = env["MT5_PASSWORD"]
    if env.get("MT5_SERVER"):
        cfg.mt5.server = env["MT5_SERVER"]
    if env.get("MT5_TERMINAL_PATH"):
        cfg.mt5.terminal_path = env["MT5_TERMINAL_PATH"]


def load_config(path: str | Path | None = None) -> BotConfig:
    """Load config from YAML. A missing file yields the defaults."""
    data: dict[str, Any] = {}
    base_dir = Path.cwd()
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            base_dir = p.resolve().parent
        elif str(path) != "config.yaml":
            raise ConfigError(f"Config file not found: {p}")
    cfg = _from_dict(BotConfig, data, "config")
    cfg.base_dir = str(base_dir)
    _apply_env(cfg)
    validate(cfg)
    return cfg
