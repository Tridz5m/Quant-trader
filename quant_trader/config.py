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
import re
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
    # M5 bars loaded each scan (at least ~70 days so the daily trend is known).
    history_bars: int = 20_000
    stale_data_minutes: int = 15


@dataclass(frozen=True)
class Geometry:
    """Stop, target and maximum holding time of a trade."""

    sl_atr_mult: float
    tp_atr_mult: float
    horizon_bars: int

    @property
    def label(self) -> str:
        return f"SL {self.sl_atr_mult:g}xATR / TP {self.tp_atr_mult:g}xATR / max {self.horizon_bars} bars"


@dataclass
class StrategyConfig:
    timeframe_minutes: int = 5
    atr_period: int = 14
    # Stop, target and maximum holding time. Wide stops pay relatively little
    # spread and noise; on real XAUUSD history 1.5 and 2.5 x ATR stops lost.
    sl_atr_mult: float = 4.0
    tp_atr_mult: float = 6.0
    horizon_bars: int = 144
    # Extra [stop x ATR, target x ATR, max bars] combinations the learner may
    # use instead of the one above, whichever works best on unseen data.
    # [] = always use the above.
    candidate_geometries: list = field(default_factory=list)
    # Round-trip slippage/commission allowance in price units, used in labels
    # and backtests so the model learns from realistic outcomes.
    slippage: float = 0.05
    # Only buy in a daily uptrend and only sell in a downtrend (10- vs 30-day
    # EMA of daily closes). On real XAUUSD history, signals against this
    # trend lost money for every stop size and in every period tested.
    trend_filter: bool = True
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

    @property
    def base_geometry(self) -> Geometry:
        return Geometry(float(self.sl_atr_mult), float(self.tp_atr_mult), int(self.horizon_bars))

    def geometries(self) -> list[Geometry]:
        """The base geometry followed by the configured candidates (no duplicates)."""
        out = [self.base_geometry]
        for c in self.candidate_geometries or []:
            g = Geometry(float(c[0]), float(c[1]), int(c[2]))
            if g not in out:
                out.append(g)
        return out


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_open_positions: int = 1
    max_trades_per_day: int = 8
    max_daily_loss_pct: float = 3.0
    # Hard kill switch: stop trading until `reset-halt` is run.
    max_drawdown_pct: float = 15.0
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
    # ~7 months of M5 bars per training run.
    train_bars: int = 60_000
    min_train_bars: int = 8_000
    # The newest part of the training window is held out. Its first half picks
    # the confidence thresholds and stop geometry; the second half (~7 weeks),
    # never used for any choice, decides whether the model may trade.
    validation_fraction: float = 0.45
    retrain_every_hours: float = 24.0
    # When no tradeable model exists, retry training this often.
    retry_hours: float = 6.0
    # Retrain early when live results deteriorate.
    early_retrain_trades: int = 8
    early_retrain_expectancy_r: float = -0.3
    min_hours_between_retrains: float = 4.0
    # Quality gates, measured on held-out data none of the model's settings
    # were chosen on.
    min_validation_trades: int = 20
    min_validation_expectancy_r: float = 0.05
    min_profit_factor: float = 1.10
    # Expectancy t-statistic (mean / std * sqrt(trades)) on held-out trades.
    # With wide stops a false alarm costs little, while 2.0 kept the bot out
    # of the market almost all the time on real XAUUSD history.
    min_t_stat: float = 1.5
    # A side is only enabled if its model ranks outcomes better than chance.
    min_auc: float = 0.52
    threshold_min: float = 0.30
    threshold_max: float = 0.80
    threshold_step: float = 0.01
    # A challenger replaces the champion only if it is better by at least this
    # many R per trade on data neither model was trained on.
    champion_tolerance_r: float = 0.05
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
class NewsConfig:
    # Skip new trades around high-impact news from the weekly economic calendar.
    enabled: bool = True
    currencies: list = field(default_factory=lambda: ["USD"])
    impacts: list = field(default_factory=lambda: ["High"])
    minutes_before: int = 30
    minutes_after: int = 30
    # Also close open trades this bot holds when such news is about to start.
    close_positions_before: bool = False
    refresh_hours: float = 4.0
    feed_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


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
    news: NewsConfig = field(default_factory=NewsConfig)
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


def _valid_candidates(cands: Any) -> bool:
    if cands is None:
        return True
    if not isinstance(cands, list) or len(cands) > 4:
        return False
    for c in cands:
        if not isinstance(c, (list, tuple)) or len(c) != 3:
            return False
        try:
            sl, tp, bars = float(c[0]), float(c[1]), int(c[2])
        except (TypeError, ValueError):
            return False
        if sl <= 0 or tp <= 0 or bars < 2:
            return False
    return True


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
        (_valid_candidates(s.candidate_geometries),
         "strategy.candidate_geometries must be a list of up to 4 [stop, target, bars] entries with positive values"),
        (s.timeframe_minutes == 5, "strategy.timeframe_minutes must be 5 (M5 scanning)"),
        (0 <= s.session_start_hour < s.session_end_hour <= 24, "invalid session hours"),
        # The training part must stay at least as large as the held-out part.
        (0.05 <= lc.validation_fraction <= 0.45, "learning.validation_fraction must be in [0.05, 0.45]"),
        (lc.threshold_min < lc.threshold_max, "learning.threshold_min must be < threshold_max"),
        (lc.min_train_bars >= 1000, "learning.min_train_bars must be >= 1000"),
        (cfg.schedule.scan_interval_seconds >= 60, "schedule.scan_interval_seconds must be >= 60"),
        (cfg.news.minutes_before >= 0 and cfg.news.minutes_after >= 0, "news minutes must be >= 0"),
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


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    return str(value)


def set_config_value(path: str | Path, section: str, key: str, value: Any, comment: str | None = None) -> None:
    """Set ``section.key`` in a YAML config file, keeping comments and layout.

    ``comment`` replaces the key's end-of-line comment ("" removes it);
    by default the existing comment is kept. The result is validated; on an
    invalid value the file is left unchanged and :class:`ConfigError` is raised.
    """
    path = Path(path)
    # Read raw bytes: read_text() would silently turn Windows line endings into "\n".
    original = path.read_bytes().decode("utf-8") if path.exists() else ""
    lines = original.splitlines(keepends=True)
    new_val = _yaml_scalar(value)
    newline = "\r\n" if "\r\n" in original else "\n"
    key_re = re.compile(rf"^(\s+){re.escape(key)}:(\s*)([^#]*?)(\s+#.*)?$")
    section_at = None
    done = False
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if re.match(rf"^{re.escape(section)}:\s*(#.*)?$", body):
            section_at = i
            continue
        if section_at is not None and body and not body[0].isspace() and not body.startswith("#"):
            break  # left the section
        if section_at is not None:
            m = key_re.match(body)
            if m:
                indent, space, old_val, old_comment = m.groups()
                if comment is None:
                    shown = new_val.ljust(len(old_val)) if old_comment else new_val
                    text = f"{indent}{key}:{space or ' '}{shown}{old_comment or ''}"
                else:
                    text = f"{indent}{key}:{space or ' '}{new_val}"
                    if comment:
                        column = len(body) - len(old_comment.lstrip()) if old_comment else 0
                        text = text.ljust(max(column, len(text) + 2)) + f"# {comment}"
                lines[i] = text + ending
                if not old_val.strip():
                    # A block list ("key:" then "- item" lines) is replaced as a whole.
                    item = re.compile(rf"^{indent}\s+-(\s|$)")
                    while i + 1 < len(lines) and item.match(lines[i + 1]):
                        del lines[i + 1]
                done = True
                break
    if not done:
        tail = f"  # {comment}" if comment else ""
        if section_at is not None:
            lines.insert(section_at + 1, f"  {key}: {new_val}{tail}{newline}")
        else:
            if lines and not lines[-1].endswith("\n"):
                lines.append(newline)
            lines.append(f"{newline}{section}:{newline}  {key}: {new_val}{tail}{newline}")
    path.write_text("".join(lines), encoding="utf-8", newline="")
    try:
        load_config(path)
    except ConfigError:
        path.write_text(original, encoding="utf-8", newline="")
        raise


# Shipped defaults that a later version changed, in groups of settings that
# belong together: (section, {key: (old, new, comment)}). A group is moved to
# the new defaults only if every key of it in config.yaml still holds its old
# default, so anything the user chose is kept.
CHANGED_DEFAULTS = [
    # 1.2: tight stops lost to spread and noise on real XAUUSD history.
    ("strategy", {
        "sl_atr_mult": (1.5, 4.0, "stop loss = 4 x ATR(14) on M5"),
        "tp_atr_mult": (2.25, 6.0, "take profit = 6 x ATR (1.5R)"),
        "horizon_bars": (36, 144, "close after 12 hours if neither level is hit"),
    }),
    ("strategy", {"candidate_geometries": ([[2.5, 3.75, 72], [4.0, 6.0, 144]], [], "")}),
    # 1.2: the daily trend needs 30+ days of M5 history.
    ("schedule", {"history_bars": (6000, 20000, "~70 days, so the daily trend is known")}),
    # 1.2: longer held-out tests (~7 weeks) with a less strict significance gate.
    ("learning", {"train_bars": (40000, 60000, "~7 months of M5 history per training run")}),
    ("learning", {"min_t_stat": (2.0, 1.5, "")}),
]


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def migrate_changed_defaults(path: str | Path) -> list[str]:
    """Move settings still at an old shipped default to the new default.

    Returns a description of each change (empty if nothing changed).
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    changes = []
    for section, group in CHANGED_DEFAULTS:
        values = data.get(section)
        if not isinstance(values, dict):
            continue
        present = [k for k in group if k in values]
        if not present or not all(_same_value(values[k], group[k][0]) for k in present):
            continue
        for key in present:
            old, new, comment = group[key]
            set_config_value(path, section, key, new, comment)
            changes.append(f"{section}.{key} {_yaml_scalar(old)} -> {_yaml_scalar(new)}")
    return changes


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
