"""Self-learning: scheduled retraining, champion/challenger and live feedback.

What makes the bot "learn" on its own:

1. It retrains on a rolling window of recent market data on a schedule, so
   it keeps adapting to the current gold regime.
2. Thresholds and quality gates are measured on the trades it would really
   take (only in the direction of the daily trend) and, if several
   stop/target/holding-time geometries are configured, it keeps the one that
   works best on recent unseen data.
3. Every trade it actually took is fed back into training: the simulated
   label for that bar is replaced by the real outcome (real fill, spread,
   slippage and management) and weighted more heavily.
4. A new model (challenger) only replaces the running one (champion) if it
   is better on data neither of them was fitted on.
5. If the champion starts losing out-of-sample and no better model can be
   found, trading is suspended until one is found.
6. A losing streak in live trading triggers an early retrain.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BotConfig, Geometry
from .features import WARMUP_BARS, build_features
from .journal import Journal
from .labeling import make_labels
from .model import InsufficientDataError, TrainedModel, choose_champion, train_model

log = logging.getLogger(__name__)

CHAMPION_FILE = "champion.joblib"
# Retry interval when there is no usable model at all (e.g. history still loading).
MIN_RETRY_HOURS = 1.0


def build_dataset(bars: pd.DataFrame, cfg: BotConfig, point: float) -> tuple[pd.DataFrame, dict[Geometry, pd.DataFrame]]:
    """Features, and labels for every candidate geometry, after the indicator warm-up."""
    feats = build_features(bars, point, cfg.strategy.atr_period, cfg.strategy.timeframe_minutes)
    labels = {g: make_labels(bars, point, cfg.strategy, g).iloc[WARMUP_BARS:] for g in cfg.strategy.geometries()}
    return feats.iloc[WARMUP_BARS:], labels


# Trades journaled before version 1.1 carry no geometry; they all used this one.
LEGACY_GEOMETRY = Geometry(1.5, 2.25, 36)


def trade_geometry(trade: dict) -> Geometry:
    """Geometry a journaled trade was opened with."""
    if trade.get("sl_atr_mult") and trade.get("tp_atr_mult") and trade.get("horizon_bars"):
        return Geometry(float(trade["sl_atr_mult"]), float(trade["tp_atr_mult"]), int(trade["horizon_bars"]))
    return LEGACY_GEOMETRY


def apply_live_feedback(
    labels: dict[Geometry, pd.DataFrame], trades: list[dict], weight: float
) -> tuple[dict[Geometry, pd.DataFrame], np.ndarray]:
    """Overwrite simulated labels with the outcome of trades really taken.

    A real outcome only replaces the label of the geometry it was traded
    with; the bar gets a higher training weight either way.
    """
    labels = {g: L.copy() for g, L in labels.items()}
    index = next(iter(labels.values())).index
    weights = np.ones(len(index))
    pos = {ts: i for i, ts in enumerate(index)}
    applied = 0
    for t in trades:
        if t.get("r_multiple") is None or not t.get("bar_time"):
            continue
        i = pos.get(pd.Timestamp(t["bar_time"]))
        if i is None:
            continue
        L = labels.get(trade_geometry(t))
        if L is not None:
            side = "long" if t["direction"] > 0 else "short"
            r = float(t["r_multiple"])
            L.iloc[i, L.columns.get_loc(f"{side}_win")] = 1.0 if r > 0 else 0.0
            L.iloc[i, L.columns.get_loc(f"{side}_r")] = r
        weights[i] = weight
        applied += 1
    if applied:
        log.info("Learning from %d real trade outcome(s)", applied)
    return labels, weights


class SelfLearner:
    def __init__(self, cfg: BotConfig, journal: Journal):
        self.cfg = cfg
        self.journal = journal
        self.model_dir = cfg.model_path
        self.champion: TrainedModel | None = None
        self.last_attempt: datetime | None = None
        self._load()

    @property
    def champion_path(self) -> Path:
        return self.model_dir / CHAMPION_FILE

    def _load(self) -> None:
        if self.champion_path.exists():
            try:
                self.champion = TrainedModel.load(self.champion_path)
                log.info("Loaded %s", self.champion.summary())
            except Exception as exc:
                log.error("Could not load model %s (%s); will retrain", self.champion_path, exc)
                self.champion = None
        last = self.journal.get_state("last_train_attempt")
        if last:
            self.last_attempt = datetime.fromisoformat(last)

    def retrain_reason(self, now: datetime | None = None) -> str | None:
        """Why a retrain is due right now, or None."""
        now = now or datetime.now(timezone.utc)
        lc = self.cfg.learning
        since_attempt = (now - self.last_attempt).total_seconds() / 3600 if self.last_attempt else float("inf")
        m = self.champion
        if m is None:
            return "no model yet" if since_attempt >= MIN_RETRY_HOURS else None
        if not m.is_compatible(self.cfg.strategy):
            return "strategy settings changed" if since_attempt >= MIN_RETRY_HOURS else None
        if not m.tradeable:
            return "no tradeable model; retrying" if since_attempt >= lc.retry_hours else None
        if since_attempt >= lc.retrain_every_hours:
            return f"scheduled (model {m.age_hours(now):.1f}h old)"
        # Only judge the current model on the trades it took itself.
        recent = [
            t["r_multiple"]
            for t in self.journal.closed_trades(200)
            if t["model_version"] == m.version and t["r_multiple"] is not None
        ][-lc.early_retrain_trades :]
        if (
            len(recent) >= lc.early_retrain_trades
            and float(np.mean(recent)) <= lc.early_retrain_expectancy_r
            and since_attempt >= lc.min_hours_between_retrains
        ):
            return f"live results deteriorated ({np.mean(recent):+.2f}R over {len(recent)} trades)"
        return None

    def retrain(self, bars: pd.DataFrame, point: float, reason: str = "manual") -> TrainedModel | None:
        lc = self.cfg.learning
        now = datetime.now(timezone.utc)
        self.last_attempt = now
        self.journal.set_state("last_train_attempt", now.isoformat())
        log.info("Retraining (%s) on %d bars from %s to %s", reason, len(bars), bars.index[0], bars.index[-1])

        feats, labels = build_dataset(bars, self.cfg, point)
        if len(feats) > lc.train_bars:
            feats = feats.iloc[-lc.train_bars :]
            labels = {g: L.iloc[-lc.train_bars :] for g, L in labels.items()}
        labels, weights = apply_live_feedback(labels, self.journal.closed_trades(), lc.live_trade_weight)
        try:
            challenger = train_model(feats, labels, self.cfg.strategy, lc, sample_weight=weights)
        except InsufficientDataError as exc:
            log.warning("Training skipped: %s", exc)
            return self.champion
        log.info("Challenger: %s", challenger.summary())
        for note in challenger.notes:
            log.info("  gate: %s", note)

        chosen, why = choose_champion(self.champion, challenger, feats, labels, self.cfg.strategy, lc)
        promoted = chosen is challenger
        log.info("Model selection: %s", why)
        self.journal.log_model(
            challenger.version,
            challenger.created_at,
            str(challenger.trained_until),
            challenger.passed,
            promoted,
            why,
            challenger.metrics,
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        challenger.save(self.model_dir / f"model_{challenger.version}.joblib")
        self._prune()
        self.champion = chosen
        if chosen is not None:
            chosen.save(self.champion_path)
        return chosen

    def _prune(self) -> None:
        files = sorted(self.model_dir.glob("model_*.joblib"))
        for f in files[: max(0, len(files) - self.cfg.learning.keep_models)]:
            f.unlink(missing_ok=True)

    def status(self) -> dict:
        return describe_model(self.champion)


def describe_model(m) -> dict:
    """Plain-dict summary of a model for status displays."""
    if m is None:
        return {"model": None}
    return {
        "model": m.version,
        "tradeable": m.tradeable,
        "suspended": m.suspended,
        "thr_long": m.thr_long,
        "thr_short": m.thr_short,
        "geometry": m.geometry.label if getattr(m, "geometry", None) is not None else None,
        "age_hours": round(m.age_hours(), 1),
        "validation": {k: m.metrics.get(k) for k in ("trades", "expectancy_r", "profit_factor", "win_rate")},
        "notes": list(getattr(m, "notes", []) or []),
    }
