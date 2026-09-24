"""Signal model: two gradient-boosted classifiers (long and short).

Each classifier estimates P(target is hit before the stop) for its side.
Training is walk-forward: fit on older data, then pick the confidence
threshold that maximises risk-adjusted expectancy on newer, unseen data
(with a purge gap so labels cannot leak). A model only trades if it passes
out-of-sample quality gates; otherwise the bot stays flat.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .config import LearningConfig, StrategyConfig
from .features import FEATURE_VERSION
from .policy import LONG, SHORT, decide_vec, simulate_policy, trade_stats

log = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    pass


def strategy_signature(scfg: StrategyConfig) -> dict[str, Any]:
    """Parameters the labels depend on. Changing any of them invalidates a model."""
    return {
        "feature_version": FEATURE_VERSION,
        "sl_atr_mult": scfg.sl_atr_mult,
        "tp_atr_mult": scfg.tp_atr_mult,
        "horizon_bars": scfg.horizon_bars,
        "atr_period": scfg.atr_period,
        "timeframe_minutes": scfg.timeframe_minutes,
    }


@dataclass
class TrainedModel:
    version: str
    created_at: str  # UTC ISO timestamp
    trained_until: pd.Timestamp  # last bar (server time) whose label was used
    feature_names: list[str]
    long_model: Any
    short_model: Any
    thr_long: float | None
    thr_short: float | None
    metrics: dict[str, Any]
    signature: dict[str, Any]
    passed: bool
    binner: QuantileBinner | None = None
    suspended: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def tradeable(self) -> bool:
        return self.passed and not self.suspended and (self.thr_long is not None or self.thr_short is not None)

    def thresholds(self, bump: float = 0.0) -> tuple[float, float]:
        tl = self.thr_long + bump if self.thr_long is not None else math.inf
        ts = self.thr_short + bump if self.thr_short is not None else math.inf
        return tl, ts

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.binner.transform(features)
        n = len(X)
        pl = self.long_model.predict_proba(X)[:, 1] if self.long_model is not None else np.zeros(n)
        ps = self.short_model.predict_proba(X)[:, 1] if self.short_model is not None else np.zeros(n)
        return pl, ps

    def is_compatible(self, scfg: StrategyConfig) -> bool:
        return self.signature == strategy_signature(scfg)

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - datetime.fromisoformat(self.created_at)).total_seconds() / 3600.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(self, tmp)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "TrainedModel":
        return joblib.load(path)

    def summary(self) -> str:
        m = self.metrics
        tl = f"{self.thr_long:.2f}" if self.thr_long is not None else "off"
        ts = f"{self.thr_short:.2f}" if self.thr_short is not None else "off"
        state = "TRADEABLE" if self.tradeable else ("SUSPENDED" if self.suspended else "NOT TRADEABLE")
        return (
            f"model {self.version} [{state}] thr long={tl} short={ts} | validation: "
            f"{m.get('trades', 0)} trades, exp={m.get('expectancy_r', 0):+.3f}R, "
            f"PF={m.get('profit_factor', 0):.2f}, win={m.get('win_rate', 0):.1%}, "
            f"AUC long={m.get('auc_long', float('nan')):.3f} short={m.get('auc_short', float('nan')):.3f}"
        )


class QuantileBinner:
    """Quantise each feature into at most 255 quantile levels (NaN preserved).

    HistGradientBoosting bins features like this internally; doing it up front
    without sample weights avoids a very slow weighted-percentile code path in
    recent scikit-learn versions when recency/live-trade weights are used.
    """

    def __init__(self, n_levels: int = 255):
        self.n_levels = n_levels
        self.columns: list[str] = []
        self.edges: list[np.ndarray] = []

    def fit(self, X: pd.DataFrame) -> "QuantileBinner":
        qs = np.linspace(0, 100, self.n_levels)[1:-1]
        arr = X.to_numpy(dtype=float)
        self.columns = list(X.columns)
        self.edges = []
        for j in range(arr.shape[1]):
            col = arr[:, j]
            col = col[np.isfinite(col)]
            self.edges.append(np.unique(np.percentile(col, qs)) if col.size else np.array([]))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = np.array(X.reindex(columns=self.columns), dtype=float)
        out = np.empty_like(arr)
        for j, edges in enumerate(self.edges):
            col = arr[:, j]
            out[:, j] = np.searchsorted(edges, col, side="right")
            out[~np.isfinite(col), j] = np.nan
        return out


def _classifier(lcfg: LearningConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=lcfg.learning_rate,
        max_iter=lcfg.max_iter,
        max_leaf_nodes=lcfg.max_leaf_nodes,
        min_samples_leaf=lcfg.min_samples_leaf,
        l2_regularization=lcfg.l2_regularization,
        early_stopping=False,
        random_state=lcfg.random_state,
    )


def _fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, lcfg: LearningConfig):
    if len(np.unique(y)) < 2:
        return None
    clf = _classifier(lcfg)
    clf.fit(X, y.astype(int), sample_weight=w)
    return clf


def threshold_grid(lcfg: LearningConfig) -> np.ndarray:
    return np.round(np.arange(lcfg.threshold_min, lcfg.threshold_max + 1e-9, lcfg.threshold_step), 4)


def select_threshold(
    prob: np.ndarray,
    r: np.ndarray,
    exits: np.ndarray,
    side: int,
    lcfg: LearningConfig,
) -> tuple[float | None, dict]:
    """Choose the threshold that maximises expectancy * sqrt(trades).

    Scores are smoothed across neighbouring thresholds so we pick a stable
    region rather than a lucky spike. Returns ``None`` if no threshold shows
    a positive edge with enough trades.
    """
    grid = threshold_grid(lcfg)
    min_trades = max(5, lcfg.min_validation_trades // 2)
    zeros = np.zeros_like(prob)
    scores = np.full(len(grid), -np.inf)
    stats_by_thr = []
    for gi, thr in enumerate(grid):
        direction = np.where(prob >= thr, side, 0).astype(np.int8)
        if side == LONG:
            rs, _ = simulate_policy(direction, r, zeros, exits, exits)
        else:
            rs, _ = simulate_policy(direction, zeros, r, exits, exits)
        st = trade_stats(rs)
        stats_by_thr.append(st)
        if st["trades"] >= min_trades:
            scores[gi] = st["expectancy_r"] * math.sqrt(st["trades"])
    finite = np.isfinite(scores)
    if not finite.any():
        return None, {"reason": "not enough trades at any threshold"}
    padded = np.where(finite, scores, np.nan)
    smooth = np.array(pd.Series(padded).rolling(3, center=True, min_periods=1).mean(), dtype=float)
    smooth[~finite] = -np.inf
    best = int(np.argmax(smooth))
    st = stats_by_thr[best]
    if st["expectancy_r"] <= 0 or smooth[best] <= 0:
        return None, {"reason": "no positive edge", **st}
    return float(grid[best]), st


def evaluate_model(model: TrainedModel, features: pd.DataFrame, labels: pd.DataFrame, bump: float = 0.0) -> dict:
    """Out-of-sample trade statistics of ``model`` on the given rows."""
    if len(features) == 0:
        return trade_stats(np.array([]))
    pl, ps = model.predict(features)
    tl, ts = model.thresholds(bump)
    direction = decide_vec(pl, ps, tl, ts)
    rs, _ = simulate_policy(
        direction,
        labels["long_r"].to_numpy(),
        labels["short_r"].to_numpy(),
        labels["long_exit"].to_numpy(),
        labels["short_exit"].to_numpy(),
    )
    return trade_stats(rs)


def recency_weights(n: int, half_life: int) -> np.ndarray:
    if half_life <= 0:
        return np.ones(n)
    age = np.arange(n)[::-1]
    return np.power(0.5, age / half_life)


def train_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    scfg: StrategyConfig,
    lcfg: LearningConfig,
    sample_weight: np.ndarray | None = None,
    version: str | None = None,
) -> TrainedModel:
    """Walk-forward train + validate + gate. ``features``/``labels`` are aligned."""
    mask = labels["long_r"].notna() & labels["short_r"].notna() & features.notna().any(axis=1)
    X = features.loc[mask]
    L = labels.loc[mask]
    w_all = np.ones(len(features)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    w_all = w_all[mask.to_numpy()]
    n = len(X)
    if n < lcfg.min_train_bars:
        raise InsufficientDataError(f"only {n} labelled bars, need {lcfg.min_train_bars}")

    n_val = int(n * lcfg.validation_fraction)
    purge = scfg.horizon_bars
    tr_end = n - n_val - purge
    if tr_end < n_val:
        raise InsufficientDataError("not enough data for a train/validation split")

    w = w_all * recency_weights(n, lcfg.recency_half_life_bars)
    X_tr, X_val = X.iloc[:tr_end], X.iloc[n - n_val :]
    L_tr, L_val = L.iloc[:tr_end], L.iloc[n - n_val :]
    w_tr = w[:tr_end]

    binner = QuantileBinner().fit(X_tr)
    B_tr, B_val = binner.transform(X_tr), binner.transform(X_val)
    long_m = _fit(B_tr, L_tr["long_win"].to_numpy(), w_tr, lcfg)
    short_m = _fit(B_tr, L_tr["short_win"].to_numpy(), w_tr, lcfg)
    n_val_rows = len(X_val)
    pl = long_m.predict_proba(B_val)[:, 1] if long_m is not None else np.zeros(n_val_rows)
    ps = short_m.predict_proba(B_val)[:, 1] if short_m is not None else np.zeros(n_val_rows)

    auc_l = _auc(L_val["long_win"].to_numpy(), pl)
    auc_s = _auc(L_val["short_win"].to_numpy(), ps)
    thr_l, info_l = select_threshold(pl, L_val["long_r"].to_numpy(), L_val["long_exit"].to_numpy(), LONG, lcfg)
    thr_s, info_s = select_threshold(ps, L_val["short_r"].to_numpy(), L_val["short_exit"].to_numpy(), SHORT, lcfg)
    # A side whose model cannot rank outcomes is noise, whatever its best threshold says.
    if long_m is None or not auc_l >= lcfg.min_auc:
        thr_l, info_l = None, {**info_l, "disabled": f"AUC {auc_l:.3f} < {lcfg.min_auc}"}
    if short_m is None or not auc_s >= lcfg.min_auc:
        thr_s, info_s = None, {**info_s, "disabled": f"AUC {auc_s:.3f} < {lcfg.min_auc}"}

    direction = decide_vec(
        pl, ps, thr_l if thr_l is not None else math.inf, thr_s if thr_s is not None else math.inf
    )
    rs, _ = simulate_policy(
        direction,
        L_val["long_r"].to_numpy(),
        L_val["short_r"].to_numpy(),
        L_val["long_exit"].to_numpy(),
        L_val["short_exit"].to_numpy(),
    )
    metrics = trade_stats(rs)
    metrics.update(
        {
            "auc_long": auc_l,
            "auc_short": auc_s,
            "long_side": info_l,
            "short_side": info_s,
            "train_rows": int(tr_end),
            "val_rows": int(n_val_rows),
            "val_start": str(X_val.index[0]),
            "val_end": str(X_val.index[-1]),
            "base_win_long": float(np.nanmean(L_val["long_win"])),
            "base_win_short": float(np.nanmean(L_val["short_win"])),
        }
    )
    notes = []
    passed = True
    if thr_l is None and thr_s is None:
        passed = False
        notes.append("no side shows a positive out-of-sample edge")
    if metrics["trades"] < lcfg.min_validation_trades:
        passed = False
        notes.append(f"only {metrics['trades']} validation trades (< {lcfg.min_validation_trades})")
    if metrics["expectancy_r"] < lcfg.min_validation_expectancy_r:
        passed = False
        notes.append(f"expectancy {metrics['expectancy_r']:+.3f}R below {lcfg.min_validation_expectancy_r:+.3f}R")
    if metrics["profit_factor"] < lcfg.min_profit_factor:
        passed = False
        notes.append(f"profit factor {metrics['profit_factor']:.2f} below {lcfg.min_profit_factor:.2f}")
    if metrics["t_stat"] < lcfg.min_t_stat:
        passed = False
        notes.append(f"t-stat {metrics['t_stat']:.2f} below {lcfg.min_t_stat:.2f} (edge not significant)")

    if lcfg.refit_on_full_data:
        # Use the most recent data too; thresholds come from the honest split above.
        binner_full = QuantileBinner().fit(X)
        B = binner_full.transform(X)
        long_full = _fit(B, L["long_win"].to_numpy(), w, lcfg)
        short_full = _fit(B, L["short_win"].to_numpy(), w, lcfg)
        if long_full is not None and short_full is not None:
            binner, long_m, short_m = binner_full, long_full, short_full

    now = datetime.now(timezone.utc)
    return TrainedModel(
        version=version or now.strftime("%Y%m%d-%H%M%S"),
        created_at=now.isoformat(),
        trained_until=X.index[-1],
        feature_names=list(X.columns),
        long_model=long_m,
        short_model=short_m,
        thr_long=thr_l,
        thr_short=thr_s,
        metrics=metrics,
        signature=strategy_signature(scfg),
        passed=passed,
        binner=binner,
        notes=notes,
    )


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def choose_champion(
    champion: TrainedModel | None,
    challenger: TrainedModel,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    scfg: StrategyConfig,
    lcfg: LearningConfig,
) -> tuple[TrainedModel | None, str]:
    """Champion/challenger selection on data neither model was fitted on.

    The champion is scored only on bars after its own training cut-off, so
    both models are judged out-of-sample. Returns the model to use and why.
    """
    if champion is None or not champion.is_compatible(scfg) or not set(champion.feature_names) <= set(features.columns):
        if challenger.passed:
            return challenger, "promoted: no compatible champion"
        return challenger, "no model passed quality gates; staying flat"

    val_start = pd.Timestamp(challenger.metrics["val_start"])
    oos = (features.index >= val_start) & (features.index > champion.trained_until)
    known = labels["long_r"].notna().to_numpy() & labels["short_r"].notna().to_numpy()
    rows = oos & known
    champ = evaluate_model(champion, features.loc[rows], labels.loc[rows])
    champ_n = champ["trades"]
    enough = champ_n >= max(5, lcfg.min_validation_trades // 2)

    if not challenger.passed:
        if enough and champ["expectancy_r"] < 0:
            champion.suspended = True
            return champion, (
                f"challenger failed gates and champion is losing out-of-sample "
                f"({champ_n} trades, {champ['expectancy_r']:+.3f}R): trading suspended"
            )
        return champion, "challenger failed gates; keeping champion"

    chal_exp = challenger.metrics["expectancy_r"]
    if enough and not champion.suspended and champ["expectancy_r"] > chal_exp + lcfg.champion_tolerance_r:
        return champion, (
            f"champion better out-of-sample ({champ['expectancy_r']:+.3f}R vs {chal_exp:+.3f}R); keeping it"
        )
    return challenger, f"promoted: challenger {chal_exp:+.3f}R vs champion OOS {champ['expectancy_r']:+.3f}R ({champ_n} trades)"
