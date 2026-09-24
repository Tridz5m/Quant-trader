"""Walk-forward backtest of the complete self-learning system.

The model is retrained periodically using only data available at that point
in time (with the same champion/challenger logic as live), and trades are
simulated bar by bar with the same filters, risk rules and position
management as the live bot.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .broker.base import SymbolSpec
from .config import BotConfig
from .features import WARMUP_BARS, build_features, compute_atr
from .labeling import make_labels
from .model import InsufficientDataError, TrainedModel, choose_champion, train_model
from .policy import LONG, check_bar_exit, decide, entry_block_reason, manage_position
from .risk import RiskManager, RiskState

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    models: list[dict] = field(default_factory=list)


@dataclass
class Signals:
    """Out-of-sample model output and trade geometry for every bar.

    ``p_long``/``p_short`` are NaN where no model trades or the side is
    blocked by the daily trend filter.
    """

    p_long: np.ndarray
    p_short: np.ndarray
    thr_long: np.ndarray
    thr_short: np.ndarray
    sl_mult: np.ndarray
    tp_mult: np.ndarray
    horizon: np.ndarray
    history: list[dict]


def walk_forward_start(cfg: BotConfig) -> int:
    """First bar that can be traded out-of-sample."""
    longest = max(g.horizon_bars for g in cfg.strategy.geometries())
    return WARMUP_BARS + cfg.learning.min_train_bars + longest


def walk_forward_signals(bars: pd.DataFrame, cfg: BotConfig, point: float, retrain_every_bars: int) -> Signals:
    """Retrain every ``retrain_every_bars`` using only data available at that time."""
    lc, sc = cfg.learning, cfg.strategy
    feats = build_features(bars, point, sc.atr_period, sc.timeframe_minutes)
    geos = sc.geometries()
    labels = {g: make_labels(bars, point, sc, g) for g in geos}
    n = len(bars)
    H = max(g.horizon_bars for g in geos)
    start = walk_forward_start(cfg)
    if start >= n:
        raise InsufficientDataError(f"need more than {start} bars for a walk-forward backtest, got {n}")

    sig = Signals(
        p_long=np.full(n, np.nan),
        p_short=np.full(n, np.nan),
        thr_long=np.full(n, np.inf),
        thr_short=np.full(n, np.inf),
        sl_mult=np.full(n, sc.sl_atr_mult, dtype=float),
        tp_mult=np.full(n, sc.tp_atr_mult, dtype=float),
        horizon=np.full(n, sc.horizon_bars, dtype=int),
        history=[],
    )
    champion: TrainedModel | None = None
    for s in range(start, n, retrain_every_bars):
        hi = s - H  # labels of rows < hi only use bars before s
        lo = max(WARMUP_BARS, hi - lc.train_bars)
        X = feats.iloc[lo:hi]
        L = {g: lab.iloc[lo:hi] for g, lab in labels.items()}
        try:
            challenger = train_model(X, L, sc, lc, version=str(bars.index[s]))
        except InsufficientDataError as exc:
            log.info("Skipping retrain at %s: %s", bars.index[s], exc)
            continue
        champion, why = choose_champion(champion, challenger, X, L, sc, lc)
        e = min(s + retrain_every_bars, n)
        trading = bool(champion is not None and champion.tradeable)
        if trading:
            pl, ps = champion.signals(feats.iloc[s:e])
            sig.p_long[s:e], sig.p_short[s:e] = pl, ps
            sig.thr_long[s:e], sig.thr_short[s:e] = champion.thresholds()
            g = champion.geometry or sc.base_geometry
            sig.sl_mult[s:e], sig.tp_mult[s:e], sig.horizon[s:e] = g.sl_atr_mult, g.tp_atr_mult, g.horizon_bars
        m = challenger.metrics
        sig.history.append(
            {
                "time": bars.index[s],
                "challenger_passed": challenger.passed,
                "challenger_geometry": m.get("geometry"),
                "tune_expectancy_r": m.get("tune_expectancy_r"),
                "val_trades": m["trades"],
                "val_expectancy_r": m["expectancy_r"],
                "decision": why,
                "trading": trading,
                "trading_geometry": champion.geometry.label if trading and champion.geometry else None,
            }
        )
        log.info("[%s] %s | %s", bars.index[s], challenger.summary(), why)
    return sig


def simulate(
    bars: pd.DataFrame,
    p_long: np.ndarray,
    p_short: np.ndarray,
    thr_long: np.ndarray,
    thr_short: np.ndarray,
    cfg: BotConfig,
    spec: SymbolSpec,
    start_equity: float = 10_000.0,
    commission_per_lot: float = 0.0,
    margin_rate: float | None = None,
    sl_mult: np.ndarray | None = None,
    tp_mult: np.ndarray | None = None,
    horizon: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Bar-by-bar trade simulation mirroring the live bot.

    ``margin_rate`` is the margin per lot per 1.0 of price (the broker's
    contract size / leverage); by default 1:100 leverage is assumed.
    ``sl_mult``/``tp_mult``/``horizon`` give each bar's trade geometry
    (default: the configured base geometry).
    """
    sc = cfg.strategy
    if margin_rate is None:
        margin_rate = spec.contract_size / 100.0
    rm = RiskManager(cfg.risk, quiet=True)
    state = RiskState()
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    sp = bars["spread"].to_numpy(float) * spec.point
    atr = compute_atr(bars, sc.atr_period).to_numpy(float)
    times = bars.index
    delta = pd.Timedelta(minutes=sc.timeframe_minutes)
    vpu = spec.value_per_price_unit

    balance = start_equity
    pos: dict | None = None
    trades: list[dict] = []
    recent_r: list[float] = []
    equity_curve = np.empty(len(c))

    def close(i: int, price: float, reason: str, when: pd.Timestamp) -> None:
        nonlocal balance, pos
        pnl = (price - pos["entry"]) * pos["dir"] * pos["volume"] * vpu - commission_per_lot * pos["volume"]
        balance += pnl
        r = pnl / pos["risk_money"] if pos["risk_money"] > 0 else 0.0
        recent_r.append(r)
        rm.register_exit(state, when, recent_r[-max(1, cfg.risk.max_consecutive_losses):])
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": when,
                "direction": pos["dir"],
                "volume": pos["volume"],
                "entry": pos["entry"],
                "exit": price,
                "sl_initial": pos["sl0"],
                "tp": pos["tp"],
                "bars_held": i - pos["i"],
                "profit": pnl,
                "r_multiple": r,
                "reason": reason,
                "p": pos["p"],
                "geometry": pos["geo"],
            }
        )
        pos = None

    for i in range(len(c)):
        now = times[i] + delta
        if pos is not None and i > pos["i"]:
            hit = check_bar_exit(pos["dir"], pos["sl"], pos["tp"], o[i], h[i], l[i], sp[i])
            if hit:
                close(i, hit[0], hit[1], times[i] + delta / 2)

        bid, ask = c[i], c[i] + sp[i]
        floating = 0.0
        if pos is not None:
            px = bid if pos["dir"] == LONG else ask
            floating = (px - pos["entry"]) * pos["dir"] * pos["volume"] * vpu
        rm.update_equity(state, now, balance + floating)

        if pos is not None:
            act = manage_position(
                pos["dir"], pos["entry"], pos["sl"], pos["risk"], bid, ask, atr[i], i - pos["i"], now, sc,
                horizon_bars=pos["horizon"],
            )
            if act is not None:
                if act.kind == "close":
                    close(i, bid if pos["dir"] == LONG else ask, act.reason, now)
                else:
                    pos["sl"] = act.sl

        if pos is None and (math.isfinite(p_long[i]) or math.isfinite(p_short[i])):
            eq = balance
            mult, bump = rm.adaptive(recent_r, state, eq)
            d = decide(p_long[i], p_short[i], thr_long[i] + bump, thr_short[i] + bump)
            if d != 0 and entry_block_reason(now, sp[i], atr[i], h[i] - l[i], sc) is None:
                ok, _ = rm.can_open(state, now, 0)
                if ok:
                    slm = float(sl_mult[i]) if sl_mult is not None else sc.sl_atr_mult
                    tpm = float(tp_mult[i]) if tp_mult is not None else sc.tp_atr_mult
                    hz = int(horizon[i]) if horizon is not None else sc.horizon_bars
                    entry = ask + sc.slippage if d == LONG else bid - sc.slippage
                    sl_dist = slm * atr[i]
                    volume, _ = rm.position_size(eq, sl_dist, spec, mult)
                    volume = rm.cap_by_margin(volume, volume * margin_rate * entry, eq, spec)
                    if volume > 0:
                        sl = entry - d * sl_dist
                        tp = entry + d * tpm * atr[i]
                        pos = {
                            "i": i, "dir": d, "entry": entry, "sl": sl, "sl0": sl, "tp": tp,
                            "volume": volume, "risk": sl_dist, "risk_money": volume * sl_dist * vpu,
                            "entry_time": now, "p": p_long[i] if d == LONG else p_short[i],
                            "horizon": hz, "geo": f"{slm:g}/{tpm:g}/{hz}",
                        }
                        rm.register_entry(state)

        floating = 0.0
        if pos is not None:
            px = bid if pos["dir"] == LONG else ask
            floating = (px - pos["entry"]) * pos["dir"] * pos["volume"] * vpu
        equity_curve[i] = balance + floating

    if pos is not None:
        last = len(c) - 1
        close(last, c[last] if pos["dir"] == LONG else c[last] + sp[last], "end_of_data", times[last] + delta)
        equity_curve[last] = balance

    return pd.DataFrame(trades), pd.Series(equity_curve, index=times + delta, name="equity")


def compute_stats(trades: pd.DataFrame, equity: pd.Series, start_equity: float) -> dict:
    stats: dict = {"start_equity": start_equity, "final_equity": float(equity.iloc[-1]) if len(equity) else start_equity}
    stats["total_return_pct"] = (stats["final_equity"] / start_equity - 1) * 100
    peak = equity.cummax()
    stats["max_drawdown_pct"] = float(((peak - equity) / peak).max() * 100) if len(equity) else 0.0
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    stats["sharpe_daily"] = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else 0.0
    n = len(trades)
    stats["trades"] = n
    if n:
        r = trades["r_multiple"].to_numpy()
        wins = trades.loc[trades["profit"] > 0, "profit"].sum()
        losses = -trades.loc[trades["profit"] < 0, "profit"].sum()
        weeks = max(1e-9, (equity.index[-1] - equity.index[0]).days / 7)
        stats.update(
            {
                "win_rate": float((trades["profit"] > 0).mean()),
                "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
                "expectancy_r": float(r.mean()),
                "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
                "avg_loss_r": float(r[r < 0].mean()) if (r < 0).any() else 0.0,
                "trades_per_week": n / weeks,
                "longs": int((trades["direction"] > 0).sum()),
                "shorts": int((trades["direction"] < 0).sum()),
                "avg_bars_held": float(trades["bars_held"].mean()),
                "exit_reasons": trades["reason"].value_counts().to_dict(),
            }
        )
        if "geometry" in trades:
            stats["geometries"] = trades["geometry"].value_counts().to_dict()
    return stats


def run_backtest(
    bars: pd.DataFrame,
    cfg: BotConfig,
    spec: SymbolSpec,
    retrain_every_bars: int = 1440,
    start_equity: float = 10_000.0,
    commission_per_lot: float = 0.0,
    margin_rate: float | None = None,
) -> BacktestResult:
    sig = walk_forward_signals(bars, cfg, spec.point, retrain_every_bars)
    history = sig.history
    trades, equity = simulate(
        bars, sig.p_long, sig.p_short, sig.thr_long, sig.thr_short, cfg, spec, start_equity, commission_per_lot,
        margin_rate=margin_rate, sl_mult=sig.sl_mult, tp_mult=sig.tp_mult, horizon=sig.horizon,
    )
    # Stats cover the whole out-of-sample period, including stretches where
    # the learner found no edge and stayed flat.
    start_i = walk_forward_start(cfg)
    stats = compute_stats(trades, equity.iloc[start_i:], start_equity)
    stats["test_period"] = f"{bars.index[start_i]} -> {bars.index[-1]}"
    stats["segments_trading"] = f"{sum(m['trading'] for m in history)}/{len(history)}"
    return BacktestResult(trades=trades, equity=equity, stats=stats, models=history)


def format_stats(stats: dict) -> str:
    lines = ["Backtest results (walk-forward, out-of-sample)", "-" * 48]
    order = [
        ("test_period", "Test period", "{}"),
        ("segments_trading", "Segments trading", "{}"),
        ("start_equity", "Start equity", "{:,.2f}"),
        ("final_equity", "Final equity", "{:,.2f}"),
        ("total_return_pct", "Total return %", "{:+.2f}"),
        ("max_drawdown_pct", "Max drawdown %", "{:.2f}"),
        ("sharpe_daily", "Sharpe (daily)", "{:.2f}"),
        ("trades", "Trades", "{}"),
        ("trades_per_week", "Trades / week", "{:.1f}"),
        ("win_rate", "Win rate", "{:.1%}"),
        ("profit_factor", "Profit factor", "{:.2f}"),
        ("expectancy_r", "Expectancy (R)", "{:+.3f}"),
        ("avg_win_r", "Avg win (R)", "{:+.2f}"),
        ("avg_loss_r", "Avg loss (R)", "{:+.2f}"),
        ("longs", "Longs", "{}"),
        ("shorts", "Shorts", "{}"),
        ("avg_bars_held", "Avg bars held", "{:.1f}"),
        ("exit_reasons", "Exit reasons", "{}"),
        ("geometries", "Stop/target/bars", "{}"),
    ]
    for key, label, fmt in order:
        if key in stats:
            lines.append(f"{label:<18}{fmt.format(stats[key])}")
    return "\n".join(lines)
