import numpy as np

from quant_trader.backtest import compute_stats, run_backtest, simulate
from quant_trader.broker.base import default_gold_spec


def test_simulate_accounting_is_consistent(noise_bars, cfg):
    n = len(noise_bars)
    rng = np.random.default_rng(1)
    p_long = rng.random(n)
    p_short = rng.random(n)
    thr = np.full(n, 0.8)
    trades, equity = simulate(noise_bars, p_long, p_short, thr, thr, cfg, default_gold_spec(), 10_000)
    assert len(trades) > 20
    assert np.isclose(equity.iloc[-1], 10_000 + trades["profit"].sum())
    # One position at a time.
    assert (trades["entry_time"].iloc[1:].to_numpy() >= trades["exit_time"].iloc[:-1].to_numpy()).all()
    # Losses are capped near -1R (gaps and breakeven aside).
    assert trades["r_multiple"].min() > -1.6
    assert (trades["bars_held"] <= cfg.strategy.horizon_bars).all()
    stats = compute_stats(trades, equity, 10_000)
    assert stats["trades"] == len(trades)


def test_walk_forward_backtest_runs(trend_bars, cfg):
    res = run_backtest(trend_bars, cfg, default_gold_spec(), retrain_every_bars=2000)
    assert res.models
    assert res.stats["trades"] > 0
    assert np.isfinite(res.equity).all()


def test_simulate_uses_per_bar_geometry_and_broker_margin(noise_bars, cfg):
    n = len(noise_bars)
    p = np.full(n, 0.9)
    thr = np.full(n, 0.5)
    spec = default_gold_spec()
    sl = np.full(n, 4.0)
    tp = np.full(n, 6.0)
    hz = np.full(n, 12)
    trades, _ = simulate(noise_bars, p, np.zeros(n), thr, thr, cfg, spec, 10_000, sl_mult=sl, tp_mult=tp, horizon=hz)
    assert set(trades["geometry"]) == {"4/6/12"}
    assert trades["bars_held"].max() <= 12
    # A broker needing 10x more margin per lot caps the position size.
    cheap, _ = simulate(noise_bars, p, np.zeros(n), thr, thr, cfg, spec, 10_000, margin_rate=0.01)
    dear, _ = simulate(noise_bars, p, np.zeros(n), thr, thr, cfg, spec, 10_000, margin_rate=10.0)
    assert dear["volume"].iloc[0] < cheap["volume"].iloc[0]
