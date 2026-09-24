import numpy as np
import pandas as pd

from quant_trader.features import (
    LIVE_HISTORY_BARS,
    TREND_COLUMN,
    TREND_FAST_DAYS,
    TREND_SLOW_DAYS,
    WARMUP_BARS,
    build_features,
    daily_trend,
    prepare_bars,
)
from quant_trader.indicators import ema
from quant_trader.synthetic import synthetic_gold_bars


def test_features_do_not_look_ahead(noise_bars):
    full = build_features(noise_bars, 0.01)
    k = 10_500  # late enough for the daily trend to be known
    part = build_features(noise_bars.iloc[:k], 0.01)
    assert part[TREND_COLUMN].notna().any()
    pd.testing.assert_frame_equal(full.iloc[:k], part, check_exact=False, rtol=1e-9, atol=1e-9)


def test_live_window_matches_training_features():
    """Features on the history the live bot loads equal those from long history."""
    bars = prepare_bars(synthetic_gold_bars(LIVE_HISTORY_BARS + 8000, seed=3), 25)
    full = build_features(bars, 0.01)
    window = build_features(bars.iloc[-LIVE_HISTORY_BARS:], 0.01)
    a, b = full.iloc[-1], window.iloc[-1]
    assert np.isfinite(a[TREND_COLUMN])
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-3, equal_nan=True), (a - b).abs().sort_values().tail()


def test_daily_trend_uses_only_completed_days(noise_bars):
    trend = daily_trend(noise_bars)
    closes = noise_bars["close"].resample("1D").last().dropna()
    sign = np.sign(ema(closes, TREND_FAST_DAYS) - ema(closes, TREND_SLOW_DAYS))
    days = closes.index
    checked = 0
    for prev, day in zip(days[:-1], days[1:]):
        if not np.isfinite(sign[prev]):
            continue
        # Until the day's last bar closes, the trend comes from the days before.
        during = trend[(trend.index >= day) & (trend.index < day + pd.Timedelta(hours=23, minutes=55))]
        assert (during == sign[prev]).all()
        checked += 1
    assert checked >= 10
    assert trend.iloc[: TREND_SLOW_DAYS * 200].isna().all()  # unknown during the first month


def test_features_are_finite_after_warmup(noise_bars):
    # The daily trend has its own, longer warm-up (see test_daily_trend_uses_only_completed_days).
    f = build_features(noise_bars, 0.01).iloc[WARMUP_BARS:].drop(columns=[TREND_COLUMN])
    assert not np.isinf(f.to_numpy()).any()
    assert f.isna().mean().max() < 0.01


def test_features_are_scale_free(noise_bars):
    """Doubling the gold price must not change ATR-normalised features."""
    scaled = noise_bars.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = scaled[col] * 2
    scaled["spread"] = scaled["spread"] * 2
    a = build_features(noise_bars, 0.01).iloc[WARMUP_BARS:].drop(columns=["atr_pct"])
    b = build_features(scaled, 0.01).iloc[WARMUP_BARS:].drop(columns=["atr_pct"])
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-6, equal_nan=True)
