import numpy as np
import pandas as pd

from quant_trader.features import WARMUP_BARS, build_features


def test_features_do_not_look_ahead(noise_bars):
    full = build_features(noise_bars, 0.01)
    k = 7000
    part = build_features(noise_bars.iloc[:k], 0.01)
    pd.testing.assert_frame_equal(full.iloc[:k], part, check_exact=False, rtol=1e-9, atol=1e-9)


def test_live_window_matches_training_features(noise_bars):
    """Features computed on the live window equal those from long history."""
    full = build_features(noise_bars, 0.01)
    window = build_features(noise_bars.iloc[-6000:], 0.01)
    a, b = full.iloc[-1], window.iloc[-1]
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-3, equal_nan=True), (a - b).abs().sort_values().tail()


def test_features_are_finite_after_warmup(noise_bars):
    f = build_features(noise_bars, 0.01).iloc[WARMUP_BARS:]
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
