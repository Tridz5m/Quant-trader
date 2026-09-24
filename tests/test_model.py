import numpy as np
import pandas as pd

from quant_trader.learner import build_dataset
from quant_trader.model import QuantileBinner, TrainedModel, choose_champion, train_model


def test_learns_a_real_edge_and_round_trips(trend_bars, cfg, tmp_path):
    feats, labels = build_dataset(trend_bars, cfg, 0.01)
    model = train_model(feats, labels, cfg.strategy, cfg.learning)
    assert model.tradeable, model.notes
    assert model.metrics["expectancy_r"] > 0
    assert model.metrics["auc_long"] > 0.55 or model.metrics["auc_short"] > 0.55

    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = TrainedModel.load(path)
    a = model.predict(feats.iloc[-50:])
    b = loaded.predict(feats.iloc[-50:])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_refuses_to_trade_pure_noise(noise_bars, cfg):
    feats, labels = build_dataset(noise_bars, cfg, 0.01)
    model = train_model(feats, labels, cfg.strategy, cfg.learning)
    assert not model.tradeable
    assert model.notes


def test_incompatible_champion_is_replaced(trend_bars, cfg):
    feats, labels = build_dataset(trend_bars, cfg, 0.01)
    champ = train_model(feats, labels, cfg.strategy, cfg.learning)
    champ.signature = {**champ.signature, "sl_atr_mult": 99}
    chal = train_model(feats, labels, cfg.strategy, cfg.learning)
    chosen, why = choose_champion(champ, chal, feats, labels, cfg.strategy, cfg.learning)
    assert chosen is chal and "promoted" in why


def test_losing_champion_is_suspended_when_no_replacement(trend_bars, noise_bars, cfg):
    # Champion fitted on trending data; the new market is pure noise so the
    # challenger fails its gates. If the champion is also losing on the new
    # data, trading must be suspended.
    t_feats, t_labels = build_dataset(trend_bars, cfg, 0.01)
    champ = train_model(t_feats, t_labels, cfg.strategy, cfg.learning)
    champ.trained_until = pd.Timestamp("2000-01-01")
    n_feats, n_labels = build_dataset(noise_bars, cfg, 0.01)
    chal = train_model(n_feats, n_labels, cfg.strategy, cfg.learning)
    assert not chal.passed
    # Make the champion's out-of-sample results unambiguously negative.
    losing = n_labels[champ.geometry].copy()
    losing["long_r"] = -1.0
    losing["short_r"] = -1.0
    n_labels = {**n_labels, champ.geometry: losing}
    chosen, why = choose_champion(champ, chal, n_feats, n_labels, cfg.strategy, cfg.learning)
    assert chosen is champ and chosen.suspended and not chosen.tradeable, why


def test_quantile_binner_levels_and_nans():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.integers(0, 3, 5000).astype(float)})
    X.iloc[:10, 0] = np.nan
    B = QuantileBinner().fit(X).transform(X)
    assert np.isnan(B[:10, 0]).all()
    assert len(np.unique(B[10:, 0])) <= 255
    assert len(np.unique(B[:, 1])) == 3
    # Monotone: larger raw values never get a smaller level.
    order = np.argsort(X["a"].to_numpy()[10:])
    assert np.all(np.diff(B[10:, 0][order]) >= 0)


def test_gates_are_measured_on_held_out_data(trend_bars, cfg):
    """Thresholds/geometry come from the tune segment; the gates from a later test segment."""
    feats, labels = build_dataset(trend_bars, cfg, 0.01)
    model = train_model(feats, labels, cfg.strategy, cfg.learning)
    m = model.metrics
    assert model.geometry in cfg.strategy.geometries()
    assert len(m["candidates"]) == len(cfg.strategy.geometries())
    assert m["test_rows"] > 500 and m["tune_rows"] > 500
    # The test segment starts after the tune segment plus a purge gap.
    longest = max(g.horizon_bars for g in cfg.strategy.geometries())
    assert m["test_rows"] <= int(len(feats) * cfg.learning.validation_fraction) - m["tune_rows"] - longest + 1
    assert pd.Timestamp(m["val_start"]) > feats.index[len(feats) - int(len(feats) * cfg.learning.validation_fraction)]


def test_old_style_validation_was_optimistic_on_noise(cfg):
    """On pure noise, grading on the threshold-picking data looks profitable; the held-out test does not."""
    from quant_trader.features import prepare_bars
    from quant_trader.synthetic import synthetic_gold_bars

    tune, test = [], []
    for seed in (100, 102, 107):
        bars = prepare_bars(synthetic_gold_bars(15_000, seed=seed, trend_strength=0.0), 25)
        feats, labels = build_dataset(bars, cfg, 0.01)
        model = train_model(feats, labels, cfg.strategy, cfg.learning)
        assert not model.tradeable
        tune.append(model.metrics["tune_expectancy_r"])
        test.append(model.metrics["expectancy_r"])
    assert np.mean(tune) > 0.1 > np.mean(test)


def test_only_base_geometry_when_no_candidates(trend_bars, cfg):
    cfg.strategy.candidate_geometries = []
    feats, labels = build_dataset(trend_bars, cfg, 0.01)
    assert list(labels) == [cfg.strategy.base_geometry]
    model = train_model(feats, labels, cfg.strategy, cfg.learning)
    assert model.geometry == cfg.strategy.base_geometry


def test_model_incompatible_when_its_geometry_is_no_longer_allowed(trend_bars, cfg):
    feats, labels = build_dataset(trend_bars, cfg, 0.01)
    model = train_model(feats, labels, cfg.strategy, cfg.learning)
    assert model.is_compatible(cfg.strategy)
    cfg.strategy.candidate_geometries = []
    cfg.strategy.sl_atr_mult = 9.0
    assert not model.is_compatible(cfg.strategy)
