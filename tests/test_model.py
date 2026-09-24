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
    n_labels = n_labels.copy()
    n_labels["long_r"] = -1.0
    n_labels["short_r"] = -1.0
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
