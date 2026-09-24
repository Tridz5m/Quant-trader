import pytest

from quant_trader.config import BotConfig
from quant_trader.features import prepare_bars
from quant_trader.synthetic import synthetic_gold_bars


def fast_config(tmp_path=None) -> BotConfig:
    """Small, fast settings for tests."""
    cfg = BotConfig()
    cfg.learning.min_train_bars = 3000
    cfg.learning.train_bars = 9000
    cfg.learning.max_iter = 60
    cfg.learning.min_samples_leaf = 100
    cfg.news.enabled = False  # tests must not depend on this week's real news
    if tmp_path is not None:
        cfg.base_dir = str(tmp_path)
    return cfg


@pytest.fixture
def cfg(tmp_path):
    return fast_config(tmp_path)


@pytest.fixture(scope="session")
def noise_bars():
    return prepare_bars(synthetic_gold_bars(12_000, seed=11, trend_strength=0.0), 25)


@pytest.fixture(scope="session")
def trend_bars():
    # Strong persistent trends: a learnable edge.
    return prepare_bars(synthetic_gold_bars(15_000, seed=5, trend_strength=0.45), 25)
