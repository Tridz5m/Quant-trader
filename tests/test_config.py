from pathlib import Path

import pytest

from quant_trader.config import BotConfig, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads_and_matches_defaults():
    cfg = load_config(ROOT / "config.example.yaml")
    defaults = BotConfig()
    assert cfg.symbol.name == "XAUUSD"
    assert cfg.risk.risk_per_trade_pct == defaults.risk.risk_per_trade_pct
    assert cfg.strategy.sl_atr_mult == defaults.strategy.sl_atr_mult
    assert cfg.mt5.allow_real_account is False


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk:\n  risk_per_trade_pcnt: 1.0\n")
    with pytest.raises(ConfigError, match="risk_per_trade_pcnt"):
        load_config(p)


def test_invalid_values_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk:\n  risk_per_trade_pct: 25\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_env_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "12345")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    cfg = load_config(None)
    assert cfg.mt5.login == 12345 and cfg.mt5.password == "secret"
    assert cfg.to_dict()["mt5"]["password"] == "***"
