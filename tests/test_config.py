from pathlib import Path

import pytest

from quant_trader.config import Geometry, BotConfig, ConfigError, load_config

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


def test_set_config_value_keeps_comments_and_validates(tmp_path):
    from quant_trader.config import set_config_value

    p = tmp_path / "config.yaml"
    p.write_text((ROOT / "config.example.yaml").read_text(encoding="utf-8").replace("\n", "\r\n"), encoding="utf-8", newline="")
    set_config_value(p, "risk", "risk_per_trade_pct", 0.75)
    text = p.read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if "risk_per_trade_pct" in l)
    assert line.startswith("  risk_per_trade_pct: 0.75") and "# % of equity" in line
    assert "\r\n" in p.read_bytes().decode() and load_config(p).risk.risk_per_trade_pct == 0.75
    before = p.read_bytes()
    with pytest.raises(ConfigError):
        set_config_value(p, "risk", "risk_per_trade_pct", 50.0)
    assert p.read_bytes() == before  # invalid value: file untouched
    set_config_value(p, "risk", "cooldown_minutes", 30)  # existing key elsewhere in the section
    set_config_value(p, "mt5", "server", "Demo-Server")
    cfg = load_config(p)
    assert cfg.risk.cooldown_minutes == 30 and cfg.mt5.server == "Demo-Server"


def test_set_config_value_adds_missing_section(tmp_path):
    from quant_trader.config import set_config_value

    p = tmp_path / "config.yaml"
    p.write_text("dry_run: false\n", encoding="utf-8")
    set_config_value(p, "risk", "risk_per_trade_pct", 1.5)
    assert load_config(p).risk.risk_per_trade_pct == 1.5


def test_untouched_old_config_is_upgraded_edited_one_is_not(tmp_path):
    from quant_trader.paths import upgrade_untouched_config

    # The config.example.yaml shipped with version 1.0 (0.5% risk).
    old_text = (ROOT / "tests" / "data" / "config_v1.0.example.yaml").read_text(encoding="utf-8")
    home = tmp_path / "untouched"
    home.mkdir()
    (home / "config.yaml").write_text(old_text.replace("\n", "\r\n"), encoding="utf-8", newline="")  # as on Windows
    assert load_config(home / "config.yaml").risk.risk_per_trade_pct == 0.5
    assert upgrade_untouched_config(home)
    assert load_config(home / "config.yaml").risk.risk_per_trade_pct == 1.0
    assert (home / "config.old.yaml").exists()
    assert not upgrade_untouched_config(home)  # already current

    edited = tmp_path / "edited"
    edited.mkdir()
    (edited / "config.yaml").write_text(old_text.replace("risk_per_trade_pct: 0.5", "risk_per_trade_pct: 0.4"), encoding="utf-8")
    assert not upgrade_untouched_config(edited)
    assert load_config(edited / "config.yaml").risk.risk_per_trade_pct == 0.4


def test_untouched_1_1_config_gets_the_new_strategy_defaults(tmp_path):
    from quant_trader.paths import upgrade_config

    old_text = (ROOT / "tests" / "data" / "config_v1.1.example.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(old_text, encoding="utf-8")
    assert upgrade_config(tmp_path)
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.strategy.geometries() == [Geometry(4.0, 6.0, 144)]
    assert cfg.strategy.trend_filter
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    assert not upgrade_config(tmp_path)


def test_edited_config_keeps_user_choices_but_drops_old_defaults(tmp_path):
    from quant_trader.paths import upgrade_config

    old_text = (ROOT / "tests" / "data" / "config_v1.1.example.yaml").read_text(encoding="utf-8")
    edited = old_text.replace("risk_per_trade_pct: 1.0 ", "risk_per_trade_pct: 0.7 ").replace("\n", "\r\n")
    path = tmp_path / "config.yaml"
    path.write_text(edited, encoding="utf-8", newline="")
    changes = upgrade_config(tmp_path)
    assert any("sl_atr_mult" in c for c in changes) and any("candidate_geometries" in c for c in changes)
    cfg = load_config(path)
    assert cfg.risk.risk_per_trade_pct == 0.7  # the user's own setting stays
    assert cfg.strategy.geometries() == [Geometry(4.0, 6.0, 144)]
    assert cfg.schedule.history_bars == 20000
    assert cfg.learning.train_bars == 60000 and cfg.learning.min_t_stat == 1.5
    raw = path.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")  # Windows line endings kept
    assert b"1.5 x ATR" not in raw  # no stale comments
    assert (tmp_path / "config.old.yaml").read_bytes() == edited.encode("utf-8")
    assert not upgrade_config(tmp_path)  # nothing left to do

    # A stop/target/holding time the user picked is left alone as a whole.
    path.write_text("strategy:\n  sl_atr_mult: 2.0\n  tp_atr_mult: 3.0\n  horizon_bars: 36\n", encoding="utf-8")
    assert upgrade_config(tmp_path) == []
    assert load_config(path).strategy.base_geometry == Geometry(2.0, 3.0, 36)


def test_upgrade_leaves_a_broken_config_alone(tmp_path):
    from quant_trader.paths import upgrade_config

    path = tmp_path / "config.yaml"
    broken = "strategy:\n  sl_atr_mult: 1.5\n  horizon_bars: [36\n"
    path.write_text(broken, encoding="utf-8")
    assert upgrade_config(tmp_path) == []
    assert path.read_text(encoding="utf-8") == broken
    assert not (tmp_path / "config.old.yaml").exists()


def test_candidate_geometry_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("strategy:\n  candidate_geometries:\n    - [2.0, 3.0, 0]\n")
    with pytest.raises(ConfigError, match="candidate_geometries"):
        load_config(p)
    p.write_text("strategy:\n  candidate_geometries:\n    - [4.0, 6.0, 144]\n    - [3, 4.5, 96]\n")
    cfg = load_config(p)
    assert [g.horizon_bars for g in cfg.strategy.geometries()] == [144, 96]  # duplicate of base dropped
