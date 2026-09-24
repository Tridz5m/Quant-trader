import pandas as pd
import pytest

from quant_trader import services


def test_load_csv_mt5_export_format(tmp_path):
    p = tmp_path / "XAUUSD_M5.csv"
    p.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2025.03.05\t10:00:00\t2900.10\t2901.50\t2899.80\t2901.00\t350\t0\t20\n"
        "2025.03.05\t10:05:00\t2901.00\t2902.20\t2900.40\t2900.90\t410\t0\t22\n"
    )
    df = services.load_csv(p)
    assert list(df.columns) == ["open", "high", "low", "close", "tick_volume", "spread"]
    assert df.index[1] == pd.Timestamp("2025-03-05 10:05")
    assert df["close"].iloc[0] == 2901.00 and df["tick_volume"].iloc[1] == 410


def test_load_csv_download_format(tmp_path):
    p = tmp_path / "bars.csv"
    p.write_text("time,open,high,low,close,tick_volume,spread\n2025-03-05 10:00:00,1,2,0.5,1.5,10,25\n")
    df = services.load_csv(p)
    assert df.index[0] == pd.Timestamp("2025-03-05 10:00") and df["spread"].iloc[0] == 25


def test_reset_halt_clears_kill_switch(cfg):
    from quant_trader.journal import Journal

    j = Journal(cfg.db_path)
    j.set_state("risk_state", {"halted": True, "halt_reason": "dd", "peak_equity": 12000, "cooldown_until": "x"})
    j.close()
    services.reset_halt(cfg)
    j = Journal(cfg.db_path)
    state = j.get_state("risk_state")
    j.close()
    assert not state["halted"] and state["peak_equity"] == 0 and state["cooldown_until"] == ""


def test_mt5_backtest_data_uses_broker_margin_and_saves_bars(cfg, noise_bars, monkeypatch, tmp_path):
    from quant_trader.broker.sim import SimBroker

    sim = SimBroker(noise_bars, leverage=500)
    monkeypatch.setattr(services, "mt5_broker", lambda c: sim)
    bars, spec, margin_rate = services.load_backtest_bars(cfg, "mt5", n_bars=5000)
    assert len(bars) == 5000
    assert margin_rate == pytest.approx(spec.contract_size / 500)  # 1:500, not the 1:100 default

    from quant_trader.backtest import BacktestResult

    res = BacktestResult(trades=pd.DataFrame(), equity=pd.Series(dtype=float), stats={}, models=[])
    out = services.save_backtest(res, tmp_path / "bt", bars)
    assert (out / "bars.csv").exists() and len(services.load_csv(out / "bars.csv")) == 5000
