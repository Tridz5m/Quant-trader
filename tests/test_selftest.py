from quant_trader.selftest import run_selftest


def test_selftest_passes(tmp_path):
    log = tmp_path / "selftest.log"
    assert run_selftest(log, check_gui=False) == 0, log.read_text()
    text = log.read_text()
    assert "SELFTEST PASSED" in text and "FAIL" not in text
