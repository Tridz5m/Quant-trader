import importlib.util
import subprocess
import sys

import pytest

from quant_trader import _deps


def test_all_dependencies_present_in_test_env():
    assert _deps.missing_packages() == []
    assert _deps.REQUIREMENTS.exists()


def test_missing_package_prints_exact_install_command(monkeypatch, capsys):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "yaml" else real(name, *a))
    with pytest.raises(SystemExit) as exc:
        _deps.check_dependencies()
    assert exc.value.code == 2  # run_bot.bat stops instead of restarting
    err = capsys.readouterr().err
    assert "PyYAML" in err
    assert f'"{sys.executable}" -m pip install -r "{_deps.REQUIREMENTS}"' in err


def test_module_entry_point_runs():
    out = subprocess.run([sys.executable, "-m", "quant_trader", "--help"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "backtest" in out.stdout
