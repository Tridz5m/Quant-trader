import queue
import threading
import time
from pathlib import Path

import pytest

from quant_trader.app import Worker, fmt_time, friendly_error, model_label, shorten_path


def _wait_event(q: queue.Queue, timeout: float = 5.0):
    return q.get(timeout=timeout)


def test_worker_reports_results_and_errors():
    events = queue.Queue()
    w = Worker(events)
    w.submit("train", lambda: 42)
    assert _wait_event(events) == ("done", "train", 42)
    w.thread.join(1)

    def boom():
        raise ValueError("bad")

    w.submit("backtest", boom)
    kind, job, exc = _wait_event(events)
    assert (kind, job) == ("error", "backtest") and isinstance(exc, ValueError)


def test_worker_runs_one_job_at_a_time_and_stops_the_bot():
    events = queue.Queue()
    w = Worker(events)
    release = threading.Event()
    stopped = []

    class FakeBot:
        def stop(self):
            stopped.append(True)
            release.set()

    def job():
        w.bot = FakeBot()
        release.wait(5)

    w.submit("bot", job)
    for _ in range(100):
        if w.bot is not None:
            break
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        w.submit("train", lambda: None)
    w.stop()
    assert w.stop_event.is_set() and stopped == [True]
    assert _wait_event(events)[0] == "done"
    w.thread.join(1)
    assert not w.busy and w.bot is None


def test_friendly_errors():
    from quant_trader.bot import SafetyError
    from quant_trader.broker.base import BrokerError
    from quant_trader.config import ConfigError
    from quant_trader.model import InsufficientDataError

    assert friendly_error(SafetyError("REAL account"))[0] == "Safety stop"
    assert friendly_error(BrokerError("no terminal"))[0] == "MetaTrader 5"
    assert "Settings" in friendly_error(ConfigError("bad key"))[1]
    assert "Max bars in chart" in friendly_error(InsufficientDataError("need more"))[1]
    assert "log" in friendly_error(KeyError("x"))[1]


def test_formatting_helpers():
    assert fmt_time("2025-03-12T10:32:30") == "12 Mar 10:32"
    assert fmt_time(None) == "-"
    assert model_label({"model": "20260924-143850", "age_hours": 5.2}) == "24 Sep 14:38 UTC (5h old)"
    assert model_label({"model": "custom", "age_hours": 1}) == "custom (1h old)"
    long = Path("/a" * 60)
    assert len(shorten_path(long, 40)) == 40 and shorten_path(Path("/x/y")) == str(Path("/x/y"))


def test_desktop_app_runs_the_bot(tmp_path, monkeypatch, trend_bars):
    """Drive the real window: start a dry run against a simulated broker, then stop."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")

    import numpy as np
    import yaml
    from tkinter import messagebox

    from quant_trader import bot as botmod
    from quant_trader import services
    from quant_trader.app import App
    from quant_trader.broker.sim import SimBroker
    from quant_trader.paths import ensure_config

    cfg_path = ensure_config(tmp_path)
    data = yaml.safe_load(cfg_path.read_text())
    data["learning"].update(min_train_bars=3000, train_bars=9000)
    cfg_path.write_text(yaml.safe_dump(data))

    idx = trend_bars.index
    start = int(np.flatnonzero((idx.dayofweek == 2) & (idx.hour == 9) & (idx.minute == 0) & (np.arange(len(idx)) > 12000))[0])
    sim = SimBroker(trend_bars, start_index=start, magic=26092401)
    monkeypatch.setattr(services, "mt5_broker", lambda cfg: sim)
    original = botmod.TradingBot.run_cycle

    def fast_cycle(self):
        self.broker.advance()
        return original(self)

    monkeypatch.setattr(botmod.TradingBot, "run_cycle", fast_cycle)
    monkeypatch.setattr(botmod.TradingBot, "seconds_to_next_scan", lambda self: 0.05)

    import logging

    saved = logging.getLogger().handlers[:], logging.getLogger().level
    shown = []  # a modal dialog would block the test forever, so record them instead
    for name in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(messagebox, name, lambda *a, _n=name, **k: shown.append((_n, a)))
    app = App(root, tmp_path)

    def pump(until, timeout=120.0):
        end = time.time() + timeout
        while time.time() < end:
            root.update()
            if until():
                return True
            time.sleep(0.02)
        return False

    try:
        app.dry_run.set(True)
        app.on_start()
        assert pump(lambda: app.state_lbl.cget("text") == "DRY RUN"), (app.state_lbl.cget("text"), shown)
        assert pump(lambda: app.fields[("Account", "Account")].cget("text") == "1 DEMO")
        assert pump(lambda: app.fields[("Bot", "Last scan")].cget("text") != "-")
        assert str(app.stop_btn.cget("state")) == "normal" and str(app.start_btn.cget("state")) == "disabled"
        app.on_stop()
        assert pump(lambda: not app.worker.busy and app.state_lbl.cget("text") == "STOPPED")
        assert sim.sent == []  # dry run: nothing was sent to the broker
        assert shown == []
        assert (tmp_path / "logs" / "bot.log").exists()
    finally:
        app._closing = True
        root.destroy()
        rootlog = logging.getLogger()
        for h in rootlog.handlers[:]:
            if h not in saved[0]:
                rootlog.removeHandler(h)
                h.close()
        for h in saved[0]:
            if h not in rootlog.handlers:
                rootlog.addHandler(h)
        rootlog.setLevel(saved[1])
