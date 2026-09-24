"""Headless self-test of the installed or packaged app.

``QuantTrader.exe --selftest [log path]`` runs the complete pipeline on
synthetic data (features, training, save/load, walk-forward backtest, a
simulated bot session, the GUI toolkit) and exits with 0 if everything
works. The Windows build runs it on every freshly built exe.
"""

from __future__ import annotations

import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _fast_config(base_dir: str):
    from .config import BotConfig

    cfg = BotConfig()
    cfg.base_dir = base_dir
    cfg.learning.min_train_bars = 3000
    cfg.learning.train_bars = 9000
    cfg.learning.max_iter = 60
    cfg.learning.min_samples_leaf = 100
    cfg.schedule.history_bars = 6000
    return cfg


def run_selftest(log_path: Path, check_gui: bool = True) -> int:
    lines: list[str] = []

    def out(msg: str) -> None:
        lines.append(msg)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        if sys.stdout is not None:
            print(msg, flush=True)

    state: dict = {}
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def environment():
        import numpy
        import pandas
        import sklearn
        import yaml

        from . import __version__
        from .paths import is_frozen

        return (f"quant_trader {__version__}, Python {platform.python_version()} ({platform.architecture()[0]}), "
                f"{platform.platform()}, frozen={is_frozen()}, numpy {numpy.__version__}, pandas {pandas.__version__}, "
                f"scikit-learn {sklearn.__version__}, PyYAML {yaml.__version__}")

    def metatrader5_package():
        if sys.platform != "win32":
            return "skipped (not Windows)"
        import MetaTrader5 as mt5

        return f"MetaTrader5 {getattr(mt5, '__version__', '?')} importable"

    def example_config():
        from .config import load_config
        from .paths import bundled_example_config

        path = bundled_example_config()
        cfg = load_config(path)
        assert cfg.symbol.name == "XAUUSD" and not cfg.mt5.allow_real_account
        return str(path)

    def data_pipeline():
        from .features import prepare_bars
        from .learner import build_dataset
        from .synthetic import synthetic_gold_bars

        cfg = _fast_config(tmp.name)
        bars = prepare_bars(synthetic_gold_bars(15_000, seed=5, trend_strength=0.45), 25)
        feats, labels = build_dataset(bars, cfg, 0.01)
        assert len(feats) == len(labels) > 10_000
        state.update(cfg=cfg, bars=bars, feats=feats, labels=labels)
        return f"{feats.shape[1]} features x {len(feats)} bars"

    def train_and_reload():
        import numpy as np

        from .model import TrainedModel, train_model

        cfg, feats, labels = state["cfg"], state["feats"], state["labels"]
        model = train_model(feats, labels, cfg.strategy, cfg.learning)
        assert model.tradeable, f"expected a tradeable model on trending data: {model.notes}"
        path = Path(tmp.name) / "model.joblib"
        model.save(path)
        loaded = TrainedModel.load(path)
        a, b = model.predict(feats.iloc[-100:]), loaded.predict(feats.iloc[-100:])
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
        state["model"] = loaded
        return model.summary()

    def walk_forward_backtest():
        from .backtest import run_backtest
        from .broker.base import default_gold_spec

        res = run_backtest(state["bars"], state["cfg"], default_gold_spec(), retrain_every_bars=2000)
        assert res.stats["trades"] > 0
        return f"{res.stats['trades']} trades, return {res.stats['total_return_pct']:+.1f}%"

    def simulated_bot_session():
        from datetime import datetime, timezone

        import numpy as np

        from .bot import TradingBot
        from .broker.sim import SimBroker
        from .journal import Journal
        from .learner import SelfLearner

        cfg, bars = state["cfg"], state["bars"]
        idx = bars.index
        start = int(np.flatnonzero((idx.dayofweek == 1) & (idx.hour == 9) & (idx.minute == 0) & (np.arange(len(idx)) > 7000))[0])
        broker = SimBroker(bars, start_index=start, magic=cfg.symbol.magic)
        journal = Journal(Path(tmp.name) / "journal.sqlite")
        learner = SelfLearner(cfg, journal)
        learner.champion = state["model"]
        learner.last_attempt = datetime.now(timezone.utc)
        bot = TradingBot(cfg, broker, journal=journal, learner=learner, sleep=lambda s: None)
        bot.start()
        actions = []
        for _ in range(250):
            actions.append(bot.run_cycle()["action"])
            broker.advance()
        opened = actions.count("open")
        closed = len(journal.closed_trades())
        journal.close()
        assert opened > 0 and closed > 0, f"opened={opened} closed={closed}"
        return f"{opened} trades opened, {closed} closed and journaled"

    def instance_lock():
        from .instance import InstanceLock

        a = InstanceLock(Path(tmp.name) / "bot.lock")
        b = InstanceLock(Path(tmp.name) / "bot.lock")
        assert a.acquire() and not b.acquire()
        a.release()
        assert b.acquire()
        b.release()
        return "second instance correctly refused"

    def gui_toolkit():
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            if "display" in str(exc).lower():
                tk.Tcl().eval("info patchlevel")
                return f"Tcl ok, no display to open a window ({exc})"
            raise
        version = root.tk.call("info", "patchlevel")
        # Build the real app window in a scratch folder and let it refresh.
        import logging

        from .app import App

        saved = logging.getLogger().handlers[:]
        try:
            home = Path(tmp.name) / "app_home"
            app = App(root, home)
            end = time.time() + 3
            while time.time() < end:
                root.update()
                time.sleep(0.05)
            state = app.state_lbl.cget("text")
            assert state == "STOPPED", state
            assert (home / "config.yaml").exists()
            app._closing = True
        finally:
            root.destroy()
            for h in logging.getLogger().handlers[:]:
                if h not in saved:
                    logging.getLogger().removeHandler(h)
                    h.close()
        return f"Tk {version}; app window built and refreshed"

    steps = [
        ("environment", environment),
        ("MetaTrader5 package", metatrader5_package),
        ("example config", example_config),
        ("features + labels", data_pipeline),
        ("train + save/load model", train_and_reload),
        ("walk-forward backtest", walk_forward_backtest),
        ("simulated bot session", simulated_bot_session),
        ("single-instance lock", instance_lock),
    ]
    if check_gui:
        steps.append(("desktop window", gui_toolkit))

    ok = True
    t_all = time.time()
    for name, fn in steps:
        t0 = time.time()
        try:
            detail = fn() or ""
            out(f"PASS  {name} ({time.time() - t0:.1f}s) {detail}")
        except Exception:
            ok = False
            out(f"FAIL  {name} ({time.time() - t0:.1f}s)\n{traceback.format_exc()}")
    out(f"SELFTEST {'PASSED' if ok else 'FAILED'} in {time.time() - t_all:.1f}s")
    tmp.cleanup()
    return 0 if ok else 1
