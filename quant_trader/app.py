"""Quant Trader desktop app.

A window to start and stop the autonomous XAUUSD bot, watch what it is
doing (account, model, risk, positions, trades, live log) and run training
or backtests - no command line needed. Packaged as QuantTrader.exe.

Threading: Tk widgets are only touched on the main thread. Long jobs (the
bot loop, training, backtests) run one at a time on a worker thread that
owns every MetaTrader 5 call and every SQLite connection it uses; results
come back through a queue that the UI polls.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger("quant_trader.app")

APP_NAME = "Quant Trader"
MAX_LOG_LINES = 4000

COLORS = {
    "RUNNING": "#1e8e3e",
    "DRY RUN": "#1a73e8",
    "STOPPED": "#6b7280",
    "STARTING": "#b45309",
    "STOPPING": "#b45309",
    "TRAINING": "#7c3aed",
    "BACKTESTING": "#7c3aed",
    "EXPORTING": "#7c3aed",
}
GREEN, RED, AMBER = "#15803d", "#b91c1c", "#b45309"

GUIDE = """\
1. Install MetaTrader 5 from your broker and log in to a DEMO account.

2. In MT5, click "Algo Trading" in the toolbar so it turns green.

3. In MT5, open Tools > Options > Charts and set "Max bars in chart" to Unlimited, so the bot can load enough gold history to learn from.

4. Here, tick "Dry run" and click Start trading to watch the bot think without placing orders. Untick it to let it trade by itself.

On the first start the bot downloads XAUUSD history and trains its model, which takes a few minutes. If it finds no reliable edge it stays flat and retries every 6 hours. That is deliberate, not an error.

The bot scans on every 5-minute candle close while this window is open. It opens no new trades from 30 minutes before to 30 minutes after high-impact US news (NFP, CPI, FOMC...) from the weekly economic calendar. Every trade has a stop loss and take profit on the broker's server, so open trades stay protected if you close the app.

Risk per trade is set with the "Change risk..." button (default 1% of the account). Other settings such as the daily loss limit and trading hours are in config.yaml (File > Settings). Real-money accounts are refused until you set allow_real_account: true there. Test on demo for several weeks first."""


class QueueLogHandler(logging.Handler):
    """Forwards log records to the UI thread."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait((record.levelno, self.format(record)))
        except Exception:
            pass


class Worker:
    """Runs one long job at a time (bot, training, backtest) off the UI thread."""

    def __init__(self, events: queue.Queue):
        self.events = events
        self.thread: threading.Thread | None = None
        self.job = ""
        self.bot = None
        self.stop_event = threading.Event()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def submit(self, job: str, fn: Callable[[], object]) -> None:
        if self.busy:
            raise RuntimeError(f"already busy: {self.job}")
        self.stop_event.clear()
        self.job = job

        def run() -> None:
            try:
                result = fn()
            except BaseException as exc:  # reported to the user, never silently lost
                log.debug("%s failed", job, exc_info=True)
                self.events.put(("error", job, exc))
            else:
                self.events.put(("done", job, result))
            finally:
                self.bot = None

        self.thread = threading.Thread(target=run, name=f"quant-trader-{job}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        bot = self.bot
        if bot is not None:
            bot.stop()


def friendly_error(exc: BaseException) -> tuple[str, str]:
    """(title, message) for an exception from a job."""
    from .bot import SafetyError
    from .broker.base import BrokerError
    from .config import ConfigError
    from .model import InsufficientDataError
    from .services import MORE_HISTORY_HELP

    if isinstance(exc, ConfigError):
        return "Settings problem", f"{exc}\n\nFix it in File > Settings (config.yaml)."
    if isinstance(exc, SafetyError):
        return "Safety stop", str(exc)
    if isinstance(exc, BrokerError):
        return "MetaTrader 5", str(exc)
    if isinstance(exc, InsufficientDataError):
        return "Not enough history", f"{exc}.\n\n{MORE_HISTORY_HELP}"
    return "Error", f"{type(exc).__name__}: {exc}\n\nDetails are in the log."


def open_path(path: Path) -> None:
    """Open a file or folder with the system's default program."""
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def edit_file(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.Popen(["notepad.exe", str(path)])
    else:
        open_path(path)


def fmt_money(v) -> str:
    return "-" if v is None else f"{v:,.2f}"


def fmt_time(value) -> str:
    """Short 'dd Mon HH:MM' for timestamps/ISO strings; '-' if missing."""
    if value in (None, ""):
        return "-"
    try:
        import pandas as pd

        return pd.Timestamp(value).strftime("%d %b %H:%M")
    except Exception:
        return str(value)[:16]


def model_label(m: dict) -> str:
    try:
        trained = datetime.strptime(str(m["model"]), "%Y%m%d-%H%M%S").strftime("%d %b %H:%M UTC")
    except ValueError:
        trained = str(m["model"])
    return f"{trained} ({m.get('age_hours', 0):.0f}h old)"


def shorten_path(path: Path, limit: int = 70) -> str:
    s = str(path)
    return s if len(s) <= limit else "..." + s[-(limit - 3):]


class App:
    POLL_MS = 200
    REFRESH_MS = 1000
    SLOW_REFRESH_S = 5.0

    def __init__(self, root, home: Path):
        import tkinter as tk
        from tkinter import ttk

        from .paths import ensure_config

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.home = home
        from .paths import upgrade_untouched_config

        first_run = not (home / "config.yaml").exists()
        upgraded = upgrade_untouched_config(home)
        self.config_path = ensure_config(home)
        self.events: queue.Queue = queue.Queue()
        self.log_queue: queue.Queue = queue.Queue()
        self.worker = Worker(self.events)
        self.dry_run = tk.BooleanVar(value=False)
        self.last_snapshot: dict = {}
        self._cfg = None
        self._cfg_mtime = None
        self._cfg_error = ""
        self._model_cache: tuple = (None, {})
        self._journal = None
        self._journal_path = None
        self._slow_at = 0.0
        self._risk_state: dict = {}
        self._trades_key = None
        self._closing = False
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0)

        self.queue_handler = QueueLogHandler(self.log_queue)
        self.queue_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        self._build_ui()
        self._configure_logging()
        log.info("%s ready. Files are kept in %s", APP_NAME, self.home)
        if upgraded:
            log.info("config.yaml updated to the new defaults (1%% risk per trade, smarter validation). "
                     "Your previous file is saved as config.old.yaml.")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(self.POLL_MS, self._poll)
        self.root.after(50, self._refresh)
        if first_run:
            self.root.after(600, self.show_guide)

    # --- config & logging -------------------------------------------------
    def load_cfg(self):
        """Current config (re-read when config.yaml changes); raises ConfigError."""
        from .config import load_config

        mtime = self.config_path.stat().st_mtime if self.config_path.exists() else None
        if self._cfg is None or mtime != self._cfg_mtime:
            self._cfg_mtime = mtime
            try:
                self._cfg = load_config(self.config_path if mtime is not None else None)
                if mtime is None:
                    self._cfg.base_dir = str(self.home)
                self._cfg_error = ""
            except Exception as exc:
                self._cfg = None
                self._cfg_error = str(exc)
                raise
        return self._cfg

    def _configure_logging(self) -> None:
        from .logsetup import setup_logging

        try:
            setup_logging(self.load_cfg(), "bot", extra_handlers=[self.queue_handler])
        except Exception as exc:
            root = logging.getLogger()
            root.setLevel(logging.INFO)
            if self.queue_handler not in root.handlers:
                root.addHandler(self.queue_handler)
            log.error("Settings problem: %s", exc)

    # --- UI construction --------------------------------------------------
    def _px(self, v: float) -> int:
        return int(round(v * self.scale))

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root
        root.title(f"{APP_NAME} - XAUUSD self-learning bot")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(self._px(1100), int(sw * 0.95)), min(self._px(720), int(sh * 0.9))
        root.geometry(f"{w}x{h}")
        root.minsize(min(self._px(860), w), min(self._px(560), h))
        style = ttk.Style(root)
        if sys.platform != "win32" and "clam" in style.theme_names():
            style.theme_use("clam")
        bold = ("Segoe UI", 10, "bold") if sys.platform == "win32" else ("DejaVu Sans", 10, "bold")
        style.configure("Key.TLabel", foreground="#6b7280")
        style.configure("Val.TLabel", font=bold)
        style.configure("Treeview", rowheight=self._px(22))
        style.configure("Accent.TButton", font=bold)

        # Menus hold the less frequent actions.
        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Settings (config.yaml)...", command=self.on_settings)
        file_menu.add_command(label="Open app folder", command=lambda: open_path(self.home))
        file_menu.add_command(label="Open log file", command=self.on_open_log)
        file_menu.add_command(label="Export MT5 history (CSV)...", command=self.on_export_history)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.on_close)
        self.tools_menu = tk.Menu(menubar, tearoff=False)
        self.tools_menu.add_command(label="Train now", command=self.on_train)
        self.tools_menu.add_command(label="Backtest...", command=self.on_backtest)
        self.tools_menu.add_command(label="Risk per trade...", command=self.on_change_risk)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Reset kill switch...", command=self.on_reset_halt)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Getting started", command=self.show_guide)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Tools", menu=self.tools_menu)
        menubar.add_cascade(label="Help", menu=help_menu)
        root.configure(menu=menubar)

        bar = ttk.Frame(root, padding=(10, 8, 10, 4))
        bar.pack(fill="x")
        self.state_lbl = tk.Label(bar, text="STOPPED", fg="white", bg=COLORS["STOPPED"], font=bold,
                                  width=12, padx=6, pady=3)
        self.state_lbl.pack(side="left")
        self.start_btn = ttk.Button(bar, text="Start trading", style="Accent.TButton", command=self.on_start)
        self.start_btn.pack(side="left", padx=(12, 0))
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.on_stop)
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.dry_chk = ttk.Checkbutton(bar, text="Dry run (no orders)", variable=self.dry_run)
        self.dry_chk.pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Settings", command=self.on_settings).pack(side="right")
        self.backtest_btn = ttk.Button(bar, text="Backtest...", command=self.on_backtest)
        self.backtest_btn.pack(side="right", padx=(0, 6))
        self.train_btn = ttk.Button(bar, text="Train now", command=self.on_train)
        self.train_btn.pack(side="right", padx=(0, 6))

        grid = ttk.Frame(root, padding=(10, 4, 10, 4))
        grid.pack(fill="x")
        sections = [
            ("Bot", ["State", "Last scan", "Decision", "Next scan", "News"]),
            ("Account", ["Account", "Balance", "Equity", "Open P/L"]),
            ("Model", ["Trained", "Status", "Stop / target", "Thresholds", "Validation"]),
            ("Risk", ["Per trade", "Trades today", "Daily limit", "Kill switch"]),
        ]
        self.fields: dict[tuple[str, str], object] = {}
        for col, (title, keys) in enumerate(sections):
            lf = ttk.LabelFrame(grid, text=title, padding=(10, 6))
            lf.grid(row=0, column=col, sticky="nsew", padx=4)
            grid.columnconfigure(col, weight=1, uniform="status")
            lf.columnconfigure(1, weight=1)
            key_labels, value_labels = [], []
            for r, key in enumerate(keys):
                k = ttk.Label(lf, text=key, style="Key.TLabel")
                k.grid(row=r, column=0, sticky="nw", pady=1)
                v = ttk.Label(lf, text="-", style="Val.TLabel", wraplength=self._px(150), justify="left")
                v.grid(row=r, column=1, sticky="nw", padx=(8, 0), pady=1)
                key_labels.append(k)
                value_labels.append(v)
                self.fields[(title, key)] = v
            lf.bind("<Configure>", self._rewrap(key_labels, value_labels))
            if title == "Risk":
                self.risk_btn = ttk.Button(lf, text="Change risk...", command=self.on_change_risk)
                self.risk_btn.grid(row=len(keys), column=0, columnspan=2, sticky="w", pady=(6, 0))

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=(6, 4))
        self.notebook = nb
        log_frame = ttk.Frame(nb)
        mono = ("Consolas", 9) if sys.platform == "win32" else ("DejaVu Sans Mono", 9)
        self.log_text = tk.Text(log_frame, wrap="none", height=10, font=mono, state="disabled",
                                background="#0f172a", foreground="#e2e8f0", insertbackground="#e2e8f0")
        ys = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        xs = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_configure("warn", foreground="#fbbf24")
        self.log_text.tag_configure("error", foreground="#f87171")
        self.log_text.tag_configure("trade", foreground="#4ade80")
        nb.add(log_frame, text="Live log")

        self.pos_tree = self._make_tree(nb, "Open positions", [
            ("ticket", "Ticket", 90), ("side", "Side", 60), ("volume", "Lots", 60), ("entry", "Entry", 90),
            ("sl", "Stop loss", 90), ("tp", "Take profit", 90), ("profit", "P/L", 90), ("opened", "Opened (server)", 130),
        ])
        self.trade_tree = self._make_tree(nb, "Trade history", [
            ("closed", "Closed (server)", 130), ("side", "Side", 60), ("volume", "Lots", 60), ("entry", "Entry", 90),
            ("exit", "Exit", 90), ("profit", "Profit", 90), ("r", "R", 70), ("reason", "Exit reason", 100),
        ])

        self.footer = ttk.Label(root, text="", padding=(12, 2, 12, 6), foreground="#6b7280")
        self.footer.pack(fill="x")

    def _rewrap(self, key_labels, value_labels):
        def on_resize(event) -> None:
            width = max(self._px(60), event.width - max(k.winfo_reqwidth() for k in key_labels) - self._px(34))
            for v in value_labels:
                v.configure(wraplength=width)

        return on_resize

    def _make_tree(self, nb, title: str, columns: list[tuple[str, str, int]]):
        ttk = self.ttk
        frame = ttk.Frame(nb)
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=8)
        for cid, heading, width in columns:
            tree.heading(cid, text=heading)
            anchor = "w" if cid in ("side", "reason", "opened", "closed") else "e"
            tree.column(cid, width=self._px(width), anchor=anchor)
        ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)
        tree.tag_configure("win", foreground=GREEN)
        tree.tag_configure("loss", foreground=RED)
        nb.add(frame, text=title)
        return tree

    # --- actions ----------------------------------------------------------
    def _cfg_or_warn(self):
        from tkinter import messagebox

        try:
            return self.load_cfg()
        except Exception as exc:
            messagebox.showerror("Settings problem", f"{exc}\n\nFix it in File > Settings (config.yaml).")
            return None

    def on_start(self) -> None:
        from tkinter import messagebox

        cfg = self._cfg_or_warn()
        if cfg is None or self.worker.busy:
            return
        dry = bool(self.dry_run.get())
        if not dry and not messagebox.askyesno(
            "Start autonomous trading",
            "The bot will now trade XAUUSD on the MetaTrader 5 account that is logged in, "
            "opening and closing orders by itself every 5 minutes.\n\n"
            "Make sure MT5 is open, logged in and Algo Trading is ON.\n"
            "Real-money accounts are refused unless allowed in Settings.\n\nStart now?",
        ):
            return
        self._configure_logging()
        self.notebook.select(0)
        self.worker.submit("bot", lambda: self._job_bot(dry))

    def _job_bot(self, dry_run: bool) -> None:
        from .bot import TradingBot
        from .config import load_config
        from .services import mt5_broker

        cfg = load_config(self.config_path)
        cfg.dry_run = dry_run
        bot = TradingBot(cfg, mt5_broker(cfg), sleep=self.worker.stop_event.wait)
        self.worker.bot = bot
        if self.worker.stop_event.is_set():
            return
        bot.run_forever()

    def on_stop(self) -> None:
        if self.worker.busy and self.worker.job == "bot":
            log.info("Stopping the bot (finishing the current step)...")
            self.worker.stop()

    def on_train(self) -> None:
        cfg = self._cfg_or_warn()
        if cfg is None or self.worker.busy:
            return
        self._configure_logging()
        self.notebook.select(0)
        log.info("Training on MT5 history - this takes a minute or two...")

        def job():
            from .config import load_config
            from .services import train_from_mt5

            return train_from_mt5(load_config(self.config_path))

        self.worker.submit("train", job)

    def on_backtest(self) -> None:
        if self._cfg_or_warn() is None or self.worker.busy:
            return
        BacktestDialog(self)

    def start_backtest(self, source: str, csv_path: str, n_bars: int, retrain_days: float, equity: float, commission: float) -> None:
        self._configure_logging()
        self.notebook.select(0)
        out_dir = self.home / "backtest_results" / datetime.now().strftime("%Y-%m-%d_%H%M%S")

        def job():
            from .config import load_config
            from .services import backtest, load_backtest_bars, save_backtest

            cfg = load_config(self.config_path)
            log.info("Backtest: loading %s data...", source)
            bars, spec, margin_rate = load_backtest_bars(cfg, source, csv_path or None, n_bars or None)
            log.info("Backtesting %d bars %s -> %s (retraining every %s days)...",
                     len(bars), bars.index[0], bars.index[-1], retrain_days)
            res = backtest(cfg, bars, spec, retrain_days, equity, commission, margin_rate)
            save_backtest(res, out_dir, bars if source == "mt5" else None)
            return res, out_dir, source

        self.worker.submit("backtest", job)

    def on_change_risk(self) -> None:
        from tkinter import messagebox, simpledialog

        from .config import ConfigError, set_config_value

        cfg = self._cfg_or_warn()
        if cfg is None:
            return
        value = simpledialog.askfloat(
            "Risk per trade",
            "Percent of your account balance to risk on each trade\n"
            "(what you lose when the stop-loss is hit):",
            initialvalue=cfg.risk.risk_per_trade_pct, minvalue=0.1, maxvalue=5.0, parent=self.root,
        )
        if value is None:
            return
        value = round(value, 2)
        if value > 2 and not messagebox.askyesno(
            "High risk",
            f"{value:g}% per trade is aggressive: ten losing trades in a row would cost about "
            f"{100 * (1 - (1 - value / 100) ** 10):.0f}% of the account.\n\nUse {value:g}% anyway?",
        ):
            return
        try:
            set_config_value(self.config_path, "risk", "risk_per_trade_pct", float(value))
        except (ConfigError, OSError) as exc:
            messagebox.showerror("Risk per trade", str(exc))
            return
        self._slow_at = 0.0
        running = self.worker.busy and self.worker.job == "bot"
        log.info("Risk per trade set to %g%%%s.", value, " - stop and start the bot to apply it" if running else "")

    def on_export_history(self) -> None:
        cfg = self._cfg_or_warn()
        if cfg is None or self.worker.busy:
            return
        out = self.home / "exports" / f"xauusd_m5_{datetime.now():%Y-%m-%d}.csv"
        self.notebook.select(0)
        log.info("Exporting XAUUSD M5 history from MT5...")

        def job():
            from .config import load_config
            from .services import export_history

            return export_history(load_config(self.config_path), out)

        self.worker.submit("export", job)

    def on_reset_halt(self) -> None:
        from tkinter import messagebox

        cfg = self._cfg_or_warn()
        if cfg is None or self.worker.busy:
            return
        if messagebox.askyesno(
            "Reset kill switch",
            "Clear the drawdown kill switch and loss cooldown?\n\nDrawdown will be measured again from the current equity.",
        ):
            from .services import reset_halt

            reset_halt(cfg)
            self._slow_at = 0.0
            log.info("Kill switch and cooldown cleared.")

    def on_settings(self) -> None:
        from .paths import ensure_config

        path = ensure_config(self.home)
        edit_file(path)
        log.info("Opened %s - changes apply the next time you start the bot, train or backtest.", path)

    def on_open_log(self) -> None:
        cfg = self._cfg_or_warn()
        if cfg is not None:
            path = cfg.path(cfg.log_dir) / "bot.log"
            open_path(path if path.exists() else path.parent)

    def show_guide(self) -> None:
        TextDialog(self, f"{APP_NAME} - Getting started", GUIDE)

    def show_about(self) -> None:
        from tkinter import messagebox

        from . import __version__

        messagebox.showinfo(
            f"About {APP_NAME}",
            f"{APP_NAME} {__version__}\nSelf-learning XAUUSD trading bot for MetaTrader 5.\n\n"
            "Trading gold on leverage is risky and can lose money quickly. No profit is guaranteed; "
            "past results do not predict future results. Use a demo account first.",
        )

    def on_close(self) -> None:
        from tkinter import messagebox

        if self._closing:
            return
        if self.worker.busy:
            if self.worker.job == "bot":
                if not messagebox.askyesno("Quit", "Stop the bot and quit?\n\nOpen trades keep their stop loss and take profit on the server."):
                    return
                self.worker.stop()
            elif not messagebox.askyesno("Quit", f"A {self.worker.job} is still running. Quit anyway?"):
                return
        self._closing = True
        self._close_when_idle(time.time() + 30)

    def _close_when_idle(self, deadline: float) -> None:
        if self.worker.busy and self.worker.job == "bot" and time.time() < deadline:
            self.root.after(200, lambda: self._close_when_idle(deadline))
            return
        if self._journal is not None:
            self._journal.close()
        self.root.destroy()

    # --- event & log pump -------------------------------------------------
    def _poll(self) -> None:
        self._drain_logs()
        while True:
            try:
                kind, job, payload = self.events.get_nowait()
            except queue.Empty:
                break
            self._on_job_finished(kind, job, payload)
        self.root.after(self.POLL_MS, self._poll)

    def _drain_logs(self) -> None:
        lines = []
        try:
            for _ in range(500):
                lines.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        if not lines:
            return
        text = self.log_text
        at_bottom = text.yview()[1] > 0.999
        text.configure(state="normal")
        for level, line in lines:
            tag = "error" if level >= logging.ERROR else "warn" if level >= logging.WARNING else ""
            if not tag and ("OPENED" in line or "closed (" in line):
                tag = "trade"
            text.insert("end", line + "\n", tag)
        excess = int(text.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if excess > 0:
            text.delete("1.0", f"{excess + 1}.0")
        text.configure(state="disabled")
        if at_bottom:
            text.see("end")

    def _on_job_finished(self, kind: str, job: str, payload) -> None:
        from tkinter import messagebox

        self._slow_at = 0.0
        if self._closing:
            return
        if kind == "error":
            title, msg = friendly_error(payload)
            log.error("%s: %s", title, str(payload).splitlines()[0] if str(payload) else type(payload).__name__)
            messagebox.showerror(title, msg)
            return
        if job == "train":
            model = payload
            if model is None:
                messagebox.showwarning("Training", "No model could be trained. See the log for details.")
            elif model.tradeable:
                messagebox.showinfo("Training finished", f"The model found an edge and will trade.\n\n{model.summary()}")
            else:
                reasons = "\n".join(f"- {n}" for n in model.notes) or "- see log"
                tried = "\n".join(
                    f"- {c['geometry']}: {c['tune_trades']} trades, {c['tune_expectancy_r']:+.2f}R per trade"
                    for c in model.metrics.get("candidates", [])
                )
                messagebox.showinfo(
                    "Training finished",
                    "No reliable edge in recent data, so the bot will stay flat (no trades) "
                    "and retry every 6 hours. This protects your account; it is not an error.\n\n"
                    f"Why:\n{reasons}" + (f"\n\nStop sizes tried (on the tune data):\n{tried}" if tried else ""),
                )
        elif job == "backtest":
            res, out_dir, source = payload
            BacktestResultWindow(self, res, out_dir, source)
        elif job == "export":
            path, n = payload
            messagebox.showinfo("Export finished", f"Saved {n:,} M5 bars of XAUUSD history to:\n\n{path}")
            open_path(path.parent)

    # --- status refresh ---------------------------------------------------
    def _set(self, section: str, key: str, text: str, color: str | None = None) -> None:
        self.fields[(section, key)].configure(text=text, foreground=color or "")

    def _journal_conn(self, cfg):
        from .journal import Journal

        if self._journal is None or self._journal_path != cfg.db_path:
            if self._journal is not None:
                self._journal.close()
            self._journal = Journal(cfg.db_path)
            self._journal_path = cfg.db_path
        return self._journal

    def _model_info(self, cfg) -> dict:
        from .learner import CHAMPION_FILE
        from .services import model_status

        path = cfg.model_path / CHAMPION_FILE
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime != self._model_cache[0]:
            try:
                self._model_cache = (mtime, model_status(cfg))
            except Exception as exc:
                self._model_cache = (mtime, {"model": None, "error": str(exc)})
        return self._model_cache[1]

    def _refresh(self) -> None:
        try:
            self._refresh_status()
        except Exception:
            log.debug("status refresh failed", exc_info=True)
        self.root.after(self.REFRESH_MS, self._refresh)

    def _state(self) -> str:
        busy, job, bot = self.worker.busy, self.worker.job, self.worker.bot
        if busy and job == "bot":
            if self.worker.stop_event.is_set() or self._closing:
                return "STOPPING"
            if bot is None or not bot.snapshot:
                return "STARTING"
            return "DRY RUN" if bot.cfg.dry_run else "RUNNING"
        if busy:
            return {"train": "TRAINING", "export": "EXPORTING"}.get(job, "BACKTESTING")
        return "STOPPED"

    def _refresh_status(self) -> None:
        bot = self.worker.bot
        if bot is not None and bot.snapshot:
            self.last_snapshot = bot.snapshot
        snap = self.last_snapshot
        try:
            cfg = self.load_cfg()
        except Exception:
            cfg = None

        state = self._state()
        self.state_lbl.configure(text=state, bg=COLORS[state])
        idle = not self.worker.busy and not self._closing
        can_stop = self.worker.busy and self.worker.job == "bot" and state != "STOPPING"
        self.start_btn.configure(state="normal" if idle else "disabled")
        self.stop_btn.configure(state="normal" if can_stop else "disabled")
        for w in (self.train_btn, self.backtest_btn, self.dry_chk):
            w.configure(state="normal" if idle else "disabled")
        self.risk_btn.configure(state="disabled" if self._closing else "normal")
        for label in ("Train now", "Backtest...", "Reset kill switch..."):
            self.tools_menu.entryconfigure(label, state="normal" if idle else "disabled")

        running = state in ("RUNNING", "DRY RUN", "STOPPING") and bool(snap) and bot is not None
        report = snap.get("report", {}) if running else {}
        self._set("Bot", "State", {"RUNNING": "Trading", "DRY RUN": "Dry run (no orders)"}.get(state, state.title()))
        self._set("Bot", "Last scan", f"{fmt_time(report['bar_time'])} bar" if report.get("bar_time") is not None else "-")
        self._set("Bot", "Decision", self._decision_text(report) if report else "-")
        nxt = bot.next_scan_at if running else None
        if nxt:
            secs = max(0, int(nxt - time.time()))
            self._set("Bot", "Next scan", f"in {secs // 60}:{secs % 60:02d}")
        else:
            self._set("Bot", "Next scan", "-")
        self._show_news(snap.get("news") if running else None)

        acc = snap.get("account") if snap else None  # last known values stay visible after stopping
        if acc:
            self._set("Account", "Account", f"{acc['login']} {'REAL' if acc['is_real'] else 'DEMO'}", RED if acc["is_real"] else None)
            self._set("Account", "Balance", f"{fmt_money(acc['balance'])} {acc['currency']}")
            self._set("Account", "Equity", fmt_money(acc["equity"]))
        else:
            self._set("Account", "Account", "not connected")
            self._set("Account", "Balance", "-")
            self._set("Account", "Equity", "-")
        positions = snap.get("positions", []) if running else []
        pnl = sum(p["profit"] for p in positions)
        self._set("Account", "Open P/L", fmt_money(pnl) if running else "-", GREEN if pnl > 0 else RED if pnl < 0 else None)
        self._fill_positions(positions)

        if cfg is not None and time.time() >= self._slow_at:
            self._slow_at = time.time() + self.SLOW_REFRESH_S
            journal = self._journal_conn(cfg)
            self._risk_state = journal.get_state("risk_state", {}) or {}
            self._fill_trades(journal.closed_trades(200))
        model = snap.get("model") if running else (self._model_info(cfg) if cfg is not None else {})
        self._show_model(model or {})
        self._show_risk(cfg, (snap.get("risk") if running else self._risk_state) or {})

        if cfg is None:
            self.footer.configure(text=f"Settings problem: {self._cfg_error}", foreground=RED)
        else:
            accounts = "real-money accounts ALLOWED" if cfg.mt5.allow_real_account else "demo accounts only"
            self.footer.configure(
                text=f"{cfg.symbol.name} M5, scan every {cfg.schedule.scan_interval_seconds // 60} min  |  "
                f"{accounts}  |  files: {shorten_path(self.home)}",
                foreground="#6b7280",
            )

    @staticmethod
    def _decision_text(report: dict) -> str:
        action = report.get("action", "")
        reason = report.get("reason", "")
        probs = ""
        if "p_long" in report:
            probs = f"\nlong {report['p_long']:.2f} / short {report['p_short']:.2f}"
        if action == "open":
            return f"Opened {reason}{probs}"
        if action == "dry_run":
            return f"Would open {reason}{probs}"
        if action == "started":
            return "Connected - waiting for the next candle"
        return f"{reason or action}{probs}"

    def _show_news(self, n: dict | None) -> None:
        if not n:
            self._set("Bot", "News", "-")
        elif not n.get("enabled"):
            self._set("Bot", "News", "filter off")
        elif n.get("blackout"):
            until = fmt_time(n.get("blackout_until"))
            self._set("Bot", "News", f"PAUSED for {n['blackout']} (until {until} UTC)", AMBER)
        elif not n.get("loaded"):
            self._set("Bot", "News", "calendar not available - check internet", RED)
        elif n.get("next"):
            self._set("Bot", "News", f"next: {n['next']}")
        else:
            self._set("Bot", "News", "no high-impact news left this week")

    def _show_model(self, m: dict) -> None:
        if not m or not m.get("model"):
            self._set("Model", "Trained", "not yet")
            self._set("Model", "Status", "trains on first start")
            self._set("Model", "Stop / target", "-")
            self._set("Model", "Thresholds", "-")
            self._set("Model", "Validation", "-")
            return
        self._set("Model", "Trained", model_label(m))
        if m.get("tradeable"):
            self._set("Model", "Status", "TRADEABLE", GREEN)
        elif m.get("suspended"):
            self._set("Model", "Status", "SUSPENDED (losing)", RED)
        else:
            self._set("Model", "Status", "no edge - stays flat", AMBER)
        self._set("Model", "Stop / target", m.get("geometry") or "-")
        tl, ts = m.get("thr_long"), m.get("thr_short")
        self._set("Model", "Thresholds", (f"long {tl:.2f}" if tl is not None else "long off") + " / "
                  + (f"short {ts:.2f}" if ts is not None else "short off"))
        v = m.get("validation") or {}
        if v.get("trades"):
            self._set("Model", "Validation", f"held-out: {v['trades']} trades, {v['expectancy_r']:+.2f}R/trade, "
                      f"PF {v['profit_factor']:.2f}, win {v['win_rate']:.0%}")
        else:
            self._set("Model", "Validation", "-")

    def _show_risk(self, cfg, risk: dict) -> None:
        limit = f"{cfg.risk.max_daily_loss_pct:g}%" if cfg is not None else "-"
        if cfg is not None:
            self._set("Risk", "Per trade", f"{cfg.risk.risk_per_trade_pct:g}% of equity")
            self._set("Risk", "Trades today", f"{risk.get('trades_today', 0)} of max {cfg.risk.max_trades_per_day}")
        if risk.get("daily_limit_hit"):
            self._set("Risk", "Daily limit", f"hit (-{limit}): paused today", RED)
        else:
            self._set("Risk", "Daily limit", f"ok (stops at -{limit})")
        if risk.get("halted"):
            self._set("Risk", "Kill switch", "TRIPPED: trading halted", RED)
        elif risk.get("cooldown_until"):
            self._set("Risk", "Kill switch", f"ok, cooling down until {fmt_time(risk['cooldown_until'])}", AMBER)
        else:
            self._set("Risk", "Kill switch", "ok")

    def _fill_positions(self, positions: list[dict]) -> None:
        tree = self.pos_tree
        tree.delete(*tree.get_children())
        for p in positions:
            tag = "win" if p["profit"] > 0 else "loss" if p["profit"] < 0 else ""
            tree.insert("", "end", tags=(tag,), values=(
                p["ticket"], "BUY" if p["direction"] > 0 else "SELL", f"{p['volume']:.2f}", f"{p['price_open']:.2f}",
                f"{p['sl']:.2f}", f"{p['tp']:.2f}", f"{p['profit']:+,.2f}", fmt_time(p["time"]),
            ))

    def _fill_trades(self, trades: list[dict]) -> None:
        key = (len(trades), trades[-1]["ticket"] if trades else None)
        if key == self._trades_key:
            return
        self._trades_key = key
        tree = self.trade_tree
        tree.delete(*tree.get_children())
        for t in reversed(trades):
            profit = t.get("profit") or 0.0
            r = t.get("r_multiple")
            tag = "win" if profit > 0 else "loss" if profit < 0 else ""
            tree.insert("", "end", tags=(tag,), values=(
                fmt_time(t.get("exit_time")), "BUY" if t["direction"] > 0 else "SELL", f"{t['volume']:.2f}",
                f"{t['entry_price']:.2f}", f"{t['exit_price']:.2f}" if t.get("exit_price") is not None else "-",
                f"{profit:+,.2f}", f"{r:+.2f}" if r is not None else "-", t.get("close_reason") or "",
            ))


class TextDialog:
    """A small window showing wrapped, read-only text."""

    def __init__(self, app: App, title: str, text: str):
        tk, ttk = app.tk, app.ttk
        win = tk.Toplevel(app.root)
        win.title(title)
        win.transient(app.root)
        frm = ttk.Frame(win, padding=14)
        frm.pack(fill="both", expand=True)
        font = ("Segoe UI", 10) if sys.platform == "win32" else ("DejaVu Sans", 10)
        body = tk.Text(frm, wrap="word", width=62, height=20, font=font, relief="flat", padx=6, pady=6,
                       background=app.root.cget("background"))
        body.insert("1.0", text)
        body.configure(state="disabled")
        body.pack(fill="both", expand=True)
        ok = ttk.Button(frm, text="OK", command=win.destroy)
        ok.pack(anchor="e", pady=(10, 0))
        ok.focus_set()
        win.bind("<Return>", lambda e: win.destroy())
        win.bind("<Escape>", lambda e: win.destroy())


class BacktestDialog:
    def __init__(self, app: App):
        tk, ttk = app.tk, app.ttk
        self.app = app
        self.win = win = tk.Toplevel(app.root)
        win.title("Backtest")
        win.transient(app.root)
        win.resizable(False, False)
        frm = ttk.Frame(win, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Walk-forward backtest of the complete self-learning system: the model is retrained\n"
                            "as it goes, using only data that was available at each point in time.",
                  foreground="#6b7280").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self.source = tk.StringVar(value="mt5")
        ttk.Radiobutton(frm, text="MT5 history (your broker's XAUUSD M5 data)", value="mt5",
                        variable=self.source, command=self._toggle).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(frm, text="CSV file", value="csv", variable=self.source,
                        command=self._toggle).grid(row=2, column=0, sticky="w")
        self.csv = tk.StringVar()
        self.csv_entry = ttk.Entry(frm, textvariable=self.csv, width=40)
        self.csv_entry.grid(row=2, column=1, sticky="we", padx=6)
        self.browse = ttk.Button(frm, text="Browse...", command=self._browse)
        self.browse.grid(row=2, column=2)
        ttk.Radiobutton(frm, text="Synthetic demo data (tests the pipeline, not real performance)", value="synthetic",
                        variable=self.source, command=self._toggle).grid(row=3, column=0, columnspan=3, sticky="w")

        self.vars = {}
        for r, (key, label, default) in enumerate([
            ("bars", "Bars of history (M5)", "100000"),
            ("retrain", "Retrain every (days)", "5"),
            ("equity", "Starting equity", "10000"),
            ("commission", "Commission per lot (round turn)", "0"),
        ], start=4):
            pad = (10 if r == 4 else 2, 2)
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=pad)
            var = tk.StringVar(value=default)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=6, pady=pad)
            self.vars[key] = var

        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=3, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btns, text="Run backtest", command=self._run).pack(side="right", padx=(0, 6))
        self._toggle()
        win.grab_set()

    def _toggle(self) -> None:
        state = "normal" if self.source.get() == "csv" else "disabled"
        self.csv_entry.configure(state=state)
        self.browse.configure(state=state)
        if self.source.get() == "synthetic" and self.vars["bars"].get() == "100000":
            self.vars["bars"].set("40000")

    def _browse(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.win, title="Choose a CSV with XAUUSD M5 bars",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv.set(path)

    def _run(self) -> None:
        from tkinter import messagebox

        try:
            n_bars = int(float(self.vars["bars"].get()))
            retrain = float(self.vars["retrain"].get())
            equity = float(self.vars["equity"].get())
            commission = float(self.vars["commission"].get())
            if n_bars < 1000 or retrain <= 0 or equity <= 0 or commission < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Backtest", "Please enter valid positive numbers.", parent=self.win)
            return
        if self.source.get() == "csv" and not Path(self.csv.get()).is_file():
            messagebox.showerror("Backtest", "Choose a CSV file first.", parent=self.win)
            return
        self.win.destroy()
        self.app.start_backtest(self.source.get(), self.csv.get(), n_bars, retrain, equity, commission)


class BacktestResultWindow:
    def __init__(self, app: App, res, out_dir: Path, source: str):
        from .backtest import format_stats

        tk, ttk = app.tk, app.ttk
        win = tk.Toplevel(app.root)
        win.title("Backtest results")
        win.transient(app.root)
        frm = ttk.Frame(win, padding=14)
        frm.pack(fill="both", expand=True)
        if source == "synthetic":
            ttk.Label(frm, text="Synthetic data: this only shows that the pipeline works, not real performance.",
                      foreground=AMBER).pack(anchor="w", pady=(0, 8))
        mono = ("Consolas", 10) if sys.platform == "win32" else ("DejaVu Sans Mono", 10)
        text = tk.Text(frm, width=66, height=24, font=mono)
        summary = format_stats(res.stats)
        if not res.stats.get("trades"):
            summary += ("\n\nNo trades: the learner found no reliable edge in this data, so it\n"
                        "stayed flat. That is the intended behaviour when there is nothing to trade.")
        text.insert("1.0", summary + f"\n\nFiles: {out_dir}")
        text.configure(state="disabled")
        text.pack(fill="both", expand=True)
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right")
        ttk.Button(btns, text="Open results folder", command=lambda: open_path(out_dir)).pack(side="right", padx=(0, 6))


def _enable_high_dpi() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def _close_splash() -> None:
    try:
        import pyi_splash  # type: ignore  # only exists inside the packaged exe

        pyi_splash.close()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()
    argv = sys.argv[1:] if argv is None else argv
    from .paths import app_home

    home = app_home()
    if "--selftest" in argv:
        from .selftest import run_selftest

        _close_splash()
        i = argv.index("--selftest")
        out = Path(argv[i + 1]) if len(argv) > i + 1 else home / "selftest.log"
        return run_selftest(out)

    _enable_high_dpi()
    import tkinter as tk

    root = tk.Tk()
    try:
        _set_icon(root)
        App(root, home)
    except Exception as exc:  # a windowed exe has no console: always explain
        from tkinter import messagebox

        _close_splash()
        root.withdraw()
        messagebox.showerror(APP_NAME, f"{APP_NAME} could not start:\n\n{type(exc).__name__}: {exc}\n\nFolder: {home}")
        root.destroy()
        return 1
    finally:
        _close_splash()
    root.mainloop()
    return 0


def _set_icon(root) -> None:
    from .paths import resource

    try:
        import tkinter as tk

        icon = resource("packaging/icon.png")
        if icon.exists():
            root.iconphoto(True, tk.PhotoImage(file=str(icon)))
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
