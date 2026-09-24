"""Logging to console (when there is one), a rotating file and optional extra handlers."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import BotConfig


def setup_logging(cfg: BotConfig, name: str = "bot", extra_handlers: list[logging.Handler] | None = None) -> None:
    extra = list(extra_handlers or [])
    log_dir = cfg.path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
        if h not in extra:
            h.close()  # release the log file (Windows cannot rotate an open file)
    if sys.stdout is not None:  # None in the windowed exe
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)
    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    for h in extra:
        root.addHandler(h)
