"""Where the app keeps its files."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running as the packaged QuantTrader.exe."""
    return bool(getattr(sys, "frozen", False))


def app_home() -> Path:
    """Folder for config.yaml, data, logs and models.

    The packaged exe keeps everything next to itself; from source it is the
    current directory (the Quant-trader folder).
    """
    return Path(sys.executable).resolve().parent if is_frozen() else Path.cwd()


def resource(relative: str) -> Path:
    """A file shipped with the app (inside the exe, or in the source checkout)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


def bundled_example_config() -> Path:
    return resource("config.example.yaml")


def ensure_config(home: Path) -> Path:
    """Return ``home/config.yaml``, creating it from the example on first run."""
    path = home / "config.yaml"
    if not path.exists():
        example = bundled_example_config()
        if example.exists():
            home.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(example, path)
    return path
