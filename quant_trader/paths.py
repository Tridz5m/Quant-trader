"""Where the app keeps its files."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

# config.example.yaml files shipped by earlier versions (sha256, LF line endings).
# A config.yaml identical to one of these was never edited by the user.
SHIPPED_DEFAULT_CONFIGS = {
    "96f556226938c37452302337dae696057b5a2b67cdd2982cb2aa21a4033bfef9",  # 1.0: 0.5% risk per trade
    "9e28b066e16c9b34074a10d2ed594dd133db770f5dc07018353105e5c2c35a84",  # 1.1: 1.5/2.5/4 x ATR stops
}


def is_frozen() -> bool:
    """True when running as the packaged QuantTrader.exe."""
    return bool(getattr(sys, "frozen", False))


def writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".quant_trader_write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def app_home() -> Path:
    """Folder for config.yaml, data, logs and models.

    The packaged exe keeps everything next to itself, or in
    %LOCALAPPDATA%\\QuantTrader when its folder is read-only (e.g. Program
    Files). From source it is the current directory (the Quant-trader folder).
    """
    if not is_frozen():
        return Path.cwd()
    home = Path(sys.executable).resolve().parent
    if writable(home):
        return home
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "QuantTrader"


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


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def upgrade_untouched_config(home: Path) -> bool:
    """Replace a never-edited config.yaml from an older version with the new defaults.

    The previous file is kept as config.old.yaml. Edited files are left alone.
    """
    path = home / "config.yaml"
    example = bundled_example_config()
    if not path.exists() or not example.exists():
        return False
    current = _content_hash(path)
    if current not in SHIPPED_DEFAULT_CONFIGS or current == _content_hash(example):
        return False
    shutil.copyfile(path, home / "config.old.yaml")
    shutil.copyfile(example, path)
    return True


def upgrade_config(home: Path) -> list[str]:
    """Bring config.yaml up to date with the current defaults.

    A never-edited file is replaced by the new example. In an edited file,
    only settings still at an old default are updated (see
    ``config.CHANGED_DEFAULTS``); everything the user chose is kept. The
    previous file is saved as config.old.yaml. Returns what changed.
    """
    from .config import migrate_changed_defaults

    path = home / "config.yaml"
    try:
        if upgrade_untouched_config(home):
            return ["all settings (the file had never been edited)"]
        if not path.exists():
            return []
        backup = path.read_bytes()
    except OSError:
        return []
    try:
        changes = migrate_changed_defaults(path)
    except Exception:
        # A broken or unusual file is left exactly as it was; loading it reports the problem.
        path.write_bytes(backup)
        return []
    if changes:
        (home / "config.old.yaml").write_bytes(backup)
    return changes
