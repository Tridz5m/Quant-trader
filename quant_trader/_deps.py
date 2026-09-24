"""Startup check for the Python version and third-party packages.

Runs before anything heavy is imported so a missing package produces a clear
message with the exact install command, instead of a traceback.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

# import name -> pip package name
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "yaml": "PyYAML",
}

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def install_command() -> str:
    """pip command for the interpreter that is running right now."""
    return f'"{sys.executable}" -m pip install -r "{REQUIREMENTS}"'


def missing_packages() -> list[str]:
    return [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]


def problems() -> list[str]:
    out = []
    if sys.version_info < (3, 10):
        out.append(f"Python 3.10 or newer is required (this is {sys.version.split()[0]}).")
    if sys.platform == "win32" and struct.calcsize("P") * 8 != 64:
        out.append("MetaTrader5 needs 64-bit Python; this Python is 32-bit. Install 64-bit Python from python.org.")
    missing = missing_packages()
    if missing:
        out.append(
            f"Missing Python packages: {', '.join(missing)}.\n"
            f"Install everything the bot needs (run this once):\n\n    {install_command()}\n"
        )
    return out


def check_dependencies() -> None:
    found = problems()
    if found:
        print("Cannot start quant_trader:\n", file=sys.stderr)
        for p in found:
            print(f"- {p}", file=sys.stderr)
        sys.exit(2)
