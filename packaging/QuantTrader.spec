# PyInstaller build recipe for QuantTrader.exe (one file, windowed, with splash).
#   pyinstaller --noconfirm --clean packaging/QuantTrader.spec
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
PKG = os.path.join(ROOT, "packaging")

hiddenimports = collect_submodules("quant_trader") + collect_submodules("sklearn.ensemble._hist_gradient_boosting")
if sys.platform == "win32":
    hiddenimports.append("MetaTrader5")

a = Analysis(
    [os.path.join(PKG, "quant_trader_app.py")],
    pathex=[ROOT],
    datas=[
        (os.path.join(ROOT, "config.example.yaml"), "."),
        (os.path.join(PKG, "icon.png"), "packaging"),
    ],
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "IPython", "notebook", "pytest", "PIL", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
splash = Splash(
    os.path.join(PKG, "splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(172, 200),
    text_size=10,
    text_color="#cbd5e1",
    minify_script=True,
    always_on_top=False,
)
exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.datas,
    [],
    name="QuantTrader",
    icon=os.path.join(PKG, "icon.ico"),
    console=False,
    upx=False,
    runtime_tmpdir=None,
)
