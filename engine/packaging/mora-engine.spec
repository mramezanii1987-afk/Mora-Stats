# PyInstaller spec for the MoRa Stats engine sidecar.
#
# Build the same file on every platform:
#     pyinstaller packaging/mora-engine.spec --distpath dist --workpath build
#
# On Windows this produces dist/mora-engine.exe. That file is what the desktop
# shell spawns, and it is the only Python that ever reaches a user's machine.
# Nobody installs Python, sets a PATH or creates a virtual environment.

import sys
from PyInstaller.utils.hooks import collect_submodules

hidden = (
    collect_submodules("scipy")
    + collect_submodules("scipy.special")
    + collect_submodules("pandas")
    + ["numpy", "openpyxl"]
)

# Everything a statistics engine has no business carrying into a desktop app.
excluded = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest",
    "sphinx", "setuptools", "pip", "PyQt5", "PySide6", "wx",
]

a = Analysis(
    ["sidecar_entry.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excluded,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="mora-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX compression is a reliable way to be called a virus
    runtime_tmpdir=None,
    console=False,      # no terminal window flashing behind the app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
