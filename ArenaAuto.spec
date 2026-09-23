# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ArenaAuto (Windows)."""

block_cipher = None

project_root = SPECPATH  # noqa: F821 - provided by PyInstaller

datas = [
    ("resources", "resources"),
    ("config", "config"),
]

hiddenimports = [
    "arena_auto",
    "arena_auto.app",
    "arena_auto.paths",
    "arena_auto.exceptions",
    "arena_auto.adb",
    "arena_auto.adb.controller",
    "arena_auto.automation",
    "arena_auto.automation.controller",
    "arena_auto.automation.selection",
    "arena_auto.automation.state_machine",
    "arena_auto.automation.states",
    "arena_auto.recognition",
    "arena_auto.recognition.detector",
    "arena_auto.recognition.ocr",
    "arena_auto.recognition.ocr_engine",
    "arena_auto.recognition.screen",
    "arena_auto.recognition.template",
    "arena_auto.config",
    "arena_auto.config.manager",
    "arena_auto.logging",
    "arena_auto.logging.logger",
    "arena_auto.debug",
    "arena_auto.debug.recorder",
    "arena_auto.models",
    "arena_auto.models.config",
    "arena_auto.models.recognition",
    "arena_auto.models.state",
    "arena_auto.gui",
    "arena_auto.gui.main_window",
    "arena_auto.gui.widgets",
    "arena_auto.gui.workers",
    "arena_auto.calibration",
    "arena_auto.calibration.analyzer",
    "arena_auto.calibration.button_detector",
    "arena_auto.calibration.coordinate_mapper",
    "arena_auto.calibration.models",
    "arena_auto.calibration.opponent_detector",
    "arena_auto.calibration.roi_detector",
    "arena_auto.calibration.screen_analyzer",
    "arena_auto.calibration.template_generator",
    "arena_auto.calibration.wizard",
]

a = Analysis(
    ["main.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ArenaAuto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ArenaAuto",
)
