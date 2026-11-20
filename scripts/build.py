#!/usr/bin/env python3
"""Build Windows EXE with PyInstaller: python scripts/build.py"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _force_utf8_stdout() -> None:
    """CI Windows runners may use cp1252; Chinese logs must not crash print()."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def verify_windows_executable(exe: Path, resources: Path) -> None:
    """Reject stale/default icons or an accidentally console-mode build."""
    import pefile  # Installed by PyInstaller on Windows.

    ico = (resources / "app_icon.ico").read_bytes()
    header = ico[:6]
    reserved, kind, count = struct.unpack("<3H", header)
    if reserved or kind != 1 or not count:
        raise ValueError("invalid source ICO")

    with pefile.PE(str(exe), fast_load=True) as pe:
        if pe.OPTIONAL_HEADER.Subsystem != pefile.SUBSYSTEM_TYPE["IMAGE_SUBSYSTEM_WINDOWS_GUI"]:
            raise ValueError("expected a Windows GUI executable")
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])

        def resource(kind: str, identifier: int) -> bytes:
            tree = getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None)
            for entry in tree.entries if tree is not None else ():
                if entry.id != pefile.RESOURCE_TYPE[kind]:
                    continue
                for item in entry.directory.entries:
                    if item.id == identifier:
                        data = item.directory.entries[0].data.struct
                        return pe.get_data(data.OffsetToData, data.Size)
            raise ValueError(f"missing {kind} resource {identifier}")

        # PyInstaller copies each ICO frame into RT_ICON and points to it from
        # RT_GROUP_ICON #1. Compare the actual data, not just the frame count.
        group = resource("RT_GROUP_ICON", 1)
        if group[:6] != header or len(group) != 6 + count * 14:
            raise ValueError("embedded icon group does not match app_icon.ico")
        for index in range(count):
            ico_pos = 6 + index * 16
            group_pos = 6 + index * 14
            length, offset = struct.unpack_from("<2I", ico, ico_pos + 8)
            (icon_id,) = struct.unpack_from("<H", group, group_pos + 12)
            if (group[group_pos:group_pos + 12] != ico[ico_pos:ico_pos + 12]
                    or resource("RT_ICON", icon_id) != ico[offset:offset + length]):
                raise ValueError(f"embedded icon frame {index + 1} does not match app_icon.ico")

    bundled_png = exe.parent / "_internal" / "resources" / "app_icon.png"
    if bundled_png.read_bytes() != (resources / "app_icon.png").read_bytes():
        raise ValueError("bundled window icon does not match app_icon.png")


def main(argv=None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distpath", type=Path, default=Path("dist"), help="output directory")
    parser.add_argument("--workpath", type=Path, default=Path("build"), help="temporary build directory")
    args = parser.parse_args(argv)
    dist = (ROOT / args.distpath).resolve()
    work = (ROOT / args.workpath).resolve()

    spec = ROOT / "ArenaAuto.spec"
    if not spec.is_file():
        print(f"找不到 spec: {spec}", file=sys.stderr)
        return 1
    # PyInstaller's incremental check compares the icon path, not its bytes.
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--distpath", str(dist), "--workpath", str(work), str(spec)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode:
        return result.returncode
    if sys.platform == "win32":
        exe = dist / "ArenaAuto" / "ArenaAuto.exe"
        try:
            verify_windows_executable(exe, ROOT / "resources")
        except (OSError, ValueError, struct.error) as exc:
            print(f"EXE 图标或子系统校验失败: {exc}", file=sys.stderr)
            return 1
        print("已校验 Windows GUI 子系统及 EXE/窗口图标")
    print(f"构建完成，输出目录: {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
