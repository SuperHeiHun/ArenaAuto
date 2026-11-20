"""Build-output validation must inspect actual PE icon bytes, not spec text."""

import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_script():
    spec = importlib.util.spec_from_file_location("arena_build", ROOT / "scripts/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_executable(monkeypatch, tmp_path):
    resources = tmp_path / "resources"
    resources.mkdir()
    icon = (ROOT / "resources/app_icon.ico").read_bytes()
    (resources / "app_icon.ico").write_bytes(icon)
    (resources / "app_icon.png").write_bytes(b"window icon")
    exe = tmp_path / "dist/ArenaAuto/ArenaAuto.exe"
    bundled_png = exe.parent / "_internal/resources/app_icon.png"
    bundled_png.parent.mkdir(parents=True)
    bundled_png.write_bytes(b"window icon")

    data, icon_nodes = {}, []

    def node(identifier, payload):
        offset = len(data)
        data[offset] = payload
        leaf = SimpleNamespace(data=SimpleNamespace(struct=SimpleNamespace(OffsetToData=offset, Size=len(payload))))
        return SimpleNamespace(id=identifier, directory=SimpleNamespace(entries=[leaf]))

    group = bytearray(icon[:6])
    (count,) = struct.unpack_from("<H", icon, 4)
    for index in range(count):
        pos = 6 + index * 16
        length, offset = struct.unpack_from("<2I", icon, pos + 8)
        group.extend(icon[pos:pos + 12] + struct.pack("<H", index + 1))
        icon_nodes.append(node(index + 1, icon[offset:offset + length]))
    group_node = node(1, bytes(group))
    pe = MagicMock()
    pe.__enter__.return_value = pe
    pe.OPTIONAL_HEADER.Subsystem = 2
    pe.DIRECTORY_ENTRY_RESOURCE.entries = [
        SimpleNamespace(id=3, directory=SimpleNamespace(entries=icon_nodes)),
        SimpleNamespace(id=14, directory=SimpleNamespace(entries=[group_node])),
    ]
    pe.get_data.side_effect = lambda offset, size: data[offset][:size]
    pefile = SimpleNamespace(
        PE=MagicMock(return_value=pe),
        SUBSYSTEM_TYPE={"IMAGE_SUBSYSTEM_WINDOWS_GUI": 2},
        DIRECTORY_ENTRY={"IMAGE_DIRECTORY_ENTRY_RESOURCE": 2},
        RESOURCE_TYPE={"RT_GROUP_ICON": 14, "RT_ICON": 3},
    )
    monkeypatch.setitem(sys.modules, "pefile", pefile)
    return exe, resources, pe, data, bundled_png


def test_verifies_all_embedded_icon_frames(build_script, fake_executable):
    exe, resources, _, _, _ = fake_executable
    build_script.verify_windows_executable(exe, resources)


def test_rejects_stale_icon_frame(build_script, fake_executable):
    exe, resources, _, data, _ = fake_executable
    data[0] = b"stale icon"
    with pytest.raises(ValueError, match="frame 1"):
        build_script.verify_windows_executable(exe, resources)


def test_rejects_missing_icon_group(build_script, fake_executable):
    exe, resources, pe, _, _ = fake_executable
    pe.DIRECTORY_ENTRY_RESOURCE.entries = []
    with pytest.raises(ValueError, match="missing RT_GROUP_ICON"):
        build_script.verify_windows_executable(exe, resources)


def test_rejects_console_subsystem(build_script, fake_executable):
    exe, resources, pe, _, _ = fake_executable
    pe.OPTIONAL_HEADER.Subsystem = 3
    with pytest.raises(ValueError, match="GUI executable"):
        build_script.verify_windows_executable(exe, resources)


def test_rejects_wrong_bundled_window_icon(build_script, fake_executable):
    exe, resources, _, _, bundled_png = fake_executable
    bundled_png.write_bytes(b"old icon")
    with pytest.raises(ValueError, match="bundled window icon"):
        build_script.verify_windows_executable(exe, resources)


def test_build_requests_clean_output_and_runs_verification(build_script, monkeypatch, tmp_path):
    monkeypatch.setattr(build_script, "ROOT", tmp_path)
    (tmp_path / "ArenaAuto.spec").touch()
    monkeypatch.setattr(build_script, "sys", SimpleNamespace(
        executable="python", platform="win32", stdout=None, stderr=None
    ))
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    verify = MagicMock()
    monkeypatch.setattr(build_script.subprocess, "run", run)
    monkeypatch.setattr(build_script, "verify_windows_executable", verify)
    assert build_script.main(["--distpath", "output", "--workpath", "work"]) == 0
    command = run.call_args.args[0]
    assert "--clean" in command
    assert command[command.index("--distpath") + 1] == str(tmp_path / "output")
    verify.assert_called_once_with(tmp_path / "output/ArenaAuto/ArenaAuto.exe", tmp_path / "resources")
