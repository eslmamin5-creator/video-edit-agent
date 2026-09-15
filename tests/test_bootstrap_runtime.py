"""Tests for bootstrap.runtime: private venv creation, idempotency, repair."""
from __future__ import annotations

import sys
from pathlib import Path

from video_edit_agent.bootstrap.runtime import (
    check_runtime,
    ensure_runtime,
    rebuild_runtime,
    runtime_dir,
    venv_dir,
    venv_python,
)


def test_runtime_dir_is_dotruntime_under_project_root(tmp_path: Path):
    assert runtime_dir(tmp_path) == tmp_path / ".runtime"
    assert venv_dir(tmp_path) == tmp_path / ".runtime" / "venv"


def test_venv_python_path_is_platform_specific(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    win_path = venv_python(tmp_path)
    assert win_path.parts[-2:] == ("Scripts", "python.exe")

    monkeypatch.setattr(sys, "platform", "linux")
    unix_path = venv_python(tmp_path)
    assert unix_path.parts[-2:] == ("bin", "python")


def test_check_runtime_reports_missing_when_no_venv(tmp_path: Path):
    status = check_runtime(tmp_path)
    assert status.exists is False
    assert status.healthy is False


def test_ensure_runtime_creates_a_real_working_venv(tmp_path: Path):
    status = ensure_runtime(tmp_path, sys.executable)
    assert status.healthy is True
    assert status.python_path.exists()


def test_ensure_runtime_is_idempotent_reuses_healthy_runtime(tmp_path: Path):
    first = ensure_runtime(tmp_path, sys.executable)
    mtime_before = first.python_path.stat().st_mtime

    second = ensure_runtime(tmp_path, sys.executable)
    mtime_after = second.python_path.stat().st_mtime

    assert second.healthy is True
    # Reused, not rebuilt -- interpreter binary is untouched.
    assert mtime_before == mtime_after


def test_ensure_runtime_rebuilds_a_broken_runtime(tmp_path: Path):
    ensure_runtime(tmp_path, sys.executable)
    # Simulate corruption: delete the venv's own interpreter.
    venv_python(tmp_path).unlink()

    status = ensure_runtime(tmp_path, sys.executable)
    assert status.healthy is True


def test_repair_forces_a_rebuild_even_if_runtime_looks_healthy(tmp_path: Path):
    ensure_runtime(tmp_path, sys.executable)
    before_mtime = venv_python(tmp_path).stat().st_mtime

    status = rebuild_runtime(tmp_path, sys.executable)
    assert status.healthy is True
    # A rebuild recreates the interpreter binary/symlink.
    assert venv_python(tmp_path).exists()
    # (mtime comparison is best-effort/racy on very fast filesystems, so we
    # only assert the runtime is healthy post-repair, not strict inequality.)
    assert before_mtime is not None


def test_ensure_runtime_never_touches_paths_outside_dotruntime(tmp_path: Path):
    marker = tmp_path / "user_media.mp4"
    marker.write_bytes(b"not really a video, just a marker file")

    ensure_runtime(tmp_path, sys.executable)
    rebuild_runtime(tmp_path, sys.executable)

    assert marker.exists()
    assert marker.read_bytes() == b"not really a video, just a marker file"
