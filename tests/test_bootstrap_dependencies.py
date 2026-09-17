"""Tests for bootstrap.dependencies: install profiles."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from video_edit_agent.bootstrap.dependencies import (
    DEFAULT_PROFILE,
    PROFILE_EXTRAS,
    extras_for_profile,
    install_extras,
    known_profiles,
    pip_install_spec,
)


def test_known_profiles_include_required_set():
    profiles = known_profiles()
    for expected in ("core", "local", "subject", "motion", "full-local"):
        assert expected in profiles


def test_default_profile_is_full_local():
    assert DEFAULT_PROFILE == "full-local"


def test_core_profile_has_no_extras():
    assert extras_for_profile("core") == ()


def test_full_local_profile_maps_to_full_local_extra():
    assert extras_for_profile("full-local") == ("full-local",)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        extras_for_profile("does-not-exist")


def test_pip_install_spec_core_has_no_brackets(tmp_path: Path):
    spec = pip_install_spec(tmp_path, "core")
    assert spec == str(tmp_path)
    assert "[" not in spec


def test_pip_install_spec_includes_extras_bracket(tmp_path: Path):
    spec = pip_install_spec(tmp_path, "subject")
    assert spec == f"{tmp_path}[local,subject]"


def test_every_profile_is_a_valid_pip_spec_shape(tmp_path: Path):
    for profile in known_profiles():
        spec = pip_install_spec(tmp_path, profile)
        assert spec.startswith(str(tmp_path))


def test_profile_extras_table_has_no_duplicated_unknown_extras():
    known_pyproject_extras = {"local", "gemini", "elevenlabs", "motion", "subject", "dev", "full-local", "all"}
    for extras in PROFILE_EXTRAS.values():
        for extra in extras:
            assert extra in known_pyproject_extras, f"profile references unknown extra {extra!r}"


def test_install_extras_no_extras_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be invoked when extras is empty")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = install_extras(tmp_path, "python", ())
    assert result.ok is True
    assert called is False


def test_install_extras_builds_correct_pip_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = install_extras(tmp_path, "python", ("gemini",))
    assert result.ok is True
    assert captured_cmd["cmd"] == ["python", "-m", "pip", "install", "-e", f"{tmp_path}[gemini]"]


def test_install_extras_reports_failure_on_nonzero_returncode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = install_extras(tmp_path, "python", ("gemini", "elevenlabs"))
    assert result.ok is False
    assert "boom" in result.detail


def test_install_extras_reports_failure_on_subprocess_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(cmd, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = install_extras(tmp_path, "does-not-exist-python", ("gemini",))
    assert result.ok is False
    assert "no such interpreter" in result.detail
