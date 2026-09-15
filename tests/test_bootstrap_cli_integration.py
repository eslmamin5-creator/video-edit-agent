"""CLI integration for the v0.2.1 installation UX: `videoedit doctor` showing
bootstrap state, and `videoedit setup --check` never mutating the filesystem.
"""
from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from video_edit_agent.bootstrap.report import SetupState, save_state
from video_edit_agent.cli.main import app

runner = CliRunner()


def test_doctor_reports_no_setup_state_when_absent(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "No bootstrap setup state found" in result.stdout


def test_doctor_reports_saved_setup_state(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_state(tmp_path, SetupState(profile="full-local", python_version="3.11.9", last_verified_at="2026-01-01T00:00:00+00:00"))

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "full-local" in result.stdout
    assert "3.11.9" in result.stdout


def test_setup_check_never_creates_runtime_dir(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["setup", "--check"])
    assert not (tmp_path / ".runtime").exists()


def test_setup_check_exits_nonzero_when_ffmpeg_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")  # nothing on PATH -> ffmpeg/python probes fail
    result = runner.invoke(app, ["setup", "--check"])
    assert result.exit_code != 0
