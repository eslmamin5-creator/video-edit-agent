"""CLI smoke tests (spec sections 10, 27-29) — every command must run cleanly
with zero API keys and never print a secret value even if one is set."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from video_edit_agent.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_brands_dir(tmp_path, monkeypatch):
    # Keep brand init/validate tests from touching the real repo's brands/.
    from video_edit_agent import brand as brand_pkg

    monkeypatch.setattr(brand_pkg.loader, "DEFAULT_BRANDS_ROOT", tmp_path / "brands")


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "videoedit" in result.stdout


def test_no_args_shows_help_without_error():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_doctor_runs_and_prints_capability_matrix():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "capability matrix" in result.stdout.lower()


def test_providers_lists_transcription_and_motion_sections():
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "Transcription providers" in result.stdout
    assert "Motion engines" in result.stdout


def test_config_show_prints_json():
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "transcription" in result.stdout


def test_doctor_never_prints_a_real_api_key_value(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-super-secret-value-should-never-appear")
    result = runner.invoke(app, ["doctor"])
    assert "sk-super-secret-value-should-never-appear" not in result.stdout


def test_providers_never_prints_a_real_api_key_value(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-super-secret-value-should-never-appear")
    result = runner.invoke(app, ["providers"])
    assert "el-super-secret-value-should-never-appear" not in result.stdout


def test_brand_init_and_validate_round_trip():
    result = runner.invoke(app, ["brand", "init", "test_ci_brand"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["brand", "validate", "test_ci_brand"])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()


def test_brand_validate_on_missing_brand_fails_cleanly():
    result = runner.invoke(app, ["brand", "validate", "does_not_exist_brand"])
    assert result.exit_code != 0


def test_project_inspect_on_missing_project_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["project", "inspect", str(tmp_path)])
    assert result.exit_code != 0


def test_edit_on_nonexistent_video_fails_cleanly():
    result = runner.invoke(app, ["edit", "/no/such/video.mp4"])
    assert result.exit_code != 0
