"""Project-local `.env` support (v0.2.3, spec section 26 addendum).

Covers: loading Gemini/ElevenLabs from `.env`, OS env overriding `.env`,
blank values counting as missing, a missing `.env` being harmless,
resolution from a project subdirectory, preserving unrelated `.env`
entries on write, and secrets never leaking into CLI output or logs.

No real credentials appear anywhere in this file -- every key used below is
an obviously-fake placeholder.
"""
from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from video_edit_agent.cli.main import app
from video_edit_agent.core.env_file import find_project_env_file, resolve_secret, write_env_values
from video_edit_agent.transcription.providers.elevenlabs import ElevenLabsProvider
from video_edit_agent.transcription.providers.gemini import GeminiTranscriptionProvider

runner = CliRunner()

FAKE_GEMINI = "fake-gemini-key-should-never-appear"
FAKE_ELEVENLABS = "fake-elevenlabs-key-should-never-appear"


def _write_env(path, **pairs):
    path.write_text("\n".join(f"{k}={v}" for k, v in pairs.items()) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# resolve_secret / find_project_env_file
# ---------------------------------------------------------------------------


def test_loads_gemini_key_from_env_file(tmp_path):
    _write_env(tmp_path / ".env", GEMINI_API_KEY=FAKE_GEMINI)
    assert resolve_secret("GEMINI_API_KEY", start=tmp_path, environ={}) == FAKE_GEMINI


def test_loads_elevenlabs_key_from_env_file(tmp_path):
    _write_env(tmp_path / ".env", ELEVENLABS_API_KEY=FAKE_ELEVENLABS)
    assert resolve_secret("ELEVENLABS_API_KEY", start=tmp_path, environ={}) == FAKE_ELEVENLABS


def test_os_env_overrides_env_file(tmp_path):
    _write_env(tmp_path / ".env", GEMINI_API_KEY="from-dotenv")
    resolved = resolve_secret(
        "GEMINI_API_KEY", start=tmp_path, environ={"GEMINI_API_KEY": "from-process-env"}
    )
    assert resolved == "from-process-env"


def test_blank_env_file_value_is_missing(tmp_path):
    _write_env(tmp_path / ".env", GEMINI_API_KEY="")
    assert resolve_secret("GEMINI_API_KEY", start=tmp_path, environ={}) is None


def test_blank_os_env_value_falls_back_to_env_file(tmp_path):
    _write_env(tmp_path / ".env", GEMINI_API_KEY=FAKE_GEMINI)
    resolved = resolve_secret("GEMINI_API_KEY", start=tmp_path, environ={"GEMINI_API_KEY": ""})
    assert resolved == FAKE_GEMINI


def test_missing_env_file_is_harmless(tmp_path):
    assert find_project_env_file(tmp_path) is None
    assert resolve_secret("GEMINI_API_KEY", start=tmp_path, environ={}) is None


def test_resolves_env_file_from_project_subdirectory(tmp_path):
    _write_env(tmp_path / ".env", GEMINI_API_KEY=FAKE_GEMINI)
    subdir = tmp_path / "footage" / "take_01"
    subdir.mkdir(parents=True)

    found = find_project_env_file(subdir)
    assert found == tmp_path / ".env"
    assert resolve_secret("GEMINI_API_KEY", start=subdir, environ={}) == FAKE_GEMINI


# ---------------------------------------------------------------------------
# write_env_values
# ---------------------------------------------------------------------------


def test_write_env_values_creates_file_if_missing(tmp_path):
    env_path = tmp_path / ".env"
    write_env_values(env_path, {"GEMINI_API_KEY": FAKE_GEMINI})
    assert resolve_secret("GEMINI_API_KEY", start=tmp_path, environ={}) == FAKE_GEMINI


def test_write_env_values_preserves_unrelated_existing_entries(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# a comment to keep\nSOME_OTHER_VAR=keep-me\nGEMINI_API_KEY=old-value\n",
        encoding="utf-8",
    )

    write_env_values(env_path, {"GEMINI_API_KEY": FAKE_GEMINI})

    content = env_path.read_text(encoding="utf-8")
    assert "SOME_OTHER_VAR=keep-me" in content
    assert "# a comment to keep" in content
    assert "old-value" not in content
    assert f"GEMINI_API_KEY={FAKE_GEMINI}" in content


def test_write_env_values_appends_new_key_without_disturbing_others(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=already-here\n", encoding="utf-8")

    write_env_values(env_path, {"ELEVENLABS_API_KEY": FAKE_ELEVENLABS})

    content = env_path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=already-here" in content
    assert f"ELEVENLABS_API_KEY={FAKE_ELEVENLABS}" in content


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only permission restriction (spec: no Windows hacks)")
def test_write_env_values_restricts_permissions_on_posix(tmp_path):
    env_path = tmp_path / ".env"
    write_env_values(env_path, {"GEMINI_API_KEY": FAKE_GEMINI})
    mode = env_path.stat().st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# .gitignore coverage
# ---------------------------------------------------------------------------


def test_dotenv_is_gitignored():
    from pathlib import Path

    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    assert ".env" in lines
    assert "!.env.example" in lines


# ---------------------------------------------------------------------------
# secrets never leak into CLI output
# ---------------------------------------------------------------------------


def test_doctor_never_prints_env_file_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    _write_env(tmp_path / ".env", GEMINI_API_KEY=FAKE_GEMINI)

    result = runner.invoke(app, ["doctor"])
    assert FAKE_GEMINI not in result.stdout


def test_providers_never_prints_env_file_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    _write_env(tmp_path / ".env", ELEVENLABS_API_KEY=FAKE_ELEVENLABS)

    result = runner.invoke(app, ["providers"])
    assert FAKE_ELEVENLABS not in result.stdout
    assert "gemini_key" in result.stdout
    assert "elevenlabs_key" in result.stdout


# ---------------------------------------------------------------------------
# --offline blocks cloud usage even with a configured .env key
# ---------------------------------------------------------------------------


def test_offline_blocks_gemini_even_with_env_file_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _write_env(tmp_path / ".env", GEMINI_API_KEY=FAKE_GEMINI)

    available, reason = GeminiTranscriptionProvider().is_available(offline=True)
    assert available is False
    assert "offline" in reason.lower()


def test_offline_blocks_elevenlabs_even_with_env_file_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    _write_env(tmp_path / ".env", ELEVENLABS_API_KEY=FAKE_ELEVENLABS)

    available, reason = ElevenLabsProvider().is_available(offline=True)
    assert available is False
    assert "offline" in reason.lower()
