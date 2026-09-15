"""Tests for bootstrap.report: setup-state persistence, never secrets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_edit_agent.bootstrap.report import (
    SetupState,
    assert_no_secrets,
    load_state,
    save_state,
    state_path,
)


def test_state_path_is_under_dotruntime(tmp_path: Path):
    assert state_path(tmp_path) == tmp_path / ".runtime" / "setup_state.json"


def test_save_and_load_round_trip(tmp_path: Path):
    state = SetupState(
        setup_version="0.2.1", os="Linux 6.1", python_executable="/x/venv/bin/python",
        python_version="3.11.9", profile="full-local", ffmpeg_version="6.1.1",
        node_version="v22.0.0", npm_version="10.0.0",
        capability_summary={"ffmpeg": "verified"}, last_verified_at="2026-01-01T00:00:00+00:00",
    )
    path = save_state(tmp_path, state)
    assert path.exists()

    loaded = load_state(tmp_path)
    assert loaded is not None
    assert loaded.profile == "full-local"
    assert loaded.capability_summary == {"ffmpeg": "verified"}


def test_load_state_returns_none_when_absent(tmp_path: Path):
    assert load_state(tmp_path) is None


def test_load_state_returns_none_on_corrupt_json(tmp_path: Path):
    path = state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_state(tmp_path) is None


def test_saved_state_is_plain_json_with_no_secret_shaped_keys(tmp_path: Path):
    state = SetupState(profile="core")
    path = save_state(tmp_path, state)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in data:
        lowered = key.lower()
        assert "key" not in lowered
        assert "token" not in lowered
        assert "secret" not in lowered
        assert "password" not in lowered
        assert "credential" not in lowered


def test_assert_no_secrets_raises_on_secret_shaped_key():
    with pytest.raises(ValueError):
        assert_no_secrets({"gemini_api_key": "sk-fake-not-real"})


def test_assert_no_secrets_allows_normal_fields():
    assert_no_secrets({"profile": "core", "python_version": "3.11.9"})


def test_load_state_ignores_unknown_fields_for_forward_compat(tmp_path: Path):
    path = state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"profile": "core", "some_future_field": "xyz"}), encoding="utf-8")
    loaded = load_state(tmp_path)
    assert loaded is not None
    assert loaded.profile == "core"
