"""Tests for bootstrap.capabilities: honest state classification."""
from __future__ import annotations

from video_edit_agent.bootstrap import capabilities as cap_mod
from video_edit_agent.core.capability_router import Capability


def test_required_capability_missing_is_missing_not_optional():
    cap = Capability("ffmpeg", available=False, detail="not found")
    assert cap_mod._classify(cap) == cap_mod.STATE_MISSING


def test_fallback_capability_missing_reports_fallback_available():
    cap = Capability("mediapipe (behind-subject)", available=False, detail="not installed")
    assert cap_mod._classify(cap) == cap_mod.STATE_FALLBACK_AVAILABLE


def test_verified_capability_reports_verified():
    cap = Capability("remotion", available=True, detail="ok", verified=True)
    assert cap_mod._classify(cap) == cap_mod.STATE_VERIFIED


def test_installed_but_unverified_reports_installed():
    cap = Capability("manim", available=True, detail="installed", verified=None)
    assert cap_mod._classify(cap) == cap_mod.STATE_INSTALLED


def test_key_capability_reports_configured_not_installed():
    cap = Capability("gemini_key", available=True, detail="set")
    assert cap_mod._classify(cap) == cap_mod.STATE_CONFIGURED


def test_purely_optional_missing_capability_reports_optional():
    cap = Capability("openai-whisper", available=False, detail="not installed")
    assert cap_mod._classify(cap) == cap_mod.STATE_OPTIONAL


def test_capability_states_returns_one_entry_per_matrix_entry():
    states = cap_mod.capability_states()
    names = {s.name for s in states}
    assert "ffmpeg" in names
    assert "ffprobe" in names


def test_capability_states_as_dicts_is_json_serializable():
    import json

    data = cap_mod.capability_states_as_dicts()
    json.dumps(data)  # must not raise
    assert all({"name", "state", "detail"} <= set(d) for d in data)


def test_offline_ready_reports_missing_reasons(monkeypatch):
    from video_edit_agent.core.capability_router import Capability as C

    fake_matrix = {
        "ffmpeg": C("ffmpeg", False),
        "ffprobe": C("ffprobe", True),
        "faster-whisper": C("faster-whisper", False),
    }
    monkeypatch.setattr(cap_mod, "full_capability_matrix", lambda: fake_matrix)
    ok, missing = cap_mod.offline_ready()
    assert ok is False
    assert any("ffmpeg" in m for m in missing)
    assert any("faster-whisper" in m for m in missing)


def test_offline_ready_true_when_core_pieces_present(monkeypatch):
    from video_edit_agent.core.capability_router import Capability as C

    fake_matrix = {
        "ffmpeg": C("ffmpeg", True),
        "ffprobe": C("ffprobe", True),
        "faster-whisper": C("faster-whisper", True),
    }
    monkeypatch.setattr(cap_mod, "full_capability_matrix", lambda: fake_matrix)
    ok, missing = cap_mod.offline_ready()
    assert ok is True
    assert missing == []
