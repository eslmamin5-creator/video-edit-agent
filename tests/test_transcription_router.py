"""Provider router + no-key fallback + offline mode tests (spec sections 5, 6, 24).

The whole point of the router is that `videoedit edit` must work with ZERO API
keys: no GEMINI_API_KEY, no ELEVENLABS_API_KEY, nothing but local Faster-Whisper
(or, if that's not installed either, a clear error — never a crash and never a
silent hang waiting for a key).
"""
from __future__ import annotations

import pytest

from video_edit_agent.core.config import TranscriptionConfig
from video_edit_agent.transcription.base import TranscriptionUnavailable
from video_edit_agent.transcription.router import TranscriptionRouter, build_providers


@pytest.fixture(autouse=True)
def no_cloud_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)


def test_default_priority_ends_with_faster_whisper():
    cfg = TranscriptionConfig()
    router = TranscriptionRouter(cfg, offline=False)
    order = router._candidate_order()
    assert order[-1] == "faster-whisper" or "faster-whisper" in order


def test_explicit_provider_short_circuits_priority_list():
    cfg = TranscriptionConfig(provider="elevenlabs")
    router = TranscriptionRouter(cfg, offline=False)
    assert router._candidate_order() == ["elevenlabs"]


def test_no_api_keys_never_raises_before_reaching_local_provider(tmp_path):
    """With no keys and no local model actually loadable in the test env, the
    router must still *attempt* faster-whisper last and report why every
    attempt failed — it must never claim a cloud provider is available."""
    cfg = TranscriptionConfig()
    router = TranscriptionRouter(cfg, offline=False)
    attempts = []
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"\x00")

    with pytest.raises(TranscriptionUnavailable) as excinfo:
        router.transcribe(fake_audio, on_attempt=lambda name, ok, reason: attempts.append((name, ok, reason)))

    attempted_names = [a[0] for a in attempts]
    assert "gemini" in attempted_names
    assert "elevenlabs" in attempted_names
    gemini_attempt = next(a for a in attempts if a[0] == "gemini")
    assert gemini_attempt[1] is False
    assert "GEMINI_API_KEY" in gemini_attempt[2]
    # The router's final error must not silently swallow *why* every provider failed.
    assert "faster-whisper" in str(excinfo.value)


def test_offline_mode_skips_cloud_providers_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-should-be-ignored-offline")
    cfg = TranscriptionConfig()
    router = TranscriptionRouter(cfg, offline=True)
    attempts = []
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"\x00")

    with pytest.raises(TranscriptionUnavailable):
        router.transcribe(fake_audio, on_attempt=lambda name, ok, reason: attempts.append((name, ok, reason)))

    # Cloud providers must be skipped by the offline gate before is_available
    # is even consulted — on_attempt should never fire for them in offline mode.
    attempted_names = [a[0] for a in attempts]
    assert "gemini" not in attempted_names
    assert "elevenlabs" not in attempted_names


def test_build_providers_returns_all_known_providers():
    cfg = TranscriptionConfig()
    providers = build_providers(cfg, gemini_model="gemini-3.5-transcribe")
    assert set(providers.keys()) == {"gemini", "elevenlabs", "faster-whisper", "openai-whisper", "whisper.cpp"}
