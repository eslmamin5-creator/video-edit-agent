"""Regression test for the "false-ready" Gemini/ElevenLabs capability bug
found during real acceptance testing: an API key being set was previously
enough for `detect_gemini_key`/`detect_elevenlabs_key` to report the
provider as available, even when the corresponding SDK (google-genai /
elevenlabs) wasn't installed in the interpreter actually running the check --
e.g. the "full-local" profile deliberately excludes the "gemini" extra, so a
key saved via `videoedit setup`'s prompt existed with no SDK present. A key
alone must never be reported as "ready"; the SDK's actual importability has
to gate it too.
"""
from __future__ import annotations

from video_edit_agent.core.capability_router import detect_elevenlabs_key, detect_gemini_key


def test_gemini_key_alone_without_sdk_is_not_available(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router.get_gemini_key", lambda: "fake-key"
    )
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router._pymodule", lambda name: False
    )
    cap = detect_gemini_key()
    assert cap.available is False
    assert "not installed" in cap.detail


def test_gemini_key_with_sdk_installed_is_available(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router.get_gemini_key", lambda: "fake-key"
    )
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router._pymodule", lambda name: True
    )
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router.is_verified", lambda name: (None, "not yet verified")
    )
    cap = detect_gemini_key()
    assert cap.available is True


def test_gemini_key_missing_is_not_available_regardless_of_sdk(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router.get_gemini_key", lambda: None
    )
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router._pymodule", lambda name: True
    )
    cap = detect_gemini_key()
    assert cap.available is False
    assert "missing GEMINI_API_KEY" in cap.detail


def test_elevenlabs_key_alone_without_sdk_is_not_available(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router.get_elevenlabs_key", lambda: "fake-key"
    )
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router._pymodule", lambda name: False
    )
    cap = detect_elevenlabs_key()
    assert cap.available is False
    assert "not installed" in cap.detail


def test_elevenlabs_key_with_sdk_installed_is_available(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router.get_elevenlabs_key", lambda: "fake-key"
    )
    monkeypatch.setattr(
        "video_edit_agent.core.capability_router._pymodule", lambda name: True
    )
    cap = detect_elevenlabs_key()
    assert cap.available is True
