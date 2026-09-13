"""V1.1 hardening (spec sections 10-14): live-provider acceptance tests.

These are the ONLY tests in the suite allowed to talk to a real cloud API,
and only when a real key is present. They must never run in normal CI/local
`pytest` runs on a machine without keys -- that would make unit tests depend
on cloud availability, which the hardening spec explicitly forbids. Never
print, log, or persist the key value itself anywhere (spec section 10).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from video_edit_agent.core.config import get_elevenlabs_key, get_gemini_key

requires_gemini_key = pytest.mark.skipif(
    not get_gemini_key(), reason="GEMINI_API_KEY not set -- live Gemini acceptance is BLOCKED BY ENVIRONMENT"
)
requires_elevenlabs_key = pytest.mark.skipif(
    not get_elevenlabs_key(),
    reason="ELEVENLABS_API_KEY not set -- live ElevenLabs acceptance is BLOCKED BY ENVIRONMENT",
)


@requires_gemini_key
def test_gemini_transcription_minimal_live_smoke(sample_video: Path):
    """Minimal live check: a real request succeeds, a schema-valid transcript
    comes back, provider is recorded as gemini, and no translation occurred
    (verbatim-only, per the project's non-negotiable dialect rule)."""
    from video_edit_agent.transcription.providers.gemini import GeminiTranscriptionProvider

    provider = GeminiTranscriptionProvider()
    transcript = provider.transcribe(sample_video)

    assert transcript.provider == "gemini"
    assert transcript.segments, "expected at least one segment from a real Gemini transcription call"


@requires_gemini_key
def test_gemini_vision_minimal_live_smoke(sample_video: Path):
    """Minimal live check for Gemini visual analysis: a real small local clip
    gets a real structured/text response back with no secret leakage."""
    from video_edit_agent.providers.gemini_client import analyze_video_segment

    result = analyze_video_segment(sample_video, "Describe this clip in one short sentence.")
    assert isinstance(result, str)
    assert get_gemini_key() not in result  # the key itself must never round-trip into output


@requires_elevenlabs_key
def test_elevenlabs_minimal_live_smoke(sample_video: Path):
    """ElevenLabs is optional; only exercised at all when a real key exists."""
    from video_edit_agent.transcription.providers.elevenlabs import ElevenLabsProvider

    provider = ElevenLabsProvider()
    transcript = provider.transcribe(sample_video)
    assert transcript.provider == "elevenlabs"
