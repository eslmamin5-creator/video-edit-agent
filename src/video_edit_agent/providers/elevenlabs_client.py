"""Low-level ElevenLabs client (spec section 22): thin wrapper used only by
`transcription/providers/elevenlabs.py` today, kept here as the shared,
capability-checked entry point so any future feature (e.g. voice cleanup)
reuses the same key handling and availability check instead of duplicating
it.
"""
from __future__ import annotations

from video_edit_agent.core.config import get_elevenlabs_key


class ElevenLabsUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    if not get_elevenlabs_key():
        return False
    try:
        import elevenlabs  # noqa: F401
    except ImportError:
        return False
    return True


def client():
    if not is_available():
        raise ElevenLabsUnavailable(
            "ElevenLabs is not available: missing ELEVENLABS_API_KEY or the elevenlabs package "
            "is not installed (pip install video-edit-agent[elevenlabs])."
        )
    from elevenlabs.client import ElevenLabs  # type: ignore

    return ElevenLabs(api_key=get_elevenlabs_key())
