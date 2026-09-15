"""Maps low-level detection (ffmpeg/node/capability_router) onto the honest
per-capability states spec section 19 requires for `doctor`/`setup` output:
installed / configured / verified / optional / missing / fallback available.

Also runnable as `python -m video_edit_agent.bootstrap.capabilities` to print
capability states as JSON -- used by `bootstrap.setup.run_setup` to check
capabilities INSIDE the private runtime venv it just built via subprocess,
since `importlib`-based detection only ever sees packages installed in the
interpreter that is actually running (which, right after building a fresh
venv, is NOT that venv unless invoked this way).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from video_edit_agent.core.capability_router import Capability, full_capability_matrix

# Capabilities that are optional (system still fully functional without them)
# vs. required (rendering cannot proceed at all without them).
REQUIRED_CAPABILITIES = {"ffmpeg", "ffprobe"}

# Capabilities that have a known local fallback when unavailable.
FALLBACK_MAP = {
    "gemini_key": "faster-whisper (local transcription)",
    "elevenlabs_key": "local TTS/caption styling",
    "remotion": "local motion engine (src/video_edit_agent/motion/simple)",
    "manim": "local motion engine",
    "mediapipe (behind-subject)": "plain overlay (no behind-subject cutout)",
    "hyperframes": "local motion engine",
}

STATE_INSTALLED = "installed"
STATE_CONFIGURED = "configured"
STATE_VERIFIED = "verified"
STATE_OPTIONAL = "optional"
STATE_MISSING = "missing"
STATE_FALLBACK_AVAILABLE = "fallback available"


@dataclass
class CapabilityState:
    name: str
    state: str
    detail: str


def _classify(cap: Capability) -> str:
    if cap.available:
        if cap.verified:
            return STATE_VERIFIED
        if cap.name.endswith("_key"):
            return STATE_CONFIGURED
        return STATE_INSTALLED
    if cap.name in REQUIRED_CAPABILITIES:
        return STATE_MISSING
    if cap.name in FALLBACK_MAP:
        return STATE_FALLBACK_AVAILABLE
    return STATE_OPTIONAL


def capability_states() -> list[CapabilityState]:
    matrix = full_capability_matrix()
    states = []
    for cap in matrix.values():
        state = _classify(cap)
        detail = cap.detail
        if state == STATE_FALLBACK_AVAILABLE:
            detail = f"{cap.detail} -- fallback: {FALLBACK_MAP[cap.name]}"
        states.append(CapabilityState(cap.name, state, detail))
    return states


def capability_states_as_dicts() -> list[dict]:
    return [
        {"name": s.name, "state": s.state, "detail": s.detail}
        for s in capability_states()
    ]


def offline_ready() -> tuple[bool, list[str]]:
    """True if Editor/Creator/Assembler can all run fully offline right now:
    ffmpeg+ffprobe present, and at least one transcription path that needs no
    API key (faster-whisper) OR the user is fine without transcription-backed
    features. Missing reasons are returned for a clear report."""
    matrix = full_capability_matrix()
    missing = []
    if not matrix["ffmpeg"].available:
        missing.append("ffmpeg")
    if not matrix["ffprobe"].available:
        missing.append("ffprobe")
    if not matrix["faster-whisper"].available:
        missing.append("faster-whisper (local transcription) -- install profile 'local' or 'full-local'")
    return (len(missing) == 0, missing)


if __name__ == "__main__":  # pragma: no cover -- exercised via subprocess in tests
    print(json.dumps(capability_states_as_dicts()))
