"""Detects what is actually available on this machine (spec sections 27, 28).

This is the single source of truth for "doctor", "setup", and every graceful
degradation decision made elsewhere in the pipeline (spec section 43): a
provider is only ever used after this module confirms it is usable.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from importlib import util as importlib_util

from video_edit_agent.core.config import get_elevenlabs_key, get_gemini_key
from video_edit_agent.core.verification_store import is_verified


@dataclass
class Capability:
    name: str
    available: bool
    detail: str = ""
    verified: bool | None = None  # None = not applicable; True/False = real-render acceptance ran or not


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run_ok(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, timeout=5, check=False)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _pymodule(name: str) -> bool:
    return importlib_util.find_spec(name) is not None


def detect_ffmpeg() -> Capability:
    ok = _which("ffmpeg") and _run_ok(["ffmpeg", "-version"])
    return Capability("ffmpeg", ok, "OK" if ok else "not found on PATH")


def detect_ffprobe() -> Capability:
    ok = _which("ffprobe") and _run_ok(["ffprobe", "-version"])
    return Capability("ffprobe", ok, "OK" if ok else "not found on PATH")


def detect_node() -> Capability:
    ok = _which("node")
    return Capability("node", ok, "OK" if ok else "not found on PATH")


def detect_npm() -> Capability:
    ok = _which("npm") or _which("npm.cmd")
    return Capability("npm", ok, "OK" if ok else "not found on PATH")


def detect_faster_whisper() -> Capability:
    ok = _pymodule("faster_whisper")
    return Capability("faster-whisper", ok, "installed" if ok else "pip install video-edit-agent[local]")


def detect_openai_whisper() -> Capability:
    ok = _pymodule("whisper")
    return Capability("openai-whisper", ok, "installed" if ok else "not installed (optional)")


def detect_whisper_cpp() -> Capability:
    ok = bool(_which("whisper-cpp") or _which("whisper-cli"))
    return Capability("whisper.cpp", ok, "on PATH" if ok else "not installed (optional)")


def detect_gemini_key() -> Capability:
    key = get_gemini_key()
    if not key:
        return Capability("gemini_key", False, "missing GEMINI_API_KEY", verified=None)
    verified, verified_detail = is_verified("gemini_live")
    return Capability("gemini_key", True, f"API key detected; live {verified_detail}", verified=verified)


def detect_elevenlabs_key() -> Capability:
    key = get_elevenlabs_key()
    return Capability("elevenlabs_key", key is not None, "set" if key else "missing ELEVENLABS_API_KEY")


def detect_hyperframes() -> Capability:
    """Installed does NOT mean usable: a package literally named "hyperframes"
    exists on PyPI but is an unrelated N-dimensional DataFrame library with no
    rendering API (confirmed during V1.1 hardening due-diligence) -- so this
    checks for the actual `render_from_spec` entry point this project's
    adapter calls, not just a successful bare import."""
    installed = _pymodule("hyperframes")
    has_real_api = False
    if installed:
        try:
            import hyperframes  # type: ignore

            has_real_api = hasattr(hyperframes, "render_from_spec")
        except ImportError:
            has_real_api = False
    verified, verified_detail = is_verified("hyperframes")
    if not installed:
        detail = "not installed (optional)"
    elif not has_real_api:
        detail = "installed but missing render_from_spec -- not a real HyperFrames SDK"
    else:
        detail = f"installed, real API present; {verified_detail}"
    return Capability("hyperframes", has_real_api, detail, verified=verified if has_real_api else None)


def detect_mediapipe() -> Capability:
    """Reports mediapipe mask detection+caching verification. Full
    behind-subject video compositing is implemented in `core/pipeline.py`
    (real RGBA subject cutout layered on top of the graphic overlay via
    ordinary overlay-list order; see `tests/test_behind_subject.py`) and
    depends on this same mediapipe segmentation being available."""
    from video_edit_agent.subject.detect import is_available as mediapipe_ok

    ok = mediapipe_ok()
    if not ok:
        detail = (
            "not installed, or installed version lacks mediapipe.solutions "
            "(mediapipe>=1.0 removed it -- use mediapipe<1.0)"
        )
        return Capability("mediapipe (behind-subject)", False, detail)
    verified, verified_detail = is_verified("mediapipe_segmentation")
    detail = f"installed, legacy Solutions API present; segmentation+cache {verified_detail}."
    return Capability("mediapipe (behind-subject)", True, detail, verified=verified)


def detect_remotion() -> Capability:
    ok = detect_node().available and detect_npm().available
    verified, verified_detail = is_verified("remotion")
    detail = (f"renderable via npx; {verified_detail}") if ok else "requires Node.js + npm"
    return Capability("remotion", ok, detail, verified=verified if ok else None)


def detect_manim() -> Capability:
    ok = _pymodule("manim")
    return Capability("manim", ok, "installed" if ok else "pip install video-edit-agent[motion]")


def detect_claude_code() -> Capability:
    ok = _which("claude")
    return Capability("claude_code", ok, "detected" if ok else "not detected")


def detect_codex() -> Capability:
    ok = _which("codex")
    return Capability("codex", ok, "detected" if ok else "not detected")


def detect_gpu() -> Capability:
    ok = _which("nvidia-smi") and _run_ok(["nvidia-smi"])
    return Capability("gpu", bool(ok), "NVIDIA GPU detected" if ok else "no GPU detected (CPU mode)")


def full_capability_matrix() -> dict[str, Capability]:
    checks = [
        detect_ffmpeg, detect_ffprobe, detect_node, detect_npm, detect_gpu,
        detect_faster_whisper, detect_openai_whisper, detect_whisper_cpp,
        detect_gemini_key, detect_elevenlabs_key,
        detect_hyperframes, detect_mediapipe, detect_remotion, detect_manim,
        detect_claude_code, detect_codex,
    ]
    return {c().name: c() for c in checks}


if __name__ == "__main__":  # pragma: no cover -- exercised via subprocess in tests
    import json

    print(json.dumps([
        {"name": c.name, "available": c.available, "detail": c.detail, "verified": c.verified}
        for c in full_capability_matrix().values()
    ]))
