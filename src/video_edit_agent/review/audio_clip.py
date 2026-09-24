"""Audio for one segment, made only when the user asks to hear it.

Listening is a fallback for an unclear phrase, never the main way to review:
nothing here runs during a normal review, and the user is never sent to browse
`review/audio/`. The agent surfaces the clip it gets back in the conversation.

A clip is cut from audio the project already has (the pipeline's cached WAV) or
from the source media, padded a little so a cut-off word is not misheard.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.media import run

AUDIO_DIRNAME = "audio"
DEFAULT_PAD_SECONDS = 0.3


class AudioClipError(RuntimeError):
    """The clip could not be made (no source audio, or ffmpeg failed)."""


def clip_name(number: int, start: float, end: float) -> str:
    return f"seg_{number:02d}_{start:.2f}-{end:.2f}.wav"


def find_clip(review_dir: Path, number: int, start: float, end: float) -> Path | None:
    """An already-made clip for this segment (exact name first, else any for the number)."""
    folder = review_dir / AUDIO_DIRNAME
    exact = folder / clip_name(number, start, end)
    if exact.exists():
        return exact
    return next(iter(sorted(folder.glob(f"seg_{number:02d}_*.wav"))), None) if folder.is_dir() else None


def find_source_audio(edit_dir: Path, explicit: Path | None = None) -> Path | None:
    """Audio to cut from: `explicit`, else the pipeline's extracted WAV, else the
    first clip's source file in `edl.json`."""
    if explicit is not None and explicit.exists():
        return explicit
    cache = edit_dir / "cache"
    if cache.is_dir():
        wavs = sorted(cache.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if wavs:
            return wavs[0]
    edl = edit_dir / "edl.json"
    if edl.exists():
        import json

        try:
            clips = json.loads(edl.read_text(encoding="utf-8")).get("clips", [])
            candidate = Path(clips[0]["source_file"]) if clips else None
        except (ValueError, KeyError, IndexError, TypeError):
            candidate = None
        if candidate is not None:
            for path in (candidate, edit_dir.parent / candidate.name):
                if path.exists():
                    return path
    return None


def ensure_clip(
    review_dir: Path, number: int, start: float, end: float, source: Path | None, *,
    pad: float = DEFAULT_PAD_SECONDS,
) -> Path:
    """The clip for a segment, cutting it now if it does not exist yet."""
    existing = find_clip(review_dir, number, start, end)
    if existing is not None:
        return existing
    if source is None:
        raise AudioClipError("no source audio found for this project")
    out = review_dir / AUDIO_DIRNAME / clip_name(number, start, end)
    out.parent.mkdir(parents=True, exist_ok=True)
    begin = max(0.0, start - pad)
    result = run(
        [
            "ffmpeg", "-y", "-ss", f"{begin:.3f}", "-t", f"{end - begin + pad:.3f}", "-i", str(source),
            "-vn", "-ac", "1", "-acodec", "pcm_s16le", str(out),
        ],
        timeout=120,
    )
    if result.returncode != 0 or not out.exists():
        raise AudioClipError(f"ffmpeg could not cut the clip: {result.stderr.strip()[-300:]}")
    return out
