"""Media probing and safe subprocess helpers.

All external process calls in this project go through `run()` here, using
argument arrays (never shell strings) to avoid shell injection (spec section
47), and all inputs treated as read-only — source media is never modified.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run an external command as an argument array. Never pass shell=True."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str
    audio_codec: str | None


def probe(path: Path) -> MediaInfo:
    """Run ffprobe on a media file and return structured info."""
    result = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        timeout=30,
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise MediaError(f"No video stream found in {path}")

    fps_raw = v.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return MediaInfo(
        path=path,
        duration=float(fmt.get("duration", v.get("duration", 0.0)) or 0.0),
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        fps=fps,
        has_audio=a is not None,
        video_codec=v.get("codec_name", "unknown"),
        audio_codec=a.get("codec_name") if a else None,
    )


def content_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Cheap content-based hash for caching (spec section 34): file size +
    sampled chunks, not a full read of large video files."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(chunk_size))
        if size > chunk_size:
            f.seek(max(0, size // 2))
            h.update(f.read(chunk_size))
            f.seek(max(0, size - chunk_size))
            h.update(f.read(chunk_size))
    return h.hexdigest()[:16]


def extract_audio(video_path: Path, out_path: Path, sample_rate: int = 16000) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
            str(out_path),
        ],
        timeout=600,
    )
    if result.returncode != 0:
        raise MediaError(f"Audio extraction failed: {result.stderr.strip()[-2000:]}")
    return out_path
