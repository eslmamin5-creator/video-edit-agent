"""Post-render compatibility QA for standard MP4 exports.

A render is only "successful" if the produced file is H.264/avc1/yuv420p + AAC 48k stereo,
level within the configured ceiling, faststart-friendly, and decodes end to end."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from video_edit_agent.core.media import run
from video_edit_agent.render.export import ExportPreset

_COMPATIBLE_PROFILES = {"Baseline", "Constrained Baseline", "Main", "High"}


class ExportCompatibilityError(RuntimeError):
    pass


def _probe(path: Path) -> dict:
    result = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=60)
    if result.returncode != 0:
        raise ExportCompatibilityError(f"ffprobe failed:\n{result.stderr.strip()[-1000:]}")
    return json.loads(result.stdout or "{}")


def _level_int(preset: ExportPreset) -> int:
    major, _, minor = preset.h264_level.partition(".")
    return int(major) * 10 + int(minor or 0)


def moov_before_mdat(path: Path) -> bool | None:
    """True when the top-level `moov` box precedes `mdat` (faststart); None if undeterminable."""
    try:
        with path.open("rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    return None
                size, kind = struct.unpack(">I4s", header)
                if kind == b"moov":
                    return True
                if kind == b"mdat":
                    return False
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                    f.seek(size - 16, 1)
                elif size == 0:
                    return None
                else:
                    f.seek(size - 8, 1)
    except (OSError, struct.error):
        return None


def check_metadata(path: Path, preset: ExportPreset) -> list[str]:
    data = _probe(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    problems: list[str] = []
    if "mp4" not in data.get("format", {}).get("format_name", ""):
        problems.append(f"container is {data.get('format', {}).get('format_name')!r}, expected mp4")
    if video is None:
        return problems + ["no video stream"]
    if video.get("codec_name") != "h264":
        problems.append(f"video codec {video.get('codec_name')!r}, expected h264")
    if video.get("codec_tag_string") != "avc1":
        problems.append(f"video codec tag {video.get('codec_tag_string')!r}, expected avc1")
    if video.get("pix_fmt") != preset.pix_fmt:
        problems.append(f"pix_fmt {video.get('pix_fmt')!r}, expected {preset.pix_fmt}")
    if video.get("profile") not in _COMPATIBLE_PROFILES:
        problems.append(f"H.264 profile {video.get('profile')!r} is not broadly playable")
    if int(video.get("level") or 0) > _level_int(preset):
        problems.append(f"H.264 level {video.get('level')} exceeds ceiling {_level_int(preset)}")
    if (video.get("width"), video.get("height")) != (preset.width, preset.height):
        problems.append(f"resolution {video.get('width')}x{video.get('height')}, expected {preset.width}x{preset.height}")
    for key in ("r_frame_rate", "avg_frame_rate"):
        num, _, den = str(video.get(key, "0/1")).partition("/")
        fps = float(num) / float(den) if float(den or 1) else 0.0
        if abs(fps - preset.fps) > 0.01:
            problems.append(f"{key} {video.get(key)}, expected constant {preset.fps:g} fps")
    if audio is None:
        problems.append("no audio stream")
    else:
        if audio.get("codec_name") != "aac":
            problems.append(f"audio codec {audio.get('codec_name')!r}, expected aac")
        if int(audio.get("sample_rate") or 0) != preset.audio_sample_rate:
            problems.append(f"audio sample_rate {audio.get('sample_rate')}, expected {preset.audio_sample_rate}")
        if int(audio.get("channels") or 0) != preset.audio_channels:
            problems.append(f"audio channels {audio.get('channels')}, expected {preset.audio_channels}")
    if moov_before_mdat(path) is False:
        problems.append("moov atom is after mdat (faststart missing)")
    return problems


def check_decode(path: Path) -> list[str]:
    """Full decode plus first / middle / last frame decodes."""
    problems: list[str] = []
    full = run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], timeout=1800)
    if full.returncode != 0 or full.stderr.strip():
        problems.append(f"full decode failed: {full.stderr.strip()[-500:] or 'nonzero exit'}")
    duration = float(_probe(path).get("format", {}).get("duration") or 0)
    seeks = {"first": ["-ss", "0"], "middle": ["-ss", f"{duration / 2:.3f}"], "last": ["-sseof", "-0.2"]}
    for label, seek in seeks.items():
        frame = run(["ffmpeg", "-v", "error", *seek, "-i", str(path), "-frames:v", "1", "-f", "null", "-"], timeout=120)
        if frame.returncode != 0 or frame.stderr.strip():
            problems.append(f"{label} frame failed to decode: {frame.stderr.strip()[-300:]}")
    return problems


def assert_export_compatible(path: Path, preset: ExportPreset) -> None:
    """Raises ExportCompatibilityError unless the file passes the standard-MP4 checks.
    A preset with `standard=False` (explicit advanced export) skips them."""
    if not preset.standard:
        return
    problems = check_metadata(path, preset) + check_decode(path)
    if problems:
        raise ExportCompatibilityError(f"{path.name} is not a compatible standard MP4:\n- " + "\n- ".join(problems))
