"""Export presets and the single source of truth for the standard MP4 encode.

Every final-path encode (main render, card clips, joins) builds its codec flags here, and
`render.export_validation` checks the result against the same values. `standard=False` is
the only way to opt out (explicit advanced export)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportPreset:
    width: int
    height: int
    fps: float
    video_bitrate: str
    audio_bitrate: str = "192k"
    standard: bool = True
    h264_profile: str = "high"
    h264_level: str = "4.1"
    pix_fmt: str = "yuv420p"
    audio_sample_rate: int = 48000
    audio_channels: int = 2


PRESETS: dict[str, ExportPreset] = {
    "reel": ExportPreset(1080, 1920, 30, "8M"),
    "tiktok": ExportPreset(1080, 1920, 30, "8M"),
    "shorts": ExportPreset(1080, 1920, 30, "8M"),
    "square": ExportPreset(1080, 1080, 30, "6M"),
    "landscape": ExportPreset(1920, 1080, 30, "10M"),
}

DEFAULT_PRESET = PRESETS["reel"]


def resolve_preset(name: str) -> ExportPreset:
    return PRESETS.get(name, PRESETS["reel"])


def video_filter_tail(preset: ExportPreset | None = None) -> str:
    """Final graph stage: BT.709 limited-range conversion to the standard pixel format."""
    preset = preset or DEFAULT_PRESET
    if not preset.standard:
        return ""
    return f"scale=out_color_matrix=bt709:out_range=tv,format={preset.pix_fmt}"


def video_encode_args(preset: ExportPreset | None = None, *, crf: int = 20) -> list[str]:
    preset = preset or DEFAULT_PRESET
    args = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"]
    if preset.standard:
        args += [
            "-profile:v", preset.h264_profile, "-level:v", preset.h264_level,
            "-pix_fmt", preset.pix_fmt, "-tag:v", "avc1",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        ]
    return args


def audio_encode_args(preset: ExportPreset | None = None) -> list[str]:
    preset = preset or DEFAULT_PRESET
    args = ["-c:a", "aac", "-b:a", preset.audio_bitrate]
    if preset.standard:
        args += ["-ar", str(preset.audio_sample_rate), "-ac", str(preset.audio_channels)]
    return args
