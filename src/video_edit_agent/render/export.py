"""Export presets for common short-form platforms."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportPreset:
    width: int
    height: int
    fps: float
    video_bitrate: str
    audio_bitrate: str = "160k"


PRESETS: dict[str, ExportPreset] = {
    "reel": ExportPreset(1080, 1920, 30, "8M"),
    "tiktok": ExportPreset(1080, 1920, 30, "8M"),
    "shorts": ExportPreset(1080, 1920, 30, "8M"),
    "square": ExportPreset(1080, 1080, 30, "6M"),
    "landscape": ExportPreset(1920, 1080, 30, "10M"),
}


def resolve_preset(name: str) -> ExportPreset:
    return PRESETS.get(name, PRESETS["reel"])
