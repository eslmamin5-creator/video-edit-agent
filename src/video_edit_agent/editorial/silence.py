"""Silence analysis via ffmpeg's silencedetect filter (spec section 11)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from video_edit_agent.core.media import run


@dataclass
class SilenceSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silence(audio_path: Path, noise_db: float = -35.0, min_duration: float = 0.35) -> list[SilenceSpan]:
    result = run(
        [
            "ffmpeg", "-i", str(audio_path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f", "null", "-",
        ],
        timeout=300,
    )
    log = result.stderr
    starts = [float(m.group(1)) for m in _START_RE.finditer(log)]
    ends = [float(m.group(1)) for m in _END_RE.finditer(log)]
    spans = []
    for s, e in zip(starts, ends):
        if e > s:
            spans.append(SilenceSpan(start=s, end=e))
    return spans
