"""Pacing heuristics (spec section 11): flag long uninterrupted clips as
candidates for a cutaway/B-roll or speed adjustment. Deliberately simple —
this only flags, the planner/B-roll modules decide what to do about it."""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.core.schemas import EDLClip


@dataclass
class PacingFlag:
    clip_index: int
    reason: str


def flag_long_clips(clips: list[EDLClip], max_seconds: float = 8.0) -> list[PacingFlag]:
    return [
        PacingFlag(clip_index=i, reason=f"clip runs {c.duration:.1f}s (> {max_seconds:.0f}s) — consider a cutaway")
        for i, c in enumerate(clips)
        if c.duration > max_seconds
    ]
