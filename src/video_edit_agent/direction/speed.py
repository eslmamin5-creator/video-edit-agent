"""Speed is never automatic.

Pacing priority (the order matters):

    1. remove genuine dead air, where editorially safe
    2. preserve the speaker's natural cadence
    3. create visual rhythm through framing, cuts and replacements
    4. change playback speed ONLY when explicitly justified

A speed change is therefore an explicit, reviewable event with a stated
justification; the planner below returns no events unless it is handed some,
and nothing multiplies the talking head by 1.1x/1.15x on its own.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator

PACING_PRIORITY = ("remove_dead_air", "preserve_cadence", "visual_rhythm", "explicit_speed_change")
MIN_FACTOR, MAX_FACTOR = 0.85, 1.25  # outside this the voice stops sounding natural


class SpeedEvent(BaseModel):
    start: float
    end: float
    factor: float
    justification: str  # required: why this beat, and only this beat, is retimed
    approved: bool = False  # the reviewer accepted it; unapproved events never render

    @field_validator("justification")
    @classmethod
    def _justified(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a speed change needs an explicit justification")
        return value.strip()

    @field_validator("factor")
    @classmethod
    def _natural(cls, value: float) -> float:
        if not MIN_FACTOR <= value <= MAX_FACTOR:
            raise ValueError(f"speed factor {value} would make the voice unnatural ({MIN_FACTOR}-{MAX_FACTOR})")
        return value


def plan_speed(justified: list[SpeedEvent] | None = None) -> list[SpeedEvent]:
    """The speed events of a plan: exactly the justified ones, or none. Never derived from the content."""
    return sorted(justified or [], key=lambda e: e.start)


def renderable(events: list[SpeedEvent]) -> list[SpeedEvent]:
    """Only events the reviewer approved may change playback."""
    return [e for e in events if e.approved]


__all__ = ["MAX_FACTOR", "MIN_FACTOR", "PACING_PRIORITY", "SpeedEvent", "plan_speed", "renderable"]
