"""Transition grammar: the default is a hard, invisible cut.

A stylized transition (whip / zoom / blur / flash / slide) is only allowed when
it supports one of four reasons, and only in the styles that fit that reason:

    motion_continuity   the outgoing and incoming shots move the same way
    semantic_change     a new idea begins
    spatial_continuity  the same space continues (slide/zoom through it)
    deliberate_reveal   something is meant to be discovered

Stylized transitions are also rate-limited. The selection is constrained on
purpose: a request that does not fit its reason simply gets the direct cut.

The renderer currently executes only hard cuts (and the crossfade the Assembler
already uses); `STYLIZED_EXECUTABLE` is empty, so a planned stylized transition
is shown to the reviewer together with the direct cut that actually plays.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class TransitionStyle(str, Enum):
    DIRECT_CUT = "direct_cut"
    WHIP = "whip"
    ZOOM = "zoom"
    BLUR = "blur"
    FLASH = "flash"
    SLIDE = "slide"


class TransitionReason(str, Enum):
    MOTION_CONTINUITY = "motion_continuity"
    SEMANTIC_CHANGE = "semantic_change"
    SPATIAL_CONTINUITY = "spatial_continuity"
    DELIBERATE_REVEAL = "deliberate_reveal"


# reason -> allowed styles, the first being the default for that reason.
ALLOWED: dict[TransitionReason, tuple[TransitionStyle, ...]] = {
    TransitionReason.MOTION_CONTINUITY: (TransitionStyle.WHIP, TransitionStyle.ZOOM),
    TransitionReason.SEMANTIC_CHANGE: (TransitionStyle.BLUR, TransitionStyle.SLIDE),
    TransitionReason.SPATIAL_CONTINUITY: (TransitionStyle.SLIDE, TransitionStyle.ZOOM),
    TransitionReason.DELIBERATE_REVEAL: (TransitionStyle.FLASH, TransitionStyle.BLUR),
}
DURATION_S = {
    TransitionStyle.DIRECT_CUT: 0.0, TransitionStyle.WHIP: 0.25, TransitionStyle.ZOOM: 0.3,
    TransitionStyle.BLUR: 0.3, TransitionStyle.FLASH: 0.15, TransitionStyle.SLIDE: 0.3,
}
MIN_GAP_BETWEEN_STYLIZED_S = 8.0
STYLIZED_EXECUTABLE: frozenset[TransitionStyle] = frozenset()


class TransitionChoice(BaseModel):
    style: TransitionStyle = TransitionStyle.DIRECT_CUT
    reason: TransitionReason | None = None
    duration_s: float = 0.0
    note: str = ""

    @property
    def stylized(self) -> bool:
        return self.style is not TransitionStyle.DIRECT_CUT

    @property
    def effective(self) -> TransitionStyle:
        """What the renderer will actually do."""
        return self.style if self.style in STYLIZED_EXECUTABLE or not self.stylized else TransitionStyle.DIRECT_CUT


DIRECT = TransitionChoice(note="default: hard, invisible cut")


def select_transition(
    reason: TransitionReason | str | None = None,
    preferred: TransitionStyle | str | None = None,
    *,
    at: float | None = None,
    previous_stylized_at: float | None = None,
) -> TransitionChoice:
    """The transition for a boundary. No justification, an unknown reason, a style that does not
    fit the reason, or a stylized transition too soon after the last one -> the direct cut."""
    if reason is None:
        return DIRECT.model_copy()
    try:
        why = TransitionReason(reason)
        style = TransitionStyle(preferred) if preferred else ALLOWED[why][0]
    except ValueError:
        return TransitionChoice(note="unknown reason or style: direct cut")
    if style is TransitionStyle.DIRECT_CUT:
        return TransitionChoice(reason=why, note="direct cut")
    if style not in ALLOWED[why]:
        return TransitionChoice(note=f"{style.value} does not support {why.value}: direct cut")
    if at is not None and previous_stylized_at is not None and at - previous_stylized_at < MIN_GAP_BETWEEN_STYLIZED_S:
        return TransitionChoice(note="another stylized transition is too close: direct cut")
    return TransitionChoice(style=style, reason=why, duration_s=DURATION_S[style])


__all__ = [
    "ALLOWED", "DIRECT", "MIN_GAP_BETWEEN_STYLIZED_S", "STYLIZED_EXECUTABLE", "TransitionChoice",
    "TransitionReason", "TransitionStyle", "select_transition",
]
