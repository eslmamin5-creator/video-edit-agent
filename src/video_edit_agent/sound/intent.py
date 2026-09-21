"""Sound intent: WHY a visual event might carry a sound, never WHICH sound.

    Visual/Edit Event -> Sound Intent -> Sound Profile -> Small SFX Registry
      -> Deterministic Asset Selection -> MasterTimeline

The director (human or AI) only ever picks an intent from this closed
vocabulary; a deterministic resolver maps `profile + intent + event type` to an
asset id (`sound.resolver`). Nobody browses sounds or names files.

`none` is the default everywhere: silence is a valid, usually correct, choice.
Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum


class SoundIntent(str, Enum):
    NONE = "none"
    SUBTLE_MOTION = "subtle_motion"  # a soft air/whoosh under a camera move
    TRANSITION = "transition"  # a sweep that leads into a scene change
    ACCENT = "accent"  # a small tick/hit on a key reveal
    IMPACT = "impact"  # a soft low hit as a reveal settles


DEFAULT_INTENT = SoundIntent.NONE

# Ordered from quietest to loudest: profiles cap the loudest intent they allow.
INTENT_ORDER = (
    SoundIntent.NONE, SoundIntent.SUBTLE_MOTION, SoundIntent.TRANSITION, SoundIntent.ACCENT, SoundIntent.IMPACT,
)


class EventType(str, Enum):
    """The visual/edit events that may carry a sound."""

    PUNCH_IN = "punch_in"
    PUNCH_OUT = "punch_out"
    SLOW_PUSH = "slow_push"
    REFRAME = "reframe"
    RESET_TO_BASE = "reset_to_base"
    REPLACEMENT_IN = "replacement_in"  # the speaker is replaced by another visual
    REPLACEMENT_OUT = "replacement_out"  # back to the speaker
    KEY_REVEAL = "key_reveal"  # a key word/idea lands on screen
    TEXT_REVEAL = "text_reveal"
    CAPTION_HIGHLIGHT = "caption_highlight"


# What each event carries when nobody said otherwise (spec: none/subtle for
# camera moves, transition for scene replacement, accent for a key reveal).
DEFAULT_EVENT_INTENT: dict[EventType, SoundIntent] = {
    EventType.PUNCH_IN: SoundIntent.SUBTLE_MOTION,
    EventType.PUNCH_OUT: SoundIntent.NONE,
    EventType.SLOW_PUSH: SoundIntent.NONE,
    EventType.REFRAME: SoundIntent.NONE,
    EventType.RESET_TO_BASE: SoundIntent.NONE,
    EventType.REPLACEMENT_IN: SoundIntent.TRANSITION,
    EventType.REPLACEMENT_OUT: SoundIntent.NONE,
    EventType.KEY_REVEAL: SoundIntent.ACCENT,
    EventType.TEXT_REVEAL: SoundIntent.NONE,
    EventType.CAPTION_HIGHLIGHT: SoundIntent.NONE,
}

# How much an event matters semantically (0..1); a profile's importance
# threshold decides whether a decorative sound is worth its place.
DEFAULT_IMPORTANCE: dict[EventType, float] = {
    EventType.PUNCH_IN: 0.45,
    EventType.PUNCH_OUT: 0.2,
    EventType.SLOW_PUSH: 0.3,
    EventType.REFRAME: 0.3,
    EventType.RESET_TO_BASE: 0.15,
    EventType.REPLACEMENT_IN: 0.8,
    EventType.REPLACEMENT_OUT: 0.4,
    EventType.KEY_REVEAL: 0.8,
    EventType.TEXT_REVEAL: 0.55,
    EventType.CAPTION_HIGHLIGHT: 0.1,
}


def default_intent(event: EventType) -> SoundIntent:
    return DEFAULT_EVENT_INTENT.get(event, DEFAULT_INTENT)


def parse_intent(value: str | SoundIntent | None) -> SoundIntent:
    """An intent from user/AI text; anything unrecognised is `none` (never a guess)."""
    if isinstance(value, SoundIntent):
        return value
    try:
        return SoundIntent((value or "none").strip().lower())
    except ValueError:
        return DEFAULT_INTENT


__all__ = [
    "DEFAULT_EVENT_INTENT", "DEFAULT_IMPORTANCE", "DEFAULT_INTENT", "INTENT_ORDER", "EventType", "SoundIntent",
    "default_intent", "parse_intent",
]
