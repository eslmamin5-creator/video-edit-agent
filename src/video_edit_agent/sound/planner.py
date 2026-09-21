"""Turns visual/edit events into a sparse, honest sound plan.

For every event the plan records what happened to its sound and why:

    none         no sound was asked for (the default)
    suppressed   asked for, but density rules dropped it (spacing, repeat,
                 stacking, below the importance threshold, per-minute cap)
    unavailable  asked for and allowed, but no asset exists -> plays nothing
    scheduled    resolved to an asset and aligned to the event's own timing

Sound timing is NOT a second timeline: `add_to_timeline` writes the scheduled
sounds as `TrackType.SFX` items on the MasterTimeline.

An event whose intent the user set explicitly (`user_locked`) skips the
importance threshold, spacing and cooldown (the user asked for it) but never
stacks a second sound onto the same beat.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from video_edit_agent.core.timeline import (
    MasterTimeline,
    Provenance,
    ProviderKind,
    TimelineItem,
    TrackType,
)
from video_edit_agent.sound import resolver
from video_edit_agent.sound.intent import DEFAULT_IMPORTANCE, EventType, SoundIntent, default_intent
from video_edit_agent.sound.mix import sfx_gain_db
from video_edit_agent.sound.profile import SoundProfile
from video_edit_agent.sound.registry import SfxRegistry

STATUS_NONE = "none"
STATUS_SUPPRESSED = "suppressed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_SCHEDULED = "scheduled"

# How a sound sits on its event (fractions of the sound, seconds otherwise).
_SWEEP_LEAD = 0.75  # a transition sweep spends 75% of its length leading INTO the cut
_MAX_SWEEP_S = 0.8


@dataclass(frozen=True)
class VisualEvent:
    """A visual/edit event on the MasterTimeline (times are timeline seconds)."""

    id: str
    type: EventType
    start: float
    duration: float = 0.0
    intent: SoundIntent | None = None  # None = the event's default intent
    importance: float | None = None  # None = the event type's default
    user_locked: bool = False  # the user chose this intent explicitly


class SoundDecision(BaseModel):
    event_id: str
    event_type: str
    intent: str
    status: str
    reason: str | None = None
    asset_id: str | None = None
    start: float | None = None  # timeline second the sound begins
    duration: float | None = None  # how long it plays (trimmed to the motion when needed)
    gain_db: float | None = None
    aligned_to: str | None = None


def align(intent: SoundIntent, event: VisualEvent, asset_duration: float) -> tuple[float, float, str]:
    """(start, duration, what it is aligned to) for a sound of `asset_duration` on `event`.

    subtle_motion  starts with the motion and never outlasts it (a 420 ms
                   punch-in gets a whoosh of at most 420 ms)
    transition     a sweep that ends just after the cut: it leads INTO the scene
    impact         lands where the reveal settles (the end of the event)
    accent         lands on the event's first frame"""
    if intent is SoundIntent.SUBTLE_MOTION:
        length = min(asset_duration, event.duration) if event.duration > 0 else asset_duration
        return event.start, round(length, 3), "motion_interval"
    if intent is SoundIntent.TRANSITION:
        length = min(asset_duration, _MAX_SWEEP_S)
        return round(max(0.0, event.start - length * _SWEEP_LEAD), 3), round(length, 3), "cut_lead_in"
    if intent is SoundIntent.IMPACT:
        settle = event.start + event.duration
        return round(settle, 3), round(asset_duration, 3), "settle"
    return event.start, round(asset_duration, 3), "event_start"


def plan_sound(
    events: list[VisualEvent], profile: SoundProfile, registry: SfxRegistry,
) -> list[SoundDecision]:
    """One decision per event, in time order. Deterministic: the same events, profile
    and registry always produce the same plan."""
    decisions: list[SoundDecision] = []
    scheduled: list[tuple[float, str]] = []  # (start, event type) of sounds already placed
    for event in sorted(events, key=lambda e: (e.start, e.id)):
        intent = event.intent if event.intent is not None else default_intent(event.type)
        base = {"event_id": event.id, "event_type": event.type.value, "intent": intent.value}
        if intent is SoundIntent.NONE:
            decisions.append(SoundDecision(**base, status=STATUS_NONE, reason=resolver.REASON_INTENT_NONE))
            continue
        importance = event.importance if event.importance is not None else DEFAULT_IMPORTANCE.get(event.type, 0.0)
        drop = _suppression(event, importance, profile, scheduled)
        if drop:
            decisions.append(SoundDecision(**base, status=STATUS_SUPPRESSED, reason=drop))
            continue
        resolution = resolver.resolve(profile, intent, event.type, registry)
        if not resolution.available:
            status = STATUS_NONE if resolution.status == resolver.NOT_WANTED else STATUS_UNAVAILABLE
            decisions.append(SoundDecision(**base, status=status, reason=resolution.reason))
            continue
        asset = registry.asset(resolution.asset_id)
        assert asset is not None
        start, length, anchor = align(intent, event, asset.duration_s)
        scheduled.append((start, event.type.value))
        decisions.append(SoundDecision(
            **base, status=STATUS_SCHEDULED, asset_id=asset.id, start=start, duration=length,
            gain_db=sfx_gain_db(profile, intent), aligned_to=anchor,
        ))
    return decisions


def _suppression(
    event: VisualEvent, importance: float, profile: SoundProfile, scheduled: list[tuple[float, str]],
) -> str | None:
    """Why this sound must not play, or None. Only meaningful for profiles that allow sound."""
    if profile.silent:
        return None  # the resolver reports it as not wanted
    if any(abs(event.start - s) < profile.stack_window_s for s, _ in scheduled):
        return "stacked_on_the_same_beat"
    if event.user_locked:
        return None
    if importance < profile.importance_threshold:
        return "below_importance_threshold"
    if any(kind == event.type.value and event.start - s < profile.same_event_cooldown_s for s, kind in scheduled):
        return "repeated_event"
    if any(0 <= event.start - s < profile.min_spacing_s for s, _ in scheduled):
        return "min_spacing"
    if sum(1 for s, _ in scheduled if 0 <= event.start - s < 60.0) >= profile.max_per_minute:
        return "density_cap"
    return None


def sfx_availability(decision: SoundDecision) -> str:
    """A short honest label for the review: what will actually be heard."""
    if decision.status == STATUS_SCHEDULED:
        return "available"
    if decision.status == STATUS_UNAVAILABLE:
        return "unavailable -> fallback to none"
    if decision.status == STATUS_SUPPRESSED:
        return f"suppressed ({decision.reason}) -> none"
    return "none"


def add_to_timeline(timeline: MasterTimeline, decisions: list[SoundDecision], registry: SfxRegistry) -> MasterTimeline:
    """Writes the scheduled sounds onto `timeline` as SFX items (idempotent: earlier
    SFX items created here are replaced). Sounds that are not scheduled add nothing."""
    timeline.items = [i for i in timeline.items if not (i.type is TrackType.SFX and i.metadata.get("origin") == "sound_plan")]
    for d in decisions:
        if d.status != STATUS_SCHEDULED or d.asset_id is None or d.start is None or d.duration is None:
            continue
        asset = registry.asset(d.asset_id)
        path = registry.path(d.asset_id)
        timeline.items.append(TimelineItem(
            id=f"sfx_{d.event_id}", type=TrackType.SFX, start=d.start, duration=d.duration,
            source=str(path) if path else "", out_point=d.duration, layer=0,
            reason=f"{d.intent} on {d.event_type}",
            provenance=Provenance(kind=ProviderKind.LOCAL_LIBRARY, detail=d.asset_id),
            metadata={
                "origin": "sound_plan", "intent": d.intent, "event_id": d.event_id, "event_type": d.event_type,
                "gain_db": d.gain_db, "aligned_to": d.aligned_to, "category": asset.category.value if asset else None,
            },
        ))
    return timeline


__all__ = [
    "STATUS_NONE", "STATUS_SCHEDULED", "STATUS_SUPPRESSED", "STATUS_UNAVAILABLE", "SoundDecision", "VisualEvent",
    "add_to_timeline", "align", "plan_sound", "sfx_availability",
]
