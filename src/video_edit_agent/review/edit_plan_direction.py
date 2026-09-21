"""Bridges the visual director into the reviewable edit plan.

`build_directed_plan` turns a `DirectionResult` (one decision per semantic beat)
into `EditPlanSlot`s carrying the full per-beat decision (speaker visibility,
camera, transition, caption behaviour, sound intent and what will actually be
heard), keeping the slots that were settled earlier exactly as they were.
`sync_sound` recomputes what each slot's sound will do after the user changes an
intent or the profile, always through the deterministic resolver.

Nothing here generates anything, renders anything or sets `ready_for_final_render`.
Nothing in this module is brand- or project-specific.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import EDL, Transcript
from video_edit_agent.direction.camera import CameraMove
from video_edit_agent.direction.director import Beat, DirectionResult, sound_event_for
from video_edit_agent.direction.transitions import TransitionStyle
from video_edit_agent.direction.vocabulary import (
    STRONG_PRIMARY,
    default_speaker_visible,
    to_plan_treatment,
    treatment_class,
)
from video_edit_agent.direction.vocabulary import (
    TEXT_TREATMENTS as VOCAB_TEXT,
)
from video_edit_agent.review import edit_plan as ep
from video_edit_agent.review.edit_plan import EditPlan, EditPlanSlot, SlotStatus
from video_edit_agent.sound.intent import DEFAULT_IMPORTANCE, EventType, parse_intent
from video_edit_agent.sound.planner import VisualEvent, plan_sound, sfx_availability
from video_edit_agent.sound.profile import get_profile
from video_edit_agent.sound.registry import SfxRegistry


def build_directed_plan(
    beats: list[Beat],
    result: DirectionResult,
    transcript: Transcript,
    edl: EDL,
    *,
    kept: list[EditPlanSlot] | None = None,
    sound_profile: str = "none",
    registry: SfxRegistry | None = None,
    caption_mode: str = "unchanged",
) -> EditPlan:
    """The reviewable plan for `result`. `kept` slots (settled decisions from earlier review
    steps) are carried over unchanged; a beat whose decision is a plain static speaker shot
    with nothing to hear is settled ("no special treatment"), everything else is numbered."""
    slots: list[EditPlanSlot] = [s.model_copy(deep=True) for s in (kept or [])]
    for beat, d in zip(sorted(beats, key=lambda b: b.start), result.decisions, strict=True):
        segs = ep._segments_for(d.start, d.end, edl, transcript)
        context = " ".join(transcript.segments[n - 1].text.strip() for n in segs)
        treatment = to_plan_treatment(d.treatment)
        plain = d.treatment == "speaker_static" and (d.sound is None or d.sound.status == "none")
        options = ep.keyword_options(context) if d.treatment in VOCAB_TEXT else []
        slot = EditPlanSlot(
            timeline_start=round(d.start, 3), timeline_end=round(d.end, 3), segments=segs, spoken_context=context,
            recommended=treatment, treatment=treatment, reason=d.reason, visual_concept=beat.visual_concept,
            fallback="stay_on_speaker" if d.needs_asset or d.needs_generation else None,
            alternatives=[a for a in dict.fromkeys(to_plan_treatment(x) for x in d.alternatives) if a != treatment],
            status=SlotStatus.APPROVED if plain else SlotStatus.PENDING_REVIEW,
            settled_reason="no special treatment: the speaker is the best visual" if plain else None,
            text_options=options,
            directed=True, speaker_visible=d.speaker_visible,
            camera=d.camera.value if d.klass != "replacement" else "n/a",
            transition=d.transition.effective.value,
            transition_requested=d.transition.style.value if d.transition.stylized else None,
            caption_mode=caption_mode, caption_behavior=d.caption_behavior,
            sound_intent=d.sound.intent if d.sound else "none",
            sound_event=d.sound.event_type if d.sound else None,
            sound_locked=beat.sound_locked, sound_importance=beat.importance, beat_kind=d.beat_kind, direction_notes=list(d.notes),
        )
        slots.append(slot)
    slots.sort(key=lambda s: (s.timeline_start, s.timeline_end))
    number = 0
    for slot in slots:
        if slot.settled_reason is None:
            number += 1
            slot.number = number
        else:
            slot.number = 0
    plan = EditPlan(slots=slots, sound_profile=sound_profile)
    return sync_sound(plan, registry)


def _event_for(slot: EditPlanSlot, ident: str) -> VisualEvent | None:
    if not slot.directed or slot.sound_event is None:
        return None
    try:
        etype = EventType(slot.sound_event)
    except ValueError:
        return None
    # A slot that was directed keeps its event type; the type fixes where the sound sits.
    replaced = etype is EventType.REPLACEMENT_IN
    duration = (slot.timeline_end - slot.timeline_start) if replaced else 0.42 if etype in _CAMERA_EVENTS else 0.6
    return VisualEvent(
        id=ident, type=etype, start=slot.timeline_start, duration=min(duration, slot.timeline_end - slot.timeline_start),
        intent=parse_intent(slot.sound_intent), user_locked=slot.sound_locked,
        importance=(max(slot.sound_importance, DEFAULT_IMPORTANCE[etype]) if slot.sound_importance is not None else None),
    )


_CAMERA_EVENTS = {EventType.PUNCH_IN, EventType.PUNCH_OUT, EventType.SLOW_PUSH, EventType.REFRAME, EventType.RESET_TO_BASE}


def sync_sound(plan: EditPlan, registry: SfxRegistry | None = None) -> EditPlan:
    """Recomputes every directed slot's sound outcome from its intent, the plan's profile and the
    registry (deterministic; a missing registry/asset means the slot stays silent, never an error)."""
    profile = get_profile(plan.sound_profile)
    reg = registry or SfxRegistry()
    events: dict[str, tuple[EditPlanSlot, VisualEvent]] = {}
    for i, slot in enumerate(plan.slots):
        ev = _event_for(slot, f"slot{i}")
        if ev is not None:
            events[ev.id] = (slot, ev)
    for slot in plan.slots:
        if slot.directed and slot.sound_event is None:
            slot.sound_status = "none"
            slot.sfx_availability = "none" if slot.sound_intent == "none" else "none (nothing moves on this beat)"
    decisions = plan_sound([ev for _, ev in events.values()], profile, reg)
    for d in decisions:
        slot, _ = events[d.event_id]
        slot.sound_status = d.status
        slot.sfx_availability = sfx_availability(d)
    return plan


def set_slot_sound(plan: EditPlan, slot: EditPlanSlot, intent: str, registry: SfxRegistry | None = None) -> EditPlanSlot:
    """The user's sound choice for one slot, then the plan's sound outcomes are recomputed."""
    ep.set_sound_intent(slot, intent)
    sync_sound(plan, registry)
    return slot


_FROM_PLAN = {"stay_on_speaker": "speaker_static", "local_broll": "real_broll", "generated_broll": "generated_visual"}


def retarget(plan: EditPlan, slot: EditPlanSlot, registry: SfxRegistry | None = None) -> EditPlanSlot:
    """After the reviewer changed a directed slot's treatment: speaker visibility, camera, caption
    behaviour and the sound event follow the new treatment (the reviewer's own sound choice stays)."""
    if not slot.directed:
        return slot
    vocab = _FROM_PLAN.get(slot.treatment, slot.treatment)
    klass = treatment_class(vocab)
    slot.speaker_visible = default_speaker_visible(vocab)
    if klass == "replacement":
        slot.camera = "n/a"
    elif vocab == "punch_in":
        slot.camera = "punch_in"
    elif klass == "speaker":
        slot.camera = "static"
    slot.caption_behavior = "reduced" if vocab in STRONG_PRIMARY else "normal"
    move = CameraMove.PUNCH_IN if slot.camera == "punch_in" else CameraMove.STATIC
    ev = sound_event_for(vocab, move, slot.speaker_visible, slot.timeline_start, slot.timeline_end, "x",
                         importance=slot.sound_importance, locked=slot.sound_locked)
    slot.sound_event = ev.type.value if ev else None
    if ev is None and not slot.sound_locked:
        slot.sound_intent = "none"
    sync_sound(plan, registry)
    return slot


def transition_label(slot: EditPlanSlot) -> str:
    if slot.transition_requested and slot.transition_requested != TransitionStyle.DIRECT_CUT.value:
        return f"{slot.transition} (planned {slot.transition_requested} not executable yet)"
    return slot.transition


__all__ = ["build_directed_plan", "retarget", "set_slot_sound", "sync_sound", "transition_label"]
