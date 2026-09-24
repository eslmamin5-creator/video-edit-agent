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
from video_edit_agent.direction.director import (
    Beat,
    DirectionResult,
    VisualDecision,
    sound_event_for,
)
from video_edit_agent.direction.hierarchy import build_hierarchy
from video_edit_agent.direction.semantic_beats import confidence_label
from video_edit_agent.direction.transitions import TransitionStyle
from video_edit_agent.direction.vocabulary import (
    TEXT_TREATMENTS as VOCAB_TEXT,
)
from video_edit_agent.direction.vocabulary import (
    default_speaker_visible,
    to_plan_treatment,
    treatment_class,
)
from video_edit_agent.review import edit_plan as ep
from video_edit_agent.review.edit_plan import BeatPlanning, EditPlan, EditPlanSlot, SlotStatus
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
    decided: list[EditPlanSlot] | None = None,
) -> EditPlan:
    """The reviewable plan for `result`. `kept` slots (settled decisions from earlier review
    steps) are carried over unchanged; a beat whose decision is a plain static speaker shot
    with nothing to hear is settled ("no special treatment"), everything else is numbered.

    `decided` are slots the reviewer already answered (approved / changed / rejected). Each is
    adopted by the beat it overlaps, so an answer survives the re-plan; when the new plan
    disagrees with it (another treatment, or a window that no longer lines up) the slot goes back
    to PENDING_REVIEW as `review_required` with the conflict spelled out, never silently
    overwritten and never auto-approved."""
    slots: list[EditPlanSlot] = [s.model_copy(deep=True) for s in (kept or [])]
    answers = [s for s in (decided or []) if s.status is not SlotStatus.PENDING_REVIEW]
    used: set[int] = set()
    for beat, d in zip(sorted(beats, key=lambda b: b.start), result.decisions, strict=True):
        segs = ep._segments_for(d.start, d.end, edl, transcript)
        context = " ".join(transcript.segments[n - 1].text.strip() for n in segs)
        treatment = to_plan_treatment(d.treatment)
        answer = _match(answers, used, d.start, d.end)
        plain = answer is None and d.treatment == "speaker_static" and (d.sound is None or d.sound.status == "none")
        options = ep.keyword_options(context) if d.treatment in VOCAB_TEXT else []
        slot = EditPlanSlot(
            timeline_start=round(d.start, 3), timeline_end=round(d.end, 3), segments=segs, spoken_context=context,
            recommended=treatment, treatment=treatment, reason=d.reason, visual_concept=beat.visual_concept,
            fallback="stay_on_speaker" if d.needs_asset or d.needs_generation else None,
            alternatives=[a for a in dict.fromkeys(to_plan_treatment(x) for x in d.alternatives) if a != treatment],
            status=SlotStatus.APPROVED if plain else SlotStatus.PENDING_REVIEW,
            settled_reason=NO_SPECIAL_TREATMENT + ": the speaker is the best visual" if plain else None,
            text_options=options,
            directed=True, speaker_visible=d.speaker_visible,
            camera=d.camera.value if d.klass != "replacement" else "n/a",
            transition=d.transition.effective.value,
            transition_requested=d.transition.style.value if d.transition.stylized else None,
            caption_mode=caption_mode, caption_behavior=d.hierarchy.caption_role if d.hierarchy else d.caption_behavior,
            sound_intent=d.sound.intent if d.sound else "none",
            sound_event=d.sound.event_type if d.sound else None,
            sound_locked=beat.sound_locked, sound_importance=beat.importance, beat_kind=d.beat_kind, direction_notes=list(d.notes),
            planning=_planning(d),
        )
        if answer is not None:
            _adopt(slot, answer, d)
        slots.append(slot)
    for i, orphan in enumerate(answers):
        if i not in used:  # an answer no beat lines up with is kept, flagged, never dropped
            copy = orphan.model_copy(deep=True)
            copy.review_required, copy.status = True, SlotStatus.PENDING_REVIEW
            copy.review_note = (f"you {orphan.status.value} {orphan.treatment} at {orphan.timeline_start:.2f}-{orphan.timeline_end:.2f}s, "
                                "but the new plan has no beat there")
            slots.append(copy)
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


NO_SPECIAL_TREATMENT = "no special treatment"
ALIGN_TOLERANCE_S = 0.5  # how far an answered window may sit from the new beat before it counts as a conflict


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _match(answers: list[EditPlanSlot], used: set[int], start: float, end: float) -> EditPlanSlot | None:
    """The answered slot this beat carries on (the largest overlap, at least half of the shorter window)."""
    best, best_ov = None, 0.0
    for i, a in enumerate(answers):
        if i in used:
            continue
        ov = _overlap(start, end, a.timeline_start, a.timeline_end)
        need = 0.5 * min(end - start, a.timeline_end - a.timeline_start)
        if ov > best_ov and ov >= need > 0:
            best, best_ov = i, ov
    if best is None:
        return None
    used.add(best)
    return answers[best]


def _adopt(slot: EditPlanSlot, answer: EditPlanSlot, d: VisualDecision) -> None:
    """Carries the reviewer's answer onto the beat's slot, or flags the conflict."""
    conflicts: list[str] = []
    if answer.treatment != slot.treatment:
        conflicts.append(f"the planner now proposes {slot.treatment} instead of {answer.treatment}")
    if abs(answer.timeline_start - slot.timeline_start) > ALIGN_TOLERANCE_S or abs(answer.timeline_end - slot.timeline_end) > ALIGN_TOLERANCE_S:
        conflicts.append(f"the beat now runs {slot.timeline_start:.2f}-{slot.timeline_end:.2f}s, not {answer.timeline_start:.2f}-{answer.timeline_end:.2f}s")
    if conflicts:
        why = "; ".join(d.notes) if d.notes else d.reason
        slot.review_required, slot.status = True, SlotStatus.PENDING_REVIEW
        slot.review_note = f"you {answer.status.value} {answer.treatment}, but {' and '.join(conflicts)} ({why})"
        return
    slot.status, slot.settled_reason = answer.status, answer.settled_reason
    slot.text = answer.text or slot.text
    slot.recommended = answer.recommended or slot.recommended
    slot.generation_approved = answer.generation_approved
    if answer.sound_locked:
        slot.sound_intent, slot.sound_locked = answer.sound_intent, True


def _planning(d: VisualDecision) -> BeatPlanning:
    """The slot-level copy of what the director knew about this beat (advisory; renders nothing)."""
    story, h, f = d.camera_story, d.hierarchy or build_hierarchy(d.treatment, d.speaker_visible), d.fatigue
    pl = BeatPlanning(
        beat_type=str(getattr(d.semantic_kind, "value", d.semantic_kind)) if d.semantic_kind else None,
        beat_confidence=d.semantic_confidence,
        beat_cues=list(d.semantic_cues),
        variation_desirable=d.variation_desirable, variation_reason=d.variation_reason,
        primary_layer=h.primary_layer, headline_role=h.headline_role, headline_placement=h.headline_placement,
        speaker_visibility=h.speaker_visibility,
    )
    if d.semantic_confidence is not None:
        pl.beat_confidence_label = confidence_label(d.semantic_confidence)
    if f is not None:
        prev = f.previous_treatment or "start of the video"
        pl.previous_state = prev + (f", camera {f.previous_camera}" if f.previous_camera else "") + \
            f"; {f.seconds_since_visual_change:.1f}s since the last visual change"
    if story is not None:
        pl.camera_incoming = f"{story.incoming_state} ({story.incoming_zoom:.2f}x)"
        pl.camera_event = story.event + (f" -> {story.event_zoom_to:.2f}x" if story.event_zoom_to is not None else "")
        pl.camera_release = story.release
        pl.camera_sequence = list(story.sequence)
    return pl


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


UNAVAILABLE = "unavailable_fallback_none"  # an intent exists, no real asset resolves: nothing is heard
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
            # an intent with nothing to attach to is never audible: say so, keep the intent
            slot.sound_status = "none" if slot.sound_intent == "none" else UNAVAILABLE
            slot.sfx_availability = "none" if slot.sound_intent == "none" else "none (nothing moves on this beat)"
    decisions = plan_sound([ev for _, ev in events.values()], profile, reg)
    for d in decisions:
        slot, _ = events[d.event_id]
        slot.sound_status = UNAVAILABLE if d.status == "unavailable" else d.status
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
    slot.caption_behavior = build_hierarchy(vocab, slot.speaker_visible).caption_role
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


__all__ = ["ALIGN_TOLERANCE_S", "NO_SPECIAL_TREATMENT", "UNAVAILABLE", "build_directed_plan", "retarget", "set_slot_sound", "sync_sound", "transition_label"]
