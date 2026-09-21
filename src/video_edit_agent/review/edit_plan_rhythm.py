"""Bridges the Visual Rhythm Engine into the reviewable edit plan.

`attach_rhythm` plans the low-semantic visual rhythm around whatever the semantic
director (or the user) already owns, then fills the review fields for every planned
state: the phrase it snaps to, why now, what came before, the composition reason, the
return, the primary layer, the caption role, the speaker's visibility, the semantic
reading, behind-subject eligibility and the sound status.

Behind-subject stays a semantic enhancement: it is only ever reported as eligible or
not, with the reasons, and is never applied here. A phrase the USER pinned keeps
`semantic_source = user_pinned` and is not dropped for a low automatic confidence; it
still has to pass every technical gate and is never approved here.

The rhythm rows are then normalised into the ONE canonical camera timeline
(`direction.camera_timeline`): legacy EDL zoom and `static` slot claims inside the
rhythm's stretch are superseded explicitly, each slot shows the camera that will
render, and the plan is render-ready only while every visual event is executable.
Nothing here renders, generates, approves or sets `ready_for_final_render`.
Nothing in this module is brand- or project-specific.
"""
from __future__ import annotations

from collections.abc import Callable

from video_edit_agent.core.schemas import EDL, Transcript
from video_edit_agent.direction.camera import CameraEvent, CameraPlan
from video_edit_agent.direction.camera_timeline import (
    build_camera_timeline,
    legacy_events,
    reconcile_slots,
    static_claims,
    timeline_from_camera_plan,
)
from video_edit_agent.direction.composition import LowerSubjectComposition, review_state
from video_edit_agent.direction.rhythm import (
    Occupied,
    RhythmPlan,
    RhythmPolicy,
    RhythmRow,
    SemanticHint,
    plan_rhythm,
    semantic_enhancement_options,
)
from video_edit_agent.direction.semantic_beats import SemanticBeat
from video_edit_agent.direction.suitability import BehindSubjectEvidence, behind_subject_eligibility
from video_edit_agent.review import edit_plan as ep
from video_edit_agent.review.edit_plan import EditPlan, EditPlanSlot, SlotStatus

NO_SPECIAL_TREATMENT = "no special treatment"
_ENHANCEMENT = {  # plan treatment -> the semantic enhancement it is
    "kinetic_typography": "kinetic_typography", "behind_subject_text": "behind_subject_text", "motion_graphic": "small_motion_graphic",
    "illustration": "diagram_process_scene", "local_broll": "user_broll", "generated_broll": "generated_visual",
}

Evidence = Callable[[RhythmRow], BehindSubjectEvidence | None]


def owned_windows(plan: EditPlan, camera: CameraPlan | None, start: float, end: float) -> list[Occupied]:
    """The windows the rhythm engine must leave alone: everything the semantic director proposed or you decided
    (a plain 'no special treatment' stretch is free). The picture changes at a window's own camera events
    (an emphasis and its reset), or at its edges when it has none."""
    out: list[Occupied] = []
    for s in plan.slots:
        if s.timeline_end <= start or s.timeline_start >= end:
            continue
        if s.settled_reason and s.settled_reason.startswith(NO_SPECIAL_TREATMENT):
            continue
        if s.treatment == "stay_on_speaker" and s.camera in ("static", "n/a"):
            continue
        changes: list[float] = []
        if camera is not None and s.camera not in ("static", "n/a"):
            for e in camera.events:
                if s.timeline_start - 0.05 <= e.start <= s.timeline_end + 0.05 or (e.beat_start is not None and abs(e.beat_start - s.timeline_start) < 0.05):
                    changes.append(e.start)
        pinned = s.status in (SlotStatus.APPROVED, SlotStatus.CHANGED, SlotStatus.GENERATION_APPROVED) and not s.settled_reason
        out.append(Occupied(start=s.timeline_start, end=s.timeline_end, label=s.camera if s.camera not in ("static", "n/a") else s.treatment,
                            pinned=pinned, changes=tuple(sorted(changes)) if changes else (s.timeline_start, s.timeline_end)))
    return out


def _slot_at(plan: EditPlan, t0: float, t1: float) -> EditPlanSlot | None:
    best, ov = None, 0.0
    for s in plan.slots:
        o = min(t1, s.timeline_end) - max(t0, s.timeline_start)
        if o > ov:
            best, ov = s, o
    return best


def _hint_at(hints: list[SemanticHint], t: float) -> SemanticHint | None:
    return next((h for h in hints if h.start - 1e-6 <= t < h.end), None)


def user_pinned(slot: EditPlanSlot | None) -> bool:
    """The user supplied the semantic evidence for this slot's treatment (they chose it, or pinned the phrase)."""
    return slot is not None and (slot.semantic_source == "user_pinned" or (
        slot.treatment == "behind_subject_text" and slot.status is SlotStatus.CHANGED))


def _approval(slot: EditPlanSlot) -> str:
    return "approved (your decision)" if slot.status in (SlotStatus.APPROVED, SlotStatus.CHANGED, SlotStatus.GENERATION_APPROVED) \
        else "pending review" if slot.status is SlotStatus.PENDING_REVIEW else slot.status.value


def enrich(plan: EditPlan, rhythm: RhythmPlan, transcript: Transcript, edl: EDL, hints: list[SemanticHint],
           evidence: Evidence | None = None) -> None:
    """Fills the review fields of every rhythm row in place."""
    for row in rhythm.rows:
        segs = ep._segments_for(row.start, row.end, edl, transcript)
        row.transcript_context = " ".join(transcript.segments[n - 1].text.strip() for n in segs)
        hint = _hint_at(hints, row.start)
        if hint is not None:
            row.semantic_kind, row.semantic_confidence = hint.kind, round(hint.confidence, 2)
        slot = _slot_at(plan, row.start, row.end) if row.source != "rhythm" else None
        if slot is not None:  # owned by the director / by you: mirror its decision
            row.semantic_enhancement = _ENHANCEMENT.get(slot.treatment, "none")
            pl = slot.planning
            row.primary_layer = pl.primary_layer if pl else "speaker"
            row.speaker_visibility = pl.speaker_visibility if pl else ("full" if slot.speaker_visible else "hidden")
            row.caption_role = slot.caption_behavior
            row.sound_intent, row.sound_status = slot.sound_intent, slot.sound_status
            row.approval_status = _approval(slot)
        phrase = None
        if row.source == "rhythm" or (slot is not None and slot.treatment == "behind_subject_text"):
            phrase = (slot.text if slot and slot.text else None) or next(iter(ep.keyword_options(row.transcript_context)), None)
        if row.source == "rhythm":
            row.semantic_options = semantic_enhancement_options(row.semantic_kind, row.semantic_confidence, phrase=phrase)
        pinned = user_pinned(slot) and slot is not None and slot.treatment == "behind_subject_text"
        verdict = behind_subject_eligibility(
            phrase=phrase, semantic_kind=row.semantic_kind, semantic_confidence=row.semantic_confidence,
            evidence=evidence(row) if evidence else None, rhythm_state=row.state if row.source == "rhythm" else "base",
            semantic_source="user_pinned" if pinned else "auto",
        )
        row.behind_subject = verdict.status
        row.behind_subject_reason = verdict.summary if verdict.status != "not_applicable" else ""
        row.semantic_source = verdict.semantic_source
        row.behind_subject_candidate = verdict.candidate
        if pinned and slot is not None:  # the slot mirrors the row so the review shows the candidate state
            slot.semantic_source, slot.behind_subject_candidate = "user_pinned", verdict.candidate
        if row.composition:  # a lower_subject that makes room for a headline: the headline leads, captions are reduced, nothing is approved
            row.semantic_enhancement, row.primary_layer = "primary_headline_typography", "headline"
            row.caption_role, row.speaker_visibility, row.approval_status = "reduced", "full", "pending_review"
            row.semantic_source = "user_pinned" if row.semantic_source == "user_pinned" else "auto"


def attach_rhythm(
    plan: EditPlan,
    transcript: Transcript,
    edl: EDL,
    *,
    semantic: list[SemanticBeat] | None = None,
    camera: CameraPlan | None = None,
    start: float = 0.0,
    end: float | None = None,
    policy: RhythmPolicy | None = None,
    face_box: tuple[float, float, float, float] | None = None,
    evidence: Evidence | None = None,
    compositions: list[LowerSubjectComposition] | None = None,
) -> RhythmPlan:
    """Plans the visual rhythm for `plan`, stores the enriched rows on it and returns the rhythm plan.
    Weak or absent semantic readings change nothing about the framing variation; they only decide which
    semantic options are listed (and none are ever applied)."""
    end = end if end is not None else max([transcript.duration, *[s.end for s in transcript.segments]])
    hints = [SemanticHint(start=b.start, end=b.end, kind=b.kind.value, confidence=b.confidence) for b in (semantic or [])]
    rhythm = plan_rhythm(transcript, start=start, end=end, policy=policy, hints=hints, face_box=face_box,
                         occupied=owned_windows(plan, camera, start, end),
                         compositions=[c for c in (compositions or []) if c.status == "ok"])
    enrich(plan, rhythm, transcript, edl, hints, evidence)
    plan.rhythm = rhythm.rows
    attach_camera_timeline(plan, rhythm, edl, camera=camera, face_box=face_box, start=start, end=end)
    return rhythm


def attach_camera_timeline(
    plan: EditPlan, rhythm: RhythmPlan, edl: EDL, *, camera: CameraPlan | None = None,
    face_box: tuple[float, float, float, float] | None = None, start: float = 0.0, end: float | None = None,
) -> None:
    """Normalises the rhythm rows (plus the director's/your camera events) into the canonical camera timeline, supersedes the
    legacy EDL zoom and static slot claims inside its stretch, and makes every slot show the camera that will render."""
    end = end if end is not None else rhythm.end
    decided = (SlotStatus.APPROVED, SlotStatus.CHANGED, SlotStatus.GENERATION_APPROVED)
    pinned: list[CameraEvent] = []
    for e in timeline_from_camera_plan(camera, "director") if camera is not None else []:
        slot = next((s for s in plan.slots if s.timeline_start - 0.05 <= e.start <= s.timeline_end + 0.05), None)
        pinned.append(e.model_copy(update={"owner": "pinned" if slot is not None and slot.status in decided and not slot.settled_reason else "director"}))
    legacy = [*legacy_events(edl), *static_claims(plan.slots)]
    windows = [(s.timeline_start, s.timeline_end) for s in plan.slots if s.treatment == "behind_subject_text"]
    plan.camera_timeline = build_camera_timeline(rhythm, face_box=face_box, legacy=legacy, pinned=pinned, behind_subject=windows,
                                                 scope=(start, end))
    reconcile_slots(plan.slots, plan.camera_timeline)


__all__ = ["attach_camera_timeline", "attach_rhythm", "enrich", "owned_windows", "review_state", "user_pinned"]
