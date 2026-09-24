"""Render-approval gate (Review-First Editing Workflow spec section 8):
`READY_FOR_FINAL_RENDER` persisted to disk so a separate "approve" step (CLI
command, Skill turn, or UI action) can flip it before the pipeline is asked
to actually render.
"""
from __future__ import annotations

import json
from pathlib import Path

from video_edit_agent.review import text_copy
from video_edit_agent.review.schemas import (
    ApprovalStatus,
    CopySource,
    ReviewApprovalState,
    ReviewStage,
    SegmentDecision,
    SegmentReviewStatus,
    TextTreatmentReview,
    UnresolvedTranscriptItem,
)

STATE_FILENAME = "review_state.json"


def save_review_state(state: ReviewApprovalState, review_dir: Path) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / STATE_FILENAME
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_review_state(review_dir: Path) -> ReviewApprovalState:
    path = review_dir / STATE_FILENAME
    if not path.exists():
        return ReviewApprovalState()
    return ReviewApprovalState.model_validate(json.loads(path.read_text(encoding="utf-8")))


class UnresolvedReviewItems(RuntimeError):
    """Raised when approval is attempted while transcript items are open."""


def _withdraw_transcript_copy(state: ReviewApprovalState, segment_id: str) -> None:
    """Copy taken from the transcript is only as good as the transcript: it goes
    back to pending (the approved visual stays); user-supplied copy is untouched."""
    state.text_treatments = [
        text_copy.pending_again(t) if t.copy_source is CopySource.APPROVED_TRANSCRIPT and segment_id in t.source_segment_ids else t
        for t in state.text_treatments
    ]


def _set_decision(state: ReviewApprovalState, segment_id: str, status: SegmentReviewStatus | None) -> None:
    state.segment_reviews = [d for d in state.segment_reviews if d.segment_id != segment_id]
    if status is not None:
        state.segment_reviews.append(SegmentDecision(segment_id=segment_id, status=status))


def flag_unresolved(review_dir: Path, item: UnresolvedTranscriptItem) -> ReviewApprovalState:
    """Marks a transcript segment as awaiting the user's confirmation (upsert
    by segment id). The project cannot be approved while any item is open."""
    state = load_review_state(review_dir)
    state.unresolved_transcript = [i for i in state.unresolved_transcript if i.segment_id != item.segment_id]
    state.unresolved_transcript.append(item)
    _set_decision(state, item.segment_id, None)  # an earlier decision no longer stands
    _withdraw_transcript_copy(state, item.segment_id)
    state.ready_for_final_render = False
    save_review_state(state, review_dir)
    return state


def resolve_unresolved(review_dir: Path, segment_id: str) -> ReviewApprovalState:
    """The user confirmed the segment: closes its flag and records it approved."""
    state = load_review_state(review_dir)
    state.unresolved_transcript = [i for i in state.unresolved_transcript if i.segment_id != segment_id]
    _set_decision(state, segment_id, SegmentReviewStatus.APPROVED)
    save_review_state(state, review_dir)
    return state


def mark_corrected_pending(review_dir: Path, segment_id: str) -> ReviewApprovalState:
    """A correction was stored but the user has not yet said the result is right
    (e.g. a single-word fix). The pending decision replaces any `unresolved` flag
    and, like it, blocks approval until the user says the result is right."""
    state = load_review_state(review_dir)
    state.unresolved_transcript = [i for i in state.unresolved_transcript if i.segment_id != segment_id]
    _set_decision(state, segment_id, SegmentReviewStatus.CORRECTED_PENDING_APPROVAL)
    _withdraw_transcript_copy(state, segment_id)
    state.ready_for_final_render = False
    save_review_state(state, review_dir)
    return state


def propose_copy(review_dir: Path, treatment: str, text: str) -> ReviewApprovalState:
    """Records wording the agent PROPOSES for `treatment` (a rewrite, labelled as
    such). Nothing is approved: copy status, source and visual stay as they were."""
    if not text.strip():
        raise ValueError("proposed copy must not be empty")
    state = load_review_state(review_dir)
    current = _treatment_or_raise(state, treatment)
    state.text_treatments = [t.model_copy(update={"proposed_copy": text}) if t is current else t for t in state.text_treatments]
    save_review_state(state, review_dir)
    return state


def approve(review_dir: Path, note: str | None = None) -> ReviewApprovalState:
    """Marks the review as approved -- the only normal way `ready_for_final_
    render` becomes True short of an explicit bypass flag. Refused while any
    transcript item is still unresolved."""
    state = load_review_state(review_dir)
    open_ids = [i.segment_id for i in state.unresolved_transcript]
    open_ids += [
        d.segment_id for d in state.segment_reviews
        if d.status is SegmentReviewStatus.CORRECTED_PENDING_APPROVAL and d.segment_id not in open_ids
    ]
    if open_ids:
        raise UnresolvedReviewItems(f"unresolved transcript segments remain open: {', '.join(open_ids)}")
    open_copy = text_copy.blocking_items(state.text_treatments)
    if open_copy:
        names = ", ".join(f"{t.treatment} ({t.blocking_reason or 'copy not approved'})" for t in open_copy)
        raise UnresolvedReviewItems(f"required on-screen copy is not approved: {names}")
    from video_edit_agent.review.edit_plan import (
        camera_blockers,  # local: avoids a review<->broll import cycle
        open_pending_slots,
    )

    open_slots = open_pending_slots(review_dir)
    if open_slots:
        names = ", ".join(str(s.number) for s in open_slots)
        raise UnresolvedReviewItems(f"edit-plan decisions are still pending: {names}")
    blockers = camera_blockers(review_dir)
    if blockers:
        raise UnresolvedReviewItems("the camera plan is not render-ready: " + "; ".join(blockers[:4]))
    state.stage = ReviewStage.APPROVED
    state.ready_for_final_render = True
    state.broll_generation_approved = True
    if note:
        state.notes.append(note)
    save_review_state(state, review_dir)
    return state


def bypass(review_dir: Path, reason: str) -> ReviewApprovalState:
    """Explicit `--yes`/`--no-review` bypass (spec section 8): skips the
    interactive gate but the bypass itself is always recorded, never silent."""
    state = ReviewApprovalState(
        stage=ReviewStage.APPROVED, ready_for_final_render=True, bypassed=True,
        broll_generation_approved=True, notes=[reason],
    )
    save_review_state(state, review_dir)
    return state


def is_ready_for_final_render(review_dir: Path) -> bool:
    return load_review_state(review_dir).ready_for_final_render


def mark_rendered(review_dir: Path, note: str | None = None) -> ReviewApprovalState:
    state = load_review_state(review_dir)
    state.stage = ReviewStage.RENDERED
    if note:
        state.notes.append(note)
    save_review_state(state, review_dir)
    return state


# --------------------------------------------------------------------------
# Text treatments: visual approval and copy approval are separate
# --------------------------------------------------------------------------


def record_text_treatment(review_dir: Path, review: TextTreatmentReview) -> ReviewApprovalState:
    """Upserts the planner's record of one treatment. A treatment whose copy is
    not approved can never leave the state `ready_for_final_render`."""
    state = load_review_state(review_dir)
    state.text_treatments = [t for t in state.text_treatments if t.treatment != review.treatment] + [review]
    if text_copy.blocking_items(state.text_treatments):
        state.ready_for_final_render = False
    save_review_state(state, review_dir)
    return state


def _treatment_or_raise(state: ReviewApprovalState, treatment: str) -> TextTreatmentReview:
    found = state.treatment(treatment)
    if found is None:
        raise KeyError(f"no text treatment '{treatment}' in the review state")
    return found


def approve_visual(review_dir: Path, treatment: str, visual_props: dict | None = None) -> ReviewApprovalState:
    """The user approved how `treatment` looks and moves. Stores the props so a
    later copy change reuses them; the copy status is not touched."""
    state = load_review_state(review_dir)
    current = _treatment_or_raise(state, treatment)
    update: dict = {"visual_status": ApprovalStatus.APPROVED}
    if visual_props is not None:
        update["visual_props"] = visual_props
    state.text_treatments = [t.model_copy(update=update) if t is current else t for t in state.text_treatments]
    save_review_state(state, review_dir)
    return state


def submit_copy(
    review_dir: Path, treatment: str, text: str, source: CopySource = CopySource.USER_SUPPLIED,
    layout_fit_issue: str | None = None, layout_adjustments: list[str] | None = None,
) -> ReviewApprovalState:
    """Approves user-supplied (or explicitly approved rewritten) copy for
    `treatment`. Only the copy fields change: visual approval is NOT reset, and
    the transcript is never touched."""
    state = load_review_state(review_dir)
    current = _treatment_or_raise(state, treatment)
    updated = text_copy.with_user_copy(current, text, source).model_copy(update={
        "layout_fit_issue": layout_fit_issue, "layout_adjustments": list(layout_adjustments or []),
    })
    state.text_treatments = [updated if t is current else t for t in state.text_treatments]
    save_review_state(state, review_dir)
    return state
