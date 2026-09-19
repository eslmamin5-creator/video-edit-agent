"""Render-approval gate (Review-First Editing Workflow spec section 8):
`READY_FOR_FINAL_RENDER` persisted to disk so a separate "approve" step (CLI
command, Skill turn, or UI action) can flip it before the pipeline is asked
to actually render.
"""
from __future__ import annotations

import json
from pathlib import Path

from video_edit_agent.review.schemas import (
    ReviewApprovalState,
    ReviewStage,
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


def flag_unresolved(review_dir: Path, item: UnresolvedTranscriptItem) -> ReviewApprovalState:
    """Marks a transcript segment as awaiting the user's confirmation (upsert
    by segment id). The project cannot be approved while any item is open."""
    state = load_review_state(review_dir)
    state.unresolved_transcript = [i for i in state.unresolved_transcript if i.segment_id != item.segment_id]
    state.unresolved_transcript.append(item)
    state.ready_for_final_render = False
    save_review_state(state, review_dir)
    return state


def resolve_unresolved(review_dir: Path, segment_id: str) -> ReviewApprovalState:
    state = load_review_state(review_dir)
    state.unresolved_transcript = [i for i in state.unresolved_transcript if i.segment_id != segment_id]
    save_review_state(state, review_dir)
    return state


def approve(review_dir: Path, note: str | None = None) -> ReviewApprovalState:
    """Marks the review as approved -- the only normal way `ready_for_final_
    render` becomes True short of an explicit bypass flag. Refused while any
    transcript item is still unresolved."""
    state = load_review_state(review_dir)
    if state.unresolved_transcript:
        open_ids = ", ".join(i.segment_id for i in state.unresolved_transcript)
        raise UnresolvedReviewItems(f"unresolved transcript segments remain open: {open_ids}")
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
