"""Copy-approval rule for text-driven motion treatments (hook titles, ...).

Raw or unresolved ASR text is never promoted into on-screen copy by itself.
Copy may come only from (1) transcript text the user has confirmed, (2) copy the
user supplied, or (3) a rewrite the user explicitly approved. While the transcript
segments a treatment is drawn from are still unresolved, its copy is
`pending_review`: the visual plan is kept, previews show a clearly marked REVIEW
placeholder, the final render gets no copy layer, and final approval is blocked.

The words and the look are approved separately (`TextTreatmentReview`). Changing
only the words reuses the approved visual props; `refit_copy` checks that the new
words still fit and, if not, moves the minimum needed and reports it.

Nothing here rewrites, improves or translates text, and nothing names a brand, a
colour, a position or a project.
"""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.captions.safe_zone import SafeZone
from video_edit_agent.core.schemas import Transcript
from video_edit_agent.review.schemas import (
    ApprovalStatus,
    CopySource,
    TextTreatmentReview,
    UnresolvedTranscriptItem,
)

HOOK_TREATMENT = "hook_title"
BLOCK_UNRESOLVED_TRANSCRIPT = "unresolved transcript"
BLOCK_NO_COPY = "no approved copy"
BLOCK_COPY_NOT_CHOSEN = "hook wording not approved"

_USER_COPY_SOURCES = (CopySource.USER_SUPPLIED, CopySource.APPROVED_REWRITE)


def placeholder_text(treatment: str) -> str:
    """The marked REVIEW placeholder shown in previews instead of unapproved
    copy, e.g. `HOOK TEXT - PENDING REVIEW` for `hook_title`."""
    label = treatment.split("_")[0].upper() or "COPY"
    return f"{label} TEXT - PENDING REVIEW"


@dataclass
class CopyDecision:
    review: TextTreatmentReview

    @property
    def final_text(self) -> str | None:
        """Text allowed into a final render; None means "no copy layer"."""
        r = self.review
        return r.approved_copy if r.copy_status is ApprovalStatus.APPROVED else None

    @property
    def preview_text(self) -> str:
        """What a review preview shows: the approved copy, else the placeholder."""
        return self.final_text or self.review.placeholder_text or placeholder_text(self.review.treatment)


def words_with_segments(transcript: Transcript, start: float, end: float) -> list[tuple[str, str]]:
    """(word, segment id) for every word starting in [start, end), in order."""
    return [(w.word, seg.id) for seg in transcript.segments for w in seg.words if start <= w.start < end]


def segment_numbers(transcript: Transcript, segment_ids: list[str], unresolved: list[UnresolvedTranscriptItem]) -> list[int]:
    """1-based segment numbers (the audio review's numbering) for `segment_ids`."""
    given = {u.segment_id: u.segment for u in unresolved if u.segment is not None}
    order = {seg.id: i + 1 for i, seg in enumerate(transcript.segments)}
    return [given.get(sid) or order[sid] for sid in segment_ids if sid in given or sid in order]


def decide_copy(
    treatment: str,
    proposed_text: str,
    source_segment_ids: list[str],
    *,
    unresolved: list[UnresolvedTranscriptItem],
    existing: TextTreatmentReview | None = None,
    source_segments: list[int] | None = None,
    required: bool = True,
    reviewed_segment_ids: set[str] | frozenset[str] = frozenset(),
) -> CopyDecision:
    """Applies the rule. `proposed_text` is what the transcript offers for the
    treatment; it is used only when every segment behind it is confirmed.
    User-supplied/approved copy already stored on `existing` always wins, and
    `existing`'s visual approval is carried over untouched.

    Approving a transcript segment is not approving on-screen wording: when a
    source segment went through the user's transcript review
    (`reviewed_segment_ids`), the copy stays pending until the user approves the
    exact words (`with_user_copy`); nothing is cut down or rewritten for them."""
    review = TextTreatmentReview(
        treatment=treatment,
        required=required,
        source_segment_ids=list(source_segment_ids),
        source_segments=list(source_segments or []),
        proposed_asr_text=proposed_text,
        placeholder_text=placeholder_text(treatment),
    )
    if existing is not None:
        review.visual_status = existing.visual_status
        review.visual_props = existing.visual_props
        review.required = existing.required
        review.proposed_copy = existing.proposed_copy

    if existing is not None and existing.copy_source in _USER_COPY_SOURCES and existing.approved_copy:
        review.copy_status = ApprovalStatus.APPROVED
        review.copy_source = existing.copy_source
        review.approved_copy = existing.approved_copy
        review.placeholder_text = None
        return CopyDecision(review)

    open_ids = {u.segment_id for u in unresolved}
    blocked = [sid for sid in source_segment_ids if sid in open_ids]
    if blocked:
        review.blocking_reason = BLOCK_UNRESOLVED_TRANSCRIPT
        review.placeholder_used_in_preview = True
    elif not proposed_text.strip():
        review.blocking_reason = BLOCK_NO_COPY
        review.placeholder_used_in_preview = True
    elif any(sid in reviewed_segment_ids for sid in source_segment_ids):
        review.blocking_reason = BLOCK_COPY_NOT_CHOSEN
        review.placeholder_used_in_preview = True
    else:
        review.copy_status = ApprovalStatus.APPROVED
        review.copy_source = CopySource.APPROVED_TRANSCRIPT
        review.approved_copy = proposed_text
        review.placeholder_text = None
    return CopyDecision(review)


def with_user_copy(
    review: TextTreatmentReview, text: str, source: CopySource = CopySource.USER_SUPPLIED,
) -> TextTreatmentReview:
    """`review` with `text` as its approved copy. Only the copy fields change:
    visual approval, visual props and the transcript reference are untouched."""
    if source not in _USER_COPY_SOURCES:
        raise ValueError(f"copy source must be one of {[s.value for s in _USER_COPY_SOURCES]}, not {source.value}")
    if not text.strip():
        raise ValueError("hook copy must not be empty")
    return review.model_copy(update={
        "copy_status": ApprovalStatus.APPROVED,
        "copy_source": source,
        "approved_copy": text,
        "placeholder_text": None,
        "placeholder_used_in_preview": False,
        "blocking_reason": None,
    })


def pending_again(review: TextTreatmentReview, reason: str = BLOCK_UNRESOLVED_TRANSCRIPT) -> TextTreatmentReview:
    """`review` with its transcript-derived copy withdrawn (its source segment
    became unresolved); the visual approval stays."""
    return review.model_copy(update={
        "copy_status": ApprovalStatus.PENDING_REVIEW,
        "copy_source": None,
        "approved_copy": None,
        "blocking_reason": reason,
        "placeholder_text": placeholder_text(review.treatment),
        "placeholder_used_in_preview": True,
    })


def blocking_items(treatments: list[TextTreatmentReview]) -> list[TextTreatmentReview]:
    """Required treatments whose copy is not approved: they block final approval."""
    return [t for t in treatments if t.required and t.copy_status is not ApprovalStatus.APPROVED]


@dataclass
class CopyFit:
    props: dict
    issue: str | None = None
    adjusted: list[str] | None = None


def refit_copy(
    props: dict, text: str, canvas_w: int, canvas_h: int, *, font_px: int = 88, safe_zone: SafeZone | None = None,
) -> CopyFit:
    """New words on an APPROVED visual: keeps the props as they are and only
    checks that the block still sits inside the safe band. If it does not, only
    `centerY` moves, by the least amount that fits; a block taller than the band
    cannot be fixed by moving, so that is reported and the block is pinned to the
    top of the band -- colour, plate, outline, width and animation never change."""
    from video_edit_agent.motion.legibility import _CAPTION_BAND, estimate_title_box

    safe = safe_zone or SafeZone()
    plate = props.get("plate") or {}
    pad = int(plate.get("padding", 0) or 0)
    max_width = int(props.get("maxWidth") or canvas_w)
    _, box_h = estimate_title_box(text, font_px, max(1, max_width - 2 * pad), canvas_w, canvas_h, pad)
    half = box_h / 2
    top_limit = safe.top_pct * canvas_h
    bottom_limit = (1.0 - max(safe.bottom_pct, _CAPTION_BAND)) * canvas_h
    center = props.get("centerY")
    if not isinstance(center, (int, float)):
        return CopyFit(dict(props))  # unpositioned title: the renderer centres it, nothing to fit
    if center - half >= top_limit - 1 and center + half <= bottom_limit + 1:
        return CopyFit(dict(props))
    if box_h > bottom_limit - top_limit:
        new_center = round(top_limit + half)
        issue = (f"new text is {box_h:.0f}px tall, taller than the {bottom_limit - top_limit:.0f}px safe band; "
                 "pinned to the top of the band -- shorten the text or approve a smaller size")
    else:
        new_center = round(min(max(center, top_limit + half), bottom_limit - half))
        issue = f"new text is {box_h:.0f}px tall and left the safe band; moved vertically by {abs(new_center - center):.0f}px"
    return CopyFit({**props, "centerY": new_center}, issue=issue, adjusted=["centerY"])
