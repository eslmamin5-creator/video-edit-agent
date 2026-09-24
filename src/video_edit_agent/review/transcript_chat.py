"""Chat-first transcript review: status model, views, paging and the text block
the agent shows the user.

The user reviews the transcript IN THE CONVERSATION -- numbered sentences with
timestamps, a status and the low-confidence words -- and answers in plain
language. They never open JSON/ASS/WAV files or see internal segment ids.

Segment status (computed, see `SegmentReviewStatus`):

- decision `approved` (user confirmed it or typed the exact wording)  -> approved
- decision `corrected_pending_approval` (partial fix, not yet OK'd)   -> corrected_pending_approval
- flagged in `unresolved_transcript`                                   -> unresolved
- a stored correction with no decision (older `review-correct` runs)  -> approved
- low-confidence words and no decision                                -> needs_review
- otherwise (clean, never flagged)                                    -> approved

`unresolved` and `corrected_pending_approval` block final approval;
`needs_review` is advisory until the segment is flagged (`open_transcript_review`).
This module only reads state; it never edits text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from video_edit_agent.core.schemas import Transcript
from video_edit_agent.review.schemas import (
    SUSPICIOUS_CONFIDENCE_THRESHOLD,
    ReviewApprovalState,
    SegmentReviewStatus,
    TranscriptCorrection,
)

DEFAULT_PAGE_SIZE = 8

VIEW_PENDING = "pending"  # everything that still needs a decision ("suspicious only")
VIEW_ALL = "all"
VIEW_RANGE = "range"
VIEW_NUMBERS = "numbers"

_S = SegmentReviewStatus
BLOCKING_STATUSES = (_S.UNRESOLVED, _S.CORRECTED_PENDING_APPROVAL)
STATUS_LABEL = {
    _S.APPROVED: "APPROVED",
    _S.NEEDS_REVIEW: "NEEDS REVIEW",
    _S.CORRECTED_PENDING_APPROVAL: "CORRECTED - WAITING FOR YOUR OK",
    _S.UNRESOLVED: "UNRESOLVED - NEEDS YOUR REVIEW",
}


@dataclass(frozen=True)
class SegmentView:
    number: int  # 1-based, the only number the user ever sees
    segment_id: str  # internal; never shown
    start: float
    end: float
    text: str  # effective text (raw ASR + persisted corrections)
    status: SegmentReviewStatus
    low_confidence: tuple[tuple[str, float], ...] = ()
    reason: str = ""

    @property
    def is_pending(self) -> bool:
        return self.status is not _S.APPROVED


@dataclass(frozen=True)
class ViewSpec:
    kind: str = VIEW_PENDING
    numbers: tuple[int, ...] = ()  # VIEW_NUMBERS: the segments; VIEW_RANGE: (first, last)
    page: int = 0
    extra: dict = field(default_factory=dict, compare=False, hash=False)


def format_timestamp(seconds: float) -> str:
    """`MM:SS.ss` (minutes are not capped, so long videos stay readable)."""
    seconds = max(0.0, seconds)
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def segment_status(
    segment_id: str,
    *,
    state: ReviewApprovalState,
    corrected_ids: set[str],
    suspicious: bool,
) -> SegmentReviewStatus:
    decision = next((d.status for d in state.segment_reviews if d.segment_id == segment_id), None)
    if decision is not None:
        return decision
    if any(i.segment_id == segment_id for i in state.unresolved_transcript):
        return _S.UNRESOLVED
    if segment_id in corrected_ids:
        return _S.APPROVED
    return _S.NEEDS_REVIEW if suspicious else _S.APPROVED


def build_views(
    effective: Transcript, corrections: list[TranscriptCorrection], state: ReviewApprovalState,
) -> list[SegmentView]:
    """One view per segment, numbered by position. `effective` is the transcript
    with corrections already applied (`apply_corrections`)."""
    corrected_ids = {c.segment_id for c in corrections}
    reasons = {i.segment_id: i.reason for i in state.unresolved_transcript}
    views: list[SegmentView] = []
    for number, seg in enumerate(effective.segments, start=1):
        low = tuple(
            (w.word, w.confidence) for w in seg.words if w.confidence < SUSPICIOUS_CONFIDENCE_THRESHOLD
        )
        status = segment_status(seg.id, state=state, corrected_ids=corrected_ids, suspicious=bool(low))
        views.append(SegmentView(
            number=number, segment_id=seg.id, start=seg.start, end=seg.end, text=seg.text,
            status=status, low_confidence=low if status is not _S.APPROVED else (),
            reason=reasons.get(seg.id, ""),
        ))
    return views


def select_view(views: list[SegmentView], spec: ViewSpec) -> list[SegmentView]:
    if spec.kind == VIEW_ALL:
        return list(views)
    if spec.kind == VIEW_RANGE and len(spec.numbers) == 2:
        lo, hi = sorted(spec.numbers)
        return [v for v in views if lo <= v.number <= hi]
    if spec.kind == VIEW_NUMBERS:
        wanted = set(spec.numbers)
        return [v for v in views if v.number in wanted]
    return [v for v in views if v.is_pending]


def paginate(items: list[SegmentView], page: int, page_size: int = DEFAULT_PAGE_SIZE) -> tuple[list[SegmentView], int, int]:
    """(items on the page, clamped page index, page count); an empty list is one page."""
    page_size = max(1, page_size)
    pages = max(1, -(-len(items) // page_size))
    page = min(max(page, 0), pages - 1)
    return items[page * page_size:(page + 1) * page_size], page, pages


def format_segment(view: SegmentView, lang: str = "ar") -> str:
    lines = [
        f"[{view.number:02d}] {format_timestamp(view.start)}–{format_timestamp(view.end)}",
        view.text,
        "",
        f"Status: {STATUS_LABEL[view.status]}",
    ]
    if view.low_confidence:
        lines.append("Low-confidence: " + "، ".join(f"{w} ({c:.2f})" for w, c in view.low_confidence))
    if view.reason and view.status is _S.UNRESOLVED:
        lines.append(("السبب: " if lang == "ar" else "Reason: ") + view.reason)
    return "\n".join(lines)


_VIEW_TITLE = {
    "ar": {VIEW_PENDING: "الجمل المشكوك فيها بس", VIEW_ALL: "كل الجمل", VIEW_RANGE: "نطاق", VIEW_NUMBERS: "جمل محددة"},
    "en": {VIEW_PENDING: "suspicious only", VIEW_ALL: "all segments", VIEW_RANGE: "range", VIEW_NUMBERS: "selected segments"},
}


def remaining_numbers(views: list[SegmentView]) -> list[int]:
    return [v.number for v in views if v.is_pending]


def blocking_numbers(views: list[SegmentView]) -> list[int]:
    return [v.number for v in views if v.status in BLOCKING_STATUSES]


def _numbers(nums: list[int]) -> str:
    return ", ".join(str(n) for n in nums)


def footer(views: list[SegmentView], shown: list[SegmentView], page: int, pages: int, lang: str) -> str:
    """Short, contextual "how to answer" text -- examples use numbers actually on screen."""
    ordered = [v.number for v in shown if v.is_pending] + [v.number for v in shown if not v.is_pending]
    ordered = ordered or [1]
    a, b, c = (ordered[min(i, len(ordered) - 1)] for i in range(3))
    must = blocking_numbers(views)
    advisory = [n for n in remaining_numbers(views) if n not in must]
    if lang == "ar":
        parts = []
        if must:
            parts.append(f"لسه محتاجة قرارك (بتمنع الاعتماد النهائي): {_numbers(must)}")
        else:
            parts.append("مفيش جمل بتمنع الاعتماد.")
        if advisory:
            parts.append(f"شكوك اختيارية (ما بتمنعش): {_numbers(advisory)}")
        parts.append(
            "ردّ بشكل طبيعي، مثلًا:\n"
            f"  {a}: <الجملة الصحيحة كاملة>\n"
            f"  {b} صح\n"
            f"  {c} غير كلمة <X> إلى <Y>\n"
            f"  اسمعني الجملة {a}   |   اعتمد الباقي   |   التالي / السابق / الكل"
        )
    else:
        parts = [f"Still needs your decision (blocks final approval): {_numbers(must)}" if must else "Nothing blocks approval."]
        if advisory:
            parts.append(f"Optional doubts (do not block): {_numbers(advisory)}")
        parts.append(
            "Reply naturally, e.g.:\n"
            f"  {a}: <the full correct sentence>\n"
            f"  {b} ok\n"
            f"  {c} change <X> to <Y>\n"
            f"  play segment {a}   |   approve the rest   |   next / previous / show all"
        )
    if pages > 1:
        parts.insert(0, f"page {page + 1}/{pages}" if lang == "en" else f"صفحة {page + 1} من {pages}")
    return "\n".join(parts)


def format_review(
    views: list[SegmentView], spec: ViewSpec, *, lang: str = "ar", page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[str, list[SegmentView], int]:
    """The chat block for one page of a view. Returns (text, segments shown, page used)."""
    chosen = select_view(views, spec)
    shown, page, pages = paginate(chosen, spec.page, page_size)
    title = _VIEW_TITLE.get(lang, _VIEW_TITLE["en"])[spec.kind]
    head = (
        f"Transcript Review — {title} · {len(chosen)}/{len(views)}"
        if lang == "en" else f"مراجعة النص — {title} · {len(chosen)}/{len(views)}"
    )
    body = "\n\n".join(format_segment(v, lang) for v in shown) if shown else (
        "Nothing to show here." if lang == "en" else "ما في جمل تنعرض هنا."
    )
    return f"{head}\n\n{body}\n\n{footer(views, shown, page, pages, lang)}", shown, page
