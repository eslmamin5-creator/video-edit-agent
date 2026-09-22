"""Phase 1.4: Semantic Motion Graphics Foundation.

Three deterministic, brand-aware, review-first visual treatments:

    primary_headline_typography   one strong concept/claim/payoff as text
    keyword_visual                one compact key term, shown small
    simple_diagram                a verbatim 2-5 item list/framework/process

This module ONLY decides *whether* and *what* to show (candidate discovery,
eligibility, selection priority, review state) and how it plugs into camera
ownership. It never renders pixels (see `render.motion_graphics` for that)
and never invents wording: every phrase/label a candidate carries is a
verbatim span of the approved transcript, produced by re-using the existing
semantic qualification system (`direction.composition`) and the existing
structural-list detector (`direction.semantic_beats.SemanticKind.PROCESS_LIST`)
-- nothing here is a new detector, a language list or a hardcoded phrase.

Brand values (colors, fonts) are never read here either: candidates only
carry the caller-supplied `brand_profile` NAME; `render.motion_graphics`
is what actually looks values up on a `Brand` instance. This keeps any
client's palette out of every generic decision in this file (sec. 3).

Nothing here is project-specific. Nothing here auto-approves anything:
`review_state()`'s `approval_status` is always `pending_review` for an
automatically discovered candidate.
"""
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import Transcript
from video_edit_agent.direction.composition import (
    ELIGIBLE_HEADLINE_ROLES,
    CompositionPolicy,
    TWord,
    discover_global_candidates,
    select_headline,
)
from video_edit_agent.direction.history import TreatmentHistory
from video_edit_agent.direction.semantic_beats import SemanticKind, detect_semantic_beats

MAX_DIAGRAM_NODES = 5
MIN_DIAGRAM_NODES = 2
_LIST_SEP_CHARS = ",،"  # comma, Arabic comma (a generic structural cue, not a language list)


class TreatmentType(str, Enum):
    PRIMARY_HEADLINE = "primary_headline_typography"
    KEYWORD_VISUAL = "keyword_visual"
    SIMPLE_DIAGRAM = "simple_diagram"


_SIMPLICITY_ORDER: dict[str, int] = {
    TreatmentType.KEYWORD_VISUAL.value: 0,
    TreatmentType.PRIMARY_HEADLINE.value: 1,
    TreatmentType.SIMPLE_DIAGRAM.value: 2,
}


class MotionGraphicCandidate(BaseModel):
    """One review-first semantic-graphic candidate (sec. 11). `phrase` is set for
    `primary_headline_typography`/`keyword_visual`; `labels` is set (2-5 verbatim
    items) for `simple_diagram`. Never both empty."""

    treatment_type: str
    semantic_source: str = "automatic"  # automatic | user_pinned
    semantic_role: str | None = None
    phrase: str | None = None
    labels: list[str] = Field(default_factory=list)
    start: float
    end: float
    layout: dict = Field(default_factory=dict)
    brand_profile: str = "default"
    technical_status: str = "not_rendered"  # not_rendered | passed | failed
    visual_status: str = "candidate"  # candidate | candidate_ready | rejected
    global_score: float = 0.0
    conflict: bool = False
    conflict_reason: str = ""

    @property
    def phrase_or_labels(self) -> str | list[str] | None:
        return self.labels or self.phrase

    def review_state(self) -> dict:
        """Sec. 11's required review-state shape. `approval_status` is always
        `pending_review` here: nothing produced by discovery is ever auto-approved."""
        return {
            "treatment_type": self.treatment_type,
            "semantic_source": self.semantic_source,
            "semantic_role": self.semantic_role,
            "phrase_or_labels": self.phrase_or_labels,
            "timing": {"start": self.start, "end": self.end},
            "layout": dict(self.layout),
            "brand_profile": self.brand_profile,
            "technical_status": self.technical_status,
            "visual_status": self.visual_status,
            "approval_status": "pending_review",
        }


def _conflicts(start: float, end: float, existing_treatments: Sequence[tuple[float, float, str]], guard_s: float) -> tuple[bool, str]:
    for t0, t1, name in existing_treatments:
        if start - guard_s < t1 and t0 < end + guard_s:
            return True, f"overlaps or is too close to the existing '{name}' treatment ({t0:.2f}-{t1:.2f}s)"
    return False, ""


def discover_headline_candidates(
    words: Sequence[TWord],
    *,
    stopwords: frozenset[str] = frozenset(),
    hints: Sequence[tuple[float, float, str, float]] = (),
    policy: CompositionPolicy | None = None,
    existing_treatments: Sequence[tuple[float, float, str]] = (),
    brand_profile: str = "default",
    top_n: int = 5,
) -> list[MotionGraphicCandidate]:
    """`primary_headline_typography` candidates: the existing global qualified-headline
    search (sec. 4: concept/claim/payoff/keyword only), never a new eligibility rule."""
    out: list[MotionGraphicCandidate] = []
    for g in discover_global_candidates(words, stopwords=stopwords, hints=hints, policy=policy,
                                         existing_treatments=existing_treatments, top_n=top_n):
        if g.conflict:
            continue
        out.append(MotionGraphicCandidate(
            treatment_type=TreatmentType.PRIMARY_HEADLINE.value, semantic_role=g.role, phrase=g.text,
            start=g.start, end=g.end, brand_profile=brand_profile, global_score=g.global_score,
            layout={"placement": "top"},
        ))
    return out


def discover_keyword_candidates(
    words: Sequence[TWord],
    *,
    stopwords: frozenset[str] = frozenset(),
    hints: Sequence[tuple[float, float, str, float]] = (),
    policy: CompositionPolicy | None = None,
    existing_treatments: Sequence[tuple[float, float, str]] = (),
    brand_profile: str = "default",
    top_n: int = 5,
) -> list[MotionGraphicCandidate]:
    """`keyword_visual` candidates: the same qualified search, restricted to the compact
    `keyword` role (sec. 4) -- a single term too small for a full headline composition."""
    out: list[MotionGraphicCandidate] = []
    for g in discover_global_candidates(words, stopwords=stopwords, hints=hints, policy=policy,
                                         existing_treatments=existing_treatments, top_n=max(top_n * 3, top_n)):
        if g.conflict or g.role != "keyword":
            continue
        out.append(MotionGraphicCandidate(
            treatment_type=TreatmentType.KEYWORD_VISUAL.value, semantic_role=g.role, phrase=g.text,
            start=g.start, end=g.end, brand_profile=brand_profile, global_score=g.global_score,
            layout={"placement": "compact_accent"},
        ))
        if len(out) >= top_n:
            break
    return out


def _split_list_words(beat_words: Sequence[TWord]) -> list[list[TWord]]:
    """Splits a PROCESS_LIST beat's own words into item groups at a comma-bearing word
    (a generic structural cue already present in the transcript's own punctuation --
    not a language list, not an invented boundary)."""
    groups: list[list[TWord]] = []
    cur: list[TWord] = []
    for w in beat_words:
        cur.append(w)
        if any(ch in w.text for ch in _LIST_SEP_CHARS):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return [g for g in groups if g]


def discover_diagram_candidates(
    transcript: Transcript,
    words: Sequence[TWord],
    *,
    stopwords: frozenset[str] = frozenset(),
    policy: CompositionPolicy | None = None,
    existing_treatments: Sequence[tuple[float, float, str]] = (),
    brand_profile: str = "default",
    max_nodes: int = MAX_DIAGRAM_NODES,
    min_nodes: int = MIN_DIAGRAM_NODES,
) -> list[MotionGraphicCandidate]:
    """`simple_diagram` candidates: only where the transcript itself carries structural
    list/framework/process evidence (sec. 2.C/4 -- `SemanticKind.PROCESS_LIST`, the
    existing generic detector; no new one is written here). Every label is a verbatim
    item, independently re-qualified through `select_headline` (sec. 1: only an eligible
    role earns a place on the diagram); an item that does not qualify is dropped rather
    than rewritten or invented. Fewer than `min_nodes` eligible items -> no candidate for
    that beat at all (sec. 4: fail closed, never force a diagram out of weak semantics)."""
    pol = policy or CompositionPolicy()
    out: list[MotionGraphicCandidate] = []
    for beat in detect_semantic_beats(transcript):
        if beat.kind != SemanticKind.PROCESS_LIST:
            continue
        beat_words = [w for w in words if beat.start - 1e-6 <= w.start and w.end <= beat.end + 1e-6]
        labels: list[str] = []
        for group in _split_list_words(beat_words):
            choice = select_headline(words, (group[0].start, group[-1].end), stopwords=stopwords, policy=pol)
            if choice is not None and choice.headline_eligible:
                labels.append(choice.text)
            if len(labels) >= max_nodes:
                break
        if len(labels) < min_nodes:
            continue  # fail closed: not enough independently-qualified items for a diagram
        conflict, reason = _conflicts(beat.start, beat.end, existing_treatments, pol.conflict_guard_s)
        out.append(MotionGraphicCandidate(
            treatment_type=TreatmentType.SIMPLE_DIAGRAM.value, semantic_role="concept", labels=labels,
            start=beat.start, end=beat.end, brand_profile=brand_profile, global_score=round(beat.confidence, 3),
            conflict=conflict, conflict_reason=reason, layout={"nodes": len(labels), "placement": "overlay_zone"},
        ))
    return out


def select_treatments(
    candidates: Sequence[MotionGraphicCandidate],
    *,
    history: TreatmentHistory | None = None,
    conflict_guard_s: float = 0.3,
    max_recent_repeats: int = 1,
) -> list[MotionGraphicCandidate]:
    """Sec. 5's selection priority, applied across candidates already gathered from all
    three discovery functions:

    1. a candidate flagged `.conflict` (it overlaps an existing/pinned treatment) never
       survives -- pinned/approved treatments (B6, Behind-Subject, ...) are preserved;
    2/6. two candidates whose windows are close in time never both survive: no crowding,
       no unauthorized stacking of major visuals at the same moment;
    3. when candidates compete for the same moment, the SIMPLEST treatment wins the slot
       (keyword_visual < primary_headline_typography < simple_diagram);
    4/5. a treatment type used too often recently (per `history`, the existing fatigue
       ledger) is skipped for this occurrence rather than repeated again.

    Deterministic: identical input -> identical output, no randomness, no clock."""
    survivors = [c.model_copy() for c in candidates if not c.conflict]
    survivors.sort(key=lambda c: (_SIMPLICITY_ORDER.get(c.treatment_type, 9), -c.global_score, c.start))
    picked: list[MotionGraphicCandidate] = []
    for c in survivors:
        if any(not (c.end + conflict_guard_s <= p.start or c.start - conflict_guard_s >= p.end) for p in picked):
            continue
        if history is not None and history.repetitions(c.treatment_type) >= max_recent_repeats:
            continue
        picked.append(c)
    picked.sort(key=lambda c: c.start)
    return picked


__all__ = [
    "ELIGIBLE_HEADLINE_ROLES",
    "MAX_DIAGRAM_NODES",
    "MIN_DIAGRAM_NODES",
    "MotionGraphicCandidate",
    "TreatmentType",
    "discover_diagram_candidates",
    "discover_headline_candidates",
    "discover_keyword_candidates",
    "select_treatments",
]
