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
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import Enum

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import Transcript
from video_edit_agent.direction.composition import (
    ELIGIBLE_HEADLINE_ROLES,
    CompositionPolicy,
    TWord,
    _bare,
    discover_global_candidates,
    select_headline,
)
from video_edit_agent.direction.history import TreatmentHistory
from video_edit_agent.direction.semantic_beats import SemanticKind, detect_semantic_beats

MAX_DIAGRAM_NODES = 5
MIN_DIAGRAM_NODES = 2
_LIST_SEP_CHARS = ",،"  # comma, Arabic comma (a generic structural cue, not a language list)

# Generic numeral words (English + Arabic), used only as an OPTIONAL supporting signal
# for framework discovery (sec. 3: "count cue (`five`, `خمس`, etc.) when present") --
# never a requirement, never a client-specific term.
_COUNT_WORDS: dict[int, tuple[str, ...]] = {
    2: ("two", "اثنين", "اثنان"),
    3: ("three", "ثلاثة", "ثلاث"),
    4: ("four", "أربعة", "أربع"),
    5: ("five", "خمسة", "خمس"),
    6: ("six", "ستة", "ست"),
    7: ("seven", "سبعة", "سبع"),
    8: ("eight", "ثمانية", "ثمان"),
}


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


@dataclass(frozen=True)
class FrameworkMember:
    """One framework label and where it came from (sec. 4: label, source type,
    supporting span)."""

    label: str
    source_type: str  # "transcript" | "approved_canonical"
    source_span: tuple[float, float] | None = None
    citation: str | None = None  # only set for "approved_canonical": the caller's own citation of the approval


@dataclass(frozen=True)
class FrameworkSpec:
    """A caller-supplied, already-approved framework to look for (sec. 3/4). The
    anchor phrase and member terms are generic DATA a caller passes in -- whatever
    the reviewed project knowledge says this video's framework actually is -- never
    a language list or a hardcoded concept baked into this module. `canonical_evidence`
    maps a member term (case-insensitive) to a citation string for when that member is
    already-approved project knowledge but is not itself a verbatim transcript span."""

    anchor: str
    member_terms: tuple[str, ...]
    canonical_evidence: dict[str, str] = dc_field(default_factory=dict)
    min_members: int = MIN_DIAGRAM_NODES


def _find_term_span(term: str, words: Sequence[TWord]) -> tuple[float, float] | None:
    """The verbatim transcript span of `term`: an exact contiguous run of words for a
    multi-word term (sec. 8.7/8.8: not limited to one clause or comma/ordinal cues --
    this searches the WHOLE word sequence given to it), or a case-insensitive substring
    match against a single token for a one-word term (covers a term that arrived glued
    to an Arabic prefix, e.g. an attached definite article, as one ASR/correction token).
    `None` when the term does not appear at all: sec. 3 -- never invent a location."""
    parts = [p for p in term.split() if p]
    if not parts:
        return None
    want = [_bare(p).casefold() for p in parts]
    have = [_bare(w.text).casefold() for w in words]
    if len(want) == 1:
        target = want[0]
        for w, bare in zip(words, have, strict=True):
            if target and target in bare:
                return w.start, w.end
        return None
    n = len(want)
    for i in range(len(have) - n + 1):
        if have[i:i + n] == want:
            return words[i].start, words[i + n - 1].end
    return None


def _has_count_cue(words: Sequence[TWord], count: int) -> bool:
    """Whether the transcript says the framework's own member count out loud (sec. 3's
    generic, optional "count cue" signal) -- informational only, never required."""
    cues = _COUNT_WORDS.get(count, ())
    if not cues:
        return False
    have = {_bare(w.text).casefold() for w in words}
    return any(c in have for c in cues)


def _find_member_evidence(
    term: str, words: Sequence[TWord], *, stopwords: frozenset[str], policy: CompositionPolicy,
) -> FrameworkMember | None:
    """`term`'s transcript evidence: found verbatim AND independently re-qualified
    through the existing `select_headline` eligibility gate (sec. 1 -- no new rule).
    A term that is found but does not qualify, or is not found at all, yields nothing
    here (the caller may still accept it via `canonical_evidence`; sec. 4/11)."""
    span = _find_term_span(term, words)
    if span is None:
        return None
    choice = select_headline(words, span, stopwords=stopwords, policy=policy)
    if choice is None or not choice.headline_eligible:
        return None
    return FrameworkMember(label=choice.text, source_type="transcript", source_span=(round(span[0], 3), round(span[1], 3)))


def discover_framework_candidates(
    transcript: Transcript,
    words: Sequence[TWord],
    frameworks: Sequence[FrameworkSpec],
    *,
    stopwords: frozenset[str] = frozenset(),
    policy: CompositionPolicy | None = None,
    existing_treatments: Sequence[tuple[float, float, str]] = (),
    brand_profile: str = "default",
    max_nodes: int = MAX_DIAGRAM_NODES,
) -> list[MotionGraphicCandidate]:
    """`simple_diagram` candidates for a named FRAMEWORK (sec. 3): a caller-supplied
    anchor phrase plus member terms, each independently searched for and re-qualified
    across the WHOLE transcript's words -- not one beat, not same-clause commas or
    ordinal words (sec. 3's gap in `discover_diagram_candidates`, which only recognises
    `SemanticKind.PROCESS_LIST`). A member earns a place only when it is a verbatim,
    independently-qualified transcript span, or when the caller's own
    `FrameworkSpec.canonical_evidence` already cites it as approved project knowledge
    (sec. 4); nothing is invented or hallucinated. Fewer than `spec.min_members`
    supportable members drops that framework entirely -- fail closed, no partial
    diagram (sec. 4/8.10)."""
    pol = policy or CompositionPolicy()
    out: list[MotionGraphicCandidate] = []
    for spec in frameworks:
        members: list[FrameworkMember] = []
        spans: list[tuple[float, float]] = []
        canonical_lookup = {k.casefold(): v for k, v in spec.canonical_evidence.items()}
        for term in spec.member_terms:
            member = _find_member_evidence(term, words, stopwords=stopwords, policy=pol)
            if member is None:
                citation = canonical_lookup.get(term.casefold())
                if citation is not None:
                    member = FrameworkMember(label=term, source_type="approved_canonical", citation=citation)
            if member is not None:
                members.append(member)
                if member.source_span is not None:
                    spans.append(member.source_span)
            if len(members) >= max_nodes:
                break
        if len(members) < spec.min_members:
            continue  # fail closed: not enough supportable members for this framework at all
        start = min((s[0] for s in spans), default=0.0)
        end = max((s[1] for s in spans), default=0.0)
        conflict, reason = _conflicts(start, end, existing_treatments, pol.conflict_guard_s)
        evidence = [
            {
                "label": m.label, "source_type": m.source_type,
                "source_span": list(m.source_span) if m.source_span else None,
                **({"citation": m.citation} if m.citation else {}),
            }
            for m in members
        ]
        out.append(MotionGraphicCandidate(
            treatment_type=TreatmentType.SIMPLE_DIAGRAM.value, semantic_role="concept",
            labels=[m.label for m in members], start=start, end=end, brand_profile=brand_profile,
            global_score=round(len(members) / max(len(spec.member_terms), 1), 3),
            conflict=conflict, conflict_reason=reason,
            layout={
                "nodes": len(members), "placement": "overlay_zone", "anchor": spec.anchor, "evidence": evidence,
                "count_cue": _has_count_cue(words, len(spec.member_terms)),
            },
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
    "FrameworkMember",
    "FrameworkSpec",
    "MotionGraphicCandidate",
    "TreatmentType",
    "discover_diagram_candidates",
    "discover_framework_candidates",
    "discover_headline_candidates",
    "discover_keyword_candidates",
    "select_treatments",
]
