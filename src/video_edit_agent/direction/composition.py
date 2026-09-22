"""`lower_subject_semantic`: lower_subject as a SEMANTIC COMPOSITION, never as free visual variety (Phase 1.3.1).

The speaker moves down to create intentional negative space, then a short verbatim phrase from the approved
transcript uses that space as the primary headline:

    base -> smooth lower_subject -> headline enters in the created space -> hold -> headline exits -> smooth reset_to_base

The visual rhythm engine never schedules `lower_subject` by itself (see `RhythmPolicy.plan_lower_subject`); this module is
the only producer. It needs (1) a headline phrase that is verbatim in the transcript and semantically worth a headline,
and (2) geometry, measured from the face box / head bounds / actual headline height, that creates usable space safely.
Without either the composition is `not_suitable_for_this_window`: nothing is faked and the rhythm engine picks another
safe state. A successful composition is a *candidate*: `approval_status` stays `pending_review` and it is never approved
here. Nothing in this module is brand- or project-specific.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import EDL, Transcript
from video_edit_agent.direction.camera import BASE_ANCHOR_X, BASE_ZOOM, MotionClass
from video_edit_agent.direction.hierarchy import (
    PRIMARY_HEADLINE,
    HeadlinePlacement,
    PrimaryLayer,
    VisualHierarchy,
)
from video_edit_agent.direction.rhythm import Boundary, RhythmPolicy, RhythmRow, RhythmState
from video_edit_agent.direction.suitability import MAX_PHRASE_WORDS
from video_edit_agent.render.reframe import face_frame

TREATMENT = "lower_subject_semantic"
NOT_SUITABLE = "not_suitable_for_this_window"
_SENTENCE = ".!?؟…"
_CLAUSE = ",،؛;:"
_PUNCT = _SENTENCE + _CLAUSE + "\"'«»()[]-–—"
_S = RhythmState


class CompositionPolicy(BaseModel):
    """Absolute limits of the composition. Fractions are of the frame height (top-anchored crop: the frame keeps its top edge)."""

    target_gain: float = 0.065  # the preferred face/head-top gain (the spec's ~5-8%)
    preferred_gain_lo: float = 0.05
    preferred_gain_hi: float = 0.08
    min_gain: float = 0.03  # less than this is not visually meaningful: reject
    max_zoom: float = 1.24  # a composition may zoom further than free rhythm (1.14) because it is validated and paired with a headline
    min_body_frac: float = 0.78  # a top-anchored crop keeps at least this fraction of the source height (natural shoulders/body)
    caption_safe_top: float = 0.70
    top_safe: float = 0.12  # the headline stays below this fraction of the frame height
    head_gap: float = 0.02  # min clear space between the headline and the head/hair
    hair_over_face: float = 0.35  # head top ~ face top - this x face height, when no head/hair mask bound is measured
    move_s: float = 0.9
    return_s: float = 0.9
    text_in_lead_s: float = 0.15  # the headline starts fading in this long before its first word
    text_in_s: float = 0.35
    text_hold_after_s: float = 0.35  # held this long after the phrase ends
    text_out_s: float = 0.30
    min_phrase_score: float = 0.5
    max_words: int = 3


class LowerSubjectGeometry(BaseModel):
    """Measured base vs target framing. All values are fractions of the frame height unless noted."""

    status: str = "ok"  # ok | rejected
    reason: str = ""
    head_top_base: float = 0.0  # top of head/hair at base (= base headroom above the head)
    face_top_base: float = 0.0
    face_bottom_base: float = 0.0
    zoom: float = BASE_ZOOM
    anchor_x: float = BASE_ANCHOR_X
    anchor_y: float = 0.0
    base_headroom: float = 0.0
    target_headroom: float = 0.0
    delta: float = 0.0  # target - base headroom above the head
    face_top_target: float = 0.0
    face_bottom_target: float = 0.0
    visible_body_frac: float = 1.0  # fraction of the source height that stays in frame
    headline_h: float = 0.0
    headline_top: float = 0.0
    headline_bottom: float = 0.0
    headline_to_head_clearance: float = 0.0
    preferred_gain_met: bool = False
    limited_by: str = ""  # what capped the zoom (empty = the preferred gain was reachable)


def solve_lower_subject(
    face_box: tuple[float, float, float, float] | None,
    headline_h: float,
    *,
    head_top: float | None = None,
    policy: CompositionPolicy | None = None,
) -> LowerSubjectGeometry:
    """The zoom (top-anchored, so the head moves DOWN in the frame) that creates a usable headline band above the head.

    `face_box` is the measured face (normalised x, y, w, h), `head_top` the measured top of the head/hair (else estimated from
    the face), `headline_h` the ACTUAL height of the headline as rendered. Rejects (status 'rejected', with the reason) when
    a safe zoom cannot give the headline usable space or a visually meaningful gain."""
    pol = policy or CompositionPolicy()
    g = LowerSubjectGeometry(headline_h=round(headline_h, 4))
    if face_box is None:
        g.status, g.reason = "rejected", "no measured face box: lower_subject cannot be proven safe"
        return g
    _, fy, _, fh = face_box
    top = head_top if head_top is not None and head_top > 0 else max(0.02, fy - pol.hair_over_face * fh)
    g.head_top_base, g.face_top_base, g.face_bottom_base, g.base_headroom = round(top, 4), round(fy, 4), round(fy + fh, 4), round(top, 4)
    need = pol.top_safe + headline_h + pol.head_gap  # the head top must end up at least this low
    z_pref = 1.0 + pol.target_gain / top
    z_need = need / top
    ceilings = {"max_zoom": pol.max_zoom, "caption_band": pol.caption_safe_top / (fy + fh), "body_crop": 1.0 / pol.min_body_frac}
    cap_name = min(ceilings, key=lambda k: ceilings[k])
    ceiling = ceilings[cap_name]
    z = max(z_pref, z_need)
    if z > ceiling:
        z, g.limited_by = ceiling, cap_name
    z = round(z, 4)
    g.zoom = z
    g.anchor_x, g.anchor_y = BASE_ANCHOR_X, 0.0
    g.target_headroom = round(top * z, 4)
    g.delta = round(g.target_headroom - g.base_headroom, 4)
    _, ft, _, fb = face_frame(face_box, z, g.anchor_x, g.anchor_y)
    g.face_top_target, g.face_bottom_target = round(ft, 4), round(fb, 4)
    g.visible_body_frac = round(1.0 / z, 4)
    band_bottom = g.target_headroom - pol.head_gap
    room = band_bottom - pol.top_safe
    if g.delta < pol.min_gain - 1e-9:
        g.status, g.reason = "rejected", f"the safe zoom ({z:.3f}, capped by {g.limited_by or 'the target'}) adds only {g.delta:.3f} of headroom (< {pol.min_gain:.2f})"
    elif room < headline_h - 1e-9:
        g.status, g.reason = "rejected", f"usable headline space is {room:.3f} but the headline needs {headline_h:.3f}"
    elif g.face_bottom_target > pol.caption_safe_top + 1e-9:
        g.status, g.reason = "rejected", f"the lowered face would reach {g.face_bottom_target:.2f}, inside the caption band"
    else:
        g.headline_top = round(pol.top_safe + (room - headline_h) / 2, 4)
        g.headline_bottom = round(g.headline_top + headline_h, 4)
        g.headline_to_head_clearance = round(g.target_headroom - g.headline_bottom, 4)
        g.preferred_gain_met = pol.preferred_gain_lo - 1e-9 <= g.delta <= pol.preferred_gain_hi + 1e-9 or g.delta > pol.preferred_gain_hi
        g.reason = (f"head top {g.base_headroom:.3f} -> {g.target_headroom:.3f} (+{g.delta:.3f}); headline {g.headline_top:.3f}-"
                    f"{g.headline_bottom:.3f}, {g.headline_to_head_clearance:.3f} clear of the head; face bottom {g.face_bottom_target:.2f} above the caption band")
    return g


# --------------------------------------------------------------------------
# The headline phrase: verbatim from the approved transcript, or nothing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TWord:
    """A transcript word on the timeline."""

    text: str
    start: float
    end: float


class HeadlineChoice(BaseModel):
    text: str
    start: float
    end: float
    score: float
    words: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    semantic_kind: str | None = None
    semantic_confidence: float | None = None
    semantic_source: str = "automatic"


def timeline_words(transcript: Transcript, edl: EDL) -> list[TWord]:
    """The approved transcript's words on the EDL timeline (a word outside every kept clip is dropped)."""
    out: list[TWord] = []
    for clip in edl.clips:
        speed = clip.speed or 1.0
        for seg in transcript.segments:
            for w in seg.words:
                if clip.source_in - 1e-6 <= w.start < clip.source_out - 1e-6:
                    a = clip.timeline_in + (w.start - clip.source_in) / speed
                    b = clip.timeline_in + (min(w.end, clip.source_out) - clip.source_in) / speed
                    out.append(TWord(w.word, round(a, 3), round(b, 3)))
    return sorted(out, key=lambda x: x.start)


def _bare(word: str) -> str:
    return word.strip(_PUNCT + " ")


def is_verbatim(phrase: str, words: Sequence[TWord]) -> bool:
    """True when `phrase` is a contiguous run of transcript words (edge punctuation aside): nothing rewritten, translated or invented."""
    want = [_bare(t) for t in phrase.split() if _bare(t)]
    have = [_bare(w.text) for w in words]
    n = len(want)
    return n > 0 and any(have[i:i + n] == want for i in range(len(have) - n + 1))


def _word_frequency(words: Sequence[TWord]) -> Counter[str]:
    """How often each bare word form recurs across the WHOLE approved transcript. A generic, data-driven stand-in for
    "is this a specific concept or a common connector/filler": no language list, no hardcoded phrase, just counting."""
    return Counter(_bare(w.text) for w in words if _bare(w.text))


def select_headline(
    words: Sequence[TWord],
    window: tuple[float, float],
    *,
    stopwords: frozenset[str] = frozenset(),
    hints: Sequence[tuple[float, float, str, float]] = (),
    policy: CompositionPolicy | None = None,
    pinned_phrase: str | None = None,
) -> HeadlineChoice | None:
    """The best short verbatim phrase inside `window`, or None when nothing is worth a headline.

    Ranks by SEMANTIC USEFULNESS, not merely brevity: a phrase that starts a clause reads as a headline, two short
    content words is the natural size, function words at the edges are penalised, a semantic hint (start, end, kind,
    confidence) covering it raises the score, and — the key generic signal for "concept vs. discourse filler" — a
    phrase built from words that are RARE across the whole approved transcript outranks one built from words that
    recur constantly (connectors, instruction verbs, discourse setup tend to repeat; a specific claim or keyword
    usually does not). Nothing here is a language list or a hardcoded phrase: rarity is counted from the transcript
    itself. `pinned_phrase` (the user's own choice) must still be verbatim; it is scored, never rewritten."""
    pol = policy or CompositionPolicy()
    span = [w for w in words if window[0] - 1e-6 <= w.start and w.end <= window[1] + 1e-6]
    if pinned_phrase and not is_verbatim(pinned_phrase, span):
        return None
    freq = _word_frequency(words)
    max_freq = max(freq.values(), default=1)
    best: HeadlineChoice | None = None
    limit = min(pol.max_words, MAX_PHRASE_WORDS)
    for i in range(len(span)):
        for n in range(1, limit + 1):
            run = span[i:i + n]
            if len(run) < n:
                break
            if any(w.text.rstrip()[-1:] in _SENTENCE + _CLAUSE for w in run[:-1]):
                break  # never straddles a clause or sentence end
            if n > 1 and any(b.start - a.end > 0.35 for a, b in pairwise(run)):
                break  # a long pause is a boundary too
            text = " ".join(_bare(w.text) for w in run)
            if pinned_phrase and [_bare(t) for t in pinned_phrase.split()] != text.split():
                continue
            score, why = 0.0, []
            prev = next((w for w in reversed(words) if w.end <= run[0].start + 1e-6 and w is not run[0]), None)  # the word before it in the whole transcript, not just this window
            if prev is None or prev.text.rstrip()[-1:] in _SENTENCE + _CLAUSE or run[0].start - prev.end >= 0.2:
                score, why = score + 0.35, [*why, "starts a clause"]
            if n == 2:
                score, why = score + 0.2, [*why, "two words: a natural headline size"]
            elif n == 1:
                score += 0.1
            bare_words = [_bare(w.text) for w in run]
            edges = (bare_words[0], bare_words[-1])
            if all(len(e) >= 3 and e not in stopwords for e in edges):
                score, why = score + 0.2, [*why, "content words at both edges"]
            dur = run[-1].end - run[0].start
            if dur >= 0.7:
                score, why = score + 0.1, [*why, f"{dur:.1f}s long: readable"]
            hint = next(((k, c) for a, b, k, c in hints if a - 0.3 <= run[0].start and run[-1].end <= b + 0.3), None)
            if hint is not None:
                score, why = score + 0.15 * hint[1], [*why, f"semantic hint {hint[0]} ({hint[1]:.2f})"]
            # Semantic usefulness (Phase 1.3.2): concept-bearing and understandable out of context outranks
            # discourse setup / instruction filler / connectors, using only signals measured from this transcript.
            rarity = sum(1.0 - freq.get(w, 0) / max_freq for w in bare_words) / max(len(bare_words), 1)
            if rarity > 0:
                score += 0.30 * rarity
                if rarity >= 0.6:
                    why = [*why, f"a specific concept, rare in the transcript (rarity {rarity:.2f})"]
            content = sum(1 for w in bare_words if len(w) >= 3 and w not in stopwords)
            density = content / max(len(bare_words), 1)
            if density == 1.0 and len(bare_words) > 1:
                score, why = score + 0.15, [*why, "every word carries content: reads as a compact idea"]
            elif density > 0:
                score += 0.15 * density
            chars = len(text.replace(" ", ""))
            if 4 <= chars <= 26:
                score, why = score + 0.05, [*why, "a readable standalone length"]
            cand = HeadlineChoice(text=text, start=run[0].start, end=run[-1].end, score=round(score, 3), words=bare_words,
                                  reasons=why, semantic_kind=hint[0] if hint else None, semantic_confidence=hint[1] if hint else None,
                                  semantic_source="user_pinned" if pinned_phrase else "automatic")
            if best is None or (cand.score, -cand.start) > (best.score, -best.start):
                best = cand
    if best is None or (best.score < pol.min_phrase_score and not pinned_phrase):
        return None
    return best


# --------------------------------------------------------------------------
# The composition (rows the rhythm engine inserts) and its review state
# --------------------------------------------------------------------------


class LowerSubjectComposition(BaseModel):
    treatment: str = TREATMENT
    status: str = "ok"  # ok | not_suitable_for_this_window
    reason: str = ""
    start: float = 0.0
    end: float = 0.0
    state: str = _S.LOWER_SUBJECT.value
    rows: list[RhythmRow] = Field(default_factory=list)
    changes: list[float] = Field(default_factory=list)
    phrase: str = ""
    phrase_start: float = 0.0
    phrase_end: float = 0.0
    headline_in: tuple[float, float] = (0.0, 0.0)  # fade-in span
    headline_out: tuple[float, float] = (0.0, 0.0)  # fade-out span
    geometry: LowerSubjectGeometry = Field(default_factory=LowerSubjectGeometry)
    choice: HeadlineChoice | None = None
    hierarchy: VisualHierarchy = Field(default_factory=VisualHierarchy)
    primary_visual: str = PRIMARY_HEADLINE
    semantic_source: str = "automatic"
    technical_status: str = "passed"
    visual_status: str = "candidate_ready"
    approval_status: str = "pending_review"  # NEVER approved here


def hierarchy() -> VisualHierarchy:
    """Headline leads; the speaker stays visible (lower in the frame); captions are reduced, never hidden."""
    return VisualHierarchy(primary_layer=PrimaryLayer.HEADLINE.value, caption_role="reduced", headline_role=PRIMARY_HEADLINE,
                           headline_placement=HeadlinePlacement.TOP.value, speaker_visibility="full",
                           notes=["the headline is the primary visual; captions stay readable but reduced; no simultaneous large icon"])


def _snap_start(t: float, words: Sequence[TWord], bounds: Sequence[Boundary]) -> float:
    """A smooth move may start on any word boundary: never inside a word. Snaps back to the nearest boundary at or before `t`."""
    inside = next((w for w in words if w.start + 1e-6 < t < w.end - 1e-6), None)
    if inside is not None:
        t = inside.start
    near = [b.t for b in bounds if b.t <= t + 1e-6 and t - b.t <= 0.35 and not any(w.start + 1e-6 < b.t < w.end - 1e-6 for w in words)]
    return round(max(near), 3) if near else round(t, 3)


def _snap_return(t: float, words: Sequence[TWord], bounds: Sequence[Boundary]) -> float:
    """The return starts on a word boundary at or after `t` (the headline has finished by then)."""
    later = [b.t for b in bounds if t - 1e-6 <= b.t <= t + 0.6 and not any(w.start + 1e-6 < b.t < w.end - 1e-6 for w in words)]
    if later:
        return round(min(later), 3)
    inside = next((w for w in words if w.start + 1e-6 < t < w.end - 1e-6), None)
    return round(inside.end if inside else t, 3)


def _rows(g: LowerSubjectGeometry, t0: float, t1: float, t2: float, t3: float, rp: RhythmPolicy) -> list[RhythmRow]:
    """base -> lower_subject (t0..t1) -> hold (t1..t2) -> reset_to_base (t2..t3): a smooth move in, a smooth glide out, back on the canonical base."""
    cls = MotionClass.SMOOTH.value
    move = RhythmRow(start=t0, end=t1, state=_S.LOWER_SUBJECT.value, zoom_from=BASE_ZOOM, zoom_to=g.zoom, anchor_x=g.anchor_x, anchor_y=g.anchor_y,
                     motion_start=t0, motion_end=t1, motion_class=cls, anchor_x_from=BASE_ANCHOR_X)
    hold = RhythmRow(start=t1, end=t2, state=_S.HOLD.value, zoom_from=g.zoom, zoom_to=g.zoom, anchor_x=g.anchor_x, anchor_y=g.anchor_y)
    back = RhythmRow(start=t2, end=t3, state=_S.RESET_TO_BASE.value, zoom_from=g.zoom, zoom_to=BASE_ZOOM, anchor_x=BASE_ANCHOR_X,
                     anchor_y=g.anchor_y, motion_start=t2, motion_end=t3, motion_class=cls, anchor_x_from=g.anchor_x, anchor_y_from=g.anchor_y)
    return [move, hold, back]


def compose_lower_subject(
    words: Sequence[TWord],
    window: tuple[float, float],
    *,
    face_box: tuple[float, float, float, float] | None,
    headline_height: Callable[[str], float],
    head_top: float | None = None,
    stopwords: frozenset[str] = frozenset(),
    hints: Sequence[tuple[float, float, str, float]] = (),
    bounds: Sequence[Boundary] | None = None,
    policy: CompositionPolicy | None = None,
    rhythm_policy: RhythmPolicy | None = None,
    pinned_phrase: str | None = None,
) -> LowerSubjectComposition:
    """A `lower_subject_semantic` composition inside `window`, or `not_suitable_for_this_window` with the reason.

    `headline_height(text)` returns the ACTUAL rendered height of the headline (fraction of the frame height), so the
    negative space is sized from what will be drawn, not from a constant."""
    pol, rp = policy or CompositionPolicy(), rhythm_policy or RhythmPolicy()
    out = LowerSubjectComposition(start=window[0], end=window[1])

    def refuse(why: str) -> LowerSubjectComposition:
        out.status, out.reason = NOT_SUITABLE, why
        out.technical_status, out.visual_status, out.rows, out.changes = "not_applicable", "not_suitable", [], []
        return out

    choice = select_headline(words, window, stopwords=stopwords, hints=hints, policy=pol, pinned_phrase=pinned_phrase)
    if choice is None:
        return refuse("no short verbatim phrase in this window is worth a headline (nothing is invented or rewritten)")
    if not is_verbatim(choice.text, words):  # belt and braces: a phrase that is not in the transcript never gets through
        return refuse("the phrase is not verbatim in the approved transcript")
    geo = solve_lower_subject(face_box, headline_height(choice.text), head_top=head_top, policy=pol)
    if geo.status != "ok":
        out.choice, out.geometry, out.phrase = choice, geo, choice.text
        return refuse(f"no safe geometry for the headline: {geo.reason}")
    bs = list(bounds) if bounds is not None else []
    t0 = _snap_start(choice.start - pol.text_in_lead_s - (pol.move_s - 0.25), words, bs)  # the fade-in lands on the last part of the move
    t1 = round(t0 + pol.move_s, 3)
    out_start = round(choice.end + pol.text_hold_after_s, 3)
    t2 = _snap_return(out_start, words, bs)
    t3 = round(t2 + pol.return_s, 3)
    if t0 < window[0] - 1.0 or t3 - t0 < rp.excursion_min_s - 1e-6:
        return refuse("the composition does not fit the window")
    rows = _rows(geo, t0, t1, t2, t3, rp)
    detail = {"phrase": choice.text, "geometry": geo.model_dump(), "treatment": TREATMENT}
    for r in rows:
        r.composition, r.composition_detail, r.semantic_source = TREATMENT, detail, "user_pinned" if pinned_phrase else "auto"
        r.composition_reason = geo.reason
    rows[0].reset_plan = rows[-1].reset_plan = f"reset_to_base glides {t2:.2f}-{t3:.2f}s and settles on the canonical base"
    out.rows, out.start, out.end = rows, t0, t3
    out.changes = [t0, t1, t2, t3]
    out.phrase, out.phrase_start, out.phrase_end, out.choice, out.geometry = choice.text, choice.start, choice.end, choice, geo
    out.headline_in = (round(choice.start - pol.text_in_lead_s, 3), round(choice.start - pol.text_in_lead_s + pol.text_in_s, 3))
    out.headline_out = (out_start, round(out_start + pol.text_out_s, 3))
    out.hierarchy = hierarchy()
    out.semantic_source = choice.semantic_source
    out.reason = f"«{choice.text}»: " + "; ".join(choice.reasons)
    return out


def review_state(c: LowerSubjectComposition) -> dict[str, str]:
    """The fields the review shows. A candidate is never approved here."""
    return {
        "treatment": c.treatment if c.status == "ok" else NOT_SUITABLE, "primary_visual": c.primary_visual if c.status == "ok" else "speaker",
        "semantic_source": c.semantic_source, "technical_status": c.technical_status, "visual_status": c.visual_status,
        "approval_status": c.approval_status, "caption_role": c.hierarchy.caption_role if c.status == "ok" else "normal",
    }


__all__ = [
    "NOT_SUITABLE",
    "TREATMENT",
    "CompositionPolicy",
    "HeadlineChoice",
    "LowerSubjectComposition",
    "LowerSubjectGeometry",
    "TWord",
    "compose_lower_subject",
    "hierarchy",
    "is_verbatim",
    "review_state",
    "select_headline",
    "solve_lower_subject",
    "timeline_words",
]
