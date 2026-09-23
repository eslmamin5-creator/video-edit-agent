"""Product Freeze (Phase 1.5): user-facing editing profiles and the Motion Graphics
feature flag, plus the real wiring that plans Visual Rhythm and (when a beat is
strongly enough qualified) a `lower_subject_semantic` headline composition for the
actual final render.

Profiles are density/eligibility knobs ONLY -- never transcript fidelity, safety,
review-first behaviour, brand rules or determinism (spec section 2). Nothing here is
project- or brand-specific: no per-client behaviour, no hardcoded phrase.

Motion Graphics (`primary_headline_typography`, `keyword_visual`, `simple_diagram`,
`direction/motion_graphics.py`) is a SEPARATE system from `lower_subject_semantic`
(this module / `direction/composition.py`). It stays experimental/off by default
(spec section 4): `motion_graphics_enabled(mode)` is the single gate a caller checks
before ever scheduling an MG treatment; nothing here removes the MG implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum

from pydantic import BaseModel

from video_edit_agent.captions.headline import (
    HeadlineSpec,
    add_headline,
    headline_style,
    reduce_captions,
)
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.schemas import EDL, Transcript
from video_edit_agent.direction.camera_timeline import (
    apply_timeline,
    build_camera_timeline,
    legacy_events,
)
from video_edit_agent.direction.composition import (
    CompositionPolicy,
    GlobalCandidate,
    LowerSubjectComposition,
    TWord,
    compose_lower_subject,
    discover_global_candidates,
    rank_verified_candidates,
    timeline_words,
)
from video_edit_agent.direction.rhythm import RhythmPolicy, plan_rhythm
from video_edit_agent.motion.legibility import estimate_title_box


class EditingProfile(str, Enum):
    MINIMAL = "minimal"
    BALANCED = "balanced"
    DYNAMIC = "dynamic"


DEFAULT_PROFILE = EditingProfile.BALANCED

# Reuses direction.camera's existing low/medium/high density grammar (spec section
# 25's motion-energy words) -- no new camera vocabulary.
_CAMERA_ENERGY: dict[EditingProfile, str] = {
    EditingProfile.MINIMAL: "low",
    EditingProfile.BALANCED: "medium",
    EditingProfile.DYNAMIC: "high",
}

# How strongly a lower_subject_semantic candidate must score (Phase 1.3.4's
# `GlobalCandidate.global_score`) before it is eligible for automatic scheduling.
# minimal: only strongly justified beats; balanced: strongly qualified; dynamic:
# more readily, but never unqualified (spec section 2) -- still fails closed below
# the base qualification bar the discovery/geometry functions already enforce.
_SEMANTIC_MIN_SCORE: dict[EditingProfile, float] = {
    EditingProfile.MINIMAL: 0.85,
    EditingProfile.BALANCED: 0.6,
    EditingProfile.DYNAMIC: 0.45,
}

# How many lower_subject_semantic headlines a single video may receive automatically.
# Never "more" for dynamic in an unbounded/random way (spec section 2: "no random or
# repetitive pattern") -- one restrained, verified headline per video, any profile.
_MAX_AUTO_HEADLINES = 1


def parse_profile(value: str | EditingProfile | None) -> EditingProfile:
    if isinstance(value, EditingProfile):
        return value
    try:
        return EditingProfile((value or DEFAULT_PROFILE.value).lower())
    except ValueError:
        return DEFAULT_PROFILE


def camera_energy_for_profile(profile: EditingProfile) -> str:
    return _CAMERA_ENERGY[profile]


def semantic_min_score_for_profile(profile: EditingProfile) -> float:
    return _SEMANTIC_MIN_SCORE[profile]


class MotionGraphicsMode(str, Enum):
    OFF = "off"
    EXPERIMENTAL = "experimental"


DEFAULT_MOTION_GRAPHICS_MODE = MotionGraphicsMode.OFF


def parse_motion_graphics_mode(value: str | MotionGraphicsMode | None) -> MotionGraphicsMode:
    if isinstance(value, MotionGraphicsMode):
        return value
    try:
        return MotionGraphicsMode((value or DEFAULT_MOTION_GRAPHICS_MODE.value).lower())
    except ValueError:
        return DEFAULT_MOTION_GRAPHICS_MODE


def motion_graphics_enabled(mode: str | MotionGraphicsMode | None) -> bool:
    """The single gate: True only under the explicit "experimental" opt-in. Any caller
    that might schedule `primary_headline_typography` / `keyword_visual` /
    `simple_diagram` (Phase 1.4's `direction/motion_graphics.py`) must check this first;
    "off" (the default) means no MG candidate is ever scheduled onto a real timeline,
    even though MG may still be discovered/rendered as a diagnostic-only candidate."""
    return parse_motion_graphics_mode(mode) is MotionGraphicsMode.EXPERIMENTAL


class HeadlineWiring(BaseModel):
    """The single `lower_subject_semantic` headline chosen for a real render, or none."""

    candidate: GlobalCandidate | None = None
    composition: LowerSubjectComposition | None = None

    @property
    def ok(self) -> bool:
        return self.composition is not None and self.composition.status == "ok"


def _headline_height_fn(style_font_px: int, width: int, height: int) -> Callable[[str], float]:
    """A real `headline_height` callback (fraction of frame height) for
    `compose_lower_subject`, using the same lightweight ink estimator the hook-title
    layout already uses (`motion.legibility.estimate_title_box`) -- no ffmpeg render
    per candidate, no new sizing heuristic."""

    def _measure(text: str) -> float:
        _, box_h = estimate_title_box(text, style_font_px, max_width_px=round(width * 0.86), canvas_w=width, canvas_h=height, pad_px=0)
        return box_h / height

    return _measure


def discover_and_verify_headline(
    words: Sequence[TWord],
    *,
    profile: EditingProfile,
    face_box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    headline_font_px: int,
    existing_treatments: Sequence[tuple[float, float, str]] = (),
    stopwords: frozenset[str] = frozenset(),
    duration: float | None = None,
    pinned: tuple[str, float, float] | None = None,
    allow_auto_discovery: bool = True,
) -> HeadlineWiring:
    """Reuses the real, generic, already-tested Phase 1.3.4/1.3.5 pipeline -- global
    candidate discovery, then targeted geometry verification, then the final
    feasibility-adjusted ranking -- to find (never invent) the ONE headline a video's
    own transcript qualifies for automatically. Fails closed: an empty/too-weak/
    ungeometrically-safe result returns no composition, exactly like a fresh project
    with no strongly qualified beat. Not tied to any particular client: any transcript
    with a qualifying concept/claim/payoff/keyword beat can produce one.

    `pinned` is an explicitly APPROVED (phrase, start, end) from the project's own
    persisted review state (spec Phase 1.5.1: approved-state integrity) -- when given,
    it is verified in place of running discovery, exactly the same technical/geometry
    gates a discovered candidate would face, and it is never displaced by a fresh
    auto-discovered candidate. `allow_auto_discovery=False` means this is an existing,
    previously-reviewed project with no approved headline for this run: fresh
    auto-discovery must not silently burn an unreviewed candidate into the render
    (a `pending_review` candidate stays a reviewable candidate only)."""
    if not words or face_box is None:
        return HeadlineWiring()
    comp_policy = CompositionPolicy()
    if pinned is not None:
        phrase, start, end = pinned
        height_fn = _headline_height_fn(headline_font_px, width, height)
        window = (max(0.0, start - 6.0), end + 10.0 if duration is None else min(duration, end + 10.0))
        comp = compose_lower_subject(
            words, window, face_box=face_box, headline_height=height_fn, stopwords=stopwords,
            policy=comp_policy, rhythm_policy=RhythmPolicy(), pinned_phrase=phrase,
        )
        if comp.status != "ok":
            return HeadlineWiring()
        comp.semantic_source, comp.approval_status = "user_pinned", "approved"
        return HeadlineWiring(candidate=None, composition=comp)
    if not allow_auto_discovery:
        return HeadlineWiring()
    candidates = discover_global_candidates(
        words, stopwords=stopwords, policy=comp_policy, existing_treatments=existing_treatments, top_n=5,
    )
    threshold = semantic_min_score_for_profile(profile)
    qualified = [c for c in candidates if not c.conflict and c.global_score >= threshold]
    if not qualified:
        return HeadlineWiring()
    height_fn = _headline_height_fn(headline_font_px, width, height)
    end = duration if duration is not None else max((c.end for c in candidates), default=0.0)
    verified: list[tuple[GlobalCandidate, LowerSubjectComposition]] = []
    for cand in qualified[: _MAX_AUTO_HEADLINES + 2]:  # a small margin of runner-ups for the ranker
        window = (max(0.0, cand.start - 6.0), min(end, cand.end + 10.0) if end else cand.end + 10.0)
        comp = compose_lower_subject(
            words, window, face_box=face_box, headline_height=height_fn, stopwords=stopwords,
            policy=comp_policy, rhythm_policy=RhythmPolicy(), pinned_phrase=cand.text,
        )
        verified.append((cand, comp))
    winner = rank_verified_candidates(verified)
    if winner is None:
        return HeadlineWiring()
    cand, comp = winner
    return HeadlineWiring(candidate=cand, composition=comp)


class VisualRhythmResult(BaseModel):
    """What real production wiring did to this video's camera + headline."""

    headline: HeadlineWiring = HeadlineWiring()
    rhythm_states: list[str] = []


def plan_and_apply_visual_rhythm(
    transcript: Transcript,
    edl: EDL,
    *,
    profile: EditingProfile,
    face_box: tuple[float, float, float, float] | None,
    headline_font_px: int,
    behind_subject_windows: Sequence[tuple[float, float]] = (),
    stopwords: frozenset[str] = frozenset(),
    pinned_headline: tuple[str, float, float] | None = None,
    allow_auto_headline_discovery: bool = True,
) -> VisualRhythmResult:
    """The real production wiring (spec section 3 step 5 / section 1's Visual Rhythm):
    plans the low-semantic camera rhythm for the whole video at the profile's density,
    discovers+verifies at most one qualified `lower_subject_semantic` headline and
    (fail-closed) inserts it as an owned composition, then normalises everything into
    the ONE canonical camera timeline and applies it to `edl.clips[*].reframe` in
    place -- the same real, generic modules the interactive edit-plan review already
    uses (`direction.rhythm`, `direction.camera_timeline`), just invoked for the
    actual render instead of only for a reviewable plan. Approved Behind-Subject
    windows are passed through untouched so a rhythm/headline excursion never
    collides with them (they stay exclusively owned, per their own approval).

    `pinned_headline` (phrase, start, end) is an explicitly approved
    `lower_subject_semantic` treatment from the project's own persisted review state:
    when present it is used instead of auto-discovery and cannot be displaced by a
    fresh candidate. `allow_auto_headline_discovery=False` (an existing, previously
    reviewed project with no approved headline this run) stops a merely
    `pending_review`/never-reviewed automatic candidate from silently entering the
    render; a genuinely fresh project (no persisted review state at all) keeps
    discovering and burning its best qualified candidate as before."""
    end = max([transcript.duration, *[s.end for s in transcript.segments]]) if transcript.segments else 0.0
    words = timeline_words(transcript, edl)
    headline = discover_and_verify_headline(
        words, profile=profile, face_box=face_box, width=edl.width, height=edl.height,
        headline_font_px=headline_font_px,
        existing_treatments=[(s, e, "behind_subject_text") for s, e in behind_subject_windows],
        stopwords=stopwords, duration=end,
        pinned=pinned_headline, allow_auto_discovery=allow_auto_headline_discovery,
    )
    rhythm_policy = RhythmPolicy(energy=camera_energy_for_profile(profile))
    compositions = [headline.composition] if headline.ok else []
    rhythm = plan_rhythm(transcript, start=0.0, end=end, policy=rhythm_policy, face_box=face_box, compositions=compositions)
    timeline = build_camera_timeline(
        rhythm, face_box=face_box, legacy=legacy_events(edl), behind_subject=list(behind_subject_windows), scope=(0.0, end),
    )
    apply_timeline(edl, timeline, face_box=face_box)
    return VisualRhythmResult(headline=headline, rhythm_states=[r.state for r in rhythm.rows])


def burn_headline_into_captions(
    normal_ass: str, reduced_ass: str, composition: LowerSubjectComposition, caption_style: CaptionStyle,
    headline_font_px: int, width: int, height: int,
) -> str:
    """The headline event + reduced captions during its window, added to the already-built normal caption ASS
    (spec: the headline leads, captions are REDUCED, never hidden). `reduced_ass` must come from the same
    transcript/EDL/chunks as `normal_ass`, styled smaller (so `reduce_captions` can line the events up 1:1)."""
    merged = reduce_captions(normal_ass, reduced_ass, (composition.start, composition.end))
    hstyle = headline_style(caption_style, headline_font_px)
    geo = composition.geometry
    y_px = geo.headline_top * height if geo is not None else height * 0.10
    spec = HeadlineSpec(
        text=composition.phrase, y_px=y_px, fade_in=composition.headline_in, fade_out=composition.headline_out,
        font_px=headline_font_px,
    )
    return add_headline(merged, hstyle, spec, width)


__all__ = [
    "DEFAULT_MOTION_GRAPHICS_MODE",
    "DEFAULT_PROFILE",
    "EditingProfile",
    "HeadlineWiring",
    "MotionGraphicsMode",
    "VisualRhythmResult",
    "burn_headline_into_captions",
    "camera_energy_for_profile",
    "discover_and_verify_headline",
    "motion_graphics_enabled",
    "parse_motion_graphics_mode",
    "parse_profile",
    "plan_and_apply_visual_rhythm",
    "semantic_min_score_for_profile",
]
