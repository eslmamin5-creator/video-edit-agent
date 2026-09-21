"""Behind-subject text: approved edit-plan phrase -> timed, judged, composited.

One implementation, used by the final render (`core/pipeline.py`) and by the
behind-subject micro-preview (`render/micro_preview.py`), so a preview shows
exactly what the final render will do:

    plan_behind_subject        approved slots -> `BehindSubjectPlan`s: WHEN the phrase
                               shows (its spoken words), WHERE it can sit and still read
                               (the subject mask against the real glyph shapes), in what
                               colour (below the captions in the hierarchy), and whether
                               the effect passes its quality gate at all
    plan_behind_subject_text   the same, as `AnimationSpec`s only
    compose_behind_subject     one spec -> the ordered overlay layers
                               (text under, refined subject cutout on top)

Editorial logic (adapted from the Arabic reference editor's behind-text pass, not its
Apple-Vision/swiftc dependency): the text is timed to the phrase's own words with a
short lead-in and a readable hold (`phrase_timing`), it is ONE coordinated group, and
the subject hides only a controlled share of it while the key word stays readable
(`occlusion`). When the gate fails the plan says `behind_subject_not_recommended`
and falls back to plain kinetic typography placed clear of the subject, or to no text
at all (the speaker punch-in alone).

The words come ONLY from a slot the user approved (`status` approved/changed,
effective treatment `behind_subject_text`, non-empty `text`); a pending, rejected
or text-less slot yields nothing, and no text is ever invented or reworded here.
Colour, size and placement derive from the Brand Profile, the caption style and the
footage; nothing here names a brand, a colour, a phrase or a timestamp.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.treatment import Treatment
from video_edit_agent.captions.safe_zone import SafeZone
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.schemas import (
    EDL,
    AnimationKind,
    AnimationSpec,
    EDLClip,
    MotionPlanItem,
    Transcript,
    Word,
)
from video_edit_agent.motion.glyph_probe import Probe, default_probe
from video_edit_agent.motion.legibility import (
    _palette,
    _worst_contrast,
    brand_safe_zone,
    relative_luminance,
)
from video_edit_agent.motion.occlusion import (
    DEFAULT_POLICY,
    GlyphLayout,
    OcclusionPolicy,
    Placement,
    SearchResult,
    caption_zone,
    key_word_indices,
    search_placement,
)
from video_edit_agent.motion.phrase_timing import (
    DEFAULT_TIMING,
    PhraseTiming,
    TimingPreset,
    WordTime,
    locate_phrase,
    plan_show_window,
)
from video_edit_agent.motion.router import render_motion
from video_edit_agent.render.composition import Overlay
from video_edit_agent.render.reframe import resolve_reframe
from video_edit_agent.review.edit_plan import EditPlan, EditPlanSlot, SlotStatus
from video_edit_agent.subject.compositor import (
    find_enclosing_clip,
    load_subject_clip,
    refined_canvas_masks,
    render_subject_cutout,
    to_source_window,
)
from video_edit_agent.subject.framing import Box, FrameAnalysis, analyze_clip
from video_edit_agent.subject.integrity import DEFAULT_MATTE_POLICY, MatteGatePolicy, assess_matte
from video_edit_agent.subject.refine import MatteReport, mask_stability

BEHIND_SUBJECT_TEXT = Treatment.BEHIND_SUBJECT_TEXT.value
_APPROVED = {SlotStatus.APPROVED, SlotStatus.CHANGED, SlotStatus.GENERATION_APPROVED}
_MAX_WORDS = 6  # the glyph probe colours one word each; a statement longer than this is not a behind-text phrase
_LARGE_TEXT_CONTRAST = 3.0  # WCAG ratio for display-size text: the text is 100px+ tall
_NEUTRAL_LIGHT, _NEUTRAL_DARK = "#FFFFFF", "#000000"
_HAIR_MARGIN = 0.45  # of the face box height, added above it: hair and forehead are not "face" to a detector

DECISION_BEHIND = "behind_subject"
DECISION_FOREGROUND = "kinetic_typography"
DECISION_PUNCH_IN = "punch_in_only"
NOT_RECOMMENDED = "behind_subject_not_recommended"


@dataclass(frozen=True)
class TreatmentPreset:
    """A generic look for the treatment (never a per-brand or per-video value)."""

    timing: TimingPreset = DEFAULT_TIMING
    occlusion: OcclusionPolicy = DEFAULT_POLICY
    opacity: float = 0.92  # below the captions' full opacity: the treatment is a temporary accent
    outline_frac: float = 0.03  # of the font size, only when contrast needs it
    max_lines: int = 3
    sample_fps: float = 10.0  # subject mask sampling over the short show window
    matte: MatteGatePolicy = DEFAULT_MATTE_POLICY  # when the subject matte is too unreliable to put text behind


DEFAULT_PRESET = TreatmentPreset()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def approved_behind_text_slots(plan: EditPlan | None) -> list[EditPlanSlot]:
    """The slots that may render behind-subject text right now: reviewable (not
    settled elsewhere), approved, effectively behind-subject text, with approved text."""
    if plan is None:
        return []
    return [
        s for s in plan.slots
        if not s.settled and s.status in _APPROVED
        and s.effective_treatment == BEHIND_SUBJECT_TEXT and bool((s.text or "").strip())
    ]


@dataclass(frozen=True)
class CaptionInfo:
    """What the phrase must stay clear of and different from: the caption band and
    the caption colours (lower-case #rrggbb)."""

    zone: tuple[int, int]
    colours: frozenset[str] = frozenset()


def ass_to_hex(value: str) -> str | None:
    """`&HAABBGGRR` / `&HBBGGRR` -> `#rrggbb`."""
    raw = value.strip().lstrip("&").lstrip("Hh").rstrip("&")
    if len(raw) == 8:
        raw = raw[2:]
    if len(raw) != 6:
        return None
    try:
        int(raw, 16)
    except ValueError:
        return None
    b, g, r = raw[0:2], raw[2:4], raw[4:6]
    return f"#{r}{g}{b}".lower()


def caption_info(style: CaptionStyle, canvas_h: int, safe: SafeZone | None = None) -> CaptionInfo:
    """The caption band (bottom-aligned above the safe-zone margin, `max_lines` lines
    of the style's size plus the plate padding) and the colours the captions use."""
    safe = safe or SafeZone()
    zone = caption_zone(canvas_h, safe.bottom_pct, style.font_size, style.max_lines, padding_px=style.outline)
    colours = {c for c in (ass_to_hex(style.primary_color), ass_to_hex(style.highlight_color)) if c}
    return CaptionInfo(zone=zone, colours=frozenset(colours))


@dataclass
class SubjectData:
    """The subject over the show window: refined canvas-size masks the placement is
    judged on (the same masks the cutout is built from), the face, the backdrop."""

    times: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    face: Box | None = None
    analysis: FrameAnalysis | None = None
    matte: list[MatteReport] = field(default_factory=list)  # what the refinement did to each mask
    reason: str = ""  # why there are no masks

    @property
    def usable(self) -> bool:
        return bool(self.masks)


SubjectLoader = Callable[[EDL, float, float], SubjectData]


def make_subject_loader(cache_dir: Path, preset: TreatmentPreset = DEFAULT_PRESET) -> SubjectLoader:
    """The MediaPipe-backed loader: masks for the timeline window `start..end` from
    the enclosing clip (cached), refined to canvas size."""

    def load(edl: EDL, start: float, end: float) -> SubjectData:
        clip = find_enclosing_clip(edl, start, end)
        if clip is None:
            return SubjectData(reason="the show window spans a cut")
        if resolve_reframe(clip) is not None:
            return SubjectData(reason="the clip has a punch-in the cutout would not follow")
        src_start, src_end = to_source_window(clip, start, end)
        source = Path(clip.source_file)
        sampled = load_subject_clip(source, src_start, src_end, cache_dir, preset.sample_fps)
        if sampled is None:
            return SubjectData(reason="no usable subject mask")
        times, masks, reports = refined_canvas_masks(sampled, (edl.width, edl.height))
        analysis = analyze_clip(source, src_start, src_end)
        return SubjectData(
            times=times, masks=masks, face=analysis.face if analysis else None, analysis=analysis, matte=reports,
        )

    return load


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def source_to_timeline(clip: EDLClip, t: float) -> float:
    speed = clip.speed if clip.speed else 1.0
    return clip.timeline_in + (t - clip.source_in) / speed


def _clip_at(edl: EDL, start: float, end: float) -> EDLClip | None:
    clip = find_enclosing_clip(edl, start, end)
    if clip is not None:
        return clip
    return next((c for c in edl.clips if c.timeline_in <= start < c.timeline_out), None)


def phrase_timing_for_slot(
    slot: EditPlanSlot, edl: EDL, words: Sequence[Word], preset: TimingPreset = DEFAULT_TIMING,
) -> PhraseTiming:
    """The show window of the slot's approved text: located in the approved
    (corrected) transcript `words`, mapped onto the timeline, clamped to the slot."""
    text = (slot.text or "").strip()
    span = (slot.timeline_start, slot.timeline_end)
    clip = _clip_at(edl, *span)
    located: tuple[WordTime, ...] | None = None
    if clip is not None and words:
        src_slot = to_source_window(clip, *span)
        found = locate_phrase(words, text, near=src_slot)
        if found is not None:
            located = tuple(
                WordTime(w.word, source_to_timeline(clip, w.start), source_to_timeline(clip, w.end)) for w in found
            )
    return plan_show_window(located, span, preset=preset)


# --------------------------------------------------------------------------
# Layout, colour
# --------------------------------------------------------------------------


def _balanced(words: list[str], n: int) -> list[str]:
    """`words` split into `n` consecutive lines of similar character length."""
    if n <= 1 or len(words) <= 1:
        return [" ".join(words)]
    total = sum(len(w) for w in words)
    lines: list[list[str]] = [[]]
    for i, word in enumerate(words):
        remaining_words = len(words) - i
        remaining_lines = n - len(lines)
        if lines[-1] and remaining_lines > 0 and (
            remaining_words <= remaining_lines or sum(len(w) for w in lines[-1]) >= total / n
        ):
            lines.append([])
        lines[-1].append(word)
    return [" ".join(line) for line in lines]


def line_options(text: str, max_lines: int = 3) -> list[tuple[str, ...]]:
    """Every distinct way to set the approved words, unchanged and in order, as one
    coordinated group: one line, or a balanced stack of 2..`max_lines` lines. Only the
    line breaks are presentation; a word is never dropped or split across lines."""
    words = text.split()
    out: list[tuple[str, ...]] = []
    for n in range(1, min(len(words), max_lines) + 1):
        option = tuple(_balanced(words, n))
        if option not in out:
            out.append(option)
    return out


def _outline_for(color: str) -> str:
    return _NEUTRAL_DARK if relative_luminance(color) > 0.3 else _NEUTRAL_LIGHT


def choose_colour(
    brand: Brand | None, backdrop: tuple[float, float, float] | None, caption_colours: frozenset[str] = frozenset(),
) -> dict:
    """The phrase's colour, below the captions in the hierarchy: a Brand-Profile
    colour that is NOT one the captions already use, the first that reads on the
    backdrop; otherwise the most legible of them with an opposite-lightness outline.
    With no usable brand colour it is a neutral light/dark picked from the backdrop."""
    candidates = [(s, c) for s, c in (_palette(brand) if brand is not None else []) if c.lower() not in caption_colours]
    if not candidates:
        dark_backdrop = backdrop is not None and backdrop[1] < 0.35
        colour = _NEUTRAL_LIGHT if dark_backdrop or backdrop is None else _NEUTRAL_DARK
        return {"color": colour, "source": "neutral", "contrast": None, "outline": _outline_for(colour) if backdrop is None else None}
    ranked: list[tuple[float, str, str]] = []
    for source, color in candidates:
        ratio = _worst_contrast(relative_luminance(color), backdrop, None, 0.0) if backdrop is not None else None
        if ratio is not None and ratio >= _LARGE_TEXT_CONTRAST:
            return {"color": color, "source": source, "contrast": ratio, "outline": None}
        ranked.append((ratio if ratio is not None else 0.0, source, color))
    ratio, source, color = max(ranked) if backdrop is not None else ranked[0]
    return {
        "color": color, "source": source, "contrast": ratio if backdrop is not None else None,
        "outline": _outline_for(color),
    }


def _grown_face(face: Box | None) -> Box | None:
    if face is None:
        return None
    x, y, w, h = face
    top = max(0.0, y - _HAIR_MARGIN * h)
    return (max(0.0, x - 0.1 * w), top, min(1.0, w * 1.2), h + (y - top))


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass
class BehindSubjectPlan:
    """What the planner decided for one approved slot and why."""

    slot: int
    decision: str  # DECISION_*
    timing: PhraseTiming
    spec: AnimationSpec | None = None
    reason: str = ""
    gate: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def recommended(self) -> bool:
        return self.decision == DECISION_BEHIND


def _gate(passed: bool, detail: str = "") -> dict:
    return {"passed": passed, "detail": detail}


def _spec(
    slot: EditPlanSlot, timing: PhraseTiming, placement: Placement, colour: dict, preset: TreatmentPreset,
    brand: Brand | None, *, behind: bool, extra: dict,
) -> AnimationSpec:
    avoid = brand.motion.avoid if brand is not None else []
    outline = colour["outline"] if "text_outline" not in avoid else None
    payload = {
        "lines": list(placement.lines), "fontPx": placement.font_px,
        "centerX": round(placement.block_center[0], 1), "centerY": round(placement.block_center[1], 1),
        "color": colour["color"], "opacity": preset.opacity,
        "outline": (
            {"color": outline, "width": round(placement.font_px * preset.outline_frac, 1)}
            if outline else None
        ),
        "shadow": "" if "text_shadow" in avoid else "0 6px 28px rgba(0,0,0,0.35)",
        "enterSec": round(timing.readable_at - timing.show_start, 3),
        "exitSec": round(timing.show_end - timing.hold_end, 3),
        "slot": slot.number,
        "phrase_timing": _timing_dict(timing),
        "placement": placement.summary(),
        "legibility": {"foreground_source": colour["source"], "contrast": colour["contrast"]},
        **extra,
    }
    return AnimationSpec(
        kind=AnimationKind.BEHIND_TEXT, timeline_start=round(timing.show_start, 3), timeline_end=round(timing.show_end, 3),
        text=(slot.text or "").strip(), behind_subject=behind, extra=payload,
    )


def _timing_dict(t: PhraseTiming) -> dict:
    return {
        "source": t.source, "phrase_start": round(t.start, 3), "phrase_end": round(t.end, 3),
        "show_start": round(t.show_start, 3), "readable_at": round(t.readable_at, 3),
        "hold_end": round(t.hold_end, 3), "show_end": round(t.show_end, 3),
        "words": [[w.word, round(w.start, 3), round(w.end, 3)] for w in t.words],
    }


def _binding(result: SearchResult) -> str:
    return result.binding_rule or "no_candidate"


def plan_behind_subject(
    plan: EditPlan | None,
    edl: EDL,
    brand: Brand | None = None,
    *,
    words: Sequence[Word] = (),
    caption: CaptionInfo | None = None,
    probe: Probe | None = None,
    load_subject: SubjectLoader | None = None,
    preset: TreatmentPreset = DEFAULT_PRESET,
) -> list[BehindSubjectPlan]:
    """One plan per approved behind-subject slot (nothing while a slot is pending).

    Without a `probe` or a `load_subject` there is nothing to judge the placement on,
    so the slot is planned as `punch_in_only` with the reason, never as an unchecked
    behind-subject text."""
    return [
        _plan_slot(slot, edl, brand, words, caption, probe, load_subject, preset)
        for slot in approved_behind_text_slots(plan)
    ]


def _plan_slot(
    slot: EditPlanSlot, edl: EDL, brand: Brand | None, words: Sequence[Word], caption: CaptionInfo | None,
    probe: Probe | None, load_subject: SubjectLoader | None, preset: TreatmentPreset,
) -> BehindSubjectPlan:
    text = (slot.text or "").strip()
    timing = phrase_timing_for_slot(slot, edl, words, preset.timing)
    gate: dict = {"phrase_timing": _gate(timing.source == "words", timing.source)}

    def punch_in(reason: str) -> BehindSubjectPlan:
        return BehindSubjectPlan(slot.number, DECISION_PUNCH_IN, timing, None, f"{NOT_RECOMMENDED}: {reason}", gate)

    if len(text.split()) > _MAX_WORDS:
        gate["phrase_length"] = _gate(False, f"{len(text.split())} words")
        return punch_in(f"the phrase has more than {_MAX_WORDS} words")
    if probe is None or load_subject is None:
        return punch_in("no glyph measurement or subject mask is available to judge the placement")

    subject = load_subject(edl, timing.show_start, timing.show_end)
    if not subject.usable:
        gate["subject_mask"] = _gate(False, subject.reason)
        return punch_in(subject.reason or "no usable subject mask")
    mean_iou, worst_iou = mask_stability(subject.masks)
    stable = mean_iou >= preset.occlusion.min_mask_stability and worst_iou >= preset.occlusion.min_worst_stability
    gate["mask_stability"] = _gate(stable, f"mean IoU {mean_iou:.3f}, worst {worst_iou:.3f}")
    if subject.matte:
        matte_ok, matte_detail = assess_matte(subject.matte, preset.matte)
        gate["matte_integrity"] = _gate(matte_ok, matte_detail)
    presence = float(np.mean([m.mean() for m in subject.masks]))
    gate["subject_present"] = _gate(presence >= preset.occlusion.min_subject_frac, f"{presence:.3f}")

    canvas = (edl.width, edl.height)
    safe = brand_safe_zone(brand) if brand is not None else SafeZone()
    zone = caption.zone if caption is not None else None
    top_limit = safe.top_pct * edl.height
    face = _grown_face(subject.face)
    layouts: list[GlyphLayout] = [probe(option) for option in line_options(text, preset.max_lines)]
    key = key_word_indices(text.split())
    search = search_placement(
        layouts, subject.masks, canvas, top_limit_px=top_limit, zone=zone, face=face, key=key, policy=preset.occlusion,
    )
    placement = search.placement
    gate["readable_occlusion"] = _gate(
        placement is not None,
        f"{search.considered} candidates; binding rule: {_binding(search)}" if placement is None else "ok",
    )
    blocking = [name for name, g in gate.items() if not g["passed"] and name != "phrase_timing"]

    if placement is not None:
        backdrop = _backdrop(subject.analysis, placement, canvas)
        colour = choose_colour(brand, backdrop, caption.colours if caption else frozenset())
        gate["caption_hierarchy"] = _gate(
            colour["color"].lower() not in (caption.colours if caption else frozenset()),
            f"{colour['source']} {colour['color']}",
        )
        gate["caption_zone_clear"] = _gate(
            placement.caption_clearance_px is None or placement.caption_clearance_px >= preset.occlusion.caption_clearance_px,
            f"{placement.caption_clearance_px} px",
        )
        blocking = [name for name, g in gate.items() if not g["passed"] and name != "phrase_timing"]
        if not blocking:
            spec = _spec(
                slot, timing, placement, colour, preset, brand, behind=True,
                extra={"decision": DECISION_BEHIND, "gate": gate},
            )
            return BehindSubjectPlan(
                slot.number, DECISION_BEHIND, timing, spec, "readable occlusion within bounds", gate, placement.summary(),
            )

    reason = f"{NOT_RECOMMENDED}: " + "; ".join(f"{n} ({gate[n]['detail']})" for n in blocking)
    return _fallback(slot, timing, subject, layouts, canvas, top_limit, zone, face, key, brand, caption, preset, gate, reason)


def _backdrop(analysis: FrameAnalysis | None, placement: Placement, canvas: tuple[int, int]) -> tuple[float, float, float] | None:
    if analysis is None:
        return None
    width, height = canvas
    x0, y0, x1, y1 = placement.bounds
    return analysis.luma_stats((x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height))


def _fallback(
    slot: EditPlanSlot, timing: PhraseTiming, subject: SubjectData, layouts: list[GlyphLayout],
    canvas: tuple[int, int], top_limit: float, zone: tuple[int, int] | None, face: Box | None,
    key: tuple[int, ...], brand: Brand | None, caption: CaptionInfo | None, preset: TreatmentPreset,
    gate: dict, reason: str,
) -> BehindSubjectPlan:
    """Behind-subject is not recommended: the same words as plain kinetic typography
    in a spot the subject never covers, or nothing when no such spot exists."""
    result = search_placement(
        layouts, subject.masks, canvas, top_limit_px=top_limit, zone=zone, face=face, key=key,
        policy=preset.occlusion, require_depth=False,
    )
    placement = result.placement
    if placement is None:
        return BehindSubjectPlan(slot.number, DECISION_PUNCH_IN, timing, None, reason, gate)
    backdrop = _backdrop(subject.analysis, placement, canvas)
    colour = choose_colour(brand, backdrop, caption.colours if caption else frozenset())
    spec = _spec(
        slot, timing, placement, colour, preset, brand, behind=False,
        extra={"decision": DECISION_FOREGROUND, NOT_RECOMMENDED: True, "reason": reason, "gate": gate},
    )
    return BehindSubjectPlan(slot.number, DECISION_FOREGROUND, timing, spec, reason, gate, placement.summary())


def plan_behind_subject_for_project(
    plan: EditPlan | None,
    edl: EDL,
    brand: Brand | None,
    *,
    transcript: Transcript,
    caption_style: CaptionStyle,
    project_root: Path,
    cache_dir: Path,
    offline: bool = False,
    preset: TreatmentPreset = DEFAULT_PRESET,
) -> list[BehindSubjectPlan]:
    """`plan_behind_subject` wired to the real project: the approved (corrected)
    transcript words, the caption style's band and colours, the renderer's own glyph
    shapes and the MediaPipe subject masks. Does nothing (and measures nothing) while
    no behind-subject slot is approved."""
    if not approved_behind_text_slots(plan):
        return []
    probe, _source = default_probe(
        project_root, brand, (edl.width, edl.height), offline=offline, cache_dir=cache_dir / "glyphs",
    )
    words = [w for seg in transcript.segments for w in seg.words]
    return plan_behind_subject(
        plan, edl, brand, words=words, caption=caption_info(caption_style, edl.height), probe=probe,
        load_subject=make_subject_loader(cache_dir / "subject_cutouts", preset), preset=preset,
    )


def plan_behind_subject_text(
    plan: EditPlan | None,
    edl: EDL,
    brand: Brand | None = None,
    **kwargs,
) -> list[AnimationSpec]:
    """The specs of `plan_behind_subject` (behind-subject text, or its fallback
    typography); slots that fall back to the punch-in alone yield none."""
    return [p.spec for p in plan_behind_subject(plan, edl, brand, **kwargs) if p.spec is not None]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


@dataclass
class BehindSubjectComposite:
    """The layers for one behind-subject spec, in draw order (graphic, then the
    subject cutout on top), and what happened on the way."""

    item: MotionPlanItem
    overlays: list[Overlay] = field(default_factory=list)
    cutout_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    decision: str = ""

    @property
    def cutout_applied(self) -> bool:
        return self.cutout_path is not None


def compose_behind_subject(
    spec: AnimationSpec,
    edl: EDL,
    project_root: Path,
    motion_dir: Path,
    cutout_dir: Path,
    *,
    brand: Brand | None,
    fps: float,
    slot_id: str,
    offline: bool = False,
    label: str | None = None,
    sample_fps: float = DEFAULT_PRESET.sample_fps,
    render_fn: Callable[..., MotionPlanItem] = render_motion,
    cutout_fn: Callable[..., Path | None] = render_subject_cutout,
) -> BehindSubjectComposite:
    """Render `spec`'s graphic and layer the refined subject cutout over it (spec
    Phase 2 section 25): the graphic is drawn on the full frame (covering the subject
    baked into the base), then the RGBA subject cutout goes on top, restoring the
    subject in front of the graphic. Falls back to a plain foreground overlay -- never
    drops the overlay -- when the window spans a cut, the clip has a punch-in (the
    cutout would not line up with the reframed base), or segmentation yields no usable
    mask (spec section 26)."""
    item = render_fn(spec, project_root, motion_dir, brand=brand, fps=fps, slot_id=slot_id, offline=offline)
    label = label if label is not None else slot_id
    result = BehindSubjectComposite(item=item)
    if not item.output_path:
        return result

    graphic = Overlay(path=Path(item.output_path), start=spec.timeline_start, end=spec.timeline_end, behind_subject=True)
    result.overlays.append(graphic)

    enclosing = find_enclosing_clip(edl, spec.timeline_start, spec.timeline_end)
    reason = ""
    if enclosing is None:
        reason = "motion window spans a cut"
    elif resolve_reframe(enclosing) is not None:
        reason = "the clip has a punch-in the cutout would not follow"
    else:
        src_start, src_end = to_source_window(enclosing, spec.timeline_start, spec.timeline_end)
        result.cutout_path = cutout_fn(
            Path(enclosing.source_file), src_start, src_end, cutout_dir,
            sample_fps=sample_fps, output_size=(edl.width, edl.height), refine=True,
        )
        if result.cutout_path is None:
            reason = "no usable subject mask"

    if result.cutout_path is not None:
        result.overlays.append(Overlay(path=result.cutout_path, start=spec.timeline_start, end=spec.timeline_end))
        result.decision = f"Behind-subject compositing applied for motion slot {label} ({spec.kind.value})"
    else:
        result.warnings.append(
            f"Behind-subject requested for motion slot {label} but unavailable ({reason}); "
            "used plain foreground overlay instead"
        )
        result.decision = f"Behind-subject fallback for motion slot {label}: {reason} -> plain overlay"
    return result
