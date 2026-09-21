"""Readable occlusion: where can a phrase sit behind the subject and still read?

Behind-subject text is about depth, not about hiding words. A candidate
placement is judged with the ACTUAL subject mask against the phrase's ACTUAL
glyph shapes (measured from the renderer, see `motion/glyph_probe.py`), and is
only usable when

- a controlled share of the glyph area is hidden (`max_hidden`), the key
  word(s) stay readable (`max_key_hidden`, `max_word_hidden`), yet the subject
  really is in front of some of it (`min_hidden`: otherwise it is just text on
  top and the treatment adds nothing);
- little of it is hidden by the face (`max_face_hidden`);
- it stays inside the safe zone and clear of the caption zone;
- the phrase is ONE coordinated group (one line, or a locked stack of lines
  centred on a single axis) and is not made smaller than a statement should be.

Nothing here knows a brand, a colour, a phrase or a timestamp: masks, glyphs,
zones and thresholds are arguments. When no candidate qualifies the search says
which rule was binding, so the caller can choose a fallback treatment instead of
forcing the effect.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]  # normalized x, y, w, h


@dataclass(frozen=True)
class OcclusionPolicy:
    max_hidden: float = 0.30  # share of ALL glyph area the subject may hide
    max_word_hidden: float = 0.45  # ... of any single word
    max_key_hidden: float = 0.20  # ... of the key word(s)
    min_hidden: float = 0.03  # depth cue: less than this is just text in front of the subject
    max_face_hidden: float = 0.05  # share of glyph area hidden by the face box
    target_hidden: float = 0.14  # what the search prefers within the bounds
    caption_clearance_px: int = 48  # gap kept between the text and the caption zone
    side_margin_frac: float = 0.06
    min_font_frac: float = 0.085  # of canvas width: smaller reads as a caption, not a statement
    max_font_frac: float = 0.17  # larger competes with the captions
    size_steps: int = 7
    y_step_px: int = 12
    min_mask_stability: float = 0.90  # mean IoU between consecutive mask samples
    min_worst_stability: float = 0.80
    min_subject_frac: float = 0.02  # of the frame: less and there is no subject to hide behind
    foreground_hidden: float = 0.005  # mask-edge tolerance when the text must sit fully clear of the subject
    # meaningful occlusion (Phase 1.3.1): the subject must really be IN FRONT of glyph bodies, not just touch them
    edge_erode_frac: float = 0.02  # of the font size: the glyph core drops the antialiased fringe and thin strokes' edges
    subject_erode_px: int = 2  # the subject core drops the matte fringe and a one-pixel hair contact
    min_overlap_ratio: float = 0.06  # core glyph px behind the subject / core glyph px
    min_occluded_glyphs: int = 2  # glyph units (connected ink shapes) that are meaningfully behind the subject
    min_glyph_overlap: float = 0.20  # a glyph unit counts as occluded when this share of its core is behind the subject
    min_overlap_rows: float = 0.20  # the overlap must reach this share of the ink's height (not a sliver along one edge)
    min_visible_ratio: float = 0.70  # share of the glyph ink that stays visible
    min_visible_per_word: float = 0.50  # every word keeps at least this much visible
    require_meaningful: bool = True  # Behind-Subject needs the subject truly in front of glyph bodies (off only for legacy checks)
    composition_sizes: int = 3  # font sizes tried per layout in the composition search (largest ... smallest)
    composition_overlaps: tuple[float, ...] = (0.20, 0.25, 0.30, 0.45, 0.60, 0.80)  # share of the text height put inside the silhouette top


DEFAULT_POLICY = OcclusionPolicy()


@dataclass(frozen=True)
class MeaningfulOcclusion:
    """How much of the glyph BODIES (an eroded core, not the antialiased fringe) is really behind the subject's silhouette,
    measured on the actual rendered geometry. Ratios are worst-case over the hold's mask samples."""

    text_total_px: int
    text_subject_overlap_px: int
    text_subject_overlap_ratio: float
    visible_text_ratio: float
    occluded_glyph_count: int
    total_glyph_count: int
    word_overlap_ratio: tuple[float, ...]
    overlap_rows_frac: float  # share of the ink's height the overlap reaches
    overlap_on_head_frac: float  # of the overlap pixels: on the head/hair (above the face's bottom edge)
    overlap_on_torso_frac: float  # ... on the shoulders/torso (below it)
    passed: bool
    reasons: tuple[str, ...] = ()

    def summary(self) -> dict:
        return {
            "text_total_px": self.text_total_px, "text_subject_overlap_px": self.text_subject_overlap_px,
            "text_subject_overlap_ratio": round(self.text_subject_overlap_ratio, 4), "visible_text_ratio": round(self.visible_text_ratio, 4),
            "occluded_glyph_count": self.occluded_glyph_count, "total_glyph_count": self.total_glyph_count,
            "word_overlap_ratio": [round(v, 3) for v in self.word_overlap_ratio], "overlap_rows_frac": round(self.overlap_rows_frac, 3),
            "overlap_on_head_frac": round(self.overlap_on_head_frac, 3), "overlap_on_torso_frac": round(self.overlap_on_torso_frac, 3),
            "passed": self.passed, "reasons": list(self.reasons),
        }


def _core(binary: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return binary
    return cv2.erode(binary.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def meaningful_occlusion(
    alphas: Sequence[np.ndarray],
    font_px: int,
    origin: tuple[int, int],
    masks: Sequence[np.ndarray],
    *,
    face_bottom_px: float | None = None,
    policy: OcclusionPolicy | None = None,
) -> MeaningfulOcclusion:
    """Judges whether the subject is meaningfully in front of the text. `alphas` are the words' ink alphas on one grid,
    `origin` the canvas pixel of that grid's top-left, `masks` the subject alphas over the hold (canvas-sized), and
    `face_bottom_px` splits the overlap into head/hair vs torso. Tiny edge contact, the antialiasing fringe and a
    one-pixel hair contact are removed by measuring an eroded glyph core against an eroded subject core."""
    import cv2

    pol = policy or DEFAULT_POLICY
    h, w = alphas[0].shape
    x0, y0 = origin
    union = np.maximum.reduce(list(alphas))
    ink = union > 0.5
    core = _core(ink, max(1, round(font_px * pol.edge_erode_frac)))
    word_core = [_core(a > 0.5, max(1, round(font_px * pol.edge_erode_frac))) for a in alphas]
    total = int(core.sum())
    n_lab, labels = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel(), minlength=n_lab)
    min_size = max(4, 0.01 * max(total, 1))
    units = [i for i in range(1, n_lab) if sizes[i] >= min_size]
    worst = None
    for mask in masks:
        crop = mask[y0:y0 + h, x0:x0 + w] if mask.shape[0] >= y0 + h and mask.shape[1] >= x0 + w else np.zeros((h, w), np.float32)
        subj = _core(crop > 0.5, pol.subject_erode_px)
        over = core & subj
        n_over = int(over.sum())
        hit = np.bincount(labels[over], minlength=n_lab)
        occluded = sum(1 for i in units if hit[i] >= pol.min_glyph_overlap * sizes[i])
        rows = float(over.any(axis=1).sum()) / max(int(ink.any(axis=1).sum()), 1)
        words = tuple(float((over & wc).sum()) / max(int(wc.sum()), 1) for wc in word_core)
        visible = 1.0 - float((union * crop).sum()) / max(float(union.sum()), 1.0)
        if face_bottom_px is not None and n_over:
            ys = np.where(over)[0] + y0
            head = float((ys < face_bottom_px).sum()) / n_over
        else:
            head = 0.0
        cand = (n_over / max(total, 1), n_over, occluded, rows, words, visible, head)
        if worst is None:
            worst = cand
        else:  # the sample with the LEAST occlusion decides whether it is meaningful; the LEAST visible decides readability
            worst = (cand if cand[0] < worst[0] else worst)[:5] + (min(visible, worst[5]),) + (cand if cand[0] < worst[0] else worst)[6:]
    ratio, n_over, occluded, rows, words, visible, head = worst or (0.0, 0, 0, 0.0, tuple(0.0 for _ in alphas), 1.0, 0.0)
    reasons: list[str] = []
    if ratio < pol.min_overlap_ratio:
        reasons.append(f"only {ratio * 100:.1f}% of the glyph bodies are behind the subject (need {pol.min_overlap_ratio * 100:.0f}%)")
    if occluded < pol.min_occluded_glyphs:
        reasons.append(f"{occluded} glyph(s) are meaningfully occluded (need {pol.min_occluded_glyphs})")
    if rows < pol.min_overlap_rows:
        reasons.append(f"the overlap reaches only {rows * 100:.0f}% of the text height: a sliver on one edge")
    if visible < pol.min_visible_ratio:
        reasons.append(f"only {visible * 100:.0f}% of the text stays visible (need {pol.min_visible_ratio * 100:.0f}%)")
    if any(1.0 - v < pol.min_visible_per_word for v in words):
        reasons.append("a word is mostly hidden")
    return MeaningfulOcclusion(
        text_total_px=total, text_subject_overlap_px=n_over, text_subject_overlap_ratio=ratio, visible_text_ratio=visible,
        occluded_glyph_count=occluded, total_glyph_count=len(units), word_overlap_ratio=words, overlap_rows_frac=rows,
        overlap_on_head_frac=head, overlap_on_torso_frac=1.0 - head if n_over else 0.0, passed=not reasons, reasons=tuple(reasons),
    )


@dataclass
class GlyphLayout:
    """One way to set the phrase: `lines` (logical order), and for each word the
    alpha of its ink measured at `ref_px`, all on one tight grid. `offset` is the
    ink centre minus the renderer's own text-block centre (at `ref_px`): the
    renderer positions the block, the search positions the ink."""

    lines: tuple[str, ...]
    ref_px: int
    words: tuple[np.ndarray, ...]
    offset: tuple[float, float] = (0.0, 0.0)

    def scaled(self, px: int) -> tuple[list[np.ndarray], tuple[float, float]]:
        import cv2

        f = px / self.ref_px
        if abs(f - 1.0) < 1e-6:
            return [w for w in self.words], self.offset
        h, w = self.words[0].shape
        size = (max(1, round(w * f)), max(1, round(h * f)))
        interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR
        return (
            [cv2.resize(a, size, interpolation=interp) for a in self.words],
            (self.offset[0] * f, self.offset[1] * f),
        )

    @property
    def aspect(self) -> float:
        h, w = self.words[0].shape
        return w / max(h, 1)

    @property
    def ink_width(self) -> int:
        return self.words[0].shape[1]


@dataclass
class Placement:
    """A judged candidate. Coordinates are canvas pixels; `center` is the INK
    centre and `block_center` where the renderer must put its own block."""

    lines: tuple[str, ...]
    font_px: int
    center: tuple[float, float]
    block_center: tuple[float, float]
    bounds: tuple[float, float, float, float]  # ink bbox x0, y0, x1, y1
    hidden: float  # share of glyph area behind the subject (worst sample)
    word_hidden: tuple[float, ...]
    key_hidden: float
    face_hidden: float
    bbox_overlap: float  # share of the ink bounding box covered by the subject
    caption_clearance_px: float | None
    score: float = 0.0

    @property
    def readable(self) -> float:
        return 1.0 - self.hidden

    def summary(self) -> dict:
        x0, y0, x1, y1 = self.bounds
        return {
            "lines": list(self.lines), "font_px": self.font_px,
            "bounds": [round(x0), round(y0), round(x1), round(y1)],
            "subject_overlap_pct": round(self.bbox_overlap * 100, 1),
            "glyph_hidden_pct": round(self.hidden * 100, 1),
            "readable_glyph_pct": round(self.readable * 100, 1),
            "word_hidden_pct": [round(v * 100, 1) for v in self.word_hidden],
            "key_hidden_pct": round(self.key_hidden * 100, 1),
            "face_hidden_pct": round(self.face_hidden * 100, 1),
            "caption_clearance_px": None if self.caption_clearance_px is None else round(self.caption_clearance_px),
        }


@dataclass
class SearchResult:
    placement: Placement | None
    considered: int = 0
    rejected: Counter = field(default_factory=Counter)  # rule -> candidates it removed

    @property
    def binding_rule(self) -> str | None:
        return self.rejected.most_common(1)[0][0] if self.rejected else None


def caption_zone(
    canvas_h: int, safe_bottom_pct: float, font_px: float, max_lines: int, *, padding_px: float = 20.0,
) -> tuple[int, int]:
    """(top, bottom) px of the band the captions occupy: bottom-aligned above the
    safe-zone margin, `max_lines` lines of `font_px` (1.25 line height) plus the
    plate padding on both sides. Derived from the caption style, never a constant."""
    bottom = round(canvas_h - canvas_h * safe_bottom_pct)
    return round(bottom - max_lines * font_px * 1.25 - 2 * padding_px), bottom


def key_word_indices(words: Sequence[str]) -> tuple[int, ...]:
    """The word(s) that carry the phrase: the longest one (ties: the last). A
    generic default; a caller with better knowledge passes its own."""
    if not words:
        return ()
    best = max(range(len(words)), key=lambda i: (len(words[i].strip()), i))
    return (best,)


def _hidden(alpha: np.ndarray, subject: np.ndarray) -> tuple[float, float]:
    total = float(alpha.sum())
    return (float((alpha * subject).sum()), total)


def evaluate(
    layout: GlyphLayout,
    font_px: int,
    center: tuple[float, float],
    masks: Sequence[np.ndarray],
    canvas: tuple[int, int],
    *,
    face: Box | None = None,
    zone: tuple[int, int] | None = None,
    key: Sequence[int] = (),
) -> Placement | None:
    """Metrics of putting the layout's ink centre at `center` with `font_px`; the
    worst sample of `masks` (canvas-sized alphas over the hold) decides. None when
    the ink would leave the canvas."""
    width, height = canvas
    alphas, offset = layout.scaled(font_px)
    h, w = alphas[0].shape
    x0, y0 = round(center[0] - w / 2), round(center[1] - h / 2)
    if x0 < 0 or y0 < 0 or x0 + w > width or y0 + h > height:
        return None
    hidden = 0.0
    word_hidden = [0.0] * len(alphas)
    face_hidden = 0.0
    bbox_overlap = 0.0
    face_px = None
    if face is not None:
        fx, fy, fw, fh = face
        face_px = (round(fx * width), round(fy * height), round((fx + fw) * width), round((fy + fh) * height))
    union = np.maximum.reduce(alphas)
    total_area = float(union.sum()) or 1.0
    for mask in masks:
        crop = mask[y0:y0 + h, x0:x0 + w]
        per_word = []
        for i, a in enumerate(alphas):
            hid, tot = _hidden(a, crop)
            per_word.append(hid / tot if tot > 0 else 0.0)
        hid_all = float((union * crop).sum()) / total_area
        hidden = max(hidden, hid_all)
        word_hidden = [max(a, b) for a, b in zip(word_hidden, per_word, strict=True)]
        bbox_overlap = max(bbox_overlap, float(crop.mean()))
        if face_px is not None:
            fx0, fy0, fx1, fy1 = face_px
            ix0, iy0, ix1, iy1 = max(fx0, x0), max(fy0, y0), min(fx1, x0 + w), min(fy1, y0 + h)
            if ix1 > ix0 and iy1 > iy0:
                sub = (union * crop)[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
                face_hidden = max(face_hidden, float(sub.sum()) / total_area)
    key_hidden = max((word_hidden[i] for i in key if i < len(word_hidden)), default=0.0)
    clearance = None if zone is None else float(zone[0] - (y0 + h))
    return Placement(
        lines=layout.lines, font_px=font_px, center=center,
        block_center=(center[0] - offset[0], center[1] - offset[1]),
        bounds=(x0, y0, x0 + w, y0 + h), hidden=hidden, word_hidden=tuple(word_hidden),
        key_hidden=key_hidden, face_hidden=face_hidden, bbox_overlap=bbox_overlap,
        caption_clearance_px=clearance,
    )


def _candidate_sizes(layout: GlyphLayout, canvas: tuple[int, int], policy: OcclusionPolicy) -> list[int]:
    width, _ = canvas
    fit = (width * (1.0 - 2 * policy.side_margin_frac)) / layout.ink_width * layout.ref_px
    top = min(width * policy.max_font_frac, fit)
    bottom = width * policy.min_font_frac
    if top < bottom:
        return []
    if policy.size_steps <= 1 or top - bottom < 2:
        return [round(top)]
    ratio = (bottom / top) ** (1.0 / (policy.size_steps - 1))
    return sorted({round(top * ratio ** i) for i in range(policy.size_steps)}, reverse=True)


def _candidate_axes(canvas: tuple[int, int], face: Box | None, masks: Sequence[np.ndarray]) -> list[float]:
    width, _ = canvas
    axes = [width / 2.0]
    if face is not None:
        axes.append((face[0] + face[2] / 2.0) * width)
    elif masks:
        cols = np.flatnonzero(np.max(np.stack([m > 0.5 for m in masks]), axis=(0, 1)))
        if cols.size:
            axes.append(float(cols.mean()))
    out: list[float] = []
    for a in axes:
        if all(abs(a - b) > 8 for b in out):
            out.append(a)
    return out


def search_placement(
    layouts: Sequence[GlyphLayout],
    masks: Sequence[np.ndarray],
    canvas: tuple[int, int],
    *,
    top_limit_px: float,
    zone: tuple[int, int] | None,
    face: Box | None = None,
    key: Sequence[int] = (),
    policy: OcclusionPolicy = DEFAULT_POLICY,
    require_depth: bool = True,
) -> SearchResult:
    """The best placement over every layout, size, axis and height, or none.
    `require_depth=False` is the foreground-typography search: the subject may
    hide at most the mask-edge tolerance (`foreground_hidden`) and the depth cue is
    not required."""
    width, height = canvas
    result = SearchResult(placement=None)
    if not masks:
        return result
    max_hidden = policy.max_hidden if require_depth else policy.foreground_hidden
    bottom_limit = (zone[0] - policy.caption_clearance_px) if zone else height * 0.8
    axes = _candidate_axes(canvas, face, masks)
    best: tuple[float, int, Placement] | None = None
    order = 0
    for layout in layouts:
        for px in _candidate_sizes(layout, canvas, policy):
            alphas, _ = layout.scaled(px)
            h, w = alphas[0].shape
            lo = top_limit_px + h / 2
            hi = bottom_limit - h / 2
            if hi < lo:
                result.rejected["no_room_above_captions"] += 1
                continue
            for cx in axes:
                if cx - w / 2 < width * policy.side_margin_frac or cx + w / 2 > width * (1 - policy.side_margin_frac):
                    result.rejected["side_margin"] += 1
                    continue
                cy = lo
                while cy <= hi + 1e-6:
                    result.considered += 1
                    order += 1
                    p = evaluate(layout, px, (cx, cy), masks, canvas, face=face, zone=zone, key=key)
                    cy += policy.y_step_px
                    if p is None:
                        result.rejected["off_canvas"] += 1
                        continue
                    reason = _violation(p, policy, max_hidden, require_depth)
                    if reason:
                        result.rejected[reason] += 1
                        continue
                    p.score = _score(p, canvas, policy, require_depth)
                    if best is None or p.score > best[0] + 1e-9:
                        best = (p.score, order, p)
    if best is not None:
        result.placement = best[2]
    return result


def _violation(p: Placement, policy: OcclusionPolicy, max_hidden: float, require_depth: bool) -> str | None:
    if p.hidden > max_hidden:
        return "too_much_hidden"
    if any(v > policy.max_word_hidden for v in p.word_hidden) and require_depth:
        return "word_too_hidden"
    if p.key_hidden > policy.max_key_hidden and require_depth:
        return "key_word_hidden"
    if p.face_hidden > policy.max_face_hidden:
        return "behind_face"
    if require_depth and p.hidden < policy.min_hidden:
        return "no_depth"
    return None


def _score(p: Placement, canvas: tuple[int, int], policy: OcclusionPolicy, require_depth: bool) -> float:
    width, _ = canvas
    impact = p.font_px / (width * policy.max_font_frac)
    depth = -abs(p.hidden - policy.target_hidden) * 2.0 if require_depth else 0.0
    off_axis = -0.15 * abs(p.center[0] - width / 2) / width
    return impact + depth + off_axis



# --------------------------------------------------------------------------
# Composition candidates (Phase 1.3.1): a SMALL deterministic search
# --------------------------------------------------------------------------

NOT_SUITABLE_FOR_SHOT = "not_suitable_for_this_shot"


@dataclass(frozen=True)
class CameraVariant:
    """A camera framing a candidate is judged under: the subject is transformed by zoom `zoom` with crop anchor
    (`anchor_x`, `anchor_y`) (the renderer's `face_frame` math). Only a static (zoom 1.0) framing is one the cutout
    renderer can follow today; the others are judged so the report can say what a camera move would have bought."""

    name: str
    zoom: float = 1.0
    anchor_x: float = 0.5
    anchor_y: float = 0.30

    @property
    def static(self) -> bool:
        return abs(self.zoom - 1.0) < 1e-6


@dataclass
class CompositionCandidate:
    camera: CameraVariant
    placement: Placement
    occlusion: MeaningfulOcclusion
    valid: bool
    score: float
    rejected_by: str = ""

    def summary(self) -> dict:
        return {
            "camera": {"name": self.camera.name, "zoom": self.camera.zoom, "anchor_x": self.camera.anchor_x, "anchor_y": self.camera.anchor_y},
            "placement": self.placement.summary(), "meaningful_occlusion": self.occlusion.summary(),
            "valid": self.valid, "score": round(self.score, 4), "rejected_by": self.rejected_by,
        }


@dataclass
class CompositionSearch:
    considered: int = 0
    valid: int = 0
    best: CompositionCandidate | None = None  # the best VALID candidate under a static camera (renderable)
    best_any: CompositionCandidate | None = None  # the best valid one under any camera
    closest: CompositionCandidate | None = None  # the candidate that came closest to meaningful, for the report
    near_miss: CompositionCandidate | None = None  # best LEGAL candidate (face, key word, captions ok) that is only not meaningful enough
    rejected: Counter = field(default_factory=Counter)

    @property
    def status(self) -> str:
        return "ok" if self.best is not None else NOT_SUITABLE_FOR_SHOT


def transform_mask(mask: np.ndarray, variant: CameraVariant) -> np.ndarray:
    if variant.static:
        return mask
    import cv2

    h, w = mask.shape[:2]
    ox, oy = (variant.zoom - 1.0) * w * variant.anchor_x, (variant.zoom - 1.0) * h * variant.anchor_y
    m = np.array([[variant.zoom, 0.0, -ox], [0.0, variant.zoom, -oy]], np.float32)
    return cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def transform_box(box: Box | None, variant: CameraVariant) -> Box | None:
    if box is None or variant.static:
        return box
    z = variant.zoom
    ox, oy = (z - 1.0) * variant.anchor_x, (z - 1.0) * variant.anchor_y
    return (box[0] * z - ox, box[1] * z - oy, box[2] * z, box[3] * z)


def _silhouette_top(masks: Sequence[np.ndarray]) -> int | None:
    rows = np.flatnonzero(np.max(np.stack([m > 0.5 for m in masks]), axis=(0, 2)))
    return int(rows[0]) if rows.size else None


def _composition_score(p: Placement, mo: MeaningfulOcclusion, canvas: tuple[int, int], policy: OcclusionPolicy) -> float:
    """Meaningful occlusion first, then readability, size and balance; caption clearance is a hard rule elsewhere."""
    width, _ = canvas
    depth = min(mo.text_subject_overlap_ratio / max(policy.target_hidden, 1e-6), 1.5)
    size = p.font_px / (width * policy.max_font_frac)
    balance = -abs(p.center[0] - width / 2) / width
    return 1.6 * depth + 1.0 * mo.visible_text_ratio + 0.6 * size + 0.4 * balance


def _closeness(c: CompositionCandidate) -> float:
    return c.occlusion.text_subject_overlap_ratio * c.occlusion.visible_text_ratio


def search_composition(
    layouts: Sequence[GlyphLayout],
    masks: Sequence[np.ndarray],
    canvas: tuple[int, int],
    *,
    top_limit_px: float,
    zone: tuple[int, int] | None,
    face: Box | None = None,
    key: Sequence[int] = (),
    variants: Sequence[CameraVariant] = (CameraVariant("base"),),
    policy: OcclusionPolicy = DEFAULT_POLICY,
) -> CompositionSearch:
    """Judges a small deterministic grid: camera variant x layout (one or two lines) x a few sizes x centred/face axis x
    heights anchored to the subject's silhouette (the text straddling its top by a few fixed fractions of its own height),
    never a sweep of every pixel. A candidate is valid when the placement is legal (caption clearance, face, key word) AND
    its `meaningful_occlusion` passes; the best valid one wins and ties keep the earlier candidate, so it is deterministic."""
    width, height = canvas
    out = CompositionSearch()
    if not masks:
        return out
    bottom_limit = (zone[0] - policy.caption_clearance_px) if zone else height * 0.8
    for variant in variants:
        vmasks = [transform_mask(m, variant) for m in masks]
        vface = transform_box(face, variant)
        top = _silhouette_top(vmasks)
        if top is None:
            out.rejected["no_subject"] += 1
            continue
        face_bottom = None if vface is None else (vface[1] + vface[3]) * height
        axes = _candidate_axes(canvas, vface, vmasks)
        for layout in layouts:
            sizes = _candidate_sizes(layout, canvas, policy)
            if len(sizes) > policy.composition_sizes:
                idx = np.linspace(0, len(sizes) - 1, policy.composition_sizes).round().astype(int)
                sizes = [sizes[i] for i in sorted(set(idx.tolist()))]
            for px in sizes:
                alphas, _ = layout.scaled(px)
                h, w = alphas[0].shape
                for cx in axes:
                    if cx - w / 2 < width * policy.side_margin_frac or cx + w / 2 > width * (1 - policy.side_margin_frac):
                        out.rejected["side_margin"] += 1
                        continue
                    for frac in policy.composition_overlaps:
                        cy = top - h / 2 + frac * h
                        if cy - h / 2 < top_limit_px or cy + h / 2 > bottom_limit:
                            out.rejected["outside_safe_band"] += 1
                            continue
                        out.considered += 1
                        p = evaluate(layout, px, (cx, cy), vmasks, canvas, face=vface, zone=zone, key=key)
                        if p is None:
                            out.rejected["off_canvas"] += 1
                            continue
                        origin = (round(p.bounds[0]), round(p.bounds[1]))
                        mo = meaningful_occlusion(alphas, px, origin, vmasks, face_bottom_px=face_bottom, policy=policy)
                        reason = _violation(p, policy, policy.max_hidden, True)
                        if reason == "no_depth":
                            reason = None  # depth is judged by the stricter meaningful metric
                        if reason is None and not mo.passed:
                            reason = "not_meaningful"
                        cand = CompositionCandidate(variant, p, mo, reason is None, 0.0, reason or "")
                        cand.score = _composition_score(p, mo, canvas, policy)
                        if reason:
                            out.rejected[reason] += 1
                            if out.closest is None or _closeness(cand) > _closeness(out.closest):
                                out.closest = cand
                            if reason == "not_meaningful" and variant.static and (out.near_miss is None or cand.score > out.near_miss.score + 1e-9):
                                out.near_miss = cand
                            continue
                        out.valid += 1
                        if out.best_any is None or cand.score > out.best_any.score + 1e-9:
                            out.best_any = cand
                        if variant.static and (out.best is None or cand.score > out.best.score + 1e-9):
                            out.best = cand
    return out
