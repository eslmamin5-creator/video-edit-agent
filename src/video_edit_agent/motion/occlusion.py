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


DEFAULT_POLICY = OcclusionPolicy()


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
