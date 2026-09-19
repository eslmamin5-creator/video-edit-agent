"""Generic legibility rule for motion titles (hook titles and the like).

A brand colour on a backdrop of a similar colour is unreadable, and a title
placed by taste can land on the speaker's face or chest. This module decides,
from the active Brand Profile plus (when available) an analysis of the footage,

- WHERE the title sits: inside the safe zone, clear of the face, off the
  subject as far as possible, away from the caption area;
- WHAT colours it uses: the first brand-palette foreground that clears a
  contrast ratio against the backdrop; if none does, a backing plate in a brand
  colour and the first palette foreground that clears the ratio on the plate;
  a neutral white/black only when the palette cannot (and it says so).

Nothing here names a brand, a colour or a position: palettes come from the
Brand Profile, geometry from the frame analysis, and each decision is recorded
in `reasons` for the review report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.safe_zone import SafeZone
from video_edit_agent.subject.framing import Box, FrameAnalysis

MIN_CONTRAST = 4.5  # WCAG AA for normal text; hook titles are large, so this is conservative
_PLATE_OPACITY = 0.78
_NEUTRAL_LIGHT = "#FFFFFF"
_NEUTRAL_DARK = "#000000"
_FACE_MARGIN = 0.02  # fraction of the frame kept clear around the face
_CAPTION_BAND = 0.30  # bottom fraction (safe zone + caption block) a title must not enter
_HEADROOM = 0.6  # fraction of the face height added above the detected face box
_AVG_CHAR_EM = 0.58  # average glyph advance in em, Arabic and Latin


def _rgb(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0


def relative_luminance(color: str) -> float:
    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = _rgb(color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(lum_a: float, lum_b: float) -> float:
    hi, lo = max(lum_a, lum_b), min(lum_a, lum_b)
    return (hi + 0.05) / (lo + 0.05)


@dataclass
class TitleTreatment:
    """How one title is drawn; `as_props` is the Remotion prop payload."""

    foreground: str
    foreground_source: str  # e.g. "brand.accent", "neutral"
    center_y_px: int
    box: Box  # normalized box the title is expected to occupy
    max_width_px: int
    plate_color: str | None = None
    plate_opacity: float = 0.0
    plate_padding_px: int = 0
    plate_radius_px: int = 0
    outline_color: str | None = None
    outline_px: float = 0.0
    shadow: str = ""
    contrast: float | None = None  # worst-case ratio over the backdrop (None when unknown)
    reasons: list[str] = field(default_factory=list)

    def as_props(self) -> dict:
        return {
            "color": self.foreground,
            "centerY": self.center_y_px,
            "maxWidth": self.max_width_px,
            "plate": None if self.plate_color is None else {
                "color": self.plate_color, "opacity": self.plate_opacity,
                "padding": self.plate_padding_px, "radius": self.plate_radius_px,
            },
            "outline": None if not self.outline_color or self.outline_px <= 0 else {
                "color": self.outline_color, "width": self.outline_px,
            },
            "shadow": self.shadow,
        }


def _palette(brand: Brand) -> list[tuple[str, str]]:
    """Foreground candidates from the Brand Profile, in preference order."""
    candidates = [("brand.secondary", brand.colors.secondary)]
    if brand.colors.accent:
        candidates.append(("brand.accent", brand.colors.accent))
    candidates.append(("brand.primary", brand.colors.primary))
    seen: set[str] = set()
    out = []
    for name, color in candidates:
        if color.lower() not in seen:
            seen.add(color.lower())
            out.append((name, color))
    return out


def _plate_candidates(brand: Brand) -> list[tuple[str, str]]:
    plates = [("brand.primary", brand.colors.primary), ("brand.secondary", brand.colors.secondary)]
    if brand.colors.accent:
        plates.append(("brand.accent", brand.colors.accent))
    return plates


def estimate_title_box(text: str, font_px: int, max_width_px: int, canvas_w: int, canvas_h: int, pad_px: int) -> tuple[float, float]:
    """(width_px, height_px) of the title block: text wrapped to `max_width_px`."""
    single = max(len(text), 1) * font_px * _AVG_CHAR_EM
    lines = max(1, math.ceil(single / max_width_px))
    width = min(max_width_px, single) + 2 * pad_px
    height = lines * font_px * 1.25 + 2 * pad_px
    return width, height


def _placement(
    analysis: FrameAnalysis | None, safe: SafeZone, canvas_w: int, canvas_h: int, box_w: float, box_h: float,
    reasons: list[str],
) -> tuple[int, Box]:
    """Vertical centre (px) for the title block and the box it occupies."""
    nw, nh = box_w / canvas_w, box_h / canvas_h
    x = (1.0 - nw) / 2.0
    top_limit = safe.top_pct
    bottom_limit = 1.0 - max(safe.bottom_pct, _CAPTION_BAND)
    if top_limit + nh > bottom_limit:  # tall title in a short band: keep it at the top of the safe zone
        y = top_limit
        reasons.append("title block taller than the free band; pinned to the top of the safe zone")
        return round((y + nh / 2) * canvas_h), (x, y, nw, nh)

    default_y = top_limit + 0.02
    if analysis is None:
        reasons.append("no footage analysis: placed in the top safe band")
        return round((default_y + nh / 2) * canvas_h), (x, default_y, nw, nh)

    face = _head_box(analysis.face)
    best: tuple[float, float, Box] | None = None
    y = top_limit
    while y + nh <= bottom_limit + 1e-9:
        cand: Box = (x, y, nw, nh)
        penalty = analysis.occupancy_in(cand) * 10.0 + abs(y - default_y) * 2.0
        if face is not None and _intersects(cand, face, _FACE_MARGIN):
            penalty += 1000.0
        if best is None or penalty < best[0] - 1e-9:
            best = (penalty, y, cand)
        y += 0.01
    assert best is not None
    _, y, cand = best
    if face is not None and _intersects(cand, face, _FACE_MARGIN):
        reasons.append("no face-free position inside the safe zone; placed at the least-overlapping spot")
    else:
        occ = analysis.occupancy_in(cand)
        reasons.append(
            f"placed clear of the face inside the safe zone (subject overlap {occ:.0%})"
            if face is not None else f"placed on the least-occupied band inside the safe zone (subject overlap {occ:.0%})"
        )
    return round((cand[1] + cand[3] / 2) * canvas_h), cand


def _head_box(face: Box | None) -> Box | None:
    """The detected face box grown upward to cover hair/forehead, which a face
    detector leaves out but a title must not touch either."""
    if face is None:
        return None
    x, y, w, h = face
    grow = h * _HEADROOM
    return x, max(0.0, y - grow), w, h + (y - max(0.0, y - grow))


def _intersects(a: Box, b: Box, margin: float) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx - margin or bx + bw < ax - margin or ay + ah < by - margin or by + bh < ay - margin)


def _worst_contrast(fg_lum: float, backdrop: tuple[float, float, float], plate_lum: float | None, alpha: float) -> float:
    worst = math.inf
    for lum in backdrop:
        eff = lum if plate_lum is None else alpha * plate_lum + (1.0 - alpha) * lum
        worst = min(worst, contrast_ratio(fg_lum, eff))
    return worst


def plan_title_treatment(
    brand: Brand,
    text: str,
    canvas_w: int,
    canvas_h: int,
    *,
    analysis: FrameAnalysis | None = None,
    font_px: int = 88,
    safe_zone: SafeZone | None = None,
    min_contrast: float = MIN_CONTRAST,
) -> TitleTreatment:
    """The legibility treatment for a title. `analysis` (face/subject/backdrop
    grids) is optional; without it the title goes in the top safe band on a
    brand plate, because a backdrop nobody has looked at cannot be trusted."""
    reasons: list[str] = []
    safe = safe_zone or SafeZone(
        top_pct=brand.safe_zones.get("top", SafeZone().top_pct),
        bottom_pct=brand.safe_zones.get("bottom", SafeZone().bottom_pct),
        side_pct=brand.safe_zones.get("side", SafeZone().side_pct),
    )
    max_width = round(canvas_w * (1.0 - 2 * safe.side_pct))
    pad = round(font_px * 0.32)
    palette = _palette(brand)

    def _bare_choice(backdrop, occupied):
        if backdrop is None or occupied > 0.05:
            return None
        for source, color in palette:
            ratio = _worst_contrast(relative_luminance(color), backdrop, None, 0.0)
            if ratio >= min_contrast:
                return source, color, None, ratio
        return None

    # Pass 1: the bare text (no plate padding) in the best subject-free spot.
    bare_w, bare_h = estimate_title_box(text, font_px, max_width, canvas_w, canvas_h, 0)
    bare_reasons: list[str] = []
    center_y, box = _placement(analysis, safe, canvas_w, canvas_h, bare_w, bare_h, bare_reasons)
    backdrop = analysis.luma_stats(box) if analysis is not None else None
    occupied = analysis.occupancy_in(box) if analysis is not None else 0.0
    chosen: tuple[str, str, str | None, float | None] | None = _bare_choice(backdrop, occupied)
    if chosen is not None:
        reasons.extend(bare_reasons)
        reasons.append(f"{chosen[0]} clears {chosen[3]:.1f}:1 on the backdrop; no plate needed")
    else:
        # Pass 2: a backing plate is needed; the block is larger (padding), so place it again.
        if backdrop is None:
            why = "backdrop unknown; using a backing plate"
        elif occupied > 0.05:
            why = f"title box overlaps the subject ({occupied:.0%}); trying a plated placement"
        else:
            why = "no brand foreground clears the contrast ratio on the bare backdrop; using a backing plate"
        box_w, box_h = estimate_title_box(text, font_px, max_width - 2 * pad, canvas_w, canvas_h, pad)
        center_y, box = _placement(analysis, safe, canvas_w, canvas_h, box_w, box_h, reasons)
        backdrop = analysis.luma_stats(box) if analysis is not None else None
        occupied = analysis.occupancy_in(box) if analysis is not None else 0.0
        chosen = None
        reasons.append(why)

    if chosen is None:
        ref_backdrop = backdrop if backdrop is not None else (0.5, 0.5, 0.5)
        for plate_source, plate in _plate_candidates(brand):
            plate_lum = relative_luminance(plate)
            for source, color in palette:
                if color.lower() == plate.lower():
                    continue
                ratio = _worst_contrast(relative_luminance(color), ref_backdrop, plate_lum, _PLATE_OPACITY)
                if ratio >= min_contrast:
                    chosen = (source, color, plate, ratio)
                    reasons.append(f"{source} on a {plate_source} plate clears {ratio:.1f}:1")
                    break
            if chosen is not None:
                break
    if chosen is None:
        # The palette cannot make a legible pair: fall back to a neutral text on a brand plate
        plate_source, plate = _plate_candidates(brand)[0]
        plate_lum = relative_luminance(plate)
        for color in (_NEUTRAL_LIGHT, _NEUTRAL_DARK):
            ref_backdrop = backdrop if backdrop is not None else (0.5, 0.5, 0.5)
            ratio = _worst_contrast(relative_luminance(color), ref_backdrop, plate_lum, _PLATE_OPACITY)
            if ratio >= min_contrast:
                chosen = ("neutral", color, plate, ratio)
                reasons.append(f"brand palette cannot reach {min_contrast}:1; neutral text on the {plate_source} plate ({ratio:.1f}:1)")
                break
    if chosen is None:  # pathological: pick the best neutral on a neutral plate
        chosen = ("neutral", _NEUTRAL_LIGHT, _NEUTRAL_DARK, None)
        reasons.append("falling back to white text on a black plate")

    source, color, plate, ratio = chosen
    fg_lum = relative_luminance(color)
    # outline / shadow: a neutral of opposite lightness to the text -- shadows are not palette colours
    outline = _NEUTRAL_DARK if fg_lum > 0.3 else _NEUTRAL_LIGHT
    honor_outline = "text_outline" not in brand.motion.avoid
    honor_shadow = "text_shadow" not in brand.motion.avoid
    return TitleTreatment(
        foreground=color, foreground_source=source, center_y_px=center_y, box=box, max_width_px=max_width,
        plate_color=plate, plate_opacity=_PLATE_OPACITY if plate else 0.0,
        plate_padding_px=pad if plate else 0, plate_radius_px=round(font_px * 0.28) if plate else 0,
        outline_color=outline if honor_outline and plate is None else None,
        outline_px=round(font_px * 0.04, 1) if honor_outline and plate is None else 0.0,
        shadow=("0 4px 18px rgba(0,0,0,0.55)" if honor_shadow else ""),
        contrast=ratio, reasons=reasons,
    )
