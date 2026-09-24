"""Phase 1.4.1 sec. 1: face-safe placement for `primary_headline_typography`.

`motion.legibility.plan_title_treatment` already avoids the face for hook
titles, but only along one axis (it scans vertical position at a fixed
horizontal centre) and it never refuses: when nothing in the safe zone is
face-free it falls back to "the least-overlapping spot" (`motion.legibility.
_placement`). Sec. 1 asks for more than that for a primary headline: a
priority-ordered search across FOUR placement families (above the head,
upper-left/upper-right negative space, a side band, and -- only when the
caller says the window already earns it -- a lower band), reposition tried
at the full size before any size reduction, and outright REJECTION when
nothing in that search is safe, rather than covering the face.

This module reuses the same grown face/head box and the same AABB
intersection test `motion.legibility` already uses (`_head_box`,
`_intersects`) instead of inventing a second face-avoidance rule; it adds
only the placement search and the hard-rejection behaviour those private
helpers do not provide. Geometry only: nothing here names a brand, a colour,
a phrase or a timestamp. The caller supplies the measured ink size (via
`measure`, e.g. `captions.headline.ink_bbox`) and, when available, a real
detected face box (`subject.framing.analyze_clip`); with no face this module
still keeps the headline inside the safe zone and clear of the caption band.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from video_edit_agent.captions.safe_zone import SafeZone
from video_edit_agent.motion.legibility import _head_box, _intersects
from video_edit_agent.subject.framing import Box

_CLEARANCE = 0.015  # frame-height fraction kept clear between the headline and the protected face/head box
_CAPTION_BAND = 0.30  # bottom fraction reserved for captions (matches motion.legibility's own convention)
_FACE_SIDE_MARGIN = 0.03  # extra fraction of frame width/height kept clear sideways, on top of _head_box's headroom
_SIZE_STEPS: tuple[float, ...] = (1.0, 0.85, 0.72)  # reposition first (index 0, unchanged size); later steps shrink

PLACEMENT_ABOVE_HEAD = "above_head"
PLACEMENT_UPPER_LEFT = "upper_left"
PLACEMENT_UPPER_RIGHT = "upper_right"
PLACEMENT_SIDE = "side"
PLACEMENT_LOWER_SUBJECT = "lower_subject"
PLACEMENT_REJECTED = "rejected"

# Sec. 1's required priority order (lower_subject is appended by the caller's opt-in only).
_PRIORITY = (PLACEMENT_ABOVE_HEAD, PLACEMENT_UPPER_LEFT, PLACEMENT_UPPER_RIGHT, PLACEMENT_SIDE)


@dataclass(frozen=True)
class HeadlinePlacement:
    """One placement decision. `box` and `clearance` are `None`/`0.0` when rejected;
    `accepted` is the single check a caller needs before rendering (sec. 1: "if no
    valid placement exists, reject the treatment rather than covering the face")."""

    placement: str  # a PLACEMENT_* constant, or PLACEMENT_REJECTED
    box: Box | None  # normalized (x, y, w, h) the headline occupies
    font_scale: float  # 1.0 = unchanged; < 1.0 only once repositioning alone could not clear the face
    clearance: float  # normalized frame-height/-width gap actually kept from the protected face/head box
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.placement != PLACEMENT_REJECTED


def _fits_frame(box: Box, safe: SafeZone) -> bool:
    x, y, w, _h = box
    return x >= safe.side_pct - 1e-9 and x + w <= 1.0 - safe.side_pct + 1e-9 and y >= safe.top_pct - 1e-9


def _fits_caption(box: Box, caption_band: float) -> bool:
    _x, y, _w, h = box
    return y + h <= 1.0 - caption_band + 1e-9


def _clearance_from(box: Box, face: Box) -> float:
    """The gap (normalized) actually kept between `box` and `face` when one sits
    cleanly above/below/beside the other; 0.0 if they are not cleanly separated
    (a caller only sees this after `_intersects` has already said they don't touch)."""
    ax, ay, aw, ah = box
    bx, by, bw, bh = face
    if ay + ah <= by:
        return by - (ay + ah)
    if by + bh <= ay:
        return ay - (by + bh)
    if ax + aw <= bx:
        return bx - (ax + aw)
    if bx + bw <= ax:
        return ax - (bx + bw)
    return 0.0


def _candidate_above_head(face: Box, ink_w: float, ink_h: float, safe: SafeZone) -> Box:
    fx, fy, fw, _fh = face
    x_center = fx + fw / 2 - ink_w / 2
    x = max(safe.side_pct, min(1.0 - safe.side_pct - ink_w, x_center))
    y = fy - _CLEARANCE - ink_h
    return (x, y, ink_w, ink_h)


def _candidate_top_band(ink_w: float, ink_h: float, safe: SafeZone) -> Box:
    """Used only when no face was detected at all: the top safe band is already
    face-free, so "above head" degrades to the plain top-of-safe-zone band."""
    return (max(safe.side_pct, (1.0 - ink_w) / 2), safe.top_pct, ink_w, ink_h)


def _candidate_upper_left(ink_w: float, ink_h: float, safe: SafeZone) -> Box:
    return (safe.side_pct, safe.top_pct, ink_w, ink_h)


def _candidate_upper_right(ink_w: float, ink_h: float, safe: SafeZone) -> Box:
    return (1.0 - safe.side_pct - ink_w, safe.top_pct, ink_w, ink_h)


def _candidate_sides(face: Box | None, ink_w: float, ink_h: float, safe: SafeZone, caption_band: float) -> list[Box]:
    mid_y = face[1] + face[3] / 2 if face is not None else 0.5
    top_limit, bottom_limit = safe.top_pct, 1.0 - caption_band
    y = max(top_limit, min(bottom_limit - ink_h, mid_y - ink_h / 2))
    return [(safe.side_pct, y, ink_w, ink_h), (1.0 - safe.side_pct - ink_w, y, ink_w, ink_h)]


def _candidate_lower_subject(face: Box, ink_w: float, ink_h: float, safe: SafeZone, caption_band: float) -> Box | None:
    fx, fy, fw, fh = face
    x_center = fx + fw / 2 - ink_w / 2
    x = max(safe.side_pct, min(1.0 - safe.side_pct - ink_w, x_center))
    y = fy + fh + _CLEARANCE
    if y + ink_h > 1.0 - caption_band:
        return None
    return (x, y, ink_w, ink_h)


def find_face_safe_headline(
    measure: Callable[[float], tuple[float, float]],
    *,
    face: Box | None,
    safe_zone: SafeZone | None = None,
    caption_band: float = _CAPTION_BAND,
    allow_lower_subject: bool = False,
) -> HeadlinePlacement:
    """Sec. 1's placement search: for each size (full size first, sec. 1's
    "prefer repositioning before resizing"), try above-head, upper-left,
    upper-right, side-left, side-right -- in that priority order -- and,
    only when `allow_lower_subject` says this window already carries a
    justified lower_subject composition, a lower band last of all. The first
    candidate that stays inside the safe zone, clear of the caption band, and
    (when a face was detected) outside the grown, margined face/head box wins.
    Nothing safe at any size -> `PLACEMENT_REJECTED` (`box=None`): sec. 1 asks
    for rejection, not `motion.legibility`'s "least-bad spot" fallback.

    `measure(font_scale)` returns the normalized (ink_w, ink_h) of the headline
    text at that scale (e.g. from `captions.headline.ink_bbox`, divided by the
    canvas size) -- this module never measures or renders text itself.
    """
    safe = safe_zone or SafeZone()
    protected = _head_box(face) if face is not None else None
    for step, scale in enumerate(_SIZE_STEPS):
        ink_w, ink_h = measure(scale)
        size_note = [] if step == 0 else [f"reduced to {scale:.0%} of the base size after no full-size placement was safe"]
        candidates: list[tuple[str, Box]] = []
        candidates.append((
            PLACEMENT_ABOVE_HEAD,
            _candidate_above_head(protected, ink_w, ink_h, safe) if protected is not None
            else _candidate_top_band(ink_w, ink_h, safe),
        ))
        candidates.append((PLACEMENT_UPPER_LEFT, _candidate_upper_left(ink_w, ink_h, safe)))
        candidates.append((PLACEMENT_UPPER_RIGHT, _candidate_upper_right(ink_w, ink_h, safe)))
        candidates.extend((PLACEMENT_SIDE, box) for box in _candidate_sides(protected, ink_w, ink_h, safe, caption_band))
        if allow_lower_subject and protected is not None:
            lower = _candidate_lower_subject(protected, ink_w, ink_h, safe, caption_band)
            if lower is not None:
                candidates.append((PLACEMENT_LOWER_SUBJECT, lower))

        for name, box in candidates:
            if not _fits_frame(box, safe) or not _fits_caption(box, caption_band):
                continue
            if protected is not None and _intersects(box, protected, _FACE_SIDE_MARGIN):
                continue
            clearance = round(_clearance_from(box, protected), 4) if protected is not None else 1.0
            reason = (
                f"{name} placement clears the protected face/head box by {clearance:.1%} of the frame"
                if protected is not None else f"{name} placement (no face detected in this window)"
            )
            return HeadlinePlacement(placement=name, box=box, font_scale=scale, clearance=clearance, reasons=[*size_note, reason])

    return HeadlinePlacement(
        placement=PLACEMENT_REJECTED, box=None, font_scale=_SIZE_STEPS[-1], clearance=0.0,
        reasons=["no safe placement exists in any priority position at any tried size; rejecting rather than covering the face"],
    )


__all__ = [
    "PLACEMENT_ABOVE_HEAD",
    "PLACEMENT_LOWER_SUBJECT",
    "PLACEMENT_REJECTED",
    "PLACEMENT_SIDE",
    "PLACEMENT_UPPER_LEFT",
    "PLACEMENT_UPPER_RIGHT",
    "HeadlinePlacement",
    "find_face_safe_headline",
]
