"""Caption readability modes: the backing plate is a choice, not a default.

    none      no plate. Readability comes from font weight, a stroke/shadow, a
              safe position, word highlight and a contrast check. The default
              candidate for premium/minimal looks.
    adaptive  like `none`, plus a subtle translucent backing ONLY for the
              stretches where the measured contrast is too low.
    plate     the translucent brand plate (still fully supported).

Modes only change how a caption is DRAWN. Caption timing, chunking and karaoke
word timing are never touched here.

Caption-vs-motion hierarchy: while a strong primary treatment (a motion graphic,
a typography scene, behind-subject text ...) is on screen, captions stay readable
but stop competing with it: smaller, no second word-highlight of equal strength.
They are never hidden (they are also the accessibility text).

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from video_edit_agent.captions.styles import CaptionStyle, hex_to_ass
from video_edit_agent.direction.vocabulary import STRONG_PRIMARY

# WCAG-style thresholds: large bold text needs 3:1, ordinary text 4.5:1.
MIN_CONTRAST = 3.0
_BOX_LIKE = {"box", "brand_box"}
_STROKE = "&H00000000"  # black stroke: readable on any backdrop
REDUCED_SCALE = 0.86
SUBTLE_BACKING_OPACITY = 0.35


class CaptionMode(str, Enum):
    NONE = "none"
    ADAPTIVE = "adaptive"
    PLATE = "plate"


def parse_mode(value: str | None) -> CaptionMode | None:
    if value is None:
        return None
    try:
        return CaptionMode(str(value).strip().lower())
    except ValueError:
        return None


def mode_of(style: CaptionStyle) -> CaptionMode:
    """The mode an existing style corresponds to (a box background is a plate)."""
    return CaptionMode.PLATE if style.background in _BOX_LIKE else CaptionMode.NONE


def recommended_mode(preset: str, *, premium: bool = True) -> CaptionMode:
    """The default candidate: no plate for minimal/premium looks, a plate for loud social presets."""
    if preset in {"bold-social"} and not premium:
        return CaptionMode.PLATE
    return CaptionMode.NONE


def apply_mode(style: CaptionStyle, mode: CaptionMode | str | None) -> CaptionStyle:
    """A copy of `style` drawn in `mode`. `plate` keeps an existing box background (or
    falls back to the legacy box); `none` and `adaptive` draw stroked text without one."""
    parsed = parse_mode(mode.value if isinstance(mode, CaptionMode) else mode)
    if parsed is None:
        return style
    if parsed is CaptionMode.PLATE:
        return style if style.background in _BOX_LIKE else replace(style, background="box")
    out = replace(style, background="outline")
    if style.background in _BOX_LIKE:
        out = replace(out, outline_color=_STROKE)  # the box tint is not a stroke colour
    if out.outline < 2.0:
        out = replace(out, outline=2.0)  # readability without a plate needs a real stroke
    return out


# ---- contrast -----------------------------------------------------------------------


def _channel(v: float) -> float:
    v /= 255.0
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(text: tuple[int, int, int], background: tuple[int, int, int]) -> float:
    a, b = luminance(text), luminance(background)
    hi, lo = max(a, b), min(a, b)
    return round((hi + 0.05) / (lo + 0.05), 3)


def needs_backing(text: tuple[int, int, int], background: tuple[int, int, int], *, threshold: float = MIN_CONTRAST) -> bool:
    """True when caption text on that background falls below the readable contrast."""
    return contrast_ratio(text, background) < threshold


@dataclass(frozen=True)
class ChunkLook:
    background: str  # "outline" or "brand_box"
    subtle_backing: bool
    contrast: float


def look_for_chunk(
    style: CaptionStyle, mode: CaptionMode, text_rgb: tuple[int, int, int], backdrop_rgb: tuple[int, int, int] | None,
) -> ChunkLook:
    """How ONE caption chunk is drawn. `adaptive` adds the subtle backing only when the measured
    backdrop makes the text unreadable; an unmeasured backdrop is treated as readable (no plate
    appears from a guess). `plate` always backs, `none` never does."""
    ratio = contrast_ratio(text_rgb, backdrop_rgb) if backdrop_rgb is not None else float("inf")
    if mode is CaptionMode.PLATE:
        return ChunkLook("brand_box", False, ratio)
    if mode is CaptionMode.ADAPTIVE and backdrop_rgb is not None and needs_backing(text_rgb, backdrop_rgb):
        return ChunkLook("brand_box", True, ratio)
    return ChunkLook("outline", False, ratio)


def subtle_outline_color(rgb_hex: str) -> str:
    """The translucent ASS colour of the adaptive backing (lighter than a full plate)."""
    return hex_to_ass(rgb_hex, alpha=round((1.0 - SUBTLE_BACKING_OPACITY) * 255))


# ---- caption vs. motion hierarchy ------------------------------------------------------


@dataclass(frozen=True)
class CaptionBehavior:
    name: str  # normal | reduced
    scale: float
    word_highlight: bool
    hidden: bool = False  # always False: accessibility text is never removed


NORMAL = CaptionBehavior("normal", 1.0, True)
REDUCED = CaptionBehavior("reduced", REDUCED_SCALE, False)


def behavior_for(treatment: str | None) -> CaptionBehavior:
    """Captions under a strong primary treatment are reduced (never hidden)."""
    return REDUCED if treatment in STRONG_PRIMARY else NORMAL


def apply_behavior(style: CaptionStyle, behavior: CaptionBehavior) -> CaptionStyle:
    """`style` adjusted for the hierarchy. The text stays fully readable: the size is reduced a
    little and the highlight (a second emphasis) is dropped, nothing is hidden."""
    if behavior.name == "normal":
        return style
    return replace(
        style, font_size=max(1, round(style.font_size * behavior.scale)),
        word_highlight=style.word_highlight and behavior.word_highlight,
    )


__all__ = [
    "MIN_CONTRAST", "NORMAL", "REDUCED", "CaptionBehavior", "CaptionMode", "ChunkLook", "apply_behavior",
    "apply_mode", "behavior_for", "contrast_ratio", "look_for_chunk", "luminance", "mode_of",
    "needs_backing", "parse_mode", "recommended_mode", "subtle_outline_color",
]
