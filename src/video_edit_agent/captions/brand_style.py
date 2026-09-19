"""Brand-driven caption styling: derives a `CaptionStyle` from a caption
preset plus the active Brand Profile, so captions carry the brand's
typography and palette instead of the preset's generic Arial/amber look.

Nothing here knows any specific brand: every value comes from the `Brand`
argument, and a missing brand value falls back to a neutral choice (white
text, the brand's own secondary color for the highlight) rather than an
invented accent color.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.styles import CaptionStyle, hex_to_ass, resolve_style

_NEUTRAL_TEXT = "#FFFFFF"
_DEFAULT_LINE_BREAK_CHARS = 18


@dataclass
class ResolvedCaptionStyle:
    style: CaptionStyle
    # Human-readable notes on what came from the brand vs. a neutral
    # fallback ("MISSING ..." entries feed the brand summary's warnings).
    notes: list[str] = field(default_factory=list)


def resolve_brand_caption_style(preset_name: str, brand: Brand) -> ResolvedCaptionStyle:
    caps = brand.captions
    overrides = caps.model_dump()
    notes: list[str] = []

    # Typography: explicit caption font > brand typography > preset default.
    font = caps.font or brand.arabic_font
    if font:
        overrides["font"] = font
    else:
        overrides["font"] = None
        notes.append("MISSING typography: no brand font set; captions use the preset default font")
    latin = caps.font or brand.latin_font
    overrides.pop("use_brand_colors", None)
    overrides.pop("background", None)
    overrides.pop("balance_lines", None)
    overrides.pop("box_opacity", None)
    overrides.pop("box_padding", None)
    if not caps.word_highlight:
        overrides.pop("word_highlight")  # False must not override a highlight preset
    style = resolve_style(preset_name, overrides)
    if latin and latin != style.font_ar:
        style.font_en = latin

    if caps.use_brand_colors:
        if not caps.primary_color:
            style.primary_color = hex_to_ass(_NEUTRAL_TEXT)
        if not caps.highlight_color:
            accent = brand.colors.accent
            if accent:
                style.highlight_color = hex_to_ass(accent)
            else:
                # Never invent a strong accent: fall back to the brand's own
                # secondary color and say so.
                style.highlight_color = hex_to_ass(brand.colors.secondary)
                notes.append("MISSING accent color: caption highlight falls back to the brand secondary color")
    if caps.max_chars_per_line:
        style.max_chars_per_line = caps.max_chars_per_line
    style.word_highlight = caps.word_highlight or style.word_highlight
    style.max_lines = caps.max_lines

    background = caps.background
    if background is None and caps.use_brand_colors:
        background = "brand_box"
    if background:
        style.background = background
    if style.background == "brand_box":
        # ASS alpha: 0 opaque .. 255 transparent.
        alpha = round((1.0 - caps.box_opacity) * 255)
        style.outline_color = hex_to_ass(brand.colors.primary, alpha=alpha)
        style.outline = caps.box_padding
        style.back_color = "&HFF000000"  # no separate shadow box
    elif style.background == "outline":
        style.outline_color = hex_to_ass(brand.colors.primary)

    balance = caps.balance_lines if caps.balance_lines is not None else caps.use_brand_colors
    if balance:
        style.line_break_chars = _DEFAULT_LINE_BREAK_CHARS
    return ResolvedCaptionStyle(style=style, notes=notes)
