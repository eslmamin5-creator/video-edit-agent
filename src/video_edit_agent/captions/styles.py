"""Caption style presets (spec section 15).

Colors are &HAABBGGRR (ASS format, alpha-blue-green-red, hex, 00=opaque).
Deliberately NOT porting English-only assumptions like forced uppercase or a
single Helvetica-only font (spec section 15/46) — Arabic needs a font with
full Arabic glyph coverage, so presets ship an Arabic-capable font family.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaptionStyle:
    name: str
    font_ar: str = "Arial"
    font_en: str = "Arial"
    font_size: int = 64
    primary_color: str = "&H00FFFFFF"  # white
    highlight_color: str = "&H0000D7FF"  # amber/gold highlight
    outline_color: str = "&H00000000"  # black
    back_color: str = "&H80000000"  # semi-transparent black box
    outline: float = 3.0
    shadow: float = 0.0
    bold: bool = True
    max_lines: int = 2
    max_chars_per_line: int = 26
    word_highlight: bool = False
    uppercase: bool = False  # never force True for Arabic; presets may opt in for Latin-only text


PRESETS: dict[str, CaptionStyle] = {
    "minimal": CaptionStyle(name="minimal", font_size=56, outline=2.0, word_highlight=False),
    "word-highlight": CaptionStyle(name="word-highlight", font_size=64, word_highlight=True),
    "karaoke": CaptionStyle(name="karaoke", font_size=64, word_highlight=True, highlight_color="&H0000FFFF"),
    "bold-social": CaptionStyle(
        name="bold-social", font_size=72, outline=4.0, bold=True, word_highlight=True,
        back_color="&HA0000000",
    ),
    "cinematic": CaptionStyle(
        name="cinematic", font_size=52, outline=1.5, shadow=1.0, word_highlight=False,
        back_color="&H00000000",
    ),
}


def resolve_style(name: str, brand_overrides: dict | None = None) -> CaptionStyle:
    base = PRESETS.get(name, PRESETS["minimal"])
    if not brand_overrides:
        return base
    data = base.__dict__.copy()
    data.update({k: v for k, v in brand_overrides.items() if k in data})
    return CaptionStyle(**data)
