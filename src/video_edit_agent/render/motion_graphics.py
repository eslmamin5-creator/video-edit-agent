"""Phase 1.4 renderer support: `keyword_visual` and `simple_diagram` as ASS events,
built on the SAME restrained, generic primitives `captions.headline` already uses for
`primary_headline_typography` (fade + subtle slide + scale settle; no bounce, no
kinetic typography -- sec. 2). `primary_headline_typography` itself needs no new
renderer: `captions.headline.add_headline` already IS it.

Brand values (colors, fonts) are the ONLY project-specific input anywhere in this
module, and they only ever arrive already resolved as a `Brand` instance (sec. 3):
nothing here hardcodes any client's palette. `brand_tokens()` is the one place a
`Brand` is turned into renderer-ready values (hex colors -> ASS colors), so a
generic caller never touches `Brand` fields directly.

Layout is deterministic: given the same text/labels, canvas size and safe zone, the
same pixel geometry comes out every time -- no randomness, no measurement from a
live render (pixel truth for QA still comes from `captions.headline.ink_bbox`,
unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.headline import (
    HEADLINE_STYLE,
    HeadlineSpec,
    _style_line,
    headline_event,
)
from video_edit_agent.captions.safe_zone import SafeZone, margins_px
from video_edit_agent.captions.styles import CaptionStyle, hex_to_ass

_ACCENT_LAYER = 4  # under the headline layer (5), above captions


def brand_tokens(brand: Brand) -> dict:
    """The brand's colors/fonts as renderer-ready values (sec. 3). `accent` is None
    when the brand did not set one (`BrandColors.accent` has no default): callers
    must fall back to `secondary`, never invent a strong color."""
    colors = {"primary": brand.colors.primary, "secondary": brand.colors.secondary, "accent": brand.colors.accent}
    return {
        "brand_profile": brand.name,
        "colors": colors,
        "colors_ass": {k: hex_to_ass(v) for k, v in colors.items() if v},
        "accent_ass": hex_to_ass(brand.colors.accent or brand.colors.secondary),
        "font_ar": brand.arabic_font,
        "font_en": brand.latin_font,
    }


def _keyword_style(base: CaptionStyle, font_px: int) -> CaptionStyle:
    from dataclasses import replace
    return replace(base, font_size=font_px, word_highlight=False, bold=True)


@dataclass(frozen=True)
class DiagramNode:
    text: str
    y_px: float


def _accent_bar_event(x_px: float, y_px: float, w_px: float, h_px: float, color_ass: str, window: tuple[float, float]) -> str:
    """A small static brand-accent bar (a vector rectangle), the only decorative
    accent `keyword_visual`/`simple_diagram` are allowed (sec. 2.B: 'a small
    underline / bracket / shape accent'; no icon)."""
    from video_edit_agent.captions.engine import _fmt_ass_time
    tags = f"{{\\an7\\pos({x_px:.0f},{y_px:.0f})\\1c{color_ass}\\p1}}m 0 0 l {w_px:.0f} 0 {w_px:.0f} {h_px:.0f} 0 {h_px:.0f}{{\\p0}}"
    return f"Dialogue: {_ACCENT_LAYER},{_fmt_ass_time(window[0])},{_fmt_ass_time(window[1])},{HEADLINE_STYLE},,0,0,0,,{tags}"


def add_keyword(ass: str, base_style: CaptionStyle, brand: Brand, text: str, window: tuple[float, float],
                 *, canvas: tuple[int, int] = (1080, 1920), font_px: int = 120,
                 safe_zone: SafeZone | None = None) -> str:
    """`ass` with one compact, brand-accented `keyword_visual` event added (sec. 2.B):
    the term itself (restrained fade/slide/settle, same as a headline) plus a small
    accent bar beneath it. Speaker stays visible; captions are untouched by this call
    (reduce them the same way a headline does, via `captions.headline.reduce_captions`,
    if the caller wants that)."""
    safe_zone = safe_zone or SafeZone()
    w, h = canvas
    margins = margins_px(safe_zone, w, h)
    tok = brand_tokens(brand)
    style = _keyword_style(base_style, font_px)
    y = h * 0.42  # a compact term sits in the middle third, clear of head/caption bands
    spec = HeadlineSpec(text=text, y_px=y, fade_in=(window[0], min(window[0] + 0.3, window[1])),
                         fade_out=(max(window[1] - 0.3, window[0]), window[1]), font_px=font_px, slide_px=10.0, settle_from=0.96)
    ass = ass.replace("\n\n[Events]", f"\n{_style_line(style)}\n\n[Events]", 1)
    ass = ass.rstrip("\n") + "\n" + headline_event(spec, w) + "\n"
    bar_w, bar_h = min(w - 2 * margins["left"], font_px * len(text) * 0.4), max(4.0, font_px * 0.045)
    bar_x = (w - bar_w) / 2
    bar_y = y + font_px * 0.62
    ass = ass.rstrip("\n") + "\n" + _accent_bar_event(bar_x, bar_y, bar_w, bar_h, tok["accent_ass"], window) + "\n"
    return ass


def diagram_layout(count: int, canvas: tuple[int, int] = (1080, 1920), *, safe_zone: SafeZone | None = None) -> list[float]:
    """The deterministic y (px) of each of `count` diagram nodes, evenly spaced inside
    the safe vertical band, top to bottom, transcript order preserved (sec. 10: safe
    margins, deterministic layout)."""
    safe_zone = safe_zone or SafeZone()
    w, h = canvas
    margins = margins_px(safe_zone, w, h)
    top, bottom = margins["top"], h - margins["bottom"]
    if count <= 1:
        return [top + (bottom - top) / 2]
    step = (bottom - top) / (count - 1)
    return [top + i * step for i in range(count)]


def add_diagram(ass: str, base_style: CaptionStyle, brand: Brand, labels: list[str], window: tuple[float, float],
                 *, canvas: tuple[int, int] = (1080, 1920), font_px: int = 76,
                 safe_zone: SafeZone | None = None) -> str:
    """`ass` with a `simple_diagram` added: up to `MAX_DIAGRAM_NODES` verbatim labels,
    one per line, evenly spaced top-to-bottom in the safe zone, each with a small
    brand-accent index bar (sec. 2.C/10). No chart, no invented data: `labels` is
    exactly what the caller (a qualified `discover_diagram_candidates` result) gives."""
    safe_zone = safe_zone or SafeZone()
    w, _h = canvas
    tok = brand_tokens(brand)
    style = _keyword_style(base_style, font_px)
    ys = diagram_layout(len(labels), canvas, safe_zone=safe_zone)
    ass = ass.replace("\n\n[Events]", f"\n{_style_line(style)}\n\n[Events]", 1)
    fin, fout = min(window[0] + 0.3, window[1]), max(window[1] - 0.3, window[0])
    for i, (text, y) in enumerate(zip(labels, ys, strict=True)):
        spec = HeadlineSpec(text=text, y_px=y, fade_in=(window[0], fin), fade_out=(fout, window[1]),
                             font_px=font_px, x_px=w * 0.58, slide_px=8.0, settle_from=0.97)
        ass = ass.rstrip("\n") + "\n" + headline_event(spec, w) + "\n"
        bar_x, bar_y, bar_w, bar_h = w * 0.10, y, w * 0.06, font_px * 0.75
        ass = ass.rstrip("\n") + "\n" + _accent_bar_event(bar_x, bar_y, bar_w, bar_h, tok["accent_ass"], window) + "\n"
    return ass


__all__ = ["DiagramNode", "add_diagram", "add_keyword", "brand_tokens", "diagram_layout"]
