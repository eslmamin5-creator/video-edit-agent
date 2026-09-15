"""Simple local motion engine (spec section 17): labels, arrows, basic boxes,
simple image animations. Uses PIL only — always available, the guaranteed
fallback engine per spec section 43 (no engine may ever hard-fail a render).

Colors are brand-aware (Creator spec section 9): when a `Brand` is supplied
its `colors`/`fonts` are used; the defaults below match `brand.defaults.DEFAULT_BRAND`
exactly, so existing Editor callers that don't pass a brand (or pass the
default brand) see pixel-identical output to before this was added.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from video_edit_agent.core.schemas import AnimationKind, AnimationSpec

DEFAULT_CANVAS = (1080, 1920)

_DEFAULT_ACCENT = (255, 204, 0, 255)
_DEFAULT_TEXT = (255, 255, 255, 255)
_DEFAULT_BG = (0, 0, 0, 160)
_DEFAULT_SUBTEXT = (200, 200, 200, 255)


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (255, 255, 255, alpha)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


class _Palette:
    def __init__(self, brand=None):
        if brand is None:
            self.accent = _DEFAULT_ACCENT
            self.text = _DEFAULT_TEXT
            self.subtext = _DEFAULT_SUBTEXT
            self.bg = _DEFAULT_BG
            self.font_candidates = ()
        else:
            self.accent = _hex_to_rgba(brand.colors.accent)
            self.text = _hex_to_rgba(brand.colors.secondary)
            self.subtext = _hex_to_rgba(brand.colors.secondary, alpha=200)
            self.bg = _hex_to_rgba(brand.colors.primary, alpha=160)
            self.font_candidates = tuple(brand.fonts)


def _load_font(size: int, palette: _Palette) -> ImageFont.FreeTypeFont:
    for candidate in (*palette.font_candidates, "arial.ttf", "DejaVuSans-Bold.ttf", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_label(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int], palette: _Palette) -> None:
    font = _load_font(64, palette)
    w, h = canvas
    text = spec.text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (w - tw) / 2, (h - th) / 2
    draw.rectangle([x - 24, y - 16, x + tw + 24, y + th + 16], fill=palette.bg)
    draw.text((x, y), text, font=font, fill=palette.text)


def _render_box(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int], palette: _Palette) -> None:
    w, h = canvas
    margin = int(w * 0.1)
    draw.rounded_rectangle(
        [margin, int(h * 0.35), w - margin, int(h * 0.55)], radius=24, outline=palette.accent, width=6
    )
    if spec.text:
        font = _load_font(48, palette)
        draw.text((margin + 24, int(h * 0.38)), spec.text, font=font, fill=palette.text)


def _render_arrow(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int], palette: _Palette) -> None:
    w, h = canvas
    cx, cy = w // 2, int(h * 0.5)
    draw.line([(cx - 100, cy), (cx + 60, cy)], fill=palette.accent, width=10)
    draw.polygon([(cx + 60, cy - 30), (cx + 120, cy), (cx + 60, cy + 30)], fill=palette.accent)


def _render_stat_counter(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int], palette: _Palette) -> None:
    w, h = canvas
    big_font = _load_font(140, palette)
    small_font = _load_font(44, palette)
    value = spec.value or spec.text
    bbox = draw.textbbox((0, 0), value, font=big_font)
    tw = bbox[2] - bbox[0]
    x, y = (w - tw) / 2, h * 0.35
    draw.text((x, y), value, font=big_font, fill=palette.accent)
    if spec.subtext:
        bbox2 = draw.textbbox((0, 0), spec.subtext, font=small_font)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((w - tw2) / 2, y + 160), spec.subtext, font=small_font, fill=palette.text)


def _render_quote(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int], palette: _Palette) -> None:
    w, h = canvas
    font = _load_font(56, palette)
    margin = int(w * 0.12)
    draw.text((margin, int(h * 0.4)), f"“{spec.text}”", font=font, fill=palette.text)
    if spec.subtext:
        sub_font = _load_font(36, palette)
        draw.text((margin, int(h * 0.4) + 140), f"— {spec.subtext}", font=sub_font, fill=palette.subtext)


_RENDERERS = {
    AnimationKind.LABEL: _render_label,
    AnimationKind.LOWER_THIRD: _render_label,
    AnimationKind.BOX: _render_box,
    AnimationKind.FEATURE_CARD: _render_box,
    AnimationKind.ARROW: _render_arrow,
    AnimationKind.STAT_COUNTER: _render_stat_counter,
    AnimationKind.METRIC_HIGHLIGHT: _render_stat_counter,
    AnimationKind.QUOTE: _render_quote,
    AnimationKind.CTA: _render_label,
    AnimationKind.HOOK_TITLE: _render_label,
}


class SimpleMotionError(RuntimeError):
    pass


def render_animation(
    spec: AnimationSpec, output_path: Path, canvas: tuple[int, int] = DEFAULT_CANVAS, brand: object | None = None
) -> Path:
    """Renders a single transparent PNG frame for the animation. Static PNGs
    are intentionally simple (spec section 17's 'simple engine' tier); the
    render composition layer handles fade-in/out and timing via `overlay`."""
    renderer = _RENDERERS.get(spec.kind, _render_label)
    palette = _Palette(brand)
    image = Image.new("RGBA", canvas, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        renderer(draw, spec, canvas, palette)
    except Exception as e:
        raise SimpleMotionError(f"simple engine failed to render {spec.kind}: {e}") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
