"""Simple local motion engine (spec section 17): labels, arrows, basic boxes,
simple image animations. Uses PIL only — always available, the guaranteed
fallback engine per spec section 43 (no engine may ever hard-fail a render).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from video_edit_agent.core.schemas import AnimationKind, AnimationSpec

DEFAULT_CANVAS = (1080, 1920)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in ("arial.ttf", "DejaVuSans-Bold.ttf", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_label(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int]) -> None:
    font = _load_font(64)
    w, h = canvas
    text = spec.text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (w - tw) / 2, (h - th) / 2
    draw.rectangle([x - 24, y - 16, x + tw + 24, y + th + 16], fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def _render_box(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int]) -> None:
    w, h = canvas
    margin = int(w * 0.1)
    draw.rounded_rectangle(
        [margin, int(h * 0.35), w - margin, int(h * 0.55)], radius=24, outline=(255, 204, 0, 255), width=6
    )
    if spec.text:
        font = _load_font(48)
        draw.text((margin + 24, int(h * 0.38)), spec.text, font=font, fill=(255, 255, 255, 255))


def _render_arrow(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int]) -> None:
    w, h = canvas
    cx, cy = w // 2, int(h * 0.5)
    draw.line([(cx - 100, cy), (cx + 60, cy)], fill=(255, 204, 0, 255), width=10)
    draw.polygon([(cx + 60, cy - 30), (cx + 120, cy), (cx + 60, cy + 30)], fill=(255, 204, 0, 255))


def _render_stat_counter(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int]) -> None:
    w, h = canvas
    big_font = _load_font(140)
    small_font = _load_font(44)
    value = spec.value or spec.text
    bbox = draw.textbbox((0, 0), value, font=big_font)
    tw = bbox[2] - bbox[0]
    x, y = (w - tw) / 2, h * 0.35
    draw.text((x, y), value, font=big_font, fill=(255, 204, 0, 255))
    if spec.subtext:
        bbox2 = draw.textbbox((0, 0), spec.subtext, font=small_font)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((w - tw2) / 2, y + 160), spec.subtext, font=small_font, fill=(255, 255, 255, 255))


def _render_quote(draw: ImageDraw.ImageDraw, spec: AnimationSpec, canvas: tuple[int, int]) -> None:
    w, h = canvas
    font = _load_font(56)
    margin = int(w * 0.12)
    draw.text((margin, int(h * 0.4)), f"“{spec.text}”", font=font, fill=(255, 255, 255, 255))
    if spec.subtext:
        sub_font = _load_font(36)
        draw.text((margin, int(h * 0.4) + 140), f"— {spec.subtext}", font=sub_font, fill=(200, 200, 200, 255))


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


def render_animation(spec: AnimationSpec, output_path: Path, canvas: tuple[int, int] = DEFAULT_CANVAS) -> Path:
    """Renders a single transparent PNG frame for the animation. Static PNGs
    are intentionally simple (spec section 17's 'simple engine' tier); the
    render composition layer handles fade-in/out and timing via `overlay`."""
    renderer = _RENDERERS.get(spec.kind, _render_label)
    image = Image.new("RGBA", canvas, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        renderer(draw, spec, canvas)
    except Exception as e:  # noqa: BLE001 - never let a motion graphic crash the whole render
        raise SimpleMotionError(f"simple engine failed to render {spec.kind}: {e}") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
