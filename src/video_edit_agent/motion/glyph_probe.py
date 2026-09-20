"""Measure the phrase's real glyph shapes for the occlusion search.

Judging readable occlusion needs the ink of each word, not a box: Arabic in
particular has descenders, dots and joins that a rectangle misjudges, and mixed
Latin/Arabic lines reorder. So the shapes come from the renderer that draws the
final frames (`remotion still` of the behind-text composition in probe mode: one
flat colour per word, full opacity, no decoration), classified back into one
alpha layer per word. When Remotion is not available a PIL rasterisation (with
Arabic shaping and bidi reordering) stands in, and the caller is told which
source was used.

A probe is `Callable[[Sequence[str]], GlyphLayout]`: lines in, the layout out.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import AnimationKind, AnimationSpec
from video_edit_agent.motion.occlusion import GlyphLayout

REF_PX = 200
_PROBE_COLORS = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32,
)

Probe = Callable[[Sequence[str]], GlyphLayout]


def _crop_words(word_alphas: list[np.ndarray], block_center: tuple[float, float]) -> GlyphLayout | None:
    union = np.maximum.reduce(word_alphas)
    ys, xs = np.nonzero(union > 0.02)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cropped = tuple(np.ascontiguousarray(a[y0:y1, x0:x1], dtype=np.float32) for a in word_alphas)
    ink_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return GlyphLayout(
        lines=(), ref_px=REF_PX, words=cropped,
        offset=(ink_center[0] - block_center[0], ink_center[1] - block_center[1]),
    )


def layout_from_probe_png(png: np.ndarray, n_words: int, block_center: tuple[float, float]) -> GlyphLayout | None:
    """Split a probe still (RGBA, straight alpha, one flat colour per word) into
    per-word alpha layers."""
    rgb = png[..., :3].astype(np.float32) / 255.0
    alpha = png[..., 3].astype(np.float32) / 255.0
    peak = rgb.max(axis=2, keepdims=True)
    unit = rgb / np.maximum(peak, 1e-3)
    palette = _PROBE_COLORS[:n_words]
    dist = ((unit[..., None, :] - palette[None, None]) ** 2).sum(axis=3)
    klass = dist.argmin(axis=2)
    words = [np.where((klass == i) & (peak[..., 0] > 0.12), alpha, 0.0) for i in range(n_words)]
    return _crop_words(words, block_center)


def _cache_key(lines: Sequence[str], canvas: tuple[int, int], brand: Brand | None) -> str:
    font = brand.arabic_font if brand is not None else ""
    raw = "|".join([*lines, str(canvas), str(font), str(REF_PX), "p1"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def remotion_probe(
    project_root: Path, brand: Brand | None, canvas: tuple[int, int], *,
    offline: bool = False, cache_dir: Path | None = None,
) -> Probe:
    """Probe that reads glyph shapes from the Remotion renderer (cached on disk)."""
    from PIL import Image

    from video_edit_agent.motion.remotion import adapter

    width, height = canvas

    def probe(lines: Sequence[str]) -> GlyphLayout:
        lines = tuple(lines)
        key = _cache_key(lines, canvas, brand)
        cache_file = cache_dir / f"glyphs_{key}.npz" if cache_dir else None
        if cache_file is not None and cache_file.exists():
            data = np.load(cache_file)
            n = int(data["n"])
            return GlyphLayout(
                lines=lines, ref_px=REF_PX, words=tuple(data[f"w{i}"] for i in range(n)),
                offset=(float(data["ox"]), float(data["oy"])),
            )
        words = [w for line in lines for w in line.split()]
        spec = AnimationSpec(
            kind=AnimationKind.BEHIND_TEXT, timeline_start=0.0, timeline_end=1.0, text=" ".join(words),
            extra={"lines": list(lines), "fontPx": REF_PX, "centerX": width / 2, "centerY": height / 2, "probe": True},
        )
        scratch = (cache_dir or project_root / "edit" / ".cache") / "_probe"
        png_path = adapter.render_still(
            spec, project_root, scratch / f"probe_{key}.png", slot_id=f"probe_{key}", offline=offline, brand=brand,
        )
        png = np.array(Image.open(png_path).convert("RGBA"))
        layout = layout_from_probe_png(png, len(words), (width / 2, height / 2))
        if layout is None:
            raise RuntimeError("the probe render contained no text")
        layout = GlyphLayout(lines=lines, ref_px=REF_PX, words=layout.words, offset=layout.offset)
        if cache_file is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file, n=len(layout.words), ox=layout.offset[0], oy=layout.offset[1],
                **{f"w{i}": w for i, w in enumerate(layout.words)},
            )
        return layout

    return probe


def _has_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ" for ch in text)


def pil_probe(brand: Brand | None, canvas: tuple[int, int], *, rtl: bool | None = None) -> Probe:
    """Approximate probe: PIL rasterisation with Arabic shaping + bidi. Used only
    when the real renderer is unavailable; the font is the brand's own file when
    one ships, else a system font."""
    from PIL import Image, ImageDraw, ImageFont

    from video_edit_agent.brand.loader import resolve_fonts_dir

    width, height = canvas
    font_path = None
    if brand is not None:
        fonts_dir = resolve_fonts_dir(brand.name)
        if fonts_dir is not None:
            bold = sorted(p for p in fonts_dir.iterdir() if "bold" in p.name.lower() and "semi" not in p.name.lower())
            font_path = (bold or sorted(fonts_dir.iterdir()))[0]
    use_rtl = rtl if rtl is not None else (brand.captions.rtl if brand is not None else False)

    def load():
        for candidate in (font_path, "arialbd.ttf", "DejaVuSans-Bold.ttf"):
            if candidate is None:
                continue
            try:
                return ImageFont.truetype(str(candidate), REF_PX)
            except OSError:
                continue
        return ImageFont.load_default()

    font = load()

    def shaped(word: str) -> str:
        if not _has_arabic(word):
            return word
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(word))

    def probe(lines: Sequence[str]) -> GlyphLayout:
        lines = tuple(lines)
        space = font.getlength(" ")
        line_h = REF_PX * 1.05
        layers: list[np.ndarray] = []
        words_per_line = [line.split() for line in lines]
        n_words = sum(len(w) for w in words_per_line)
        layers = [Image.new("L", (width, height), 0) for _ in range(n_words)]
        top = height / 2 - line_h * len(lines) / 2
        index = 0
        for li, words in enumerate(words_per_line):
            widths = [font.getlength(shaped(w)) for w in words]
            total = sum(widths) + space * (len(words) - 1)
            order = list(range(len(words)))
            visual = order[::-1] if use_rtl else order
            x = width / 2 - total / 2
            positions: dict[int, float] = {}
            for wi in visual:
                positions[wi] = x
                x += widths[wi] + space
            for wi, w in enumerate(words):
                ImageDraw.Draw(layers[index + wi]).text((positions[wi], top + li * line_h), shaped(w), font=font, fill=255)
            index += len(words)
        arrays = [np.asarray(layer, dtype=np.float32) / 255.0 for layer in layers]
        layout = _crop_words(arrays, (width / 2, height / 2))
        if layout is None:
            raise RuntimeError("the fallback raster contained no text")
        return GlyphLayout(lines=lines, ref_px=REF_PX, words=layout.words, offset=layout.offset)

    return probe


def default_probe(
    project_root: Path, brand: Brand | None, canvas: tuple[int, int], *,
    offline: bool = False, cache_dir: Path | None = None,
) -> tuple[Probe, str]:
    """(probe, source): the real renderer when it can run, else the PIL stand-in."""
    from video_edit_agent.motion.remotion import adapter

    if adapter.is_available():
        remote = remotion_probe(project_root, brand, canvas, offline=offline, cache_dir=cache_dir)

        def guarded(lines: Sequence[str]) -> GlyphLayout:
            try:
                return remote(lines)
            except (adapter.RemotionUnavailable, adapter.RemotionRenderError, RuntimeError):
                return pil_probe(brand, canvas)(lines)

        return guarded, "remotion"
    return pil_probe(brand, canvas), "pil"
