"""The restrained primary headline of a `lower_subject_semantic` composition, as an ASS event (Phase 1.3.1).

Motion is deliberately small: a fade, a subtle slide of a few pixels and a scale settle. No bounce, no per-word kinetics.
The captions under the headline are REDUCED (smaller, no second highlight), never hidden: `reduce_captions` swaps the
caption events that start inside the headline window for their reduced version and touches nothing else, so caption
timing and chunking are exactly those of the normal captions.

`ink_bbox` measures where the headline is actually drawn (libass on a black canvas), so the geometry is sized from the
rendered headline and QA reports pixels, not intentions. Nothing here is brand- or project-specific: colours, font and size
come from the caption style the caller passes in.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from video_edit_agent.captions.engine import _fmt_ass_time
from video_edit_agent.captions.rtl import wrap_rtl
from video_edit_agent.captions.styles import CaptionStyle

HEADLINE_STYLE = "Headline"
_HEADLINE_LAYER = 5
_TIME = re.compile(r"^Dialogue:\s*\d+,(\d+):(\d+):([\d.]+),")


@dataclass(frozen=True)
class HeadlineSpec:
    """One headline: text, where its ink top-centre sits, and its fade-in / fade-out spans (slice-relative seconds)."""

    text: str
    y_px: float  # top of the text line box (ASS \an8 anchor); use `ink_bbox` to place the INK precisely
    fade_in: tuple[float, float]
    fade_out: tuple[float, float]
    font_px: int
    x_px: float | None = None  # None = centred
    slide_px: float = 14.0
    settle_from: float = 0.94


def headline_style(base: CaptionStyle, font_px: int) -> CaptionStyle:
    """The headline look derived from the caption style (same font, colour and outline), larger and never karaoke."""
    return replace(base, font_size=font_px, word_highlight=False, bold=True)


def _style_line(st: CaptionStyle) -> str:
    return (f"Style: {HEADLINE_STYLE},{st.font_ar},{st.font_size},{st.primary_color},{st.primary_color},{st.outline_color},{st.back_color},"
            f"{-1 if st.bold else 0},0,0,0,100,100,0,0,1,{max(st.outline, 2.0)},{st.shadow},8,0,0,0,-1")


def headline_event(spec: HeadlineSpec, width: int) -> str:
    """The Dialogue line: fade, subtle slide (the position eases by `slide_px` over the fade-in) and a scale settle."""
    t0, t1 = spec.fade_in[0], spec.fade_out[1]
    fin, fout = round((spec.fade_in[1] - spec.fade_in[0]) * 1000), round((spec.fade_out[1] - spec.fade_out[0]) * 1000)
    x = width / 2 if spec.x_px is None else spec.x_px
    move = f"\\move({x:.0f},{spec.y_px + spec.slide_px:.0f},{x:.0f},{spec.y_px:.0f},0,{fin})"
    settle = f"\\fscx{spec.settle_from * 100:.0f}\\fscy{spec.settle_from * 100:.0f}\\t(0,{fin},\\fscx100\\fscy100)"
    tags = f"{{\\an8{move}\\fad({fin},{fout}){settle}}}"
    return f"Dialogue: {_HEADLINE_LAYER},{_fmt_ass_time(t0)},{_fmt_ass_time(t1)},{HEADLINE_STYLE},,0,0,0,,{wrap_rtl(tags + spec.text, spec.text)}"


def add_headline(ass: str, style: CaptionStyle, spec: HeadlineSpec, width: int) -> str:
    """`ass` with the Headline style and one headline event added."""
    ass = ass.replace("\n\n[Events]", f"\n{_style_line(style)}\n\n[Events]", 1)
    return ass.rstrip("\n") + "\n" + headline_event(spec, width) + "\n"


def _start_of(line: str) -> float | None:
    m = _TIME.match(line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def reduce_captions(normal: str, reduced: str, window: tuple[float, float]) -> str:
    """The normal caption ASS, except that every caption event starting inside `window` is taken from the `reduced` ASS.
    Both documents come from the same chunks, so the events line up one to one; a mismatch leaves `normal` untouched."""
    n_lines, r_lines = normal.split("\n"), reduced.split("\n")
    n_idx = [i for i, ln in enumerate(n_lines) if ln.startswith("Dialogue:")]
    r_idx = [i for i, ln in enumerate(r_lines) if ln.startswith("Dialogue:")]
    if len(n_idx) != len(r_idx):
        return normal
    for i, j in zip(n_idx, r_idx, strict=True):
        t = _start_of(n_lines[i])
        if t is not None and window[0] - 1e-6 <= t < window[1]:
            n_lines[i] = r_lines[j]
    return "\n".join(n_lines)


def ink_bbox(ass: str, size: tuple[int, int], t: float, *, fonts_dir: Path | None = None) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the pixels libass draws at time `t`, measured on a black canvas (None when nothing is drawn)."""
    w, h = size
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "h.ass"
        path.write_text(ass, encoding="utf-8")
        esc = str(path).replace("\\", "/").replace(":", "\\:")
        vf = f"subtitles='{esc}'" + (f":fontsdir='{Path(fonts_dir).as_posix().replace(':', chr(92) + ':')}'" if fonts_dir else "")
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=25:d={t + 1:.2f}", "-vf", vf,
                            "-ss", f"{t:.3f}", "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=False)
    if r.returncode != 0 or len(r.stdout) < w * h:
        return None
    a = np.frombuffer(r.stdout[: w * h], np.uint8).reshape(h, w) > 40
    ys, xs = np.where(a.any(axis=1))[0], np.where(a.any(axis=0))[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if ys.size and xs.size else None


__all__ = ["HEADLINE_STYLE", "HeadlineSpec", "add_headline", "headline_event", "headline_style", "ink_bbox", "reduce_captions"]
