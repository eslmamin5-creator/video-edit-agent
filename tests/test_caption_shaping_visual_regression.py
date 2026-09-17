"""Baseline Recovery Milestone item 3: real frame-level visual regression
check for the Arabic caption double-shaping bug.

This renders the ACTUAL `build_ass()` output for an Arabic line through this
project's real ffmpeg/libass build, and separately renders a hand-built ASS
file that reproduces the old (buggy) behavior -- pre-shaping the same text
with `arabic_reshaper` + `python-bidi` before handing it to libass -- using
the identical header/style/geometry. The two rendered frames must differ:
this is the same empirical method used to originally diagnose the bug
(comparing rendered pixels of pre-shaped vs. raw-logical-order ASS input
through this ffmpeg build, which is compiled with HarfBuzz + FriBidi and
performs its own shaping/BiDi reordering on raw text).

A pixel-identical result here would mean the fix regressed back to
pre-shaping (or that libass stopped doing its own shaping), either of which
is exactly the bug this test exists to catch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.captions.engine import ASS_HEADER_TEMPLATE, build_ass
from video_edit_agent.captions.styles import PRESETS
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript

pytest.importorskip("arabic_reshaper")
pytest.importorskip("bidi.algorithm")

ARABIC_TEXT = "مرحبا بكم في هذا الفيديو"
WIDTH, HEIGHT = 640, 360


def _render_ass_to_png(ass_path: Path, png_path: Path) -> None:
    ass_arg = ass_path.name  # relative to cwd to avoid Windows path-escaping issues in the filter graph
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}",
            "-vf", f"subtitles={ass_arg}",
            "-frames:v", "1",
            png_path.name,
        ],
        cwd=ass_path.parent, capture_output=True, check=True, timeout=60,
    )


@requires_ffmpeg
def test_fixed_caption_render_differs_from_known_buggy_double_shaped_render(tmp_path: Path):
    import arabic_reshaper
    from bidi.algorithm import get_display

    style = PRESETS["minimal"]
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=1.0, text=ARABIC_TEXT, words=[
            {"word": w, "start": i * 0.2, "end": (i + 1) * 0.2} for i, w in enumerate(ARABIC_TEXT.split())
        ]),
    ])
    edl = EDL(
        width=WIDTH, height=HEIGHT,
        clips=[EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"])],
    )

    fixed_ass = build_ass(transcript, edl, style)
    fixed_ass_path = tmp_path / "fixed.ass"
    fixed_ass_path.write_text(fixed_ass, encoding="utf-8")
    fixed_png = tmp_path / "fixed.png"
    _render_ass_to_png(fixed_ass_path, fixed_png)

    # Reproduce the old buggy behavior directly: pre-shape + BiDi-reorder the
    # same text before placing it in an otherwise identical ASS event.
    buggy_text = get_display(arabic_reshaper.reshape(ARABIC_TEXT))
    header = ASS_HEADER_TEMPLATE.format(
        width=WIDTH, height=HEIGHT, font=style.font_ar, size=style.font_size,
        primary=style.primary_color, highlight=style.highlight_color, outline_color=style.outline_color,
        back=style.back_color, bold=-1 if style.bold else 0, outline=style.outline, shadow=style.shadow,
        margin_l=40, margin_r=40, margin_v=60,
    )
    buggy_event = f"Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{buggy_text}"
    buggy_ass_path = tmp_path / "buggy.ass"
    buggy_ass_path.write_text(header + buggy_event + "\n", encoding="utf-8")
    buggy_png = tmp_path / "buggy.png"
    _render_ass_to_png(buggy_ass_path, buggy_png)

    fixed_bytes = fixed_png.read_bytes()
    buggy_bytes = buggy_png.read_bytes()

    assert len(fixed_bytes) > 500, "fixed render looks blank/empty"
    assert len(buggy_bytes) > 500, "buggy-reference render looks blank/empty"
    assert fixed_bytes != buggy_bytes, (
        "fixed caption render is pixel-identical to the known-buggy "
        "double-shaped render -- the double-shaping regression is back"
    )
