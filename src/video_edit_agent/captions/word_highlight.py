"""Word-level highlight / karaoke tagging for the ASS caption engine
(spec section 15).

Builds libass `\\k` karaoke tags in LOGICAL word order (so per-word timing is
always correct) and leaves the word text itself untouched. This project's
ffmpeg/libass build is compiled with HarfBuzz + FriBidi (see
`captions/rtl.py` for the full explanation), so libass does its own letter
shaping and BiDi reordering on the Dialogue text at render time -- pre-
shaping words here (or reordering the assembled line) would double-process
already-tagged, already-logical-order text and garble it. Per-word `{\\k}`
tags are BiDi-neutral override blocks, so they travel with the word they
prefix through libass's own reordering.
"""
from __future__ import annotations

from video_edit_agent.captions.chunking import CaptionChunk
from video_edit_agent.captions.styles import CaptionStyle


def build_karaoke_text(chunk: CaptionChunk, style: CaptionStyle, break_before: int | None = None) -> str:
    """`break_before` is the index of the word that starts the second line
    (an explicit ASS line break is emitted before it)."""
    out = ""
    for i, w in enumerate(chunk.words):
        duration_cs = max(1, round((w.end - w.start) * 100))
        tag = f"{{\\k{duration_cs}\\kf{duration_cs}}}" if style.word_highlight else ""
        if i:
            out += "\\N" if i == break_before else " "
        out += f"{tag}{w.word}"
    return out
