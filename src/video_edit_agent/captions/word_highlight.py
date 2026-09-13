"""Word-level highlight / karaoke tagging for the ASS caption engine
(spec section 15).

Builds libass `\\k` karaoke tags in LOGICAL word order (so per-word timing is
always correct), shapes each Arabic word independently (safe: Arabic letter
joining happens within a word, never across the space that separates words),
then runs BiDi reordering once over the fully-tagged line. `{}`/`\\` control
sequences are BiDi-neutral (class ON) so they travel with the run they sit
next to — this is the same approach production Arabic-subtitle tooling uses,
but it is a best-effort visual approximation, not a formally verified BiDi
embedding; extremely long mixed-direction lines are the main edge case worth
watching in visual regression tests (spec section 38).
"""
from __future__ import annotations

from video_edit_agent.captions.chunking import CaptionChunk
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.language.arabic import is_arabic_text

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    _HAS_ARABIC_SHAPING = True
except ImportError:
    _HAS_ARABIC_SHAPING = False


def build_karaoke_text(chunk: CaptionChunk, style: CaptionStyle) -> str:
    is_ar = is_arabic_text(chunk.text)
    parts = []
    for w in chunk.words:
        duration_cs = max(1, round((w.end - w.start) * 100))
        word_text = w.word
        if is_ar and _HAS_ARABIC_SHAPING:
            word_text = arabic_reshaper.reshape(word_text)
        tag = f"{{\\k{duration_cs}\\kf{duration_cs}}}" if style.word_highlight else ""
        parts.append(f"{tag}{word_text}")

    line = " ".join(parts)
    if is_ar and _HAS_ARABIC_SHAPING:
        line = get_display(line)
    return line
