"""RTL layout helpers for the ASS caption engine (spec section 15).

Baseline Recovery Milestone item 3 correction: this project's ffmpeg/libass
build is compiled with HarfBuzz + FriBidi (verified via `ffmpeg -version`),
so libass performs its OWN complex-script shaping and BiDi reordering on
Dialogue text -- it expects plain LOGICAL-order Unicode, not pre-shaped
visual-order glyphs. Feeding it text already run through
`arabic_reshaper`/`python-bidi` (what this module's docstring used to
recommend) double-processes it: reshaping already-presentation-form letters
breaks joining, and re-reversing already-visual-order text un-reorders it.
Verified empirically by rendering both variants through this project's
actual ffmpeg `subtitles` filter and comparing frames -- pre-shaped input
rendered garbled/disjointed glyphs, raw logical-order input rendered
correct joining and reading order.

The `{\\rtl}` override tag below is the correct mechanism instead: it tells
libass's own BiDi algorithm to treat the paragraph as RTL, and libass then
shapes + reorders the logical-order text itself.
"""
from __future__ import annotations

from video_edit_agent.language.arabic import is_arabic_text


def alignment_for_text(text: str) -> int:
    """ASS \\an alignment codes: 2 = bottom-center for both directions —
    RTL doesn't need a different anchor, just correctly shaped glyphs."""
    return 2


def rtl_override_tags(text: str) -> str:
    """Force right-to-left paragraph direction inside libass via ASS override
    tags for pure-Arabic lines; leave mixed/Latin lines to auto-detection."""
    if is_arabic_text(text):
        return "{\\rtl}"
    return ""
