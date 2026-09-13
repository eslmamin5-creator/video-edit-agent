"""RTL layout helpers for the ASS caption engine (spec section 15).

libass renders RTL text correctly when the glyph run itself is already
shaped+reordered (visual order), which is what `shaping.py` produces. This
module only decides paragraph-level alignment/justification for mixed
Arabic/English lines.
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
