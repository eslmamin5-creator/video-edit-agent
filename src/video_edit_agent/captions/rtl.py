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

from video_edit_agent.language.arabic import ARABIC_RANGE, is_arabic_text


def alignment_for_text(text: str) -> int:
    """ASS \\an alignment codes: 2 = bottom-center for both directions —
    RTL doesn't need a different anchor, just correctly shaped glyphs."""
    return 2


# A chunk is laid out right-to-left when at least this share of its words
# contain Arabic letters: an Arabic sentence with embedded Latin terms
# ("... الـbrand owner") stays RTL even when the Latin terms hold most letters.
_RTL_WORD_SHARE = 0.4


def is_rtl_chunk(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    with_arabic = sum(1 for w in words if any(ARABIC_RANGE.match(c) for c in w))
    return with_arabic / len(words) >= _RTL_WORD_SHARE


def rtl_override_tags(text: str) -> str:
    """Force right-to-left paragraph direction inside libass via ASS override
    tags for Arabic lines (including code-switched ones whose words are mostly
    Arabic); leave pure-Latin lines to auto-detection."""
    if is_arabic_text(text) or is_rtl_chunk(text):
        return "{\\rtl}"
    return ""


# Unicode RIGHT-TO-LEFT EMBEDDING / POP DIRECTIONAL FORMATTING. This project's
# libass build ignores `{\rtl}` and first-strong detection is defeated as soon
# as a line mixes scripts, so a code-switched line ("من الـbusiness owner")
# came out with the Latin run on the wrong side. Verified empirically: only an
# explicit embedding around the whole event text yields a true RTL paragraph
# (Arabic order right-to-left, Latin runs kept left-to-right inside it).
_RLE = "\u202b"
_PDF = "\u202c"


def wrap_rtl(dialogue_text: str, chunk_text: str) -> str:
    """The event text for a chunk: RTL chunks get the `{\\rtl}` tag plus an
    RTL embedding around each line of the (already karaoke-tagged,
    logical-order) text; everything else is returned untouched. libass runs
    BiDi per hard line (`\\N`), so the embedding is closed and re-opened
    around every break."""
    tags = rtl_override_tags(chunk_text)
    if not tags:
        return dialogue_text
    body = dialogue_text.replace("\\N", f"{_PDF}\\N{_RLE}")
    return f"{tags}{_RLE}{body}{_PDF}"
