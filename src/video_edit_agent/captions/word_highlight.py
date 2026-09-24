"""Word-level highlight / karaoke tagging for the ASS caption engine
(spec section 15).

Builds libass karaoke tags in LOGICAL word order (so per-word timing is
always correct) and leaves the word text itself untouched. This project's
ffmpeg/libass build is compiled with HarfBuzz + FriBidi (see
`captions/rtl.py` for the full explanation), so libass does its own letter
shaping and BiDi reordering on the Dialogue text at render time -- pre-
shaping words here (or reordering the assembled line) would double-process
already-tagged, already-logical-order text and garble it. Karaoke tags are
BiDi-neutral override blocks, so they travel with the word they prefix
through libass's own reordering.

Timing model
------------
Every ASS karaoke tag (`\\k`, `\\kf`, `\\K`, `\\ko`) CONSUMES time: libass keeps
a running clock per event and each tag advances it by its centisecond value.
Tags that share a word therefore add up -- `{\\k38\\kf38}` moves the clock 76 cs
for a 38 cs word (the first tag is folded into the skip time of the second) and
every later word lands later and later. So each spoken word carries exactly ONE
duration-bearing tag: `\\kf<cs>`, a progressive fill from the "not yet spoken"
colour to the highlight colour over the word's real start..end. A pause between
two words (or before the first one) is the only other time-consuming tag, a
`\\k<cs>` on an empty text run, and only when the pause is at least one
centisecond.

All boundaries are converted to centiseconds relative to the event start and
rounded ONCE, cumulatively, so rounding error never accumulates along an event.
"""
from __future__ import annotations

import re

from video_edit_agent.captions.chunking import CaptionChunk
from video_edit_agent.captions.styles import CaptionStyle

_KARAOKE_TAG = re.compile(r"\\(kf|ko|K|k)(\d+)")
_OVERRIDE_BLOCK = re.compile(r"\{[^}]*\}")
_BIDI_CONTROLS = dict.fromkeys([*range(0x202A, 0x202F), *range(0x2066, 0x206A), 0x200E, 0x200F])  # for str.translate


def _cs(seconds: float) -> int:
    return round(seconds * 100)


def build_karaoke_text(chunk: CaptionChunk, style: CaptionStyle, break_before: int | None = None) -> str:
    """`break_before` is the index of the word that starts the second line
    (an explicit ASS line break is emitted before it)."""
    out = ""
    origin = _cs(chunk.start)  # the Dialogue Start, at the same centisecond rounding
    clock = 0  # centiseconds consumed so far in this event
    for i, w in enumerate(chunk.words):
        if i:
            out += "\\N" if i == break_before else " "
        if style.word_highlight:
            begin = max(clock, _cs(w.start) - origin)
            finish = max(begin + 1, _cs(w.end) - origin)  # a word always sweeps for >= 1 cs
            if begin > clock:
                out += f"{{\\k{begin - clock}}}"  # silent gap: the pause before this word
            out += f"{{\\kf{finish - begin}}}"
            clock = finish
        out += w.word
    return out


def karaoke_cs(text: str) -> int:
    """Total centiseconds of karaoke time an event's text asks libass to consume
    (every karaoke tag counts; text outside override blocks is ignored)."""
    return sum(int(m.group(2)) for blk in _OVERRIDE_BLOCK.findall(text) for m in _KARAOKE_TAG.finditer(blk))


def word_timing_tags(text: str) -> list[list[str]]:
    """Karaoke tags attached to each word of an event, in logical order. A run of
    override blocks directly before a word belongs to that word; a `\\k` gap block
    counts with the word it precedes."""
    groups: list[list[str]] = []
    pending: list[str] = []
    for part in re.split(r"(\{[^}]*\})", text):
        if part.startswith("{"):
            pending += [m.group(0) for m in _KARAOKE_TAG.finditer(part)]
        elif part.translate(_BIDI_CONTROLS).replace("\\N", "").strip():
            groups.append(pending)
            pending = []
    return groups


def check_karaoke_event(text: str, start_s: float, end_s: float, *, tolerance_cs: int = 2) -> list[str]:
    """Problems with one Dialogue event's karaoke timing (empty list = fine):
    more than one duration tag on a word (the old `\\k+\\kf` doubling), or a
    total karaoke time that is not the event's timed span within `tolerance_cs`."""
    problems = []
    for n, tags in enumerate(word_timing_tags(text)):
        sweeps = [t for t in tags if not re.fullmatch(r"\\k\d+", t)]  # everything but a plain `\k` pause
        if len(sweeps) != 1 or len(tags) > 2:
            problems.append(f"word {n} carries {len(sweeps)} sweep tags among {tags}")
    total, span = karaoke_cs(text), _cs(end_s) - _cs(start_s)
    if abs(total - span) > tolerance_cs:
        problems.append(f"karaoke total {total} cs vs event span {span} cs")
    return problems
