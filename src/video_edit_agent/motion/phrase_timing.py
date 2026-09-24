"""Phrase-anchored timing for text treatments.

A text treatment belongs to the words that motivate it, not to the editorial
slot that happens to contain them. Given the approved (corrected) transcript
words, `locate_phrase` finds the phrase and its exact spoken span, and
`plan_show_window` turns that span into a show window:

    lead-in  ->  entrance  ->  readable at the phrase  ->  hold  ->  exit

    show_start = phrase_start - lead_in
    readable   = show_start + enter          (at or just before the phrase)
    hold_end   = phrase_end + tail           (the semantic beat completes, then a breath)
    show_end   = hold_end + exit

The lead/enter/tail/exit values come from a `TimingPreset` (a generic treatment
preset, never a per-brand or per-video constant). The window is clamped to the
slot the user approved, so a treatment never leaks outside its slot, and it is
never the whole slot unless the words themselves span it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from video_edit_agent.core.schemas import Word

_PUNCT = ".,،؛;:!?؟\"'“”«»()[]{}…-–—"


@dataclass(frozen=True)
class TimingPreset:
    """Seconds. Entrances are longer than exits (an exit that lingers reads as a
    mistake, an entrance that snaps in reads as a pop-up)."""

    lead_in: float = 0.30  # entrance starts this long before the phrase's first word
    enter: float = 0.30  # entrance length: readable at `phrase_start` with the default lead-in
    tail: float = 0.40  # hold after the last word ends
    exit: float = 0.25
    min_show: float = 1.0  # a treatment shorter than this cannot be read, so it is stretched to it
    max_unanchored: float = 2.0  # cap when the phrase has no word timing to anchor to


DEFAULT_TIMING = TimingPreset()


@dataclass(frozen=True)
class WordTime:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class PhraseTiming:
    """Where the phrase is spoken (`start`..`end`) and when the treatment shows.
    `source` is "words" when derived from approved word timings, "slot" when
    there were none and the window is a capped slot-anchored fallback."""

    start: float
    end: float
    words: tuple[WordTime, ...]
    show_start: float
    readable_at: float
    hold_end: float
    show_end: float
    source: str = "words"

    @property
    def duration(self) -> float:
        return self.show_end - self.show_start


def _key(token: str) -> str:
    return token.strip(_PUNCT).casefold()


def locate_phrase(
    words: Sequence[Word], phrase: str, near: tuple[float, float] | None = None,
) -> tuple[WordTime, ...] | None:
    """The run of consecutive transcript `words` that spells `phrase` (whole
    tokens, ignoring case and punctuation), as `WordTime`s in the words' own
    time base; None when the phrase is not spoken. With several matches the one
    overlapping `near` (or, failing that, closest to it) wins."""
    target = [_key(t) for t in phrase.split()]
    target = [t for t in target if t]
    if not target:
        return None
    keys = [_key(w.word) for w in words]
    hits = [i for i in range(len(words) - len(target) + 1) if keys[i:i + len(target)] == target]
    if not hits:
        return None

    def span(i: int) -> tuple[float, float]:
        return words[i].start, words[i + len(target) - 1].end

    if near is not None and len(hits) > 1:
        lo, hi = near

        def distance(i: int) -> float:
            s, e = span(i)
            return 0.0 if s < hi and e > lo else min(abs(s - hi), abs(e - lo))

        hits.sort(key=lambda i: (distance(i), i))
    i = hits[0]
    return tuple(WordTime(w.word, w.start, w.end) for w in words[i:i + len(target)])


def plan_show_window(
    phrase: Sequence[WordTime] | None,
    slot: tuple[float, float],
    *,
    preset: TimingPreset = DEFAULT_TIMING,
    offset: float = 0.0,
) -> PhraseTiming:
    """The show window for `phrase` (word times shifted by `offset` onto the
    timeline axis), clamped to the approved `slot`. Without a phrase the window
    is the first `preset.max_unanchored` seconds of the slot."""
    slot_start, slot_end = slot
    if phrase:
        words = tuple(WordTime(w.word, w.start + offset, w.end + offset) for w in phrase)
        start, end, source = words[0].start, words[-1].end, "words"
        show_start = start - preset.lead_in
        hold_end = end + preset.tail
        show_end = hold_end + preset.exit
    else:
        words, source = (), "slot"
        start = slot_start + preset.lead_in
        end = min(slot_end, slot_start + preset.max_unanchored)
        show_start = slot_start
        hold_end = end
        show_end = end + preset.exit
    show_start = max(show_start, slot_start)
    if show_end - show_start < preset.min_show:
        show_end = show_start + preset.min_show
        hold_end = show_end - preset.exit
    show_end = min(show_end, slot_end)
    hold_end = min(hold_end, show_end - min(preset.exit, show_end - show_start))
    readable_at = min(show_start + preset.enter, hold_end)
    return PhraseTiming(
        start=start, end=end, words=words, show_start=show_start, readable_at=readable_at,
        hold_end=hold_end, show_end=show_end, source=source,
    )
