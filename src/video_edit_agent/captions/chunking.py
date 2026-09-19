"""Semantic, timing-aware caption chunking (spec section 15).

Deliberately NOT fixed two-word chunks (an English-only assumption per spec
section 46/55) — chunks are built from punctuation + a max-chars budget so
Arabic and English both read naturally.

The budget alone cuts wherever the line happens to fill, which strands a
preposition at the end of one event, or leaves "وجهة | نظر" in two. So when the
budget forces a cut, the cut moves back to the nearest boundary that does not
split a phrase (`captions/phrasing.py`: function words and construct-phrase
heads bind to what follows, Latin runs stay whole; a real pause overrides), and
a lone trailing word is folded back into a neighbour when the budget allows.
Words and their timings are never altered -- only where events begin and end.
"""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.captions.phrasing import splits_phrase
from video_edit_agent.core.schemas import Word

SENTENCE_BREAK_CHARS = set(".!?؟!،,")

MIN_WORDS_PER_EVENT = 2  # a cut never leaves fewer, unless the run itself is shorter
_OVERFLOW_FACTOR = 1.3  # a lone word may push an event this far past the budget
_PAUSE_BREAK_S = 0.4  # a pause this long is a boundary whatever the words are


@dataclass
class CaptionChunk:
    words: list[Word]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words).strip()


def _text_len(words: list[Word]) -> int:
    return len(" ".join(w.word for w in words))


def _ends_sentence(word: Word) -> bool:
    stripped = word.word.strip()
    return bool(stripped) and stripped[-1] in SENTENCE_BREAK_CHARS


def _boundary_splits_phrase(prev: Word, nxt: Word) -> bool:
    return splits_phrase(prev, nxt, pause_s=nxt.start - prev.end, pause_break_s=_PAUSE_BREAK_S)


def _cut_index(current: list[Word], incoming: Word) -> int | None:
    """How many of `current` stay in the event that is being closed because
    `incoming` does not fit. The fullest phrase-safe cut wins; None when every
    cut would split a phrase."""
    for k in range(len(current), 0, -1):
        if k < len(current) and k < MIN_WORDS_PER_EVENT <= len(current):
            break
        nxt = current[k] if k < len(current) else incoming
        if not _boundary_splits_phrase(current[k - 1], nxt):
            return k
    return None


def _absorb_lone_words(chunks: list[CaptionChunk], max_chars: int, max_duration: float) -> list[CaptionChunk]:
    """A one-word event (usually the tail of a phrase) reads as a flicker.
    Fold it into the previous event when the budget (with some slack) allows,
    otherwise borrow the previous event's last word if that leaves it intact."""
    out: list[CaptionChunk] = []
    for chunk in chunks:
        prev = out[-1] if out else None
        if prev is None or len(chunk.words) != 1 or _ends_sentence(prev.words[-1]):
            out.append(chunk)
            continue
        merged = [*prev.words, *chunk.words]
        if _text_len(merged) <= max_chars * _OVERFLOW_FACTOR and (merged[-1].end - merged[0].start) <= max_duration * 1.25:
            out[-1] = CaptionChunk(words=merged, start=merged[0].start, end=merged[-1].end)
        elif len(prev.words) > MIN_WORDS_PER_EVENT and not _boundary_splits_phrase(prev.words[-2], prev.words[-1]):
            kept, moved = prev.words[:-1], [prev.words[-1], *chunk.words]
            out[-1] = CaptionChunk(words=kept, start=kept[0].start, end=kept[-1].end)
            out.append(CaptionChunk(words=moved, start=moved[0].start, end=moved[-1].end))
        else:
            out.append(chunk)
    return out


def chunk_words(words: list[Word], max_chars: int = 26, max_duration: float = 3.2) -> list[CaptionChunk]:
    if not words:
        return []

    chunks: list[CaptionChunk] = []
    current: list[Word] = []

    def close(count: int) -> None:
        """Emit the first `count` words of `current` as an event; the rest
        carries over into the next one."""
        emitted = current[:count]
        del current[:count]
        if emitted:
            chunks.append(CaptionChunk(words=emitted, start=emitted[0].start, end=emitted[-1].end))

    def overflows(incoming: Word) -> bool:
        return bool(current) and (
            _text_len([*current, incoming]) > max_chars or (incoming.end - current[0].start) > max_duration
        )

    def within_slack(incoming: Word) -> bool:
        return (
            _text_len([*current, incoming]) <= max_chars * _OVERFLOW_FACTOR
            and (incoming.end - current[0].start) <= max_duration * 1.25
        )

    for w in words:
        while overflows(w):
            cut = _cut_index(current, w)
            if cut is None:
                # No phrase-safe cut: keep the phrase whole if the event can
                # stretch a little, otherwise fall back to a plain greedy cut.
                if within_slack(w):
                    break
                cut = len(current)
            close(cut)
        current.append(w)
        if _ends_sentence(w):
            close(len(current))

    close(len(current))
    return _absorb_lone_words(chunks, max_chars, max_duration)
