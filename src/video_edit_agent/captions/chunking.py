"""Semantic, timing-aware caption chunking (spec section 15).

Deliberately NOT fixed two-word chunks (an English-only assumption per spec
section 46/55) — chunks are built from punctuation + a max-chars budget so
Arabic and English both read naturally.
"""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.core.schemas import Word

SENTENCE_BREAK_CHARS = set(".!?؟!،,")


@dataclass
class CaptionChunk:
    words: list[Word]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words).strip()


def chunk_words(words: list[Word], max_chars: int = 26, max_duration: float = 3.2) -> list[CaptionChunk]:
    if not words:
        return []

    chunks: list[CaptionChunk] = []
    current: list[Word] = []

    def flush():
        if current:
            chunks.append(CaptionChunk(words=list(current), start=current[0].start, end=current[-1].end))
            current.clear()

    for w in words:
        candidate_text = " ".join([*(x.word for x in current), w.word])
        candidate_duration = (w.end - current[0].start) if current else 0.0

        if current and (len(candidate_text) > max_chars or candidate_duration > max_duration):
            flush()

        current.append(w)

        stripped = w.word.strip()
        if stripped and stripped[-1] in SENTENCE_BREAK_CHARS:
            flush()

    flush()
    return chunks
