"""Phrase-aware two-line caption wrapping.

A caption chunk that is too wide for one line is split in two. A plain
character-count split happily cuts a preposition off from its noun or splits
"business owner" across lines; this module prefers a break that respects the
phrase: it never separates a function word from the word it governs, never
splits a run of Latin (code-switched) words, prefers a real pause or trailing
punctuation, and only among such breaks balances line widths. When no
phrase-safe break fits within the line cap it falls back to the plain
width-balanced split, so a chunk is always wrapped somehow.

Nothing here is brand-, project- or video-specific: the word lists are
language-level (Arabic/English function words), and timing is never touched --
the result is just the index of the word that starts line two.
"""
from __future__ import annotations

import re

# Words that grammatically bind to the FOLLOWING word: a line must not end on
# them (prepositions, conjunctions, negation/relative/demonstrative openers,
# the definite-article prefix token used in code-switching, and their English
# counterparts).
_BINDS_FORWARD = frozenset({
    # Arabic prepositions / particles / conjunctions
    "في", "من", "على", "عن", "إلى", "الى", "مع", "بـ", "لـ", "كـ", "الـ", "و", "أو", "او", "ثم", "ف", "ولا",
    "ما", "لا", "لم", "لن", "إن", "أن", "ان", "هل", "لو", "لما", "علشان", "عشان", "إنت", "انت", "أنت",
    "اللي", "الذي", "التي", "كل", "غاية", "زي", "بين", "حتى", "قبل", "بعد", "عند", "يعني",
    # English
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "and", "or", "but", "is", "are",
    "your", "our", "their", "my", "this", "that",
})
# Words that bind to the PRECEDING word: a line must not start with them
# (demonstratives/pronoun suffixes standing alone after a noun).
# Head words of a fixed two-word unit (Arabic construct phrases such as
# "وجهة نظر"): they need their complement, so a break right after them is a split.
_UNIT_HEADS = frozenset({"وجهة", "وجهات", "صاحب", "أصحاب", "اصحاب"})
_BINDS_BACKWARD = frozenset({"ده", "دي", "دة", "دا", "دول", "نفسه", "نفسها", "كمان", "برضه", "بس"})

_LATIN = re.compile(r"[A-Za-z]")
_TRAILING_PUNCT = ".!?؟،,;:؛"
_PAUSE_S = 0.25
_HARD_CAP_FACTOR = 1.35  # a line may exceed the nominal width by this much


def _is_latin(word: str) -> bool:
    return bool(_LATIN.search(word))


def _line_len(words: list) -> int:
    return len(" ".join(w.word for w in words))


def _binds_forward(word: str) -> bool:
    """A function word, alone or fused with a leading conjunction letter
    ("وفي" = "و" + "في", "فاللي" = "ف" + "اللي")."""
    if word in _BINDS_FORWARD or word in _UNIT_HEADS:
        return True
    return len(word) > 2 and word[0] in "وف" and word[1:] in _BINDS_FORWARD


def _phrase_violation(prev, nxt) -> bool:
    """True when breaking between `prev` and `nxt` would split a phrase."""
    p, n = prev.word.strip(_TRAILING_PUNCT), nxt.word
    if prev.word[-1:] in _TRAILING_PUNCT:
        return False  # punctuation is an explicit phrase boundary
    if _binds_forward(p) or n in _BINDS_BACKWARD:
        return True
    if _is_latin(p) and _is_latin(n):
        return True  # keep a code-switched Latin run ("business owner") whole
    return p.endswith("ـ")  # "الـ" + Latin word


def splits_phrase(prev, nxt, *, pause_s: float | None = None, pause_break_s: float = 0.4) -> bool:
    """Public form of the phrase test for callers that build chunks. A real
    pause between the words (>= `pause_break_s`) is treated as a boundary even
    between a function word and its neighbour: the speaker separated them."""
    if pause_s is not None and pause_s >= pause_break_s:
        return False
    return _phrase_violation(prev, nxt)


def width_break_index(words: list, max_line_chars: int) -> int | None:
    """The plain width-balanced split: the word index that starts line two so
    both lines have similar length, or None when the chunk fits on one line."""
    if max_line_chars <= 0 or len(words) < 2:
        return None
    if _line_len(words) <= max_line_chars:
        return None
    best_i, best_gap = None, None
    for i in range(1, len(words)):
        gap = abs(_line_len(words[:i]) - _line_len(words[i:]))
        if best_gap is None or gap < best_gap:
            best_i, best_gap = i, gap
    return best_i


def phrase_break_index(words: list, max_line_chars: int) -> int | None:
    """Index of the word that starts line two, preferring phrase-safe breaks;
    None when the chunk fits on one line (or cannot be split)."""
    if max_line_chars <= 0 or len(words) < 2 or _line_len(words) <= max_line_chars:
        return None
    hard_cap = max_line_chars * _HARD_CAP_FACTOR
    best_i, best_score = None, None
    for i in range(1, len(words)):
        first, second = _line_len(words[:i]), _line_len(words[i:])
        if max(first, second) > hard_cap:
            continue
        if _phrase_violation(words[i - 1], words[i]):
            continue
        score = float(abs(first - second))
        pause = words[i].start - words[i - 1].end
        if words[i - 1].word[-1:] in _TRAILING_PUNCT:
            score -= 8.0
        elif pause >= _PAUSE_S:
            score -= 6.0
        if best_score is None or score < best_score:
            best_i, best_score = i, score
    if best_i is not None:
        return best_i
    if _line_len(words) <= hard_cap:
        return None  # no phrase-safe break: one slightly long line beats a split phrase
    return width_break_index(words, max_line_chars)
