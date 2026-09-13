"""False-start / filler detection (spec section 11).

Detects word-level filler tokens and abrupt self-interrupted sentences so the
planner can trim them. Deliberately conservative: only flags, never rewrites
the underlying verbatim transcript text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from video_edit_agent.core.schemas import Segment, Word

ARABIC_FILLERS = {"يعني", "امم", "اه", "آه", "ايه", "طيب", "خلاص كده"}
ENGLISH_FILLERS = {"um", "uh", "uhh", "umm", "like", "you know"}

_SELF_CORRECTION_RE = re.compile(r"\b(\w+)[-–]\s*\1\b", re.IGNORECASE)


@dataclass
class FalseStart:
    segment_id: str
    word_index: int
    word: str
    kind: str  # "filler" | "self_correction"


def detect_fillers(words: list[Word], segment_id: str) -> list[FalseStart]:
    out = []
    for i, w in enumerate(words):
        token = w.word.strip().strip(".,،!؟?").lower()
        if token in ARABIC_FILLERS or token in ENGLISH_FILLERS:
            out.append(FalseStart(segment_id=segment_id, word_index=i, word=w.word, kind="filler"))
    return out


def detect_self_corrections(segment: Segment) -> list[FalseStart]:
    out = []
    for m in _SELF_CORRECTION_RE.finditer(segment.text):
        out.append(FalseStart(segment_id=segment.id, word_index=-1, word=m.group(0), kind="self_correction"))
    return out


def analyze_segment(segment: Segment) -> list[FalseStart]:
    return detect_fillers(segment.words, segment.id) + detect_self_corrections(segment)
