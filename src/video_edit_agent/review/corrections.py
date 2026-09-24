"""Persisted transcript corrections (project state, not ASS editing).

Corrections live in `<edit>/review/transcript_corrections.json`; the pipeline
applies them on every run so a user fixes a word once and every later preview
and render reflects it. Applying corrections never changes timing: same word
count maps 1:1 onto the original word slots; otherwise tokens the correction
leaves unchanged keep their original timing and only the replaced blocks are
spread over the original span of the words they replace.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.review.schemas import TranscriptCorrection

CORRECTIONS_FILENAME = "transcript_corrections.json"


def load_corrections(review_dir: Path) -> list[TranscriptCorrection]:
    path = review_dir / CORRECTIONS_FILENAME
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptCorrection.model_validate(c) for c in raw.get("corrections", [])]


def save_corrections(review_dir: Path, corrections: list[TranscriptCorrection]) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / CORRECTIONS_FILENAME
    payload = {"corrections": [c.model_dump(mode="json") for c in corrections]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def add_correction(review_dir: Path, correction: TranscriptCorrection) -> list[TranscriptCorrection]:
    """Upserts by (segment_id, word_index); later corrections replace earlier ones."""
    existing = [
        c for c in load_corrections(review_dir)
        if (c.segment_id, c.word_index) != (correction.segment_id, correction.word_index)
    ]
    existing.append(correction)
    save_corrections(review_dir, existing)
    return existing


_PUNCT = ".,،؛;:!?؟\"'“”«»()[]{}…"


def _key(token: str) -> str:
    return token.strip(_PUNCT).casefold()


def replace_phrase(text: str, old: str, new: str) -> tuple[str | None, int]:
    """Replaces exactly one occurrence of the word/phrase `old` in `text` with `new`
    (used verbatim -- never formalized or translated). Returns `(new_text, count)`.

    Matching is whole-token, ignoring case and surrounding punctuation; the
    punctuation around the matched tokens is kept. If there is no whole-token
    match, a lone word is matched inside a longer token (an Arabic clitic such
    as a leading `ال`) when that is unambiguous. `new_text` is None when there
    are zero matches or more than one (`count` says which): the caller must ask
    instead of guessing which one the user meant."""
    tokens = text.split()
    old_keys = [_key(t) for t in old.split()]
    n = len(old_keys)
    if n == 0 or not new.strip():
        return None, 0
    hits = [i for i in range(len(tokens) - n + 1) if [_key(t) for t in tokens[i:i + n]] == old_keys]
    if len(hits) > 1:
        return None, len(hits)
    if not hits:
        if n != 1:
            return None, 0
        pattern = re.compile(re.escape(old.strip(_PUNCT)), re.IGNORECASE)
        inside = [(i, len(pattern.findall(t))) for i, t in enumerate(tokens) if pattern.search(t)]
        total = sum(c for _, c in inside)
        if total != 1:
            return None, total
        i = inside[0][0]
        tokens[i] = pattern.sub(lambda _m: new.strip(), tokens[i], count=1)
        return " ".join(tokens), 1
    i, j = hits[0], hits[0] + n - 1
    lead = tokens[i][:len(tokens[i]) - len(tokens[i].lstrip(_PUNCT))]
    trail = tokens[j][len(tokens[j].rstrip(_PUNCT)):]
    fresh = new.split()
    if lead and fresh[0][0] not in _PUNCT:
        fresh[0] = lead + fresh[0]
    if trail and fresh[-1][-1] not in _PUNCT:
        fresh[-1] += trail
    tokens[i:j + 1] = fresh
    return " ".join(tokens), 1


def set_segment_text(review_dir: Path, segment_id: str, text: str) -> list[TranscriptCorrection]:
    """Stores `text` as the whole corrected text of a segment. Earlier word-level
    corrections for that segment are dropped (they indexed the previous wording)."""
    kept = [c for c in load_corrections(review_dir) if c.segment_id != segment_id]
    kept.append(TranscriptCorrection(segment_id=segment_id, corrected_text=text))
    save_corrections(review_dir, kept)
    return kept


def clear_corrections(review_dir: Path, segment_id: str) -> list[TranscriptCorrection]:
    kept = [c for c in load_corrections(review_dir) if c.segment_id != segment_id]
    save_corrections(review_dir, kept)
    return kept


def _spread(words: list[str], start: float, end: float, confidence: float = 1.0) -> list[Word]:
    """Distributes `words` over [start, end] proportionally to their length."""
    weights = [max(len(w), 1) for w in words]
    total = sum(weights)
    out: list[Word] = []
    cursor = start
    for w, weight in zip(words, weights, strict=True):
        span = (end - start) * weight / total
        out.append(Word(word=w, start=cursor, end=cursor + span, confidence=confidence))
        cursor += span
    if out:
        out[-1].end = end
    return out


def _align_words(words: list[Word], new_tokens: list[str]) -> list[Word]:
    """Re-words `words` as `new_tokens` while preserving timing.

    Unchanged tokens keep their exact timing and confidence; a replaced block
    is spread over the span of the original words it replaces; an inserted
    block shares the span of the neighbouring word; dropped words hand their
    span to the previous kept word. The overall segment span never changes.
    """
    old_tokens = [w.word for w in words]
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    out: list[Word] = []
    pending_start: float | None = None  # span of dropped leading words
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            kept = list(words[i1:i2])
            if pending_start is not None and kept:
                kept[0] = kept[0].model_copy(update={"start": pending_start})
                pending_start = None
            out.extend(kept)
        elif tag == "delete":
            if out:
                out[-1] = out[-1].model_copy(update={"end": words[i2 - 1].end})
            elif pending_start is None:
                pending_start = words[i1].start
        elif tag == "replace":
            start = words[i1].start if pending_start is None else pending_start
            pending_start = None
            out.extend(_spread(new_tokens[j1:j2], start, words[i2 - 1].end))
        else:  # insert: share the previous word's span (or the next word's)
            if out:
                prev = out.pop()
                out.extend(_spread([prev.word, *new_tokens[j1:j2]], prev.start, prev.end, prev.confidence))
            else:
                nxt = words[i1]
                out.extend(_spread(new_tokens[j1:j2], nxt.start, nxt.start + (nxt.end - nxt.start) / 2))
                words = [*words[:i1], nxt.model_copy(update={"start": out[-1].end}), *words[i1 + 1:]]
    return out


def _correct_segment(seg: Segment, corrections: list[TranscriptCorrection]) -> Segment:
    words = list(seg.words)
    text_fix = next((c for c in corrections if c.word_index is None), None)
    if text_fix is not None:
        new_tokens = text_fix.corrected_text.split()
        if words and len(new_tokens) == len(words):
            words = [w.model_copy(update={"word": t, "confidence": 1.0}) for w, t in zip(words, new_tokens, strict=True)]
        elif words and new_tokens:
            words = _align_words(words, new_tokens)
        elif words:
            words = []
        else:
            words = _spread(new_tokens, seg.start, seg.end) if new_tokens else []
    for c in corrections:
        if c.word_index is None or not (0 <= c.word_index < len(words)):
            continue
        new_tokens = c.corrected_text.split()
        old = words[c.word_index]
        replacement = (
            [old.model_copy(update={"word": new_tokens[0], "confidence": 1.0})]
            if len(new_tokens) == 1
            else _spread(new_tokens, old.start, old.end)
        )
        words[c.word_index:c.word_index + 1] = replacement
    return seg.model_copy(update={"text": " ".join(w.word for w in words), "words": words})


def apply_corrections(transcript: Transcript, corrections: list[TranscriptCorrection]) -> Transcript:
    """Returns a copy of `transcript` with `corrections` applied. Segment
    start/end and every unchanged word's timing are preserved."""
    if not corrections:
        return transcript
    by_segment: dict[str, list[TranscriptCorrection]] = {}
    for c in corrections:
        by_segment.setdefault(c.segment_id, []).append(c)
    segments = [
        _correct_segment(seg, by_segment[seg.id]) if seg.id in by_segment else seg
        for seg in transcript.segments
    ]
    return transcript.model_copy(update={"segments": segments})
