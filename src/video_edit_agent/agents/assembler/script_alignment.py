"""Script-to-scene alignment (spec section 7): maps scenes to script beats
only with real evidence. Offline, this means keyword overlap between a
scene's filename and a script beat's text -- never a guessed/invented
mapping. A scene with no keyword match gets `script_beat_index=None`,
`confidence=0.0`, which callers (ordering.py) must treat as "no
recommendation," not "beat 0."
"""
from __future__ import annotations

import re

from video_edit_agent.agents.assembler.schemas import ScriptAlignmentItem, SceneInventoryItem

_WORD_RE = re.compile(r"[a-zA-Z؀-ۿ]{3,}")
_STOPWORDS = {"the", "and", "for", "with", "this", "that", "scene", "clip"}


def _beats(script_text: str) -> list[str]:
    # Paragraphs (blank-line separated) are the natural "beat" granularity
    # for a plain-text script; falls back to non-empty lines if there are
    # no blank-line breaks at all.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", script_text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [line.strip() for line in script_text.splitlines() if line.strip()]


def _keywords(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS


def align_script(script_text: str, items: list[SceneInventoryItem]) -> list[ScriptAlignmentItem]:
    beats = _beats(script_text)
    beat_keywords = [_keywords(b) for b in beats]

    results: list[ScriptAlignmentItem] = []
    for item in items:
        scene_keywords = _keywords(item.filename.rsplit(".", 1)[0])
        best_idx, best_score = None, 0.0
        for i, bk in enumerate(beat_keywords):
            if not scene_keywords or not bk:
                continue
            overlap = scene_keywords & bk
            if not overlap:
                continue
            score = len(overlap) / len(scene_keywords)
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx is not None:
            results.append(
                ScriptAlignmentItem(
                    scene_id=item.id,
                    script_beat_index=best_idx,
                    script_beat_text=beats[best_idx][:200],
                    confidence=round(min(1.0, best_score), 3),
                    rationale=f"filename keyword overlap with script beat {best_idx}",
                )
            )
        else:
            results.append(
                ScriptAlignmentItem(
                    scene_id=item.id,
                    script_beat_index=None,
                    script_beat_text=None,
                    confidence=0.0,
                    rationale="no keyword evidence linking this scene's filename to any script beat",
                )
            )
    return results
