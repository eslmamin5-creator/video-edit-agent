"""Identifies moments in the transcript worth covering with B-roll (spec
section 20). Deliberately conservative -- only proposes a B-roll slot where a
concrete, visualizable noun phrase is spoken, not for every clip -- since
over-inserting B-roll is a worse failure mode than under-inserting it.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import EDL, BrollPlanItem, Transcript

_MIN_CLIP_DURATION_FOR_BROLL = 2.5
_MAX_BROLL_SLOTS_PER_MINUTE = 4

# Concrete/visualizable concept cues; a heuristic signal, not an NLP model --
# consistent with the rest of the editorial layer's local-first philosophy.
_CONCEPT_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "these", "those",
    "و", "في", "من", "على", "إلى", "هذا", "هذه", "ذلك",
}


def _extract_concept(text: str) -> str | None:
    words = [w.strip(".,!?؟،") for w in text.split()]
    meaningful = [w for w in words if w and w.lower() not in _CONCEPT_STOPWORDS and len(w) > 2]
    if len(meaningful) < 2:
        return None
    return " ".join(meaningful[:6])


def select_broll_moments(edl: EDL, transcript: Transcript) -> list[BrollPlanItem]:
    """Returns candidate B-roll slots, capped at
    `_MAX_BROLL_SLOTS_PER_MINUTE` per minute of final timeline duration to
    avoid a visually chaotic edit."""
    total_minutes = max(edl.total_duration / 60.0, 1 / 60.0)
    max_slots = max(1, round(total_minutes * _MAX_BROLL_SLOTS_PER_MINUTE))

    candidates: list[BrollPlanItem] = []
    for clip in edl.clips:
        if clip.duration < _MIN_CLIP_DURATION_FOR_BROLL:
            continue

        spoken_words = [w.word for w in transcript.words if clip.source_in <= w.start < clip.source_out]
        joined = " ".join(spoken_words)
        concept = _extract_concept(joined)
        if not concept:
            continue

        candidates.append(
            BrollPlanItem(
                timeline_start=clip.timeline_in,
                timeline_end=clip.timeline_out,
                purpose="illustrate spoken concept",
                spoken_concept=concept,
                recommended_visual=concept,
                duration=clip.duration,
            )
        )

    candidates.sort(key=lambda item: item.duration, reverse=True)
    return candidates[:max_slots]
