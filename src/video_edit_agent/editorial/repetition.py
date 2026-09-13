"""Repetition detection (spec section 11): find segments where the speaker
re-recorded the same line ("takes"), so only the best take is kept."""
from __future__ import annotations

import re
from dataclasses import dataclass

from video_edit_agent.core.schemas import Segment
from video_edit_agent.language.arabic import normalize_for_matching


@dataclass
class RepetitionGroup:
    segment_ids: list[str]
    normalized_text: str


def _normalize(text: str) -> str:
    text = normalize_for_matching(text.lower())
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    """Cheap token-overlap similarity — good enough to catch re-takes of the
    same line without pulling in a heavy NLP dependency."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_repetition_groups(segments: list[Segment], threshold: float = 0.6) -> list[RepetitionGroup]:
    normalized = [(seg.id, _normalize(seg.text)) for seg in segments if seg.text.strip()]
    used: set[str] = set()
    groups: list[RepetitionGroup] = []

    for i, (id_a, norm_a) in enumerate(normalized):
        if id_a in used or not norm_a:
            continue
        group = [id_a]
        for id_b, norm_b in normalized[i + 1:]:
            if id_b in used:
                continue
            if _similarity(norm_a, norm_b) >= threshold:
                group.append(id_b)
                used.add(id_b)
        if len(group) > 1:
            used.add(id_a)
            groups.append(RepetitionGroup(segment_ids=group, normalized_text=norm_a))
    return groups


def pick_best_take(group: RepetitionGroup, segments_by_id: dict[str, Segment]) -> str:
    """Prefer the LAST take (speakers usually improve on retakes), unless a
    later take is drastically shorter (likely a false start), in which case
    prefer the longest/most complete one."""
    candidates = [segments_by_id[sid] for sid in group.segment_ids]
    durations = [(c.end - c.start) for c in candidates]
    max_duration = max(durations)
    for c, d in zip(reversed(candidates), reversed(durations)):
        if d >= max_duration * 0.7:
            return c.id
    return candidates[-1].id
