"""Editorial treatment choice for each planned B-roll slot.

Generated imagery is OPTIONAL. For every candidate slot the editor picks ONE
treatment: stay on the speaker, punch-in/reframe, kinetic typography,
behind-subject text, a motion graphic, local/user B-roll or generated
B-roll. Decisions come from the project (`broll_editorial.json` in the review
directory -- reviewable/editable state) and, absent a decision, from a
conservative generic heuristic that never defaults every slot to generation.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind

DECISIONS_FILENAME = "broll_editorial.json"
_DIGIT_RE = re.compile(r"\d")
_SHORT_SLOT_S = 3.0
# Fallback heuristic: at most this share of slots may default to generated
# imagery; the rest use cheaper on-speaker treatments.
_MAX_GENERATED_SHARE = 1 / 3


class Treatment(str, Enum):
    STAY_ON_SPEAKER = "stay_on_speaker"
    PUNCH_IN = "punch_in"
    KINETIC_TYPOGRAPHY = "kinetic_typography"
    BEHIND_SUBJECT_TEXT = "behind_subject_text"
    MOTION_GRAPHIC = "motion_graphic"
    LOCAL_BROLL = "local_broll"
    GENERATED_BROLL = "generated_broll"


BROLL_TREATMENTS = frozenset({Treatment.LOCAL_BROLL, Treatment.GENERATED_BROLL})


class TreatmentDecision(BaseModel):
    """One editorial decision for the slot overlapping [timeline_start, timeline_end]."""

    timeline_start: float
    timeline_end: float
    treatment: Treatment
    reason: str = ""
    visual_concept: str | None = None
    source_recommendation: str | None = None
    # Only meaningful for generated_broll; the mandatory realism/no-text rules
    # are appended by `broll.prompt.build_broll_prompt` regardless.
    prompt: str | None = None
    # Review-only hints (see `review.edit_plan`): what to do when the asset is
    # missing / generation is not approved, other treatments worth offering, and
    # candidate keywords for text treatments (each must come from the approved
    # transcript of that slot).
    fallback: Treatment | None = None
    alternatives: list[Treatment] = []
    text_options: list[str] = []


def load_decisions(review_dir: Path) -> list[TreatmentDecision]:
    path = review_dir / DECISIONS_FILENAME
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TreatmentDecision.model_validate(d) for d in raw.get("decisions", [])]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _match(item: BrollPlanItem, decisions: list[TreatmentDecision]) -> TreatmentDecision | None:
    best, best_ov = None, 0.0
    for d in decisions:
        ov = _overlap(item.timeline_start, item.timeline_end, d.timeline_start, d.timeline_end)
        if ov > best_ov and ov >= 0.5 * (item.timeline_end - item.timeline_start):
            best, best_ov = d, ov
    return best


@dataclass
class _Heuristic:
    treatment: Treatment
    reason: str


def _heuristic(items: list[BrollPlanItem]) -> dict[int, _Heuristic]:
    out: dict[int, _Heuristic] = {}
    open_slots: list[int] = []
    for i, item in enumerate(items):
        if item.source not in (BrollSourceKind.NONE, BrollSourceKind.GENERATED_IMAGE, BrollSourceKind.GENERATED_VIDEO):
            out[i] = _Heuristic(Treatment.LOCAL_BROLL, "a local/user asset already matches this slot")
        elif _DIGIT_RE.search(item.spoken_concept):
            out[i] = _Heuristic(Treatment.MOTION_GRAPHIC, "a spoken number reads better as a graphic than as imagery")
        elif item.timeline_end - item.timeline_start <= _SHORT_SLOT_S:
            out[i] = _Heuristic(Treatment.PUNCH_IN, "short beat: a reframe keeps energy without cutting away")
        else:
            open_slots.append(i)
    quota = math.ceil(len(open_slots) * _MAX_GENERATED_SHARE) if open_slots else 0
    ranked = sorted(open_slots, key=lambda i: items[i].confidence, reverse=True)
    generated = set(ranked[:quota])
    flip = False
    for i in open_slots:
        if i in generated:
            out[i] = _Heuristic(Treatment.GENERATED_BROLL, "highest-confidence open slot; generation capped to a third of slots")
        else:
            out[i] = _Heuristic(Treatment.PUNCH_IN if flip else Treatment.STAY_ON_SPEAKER, "stay with the speaker; no strong visual noun")
            flip = not flip
    return out


def assign_treatments(items: list[BrollPlanItem], decisions: list[TreatmentDecision] | None) -> list[BrollPlanItem]:
    """Sets `treatment` (+ reason/concept/prompt/generate_later) on every item."""
    fallback = _heuristic(items)
    for i, item in enumerate(items):
        decision = _match(item, decisions or [])
        if decision is not None:
            treatment, reason = decision.treatment, decision.reason or "editor decision"
            if decision.visual_concept:
                item.visual_concept = decision.visual_concept
            if decision.source_recommendation:
                item.source_recommendation = decision.source_recommendation
            if decision.prompt:
                item.prompt = decision.prompt
        else:
            treatment, reason = fallback[i].treatment, fallback[i].reason
        item.treatment = treatment.value
        item.treatment_reason = reason
        item.generate_later = treatment is Treatment.GENERATED_BROLL
        if item.generate_later:
            concept = item.visual_concept or item.recommended_visual
            item.recommended_visual = concept
            item.visual_concept = concept
    return items
