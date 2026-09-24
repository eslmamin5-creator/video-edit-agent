"""Primary visual hierarchy: which layer leads a beat, and what everything else does about it.

One beat, one primary visual. When a large headline, a motion graphic or a
replacement scene leads, the captions and any other text must not compete with
it at equal strength (large headline + equal caption + motion text is the failure
this metadata exists to prevent). This is PLANNING METADATA ONLY: the caption
renderer is not wired to it in this phase, and it changes no caption timing,
chunking or karaoke.

    primary_layer     speaker | replacement_visual | headline
    caption_role      normal | reduced  (captions are never hidden)
    headline_role     none | primary_headline_typography
    headline_placement  top | head_adjacent | behind_subject | full_screen | None
    speaker_visibility  full | partial | hidden

`primary_headline_typography` is one generic role with placement modes;
`behind_subject` is only a valid placement when the editorial suitability gate
passed, and it is never a default. Reference styles do not decide this: the
placement follows the treatment and the gate.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from video_edit_agent.direction.vocabulary import (
    STRONG_PRIMARY,
    TEXT_TREATMENTS,
    OverlayTreatment,
    ReplacementTreatment,
    default_speaker_visible,
    treatment_class,
)

PRIMARY_HEADLINE = "primary_headline_typography"


class PrimaryLayer(str, Enum):
    SPEAKER = "speaker"
    REPLACEMENT = "replacement_visual"
    HEADLINE = "headline"


class HeadlinePlacement(str, Enum):
    TOP = "top"
    HEAD_ADJACENT = "head_adjacent"
    BEHIND_SUBJECT = "behind_subject"  # only when the suitability gate passes; never a default
    FULL_SCREEN = "full_screen"  # the headline IS the replacement scene


class VisualHierarchy(BaseModel):
    primary_layer: str = PrimaryLayer.SPEAKER.value
    caption_role: str = "normal"  # normal | reduced
    headline_role: str = "none"
    headline_placement: str | None = None
    speaker_visibility: str = "full"  # full | partial | hidden
    notes: list[str] = Field(default_factory=list)

    def duplication_risks(self) -> list[str]:
        """Ways this hierarchy would put two equal-strength layers on screen (empty = none)."""
        risks: list[str] = []
        if self.primary_layer != PrimaryLayer.SPEAKER.value and self.caption_role != "reduced":
            risks.append("a strong primary layer with full-strength captions")
        if self.headline_role != "none" and self.primary_layer != PrimaryLayer.HEADLINE.value:
            risks.append("a headline that is not the primary layer")
        if self.headline_placement == HeadlinePlacement.BEHIND_SUBJECT.value and self.speaker_visibility == "hidden":
            risks.append("behind-subject text with the speaker hidden")
        return risks


def build_hierarchy(treatment: str, speaker_visible: bool | None = None, *, gate_passed: bool | None = None) -> VisualHierarchy:
    """The hierarchy for `treatment` (a vocabulary treatment). `gate_passed` is the behind-subject
    suitability verdict (None = not assessed); behind-subject placement needs it to be True,
    otherwise the headline falls back to the top placement and the note says why."""
    visible = default_speaker_visible(treatment) if speaker_visible is None else speaker_visible
    klass = treatment_class(treatment)
    h = VisualHierarchy(speaker_visibility="full" if visible else "hidden")
    if treatment in TEXT_TREATMENTS:
        h.primary_layer, h.headline_role = PrimaryLayer.HEADLINE.value, PRIMARY_HEADLINE
        if treatment == OverlayTreatment.BEHIND_SUBJECT_TEXT.value:
            if gate_passed:
                h.headline_placement = HeadlinePlacement.BEHIND_SUBJECT.value
                h.speaker_visibility = "partial"  # the speaker is in front of the words
            else:
                h.headline_placement = HeadlinePlacement.TOP.value
                h.notes.append("behind-subject placement needs the suitability gate to pass; the headline would sit at the top instead"
                               if gate_passed is False else
                               "behind-subject placement is unproven (no gate verdict); the headline would sit at the top instead")
        elif treatment == ReplacementTreatment.FULL_SCREEN_TEXT_SCENE.value:
            h.headline_placement = HeadlinePlacement.FULL_SCREEN.value
        else:  # kinetic typography layered over the speaker
            h.headline_placement = HeadlinePlacement.TOP.value if visible else HeadlinePlacement.FULL_SCREEN.value
            h.speaker_visibility = "partial" if visible else "hidden"
    elif klass == "replacement":
        h.primary_layer = PrimaryLayer.REPLACEMENT.value if not visible else PrimaryLayer.SPEAKER.value
    if treatment in STRONG_PRIMARY:
        h.caption_role = "reduced"
        h.notes.append("captions stay readable but do not compete with the primary visual")
    return h


__all__ = ["PRIMARY_HEADLINE", "HeadlinePlacement", "PrimaryLayer", "VisualHierarchy", "build_hierarchy"]
