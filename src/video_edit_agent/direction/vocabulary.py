"""The closed treatment vocabulary of the visual director.

The director's first question for every semantic beat is: *is the speaker still
the best visual for this beat?*

  yes -> a SPEAKER treatment (the speaker stays; only the framing changes)
  no  -> a REPLACEMENT treatment (the speaker is temporarily replaced by another
         visual while the voice continues uninterrupted; see `replacement`)

Behind-subject text is ONE treatment among the others, never mandatory, and
kinetic typography may either sit over the speaker or replace it. AI picks from
this vocabulary; nothing outside it is executable.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum


class SpeakerTreatment(str, Enum):
    SPEAKER_STATIC = "speaker_static"
    PUNCH_IN = "punch_in"
    PUNCH_OUT = "punch_out"
    REFRAME = "reframe"
    SLOW_PUSH = "slow_push"
    HOLD = "hold"
    RESET_TO_BASE = "reset_to_base"


class ReplacementTreatment(str, Enum):
    REAL_BROLL = "real_broll"  # stock/real footage that exists
    USER_BROLL = "user_broll"  # the user's own footage
    MOTION_GRAPHIC = "motion_graphic"
    KINETIC_TYPOGRAPHY = "kinetic_typography"
    ILLUSTRATION = "illustration"
    GENERATED_VISUAL = "generated_visual"  # only ever after explicit generation approval
    GRAPHIC_DATA_SCENE = "graphic_data_scene"
    FULL_SCREEN_TEXT_SCENE = "full_screen_text_scene"


class OverlayTreatment(str, Enum):
    """The speaker stays visible and something is layered with them."""

    BEHIND_SUBJECT_TEXT = "behind_subject_text"


SPEAKER_TREATMENTS = frozenset(t.value for t in SpeakerTreatment)
REPLACEMENT_TREATMENTS = frozenset(t.value for t in ReplacementTreatment)
OVERLAY_TREATMENTS = frozenset(t.value for t in OverlayTreatment)
VOCABULARY = SPEAKER_TREATMENTS | REPLACEMENT_TREATMENTS | OVERLAY_TREATMENTS

# Kinetic typography is a replacement in the vocabulary, but by default it is
# layered over the speaker (the established hook style); `speaker_visible` says which.
_DEFAULT_VISIBLE_REPLACEMENTS = frozenset({ReplacementTreatment.KINETIC_TYPOGRAPHY.value})

# Treatments that need footage supplied by someone.
ASSET_TREATMENTS = frozenset({ReplacementTreatment.REAL_BROLL.value, ReplacementTreatment.USER_BROLL.value})
GENERATED_TREATMENTS = frozenset({ReplacementTreatment.GENERATED_VISUAL.value})
# Treatments whose visual is authored by the pipeline itself (no external asset).
AUTHORED_TREATMENTS = frozenset({
    ReplacementTreatment.MOTION_GRAPHIC.value, ReplacementTreatment.KINETIC_TYPOGRAPHY.value,
    ReplacementTreatment.ILLUSTRATION.value, ReplacementTreatment.GRAPHIC_DATA_SCENE.value,
    ReplacementTreatment.FULL_SCREEN_TEXT_SCENE.value, OverlayTreatment.BEHIND_SUBJECT_TEXT.value,
})
TEXT_TREATMENTS = frozenset({
    ReplacementTreatment.KINETIC_TYPOGRAPHY.value, ReplacementTreatment.FULL_SCREEN_TEXT_SCENE.value,
    OverlayTreatment.BEHIND_SUBJECT_TEXT.value,
})
# The strong primary motion/text treatments: while one is active, captions must not compete with it.
STRONG_PRIMARY = frozenset({
    ReplacementTreatment.MOTION_GRAPHIC.value, ReplacementTreatment.KINETIC_TYPOGRAPHY.value,
    ReplacementTreatment.GRAPHIC_DATA_SCENE.value, ReplacementTreatment.FULL_SCREEN_TEXT_SCENE.value,
    ReplacementTreatment.ILLUSTRATION.value, OverlayTreatment.BEHIND_SUBJECT_TEXT.value,
})


# The edit-plan review keeps its established treatment names; each vocabulary
# treatment maps onto one of them (the exact camera move / footage source is
# carried in the slot's own fields).
_TO_PLAN = {
    SpeakerTreatment.SPEAKER_STATIC.value: "stay_on_speaker",
    SpeakerTreatment.HOLD.value: "stay_on_speaker",
    SpeakerTreatment.PUNCH_OUT.value: "stay_on_speaker",
    SpeakerTreatment.RESET_TO_BASE.value: "stay_on_speaker",
    SpeakerTreatment.PUNCH_IN.value: "punch_in",
    SpeakerTreatment.SLOW_PUSH.value: "punch_in",
    SpeakerTreatment.REFRAME.value: "punch_in",
    ReplacementTreatment.REAL_BROLL.value: "local_broll",
    ReplacementTreatment.USER_BROLL.value: "local_broll",
    ReplacementTreatment.GENERATED_VISUAL.value: "generated_broll",
}


def to_plan_treatment(treatment: str) -> str:
    """The edit-plan (`broll.treatment.Treatment`) name for a vocabulary treatment."""
    return _TO_PLAN.get(treatment, treatment)


def is_vocabulary(treatment: str) -> bool:
    return treatment in VOCABULARY


def treatment_class(treatment: str) -> str:
    """`speaker` | `replacement` | `overlay`."""
    if treatment in REPLACEMENT_TREATMENTS:
        return "replacement"
    if treatment in OVERLAY_TREATMENTS:
        return "overlay"
    return "speaker"


def default_speaker_visible(treatment: str) -> bool:
    if treatment in REPLACEMENT_TREATMENTS:
        return treatment in _DEFAULT_VISIBLE_REPLACEMENTS
    return True


__all__ = [
    "ASSET_TREATMENTS", "AUTHORED_TREATMENTS", "GENERATED_TREATMENTS", "OVERLAY_TREATMENTS", "REPLACEMENT_TREATMENTS",
    "SPEAKER_TREATMENTS", "STRONG_PRIMARY", "TEXT_TREATMENTS", "VOCABULARY", "OverlayTreatment",
    "ReplacementTreatment", "SpeakerTreatment", "default_speaker_visible", "is_vocabulary", "to_plan_treatment", "treatment_class",
]
