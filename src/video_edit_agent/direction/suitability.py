"""Editorial suitability gate for behind-subject text.

The technical compositing path (hair-safe matte, subject-wins z-order, matte
gate) only says the effect CAN be drawn cleanly. This gate says whether it
SHOULD be, for a given beat. It is fail-closed: a check that has not been
measured counts as unproven, and an unproven or failed check means the director
does not choose behind-subject (another treatment is stronger or simply
safer). The gate never forces the effect.

    mask_quality       the matte gate passed for the window
    readable_occlusion the subject hides the text only where the text is still legible
    phrase_timing      the word lands with the spoken phrase (word-level timing exists)
    shot_composition   there is clear space behind/around the subject for the word
    caption_hierarchy  the captions do not fight the word for the same place/strength
    visual_value       how much stronger it is than the best alternative (0..1)

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from pydantic import BaseModel

CHECKS = ("mask_quality", "readable_occlusion", "phrase_timing", "shot_composition", "caption_hierarchy")
MIN_VISUAL_VALUE = 0.5


class BehindSubjectEvidence(BaseModel):
    """Measured facts about one beat. None means not measured."""

    mask_quality: bool | None = None
    readable_occlusion: bool | None = None
    phrase_timing: bool | None = None
    shot_composition: bool | None = None
    caption_hierarchy: bool | None = None
    visual_value: float | None = None


class Suitability(BaseModel):
    suitable: bool
    failed: list[str]
    unproven: list[str]
    summary: str


def assess_behind_subject(evidence: BehindSubjectEvidence) -> Suitability:
    failed = [c for c in CHECKS if getattr(evidence, c) is False]
    unproven = [c for c in CHECKS if getattr(evidence, c) is None]
    if evidence.visual_value is None:
        unproven.append("visual_value")
    elif evidence.visual_value < MIN_VISUAL_VALUE:
        failed.append("visual_value")
    suitable = not failed and not unproven
    if suitable:
        summary = "passes the editorial gate"
    else:
        parts = []
        if failed:
            parts.append("failed: " + ", ".join(failed))
        if unproven:
            parts.append("not proven: " + ", ".join(unproven))
        summary = "; ".join(parts)
    return Suitability(suitable=suitable, failed=failed, unproven=unproven, summary=summary)


__all__ = ["CHECKS", "MIN_VISUAL_VALUE", "BehindSubjectEvidence", "Suitability", "assess_behind_subject"]
