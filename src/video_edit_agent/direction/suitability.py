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
    head_hair_integrity  the head/hair edge survives compositing (optional; an explicit False fails the gate)
    meaningful_occlusion the subject actually overlaps the word in a way that reads as depth (optional)

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from pydantic import BaseModel

CHECKS = ("mask_quality", "readable_occlusion", "phrase_timing", "shot_composition", "caption_hierarchy")
OPTIONAL_CHECKS = ("head_hair_integrity", "meaningful_occlusion")  # unmeasured is tolerated here; measured-False is not
MIN_VISUAL_VALUE = 0.5


class BehindSubjectEvidence(BaseModel):
    """Measured facts about one beat. None means not measured."""

    mask_quality: bool | None = None
    readable_occlusion: bool | None = None
    phrase_timing: bool | None = None
    shot_composition: bool | None = None
    caption_hierarchy: bool | None = None
    visual_value: float | None = None
    head_hair_integrity: bool | None = None
    meaningful_occlusion: bool | None = None


class Suitability(BaseModel):
    suitable: bool
    failed: list[str]
    unproven: list[str]
    summary: str


def assess_behind_subject(evidence: BehindSubjectEvidence) -> Suitability:
    failed = [c for c in CHECKS + OPTIONAL_CHECKS if getattr(evidence, c) is False]
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


# --------------------------------------------------------------------------
# Phase 1.2: behind-subject eligibility for one phrase (stricter than the gate above; still fail-closed)
# --------------------------------------------------------------------------

MAX_PHRASE_WORDS = 4
MIN_SEMANTIC_CONFIDENCE = 0.45  # below this a reading is a guess: it never buys behind-subject text
# the rhythm states behind-subject text may sit with (a settled or slightly reframed shot, or the lowered shot that makes room)
COMPATIBLE_STATES = ("base", "hold", "reset_to_base", "reframe_left", "reframe_right", "lower_subject")


class Eligibility(BaseModel):
    """`eligible` = passes every check; it still needs the user's approval when review-first applies.
    `not_eligible` names what failed or is unproven. `not_applicable` = there was no phrase to consider."""

    status: str  # eligible | not_eligible | not_applicable
    failed: list[str] = []
    unproven: list[str] = []
    requires_approval: bool = True
    summary: str = ""


def behind_subject_eligibility(
    *,
    phrase: str | None,
    semantic_kind: str | None,
    semantic_confidence: float | None,
    evidence: BehindSubjectEvidence | None,
    rhythm_state: str = "base",
    caption_competing: bool = False,
    better_simpler_treatment: bool = False,
    review_first: bool = True,
) -> Eligibility:
    """Whether behind-subject text may be OFFERED for this phrase. Every one of the ten conditions must hold;
    anything unmeasured counts as unproven and fails. Behind-subject is a semantic enhancement: it is never part
    of the basic rhythm loop and it is never forced onto a suitable phrase."""
    if not phrase or not phrase.strip():
        return Eligibility(status="not_applicable", requires_approval=review_first, summary="no candidate phrase in this stretch")
    ev = evidence or BehindSubjectEvidence()
    failed: list[str] = []
    unproven: list[str] = []

    def need(name: str, value: bool | None) -> None:
        if value is None:
            unproven.append(name)
        elif value is False:
            failed.append(name)

    if semantic_kind is None or semantic_confidence is None:
        unproven.append("semantic_clarity")
    elif semantic_confidence < MIN_SEMANTIC_CONFIDENCE:
        failed.append("semantic_clarity")
    words = phrase.split()
    if not 1 <= len(words) <= MAX_PHRASE_WORDS:
        failed.append("short_phrase")
    need("mask_integrity", ev.mask_quality)
    need("meaningful_occlusion", ev.meaningful_occlusion)
    need("head_hair_integrity", ev.head_hair_integrity)
    need("readable_behind_subject", ev.readable_occlusion)
    need("phrase_timing", ev.phrase_timing)
    need("composition_supports", ev.shot_composition)
    if rhythm_state not in COMPATIBLE_STATES:
        failed.append(f"composition_state:{rhythm_state}")
    need("caption_hierarchy", ev.caption_hierarchy)
    if caption_competing:
        failed.append("caption_competes")
    if better_simpler_treatment:
        failed.append("a_simpler_treatment_is_better")
    if ev.visual_value is None:
        unproven.append("visual_value")
    elif ev.visual_value < MIN_VISUAL_VALUE:
        failed.append("no_better_simpler_treatment")
    ok = not failed and not unproven
    parts = []
    if failed:
        parts.append("failed: " + ", ".join(failed))
    if unproven:
        parts.append("not proven: " + ", ".join(unproven))
    return Eligibility(
        status="eligible" if ok else "not_eligible", failed=failed, unproven=unproven, requires_approval=review_first,
        summary="passes every check" + ("; still needs your approval" if review_first else "") if ok else "; ".join(parts),
    )


__all__ = [
    "CHECKS", "COMPATIBLE_STATES", "MAX_PHRASE_WORDS", "MIN_SEMANTIC_CONFIDENCE", "MIN_VISUAL_VALUE", "OPTIONAL_CHECKS",
    "BehindSubjectEvidence", "Eligibility", "Suitability", "assess_behind_subject", "behind_subject_eligibility",
]
