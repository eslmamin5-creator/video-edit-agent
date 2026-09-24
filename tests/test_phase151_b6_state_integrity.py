"""Phase 1.5.1: Approved-State Integrity Fix.

A previously-*approved* `lower_subject_semantic` headline (e.g. B6's "Five Cs") persisted in a
project's own `edit_plan.json` must survive regeneration verbatim and can never be silently
displaced by a freshly auto-discovered candidate. Conversely, an existing, previously-reviewed
project with no approved headline this run must not let auto-discovery quietly burn a merely
`pending_review` candidate into the render -- only a genuinely fresh project (no persisted review
state at all) keeps the original auto-discover-and-burn behaviour. This file covers both the
`direction.production_profile` pinning/gating primitives and the `core.pipeline` derivation logic
that decides `pinned_headline` / `allow_auto_headline_discovery` from a loaded `EditPlan`.
Synthetic content only: no project, brand or reference specifics."""
from __future__ import annotations

from tests.test_phase131_b2_c2 import FACE, STOP, _long_edl_and_plan
from video_edit_agent.direction.composition import timeline_words
from video_edit_agent.direction.production_profile import (
    EditingProfile,
    discover_and_verify_headline,
    plan_and_apply_visual_rhythm,
)
from video_edit_agent.review.edit_plan import EditPlan, EditPlanSlot, SlotStatus

_APPROVED_STATUSES = {"approved", "changed", "generation_approved"}


def _approved_headline_slot(slots: list[EditPlanSlot]) -> EditPlanSlot | None:
    """Mirrors core.pipeline.run_pipeline's derivation exactly, for isolated unit testing."""
    return next(
        (s for s in slots if s.treatment == "lower_subject_semantic" and s.status.value in _APPROVED_STATUSES and s.text),
        None,
    )


def _pinned_headline(slots: list[EditPlanSlot]) -> tuple[str, float, float] | None:
    slot = _approved_headline_slot(slots)
    return (slot.text, slot.timeline_start, slot.timeline_end) if slot else None


def _allow_auto_discovery(existing_plan: EditPlan | None, pinned: tuple[str, float, float] | None) -> bool:
    existing_slots = existing_plan.slots if existing_plan else []
    return existing_plan is None or not existing_slots or pinned is not None


# ---- 1. an approved lower_subject_semantic slot survives regeneration, pinned verbatim ----------
def test_1_approved_headline_pins_and_verifies_instead_of_discovering():
    t, edl = _long_edl_and_plan(12.0)
    words = timeline_words(t, edl)
    # "check twice" is what auto-discovery would find on this fixture (see test_phase15_product_freeze
    # test_8); pin a DIFFERENT, already-approved phrase instead and confirm it wins.
    pinned_phrase = next(w.text for w in words if w.text not in STOP)
    start = next(w.start for w in words if w.text == pinned_phrase)
    end = next(w.end for w in words if w.text == pinned_phrase)
    wiring = discover_and_verify_headline(
        words, profile=EditingProfile.BALANCED, face_box=FACE, width=edl.width, height=edl.height,
        headline_font_px=64, stopwords=STOP, duration=12.0,
        pinned=(pinned_phrase, start, end),
    )
    assert wiring.ok and wiring.composition is not None
    assert wiring.composition.phrase.rstrip(".") == pinned_phrase.rstrip(".")
    assert wiring.composition.semantic_source == "user_pinned"
    assert wiring.composition.approval_status == "approved"


# ---- 2. a pinned headline is never displaced by the auto-discoverable "check twice" -------------
def test_2_pinned_headline_not_displaced_by_auto_discovery():
    t, edl = _long_edl_and_plan(12.0)
    words = timeline_words(t, edl)
    pinned_phrase = next(w.text for w in words if w.text not in STOP)
    start = next(w.start for w in words if w.text == pinned_phrase)
    end = next(w.end for w in words if w.text == pinned_phrase)
    wiring = discover_and_verify_headline(
        words, profile=EditingProfile.BALANCED, face_box=FACE, width=edl.width, height=edl.height,
        headline_font_px=64, stopwords=STOP, duration=12.0,
        pinned=(pinned_phrase, start, end),
    )
    assert wiring.composition.phrase != "check twice"


# ---- 3. allow_auto_discovery=False with no pin fails closed: no headline burned in ----------------
def test_3_auto_discovery_disabled_without_pin_fails_closed():
    t, edl = _long_edl_and_plan(12.0)
    words = timeline_words(t, edl)
    wiring = discover_and_verify_headline(
        words, profile=EditingProfile.BALANCED, face_box=FACE, width=edl.width, height=edl.height,
        headline_font_px=64, stopwords=STOP, duration=12.0,
        pinned=None, allow_auto_discovery=False,
    )
    assert not wiring.ok
    assert wiring.composition is None and wiring.candidate is None


# ---- 4. allow_auto_discovery=True (default) preserves the original auto-discover behaviour --------
def test_4_auto_discovery_enabled_by_default_unaffected_by_new_params():
    t, edl = _long_edl_and_plan(12.0)
    words = timeline_words(t, edl)
    wiring = discover_and_verify_headline(
        words, profile=EditingProfile.BALANCED, face_box=FACE, width=edl.width, height=edl.height,
        headline_font_px=64, stopwords=STOP, duration=12.0,
    )
    assert wiring.ok and wiring.composition.phrase == "check twice"


# ---- 5. plan_and_apply_visual_rhythm passes a pinned headline all the way through ------------------
def test_5_visual_rhythm_uses_pinned_headline_not_auto_discovery():
    t, edl = _long_edl_and_plan(12.0)
    words = timeline_words(t, edl)
    pinned_phrase = next(w.text for w in words if w.text not in STOP)
    start = next(w.start for w in words if w.text == pinned_phrase)
    end = next(w.end for w in words if w.text == pinned_phrase)
    result = plan_and_apply_visual_rhythm(
        t, edl, profile=EditingProfile.BALANCED, face_box=FACE, headline_font_px=64,
        stopwords=STOP, pinned_headline=(pinned_phrase, start, end),
    )
    assert result.headline.ok and result.headline.composition.phrase.rstrip(".") == pinned_phrase.rstrip(".")


# ---- 6. plan_and_apply_visual_rhythm with allow_auto_headline_discovery=False and no pin -----------
#         yields no headline at all (an existing reviewed project with nothing approved this run) ---
def test_6_visual_rhythm_no_pin_and_discovery_disabled_yields_no_headline():
    t, edl = _long_edl_and_plan(12.0)
    result = plan_and_apply_visual_rhythm(
        t, edl, profile=EditingProfile.BALANCED, face_box=FACE, headline_font_px=64,
        stopwords=STOP, pinned_headline=None, allow_auto_headline_discovery=False,
    )
    assert not result.headline.ok


# ---- 7. pipeline predicate: an APPROVED lower_subject_semantic slot is pinned ----------------------
def test_7_pipeline_predicate_pins_approved_slot():
    slot = EditPlanSlot(
        timeline_start=32.82, timeline_end=36.04, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.CHANGED, text="Five Cs",
    )
    plan = EditPlan(slots=[slot])
    pinned = _pinned_headline(plan.slots)
    assert pinned == ("Five Cs", 32.82, 36.04)
    assert _allow_auto_discovery(plan, pinned) is True  # a real pin always re-enables verification


# ---- 8. pipeline predicate: PENDING_REVIEW never counts as approved (never rendered) ---------------
def test_8_pipeline_predicate_ignores_pending_review_slot():
    slot = EditPlanSlot(
        timeline_start=17.88, timeline_end=21.44, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.PENDING_REVIEW, text="تأكد الأول",
    )
    plan = EditPlan(slots=[slot])
    pinned = _pinned_headline(plan.slots)
    assert pinned is None
    # an existing reviewed project (has slots) with nothing approved this run: auto-discovery is
    # disabled so the pending candidate cannot silently enter the render
    assert _allow_auto_discovery(plan, pinned) is False


# ---- 9. pipeline predicate: REJECTED never counts as approved (never rendered) ---------------------
def test_9_pipeline_predicate_ignores_rejected_slot():
    slot = EditPlanSlot(
        timeline_start=17.88, timeline_end=21.44, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.REJECTED, text="تأكد الأول",
    )
    plan = EditPlan(slots=[slot])
    pinned = _pinned_headline(plan.slots)
    assert pinned is None
    assert _allow_auto_discovery(plan, pinned) is False


# ---- 10. pipeline predicate: GENERATION_APPROVED also counts as an approved slot status -------------
def test_10_pipeline_predicate_accepts_generation_approved_status():
    slot = EditPlanSlot(
        timeline_start=32.82, timeline_end=36.04, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.GENERATION_APPROVED, text="Five Cs",
    )
    plan = EditPlan(slots=[slot])
    assert _pinned_headline(plan.slots) == ("Five Cs", 32.82, 36.04)


# ---- 11. pipeline predicate: an approved slot with no text yet is not a usable pin -------------------
def test_11_pipeline_predicate_requires_nonempty_text():
    slot = EditPlanSlot(
        timeline_start=32.82, timeline_end=36.04, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.APPROVED, text=None,
    )
    plan = EditPlan(slots=[slot])
    assert _pinned_headline(plan.slots) is None


# ---- 12. pipeline predicate: a fresh project (no persisted plan at all) keeps auto-discover-and-burn -
def test_12_fresh_project_with_no_plan_allows_auto_discovery():
    assert _allow_auto_discovery(None, None) is True


# ---- 13. pipeline predicate: an existing plan with zero slots behaves like a fresh project ------------
def test_13_existing_plan_with_no_slots_allows_auto_discovery():
    plan = EditPlan(slots=[])
    assert _allow_auto_discovery(plan, None) is True


# ---- 14. an approved Behind-Subject (C3-style) slot is untouched by the headline predicate ------------
def test_14_behind_subject_slot_never_mistaken_for_a_headline_pin():
    c3 = EditPlanSlot(
        timeline_start=58.1, timeline_end=64.96, recommended="behind_subject_text",
        treatment="behind_subject_text", status=SlotStatus.APPROVED, text="value حقيقية",
    )
    b6 = EditPlanSlot(
        timeline_start=32.82, timeline_end=36.04, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.CHANGED, text="Five Cs",
    )
    plan = EditPlan(slots=[c3, b6])
    assert _pinned_headline(plan.slots) == ("Five Cs", 32.82, 36.04)
    behind_subject_windows = [
        (s.timeline_start, s.timeline_end)
        for s in plan.slots
        if s.treatment == "behind_subject_text" and s.status.value in _APPROVED_STATUSES
    ]
    assert behind_subject_windows == [(58.1, 64.96)]


# ---- 15. rerunning the derivation twice on the same approved state is idempotent -----------------------
def test_15_pipeline_predicate_idempotent_across_reruns():
    slot = EditPlanSlot(
        timeline_start=32.82, timeline_end=36.04, recommended="stay_on_speaker",
        treatment="lower_subject_semantic", status=SlotStatus.CHANGED, text="Five Cs",
    )
    plan = EditPlan(slots=[slot])
    first = _pinned_headline(plan.slots)
    second = _pinned_headline(plan.slots)
    assert first == second == ("Five Cs", 32.82, 36.04)
    assert _allow_auto_discovery(plan, first) == _allow_auto_discovery(plan, second) is True
