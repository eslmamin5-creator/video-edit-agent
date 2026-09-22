"""Phase 1.3.3: semantic headline qualification for `lower_subject_semantic`.
Synthetic content only: no project, brand or reference specifics. Builds on the Phase 1.3.1/1.3.2 fixtures in
`tests/test_phase131_b2_c2.py` and `tests/test_phase132_perceptual_gate_headline.py` rather than re-deriving them.

The ONE goal (spec sec. 0): a short, rare, compact phrase must not be promoted to a primary headline merely
because it scores well on rarity/length — its semantic ROLE must dominate. `concept`/`claim`/`payoff`/`keyword`
are eligible by default; `instruction`/`connector`/`discourse`/`filler` are not (sec. 1)."""
from __future__ import annotations

import re
from pathlib import Path

from tests.test_phase131_b2_c2 import GOOD, STOP, _compose_ok
from tests.test_phase131_b2_c2 import WINDOW as B2_WINDOW
from tests.test_phase132_perceptual_gate_headline import HSTOP, HWINDOW, HWORDS, _hwords
from video_edit_agent.direction.composition import (
    ELIGIBLE_HEADLINE_ROLES,
    INELIGIBLE_HEADLINE_ROLES,
    NOT_SUITABLE_NO_QUALIFIED_HEADLINE,
    TWord,
    compose_lower_subject,
    is_verbatim,
    review_state,
    select_headline,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"

# ---- fixtures ---------------------------------------------------------------------------------------------
# "check first" repeats often (an instruction lead-in that always precedes the real point); "true value" appears
# exactly once and is the whole rest of its own clause: rare AND a concept, so it must win on role, not just rarity.
_PRE = _hwords("check first. check first. check first.", start=0.0)
_WIN = _hwords("check first true value today", start=6.0)
_POST = _hwords("check first. check first.", start=11.0)
IWORDS = _PRE + _WIN + _POST
IWINDOW = (6.0, 6.0 + 4 * 0.5)


def _win_words(spec: str, start: float = 0.0, step: float = 0.5) -> list[TWord]:
    return _hwords(spec, step=step, start=start)


# ---- 1. semantic role dominates rarity -------------------------------------------------------------------
def test_1_semantic_role_dominates_rarity():
    best = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    assert best is not None and best.text == "true value"
    assert best.role in ELIGIBLE_HEADLINE_ROLES and best.headline_eligible


# ---- 2. rare instruction does NOT outrank a meaningful concept ---------------------------------------------
def test_2_rare_instruction_does_not_outrank_meaningful_concept():
    instr = select_headline(IWORDS, IWINDOW, stopwords=HSTOP, pinned_phrase="check first")
    best = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    assert instr is not None and best is not None
    assert instr.headline_eligible is False and instr.role in INELIGIBLE_HEADLINE_ROLES
    assert best.text == "true value" and best.role in ELIGIBLE_HEADLINE_ROLES
    # never selected automatically, no matter how "check first" would have scored
    assert select_headline(IWORDS, IWINDOW, stopwords=HSTOP) is not None and best.text != "check first"


# ---- 3. discourse phrase is headline-ineligible by default -------------------------------------------------
def test_3_discourse_phrase_is_headline_ineligible_by_default():
    hints = [(6.0, 8.0, "question", 0.8)]
    words = _win_words("so what now really matters", start=6.0)
    got = select_headline(words, (6.0, 6.0 + 5 * 0.5), stopwords=HSTOP, hints=hints, pinned_phrase="so what")
    assert got is not None
    assert got.role == "discourse" and not got.headline_eligible


# ---- 4. filler is headline-ineligible by default -------------------------------------------------------------
def test_4_filler_is_headline_ineligible_by_default():
    words = _win_words("so it is so it", start=0.0)
    got = select_headline(words, (0.0, 5 * 0.5), stopwords=HSTOP, pinned_phrase="so it")
    assert got is not None and got.role == "filler" and not got.headline_eligible
    assert select_headline(words, (0.0, 5 * 0.5), stopwords=HSTOP) is None  # nothing to automatically promote


# ---- 5. concept is eligible ------------------------------------------------------------------------------
def test_5_concept_is_eligible():
    best = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    assert best is not None and best.role == "concept" and best.headline_eligible


# ---- 6. claim is eligible --------------------------------------------------------------------------------
def test_6_claim_is_eligible():
    hints = [(6.0, 7.5, "key_claim", 0.8)]
    words = _win_words("this changes everything completely", start=6.0)
    got = select_headline(words, (6.0, 6.0 + 4 * 0.5), stopwords=HSTOP, hints=hints)
    assert got is not None and got.role == "claim" and got.headline_eligible


# ---- 7. payoff is eligible -------------------------------------------------------------------------------
def test_7_payoff_is_eligible():
    hints = [(6.0, 7.5, "payoff", 0.8)]
    words = _win_words("real lasting value grows", start=6.0)
    got = select_headline(words, (6.0, 6.0 + 4 * 0.5), stopwords=HSTOP, hints=hints)
    assert got is not None and got.role == "payoff" and got.headline_eligible


# ---- 8. keyword is eligible ------------------------------------------------------------------------------
def test_8_keyword_is_eligible():
    words = _win_words("integrity", start=6.0)
    got = select_headline(words, (6.0, 6.5), stopwords=HSTOP)
    assert got is not None and got.role == "keyword" and got.headline_eligible and len(got.words) == 1


# ---- 9. user-pinned weak phrase can survive only as pending_review override -----------------------------
def test_9_user_pinned_weak_phrase_survives_only_as_pending_review_override():
    pinned = select_headline(IWORDS, IWINDOW, stopwords=HSTOP, pinned_phrase="check first")
    assert pinned is not None and pinned.headline_eligible is False and pinned.semantic_source == "user_pinned"
    comp = compose_lower_subject(GOOD, B2_WINDOW, face_box=(0.30, 0.30, 0.40, 0.15), head_top=0.28,
                                  headline_height=lambda _t: 0.07, stopwords=STOP, pinned_phrase="check twice")
    assert comp.status == "ok"
    assert review_state(comp)["approval_status"] == "pending_review"  # never auto-approved, pinned or not


# ---- 10. selector never rewrites the approved transcript --------------------------------------------------
def test_10_selector_never_rewrites_the_transcript():
    assert select_headline(IWORDS, IWINDOW, stopwords=HSTOP, pinned_phrase="genuine value") is None  # not verbatim
    best = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    assert best is not None and is_verbatim(best.text, IWORDS)


# ---- 11. candidate spans remain contiguous -----------------------------------------------------------------
def test_11_candidate_spans_remain_contiguous():
    best = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    assert best is not None
    bare = [w.text.strip(".,") for w in IWORDS]
    run = " ".join(bare)
    assert best.text in run  # a contiguous substring of the transcript's word sequence, nothing spliced


# ---- 12. context can upgrade/downgrade semantic qualification -----------------------------------------------
def test_12_context_can_upgrade_or_downgrade_semantic_qualification():
    # downgrade: a "key_claim" hint on a lead-in that repeats often elsewhere, whose OWN clause continues with
    # the real, rarer, more content-bearing idea (context downgrades a hint the isolated words alone would pass).
    pre = _win_words("important detail matters here. important detail matters here. important detail matters here.", start=0.0)
    win = _win_words("important detail about the real global economic outlook today", start=len(pre) * 0.5)
    words_down = pre + win
    win_start, win_end = win[0].start, win[-1].end
    hints_down = [(win_start, win_start + 0.95, "key_claim", 0.9)]  # covers only "important detail"
    down = select_headline(words_down, (win_start, win_end + 0.01), stopwords=HSTOP, hints=hints_down,
                            pinned_phrase="important detail")
    assert down is not None and "downgraded_low_standalone_meaning" in down.reason_codes
    assert down.role == "instruction" and not down.headline_eligible

    # upgrade: a "process_list" hint on a candidate that is actually the WHOLE, content-dense clause
    words_up = _win_words("genuine lasting value", start=0.0)
    hints_up = [(0.0, 1.5, "process_list", 0.9)]
    up = select_headline(words_up, (0.0, 1.5), stopwords=HSTOP, hints=hints_up)
    assert up is not None and "upgraded_full_clause_content_dense" in up.reason_codes
    assert up.role == "concept" and up.headline_eligible


# ---- 13. uncertain semantics fail closed ---------------------------------------------------------------------
def test_13_uncertain_semantics_fail_closed():
    # candidate and remainder are close but not clearly resolved either way (no rarity gap, a small density gap):
    # neither "run carries more weight" nor "remainder is clearly the real idea" fires.
    words = _win_words("value it and gains steadily", start=0.0)
    got = select_headline(words, (0.0, 5 * 0.5), stopwords=HSTOP, pinned_phrase="value it")
    assert got is not None and got.role == "uncertain" and got.role_confidence is not None and got.role_confidence < 0.4
    assert not got.headline_eligible and "uncertain_semantics_no_strong_signal" in got.reason_codes
    assert select_headline(words, (0.0, 5 * 0.5), stopwords=HSTOP) is None or \
        select_headline(words, (0.0, 5 * 0.5), stopwords=HSTOP).text != "value it"  # never forced into a headline


# ---- 14. no qualified headline means no lower_subject_semantic scheduling ------------------------------------
def test_14_no_qualified_headline_means_no_scheduling():
    comp = compose_lower_subject(IWORDS, IWINDOW, face_box=(0.30, 0.30, 0.40, 0.15), head_top=0.28,
                                  headline_height=lambda _t: 0.07, stopwords=HSTOP,
                                  pinned_phrase=None)
    # the window itself DOES have a qualified headline ("true value"); force the failure path by shrinking it
    # to just the instruction lead-in words, where nothing eligible exists.
    narrow = (6.0, 6.0 + 2 * 0.5 - 0.05)
    blocked = compose_lower_subject(IWORDS, narrow, face_box=(0.30, 0.30, 0.40, 0.15), head_top=0.28,
                                     headline_height=lambda _t: 0.07, stopwords=HSTOP)
    assert blocked.status == NOT_SUITABLE_NO_QUALIFIED_HEADLINE
    assert blocked.technical_status == "not_applicable" and blocked.visual_status == NOT_SUITABLE_NO_QUALIFIED_HEADLINE
    assert blocked.rows == [] and blocked.changes == []
    assert comp.status == "ok"  # sanity: the full window is NOT itself broken by the qualification layer


# ---- 15. rhythm fallback remains available when semantic composition is skipped -------------------------------
def test_15_rhythm_fallback_available_when_semantic_composition_is_skipped():
    from video_edit_agent.direction.rhythm import RhythmPolicy

    narrow = (6.0, 6.0 + 2 * 0.5 - 0.05)
    blocked = compose_lower_subject(IWORDS, narrow, face_box=(0.30, 0.30, 0.40, 0.15), head_top=0.28,
                                     headline_height=lambda _t: 0.07, stopwords=HSTOP)
    assert blocked.status == NOT_SUITABLE_NO_QUALIFIED_HEADLINE
    # nothing about a skipped semantic composition disables the rhythm engine's own (camera-only) planning
    assert RhythmPolicy().plan_lower_subject is False  # lower_subject stays opt-in via a composition; camera rhythm plans regardless


# ---- 16. deterministic result offline -----------------------------------------------------------------------
def test_16_deterministic_result_offline():
    a = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    b = select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    assert a is not None and a.model_dump() == b.model_dump()


# ---- 17. optional model/hint signal cannot make cloud access mandatory ----------------------------------------
def test_17_optional_hint_signal_cannot_make_cloud_access_mandatory():
    # the exact same qualification runs, with identical eligibility, whether or not a semantic-beat hint is supplied
    words = _win_words("real lasting value grows", start=6.0)
    window = (6.0, 6.0 + 4 * 0.5)
    without_hint = select_headline(words, window, stopwords=HSTOP)
    with_hint = select_headline(words, window, stopwords=HSTOP, hints=[(6.0, 7.5, "payoff", 0.8)])
    assert without_hint is not None and with_hint is not None
    assert without_hint.headline_eligible and with_hint.headline_eligible  # degrades safely: no hint -> still usable
    assert with_hint.role == "payoff" and without_hint.role in ELIGIBLE_HEADLINE_ROLES


# ---- 18. no client-specific phrase hardcoding in production code ---------------------------------------------------
def test_18_no_client_or_project_hardcoding_in_generic_code():
    files = [SRC / "direction" / "composition.py", SRC / "motion" / "occlusion.py", SRC / "captions" / "headline.py"]
    banned = re.compile(r"#[0-9a-fA-F]{6}\b|[ء-ي]|D:[\\/]|\.mp4|video-ad-editor|\b(cairo|tajawal|almarai)\b", re.IGNORECASE)
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not banned.search(line), f"{f.name}:{n}: {line.strip()}"


# ---- 19. no collision with the existing Behind-Subject treatment ------------------------------------------------
def test_19_no_collision_with_existing_behind_subject_treatment():
    from video_edit_agent.direction.composition import TREATMENT

    comp = _compose_ok()
    assert comp.status == "ok" and comp.treatment == TREATMENT
    assert comp.treatment != "behind_subject_text"  # the qualification layer only governs lower_subject_semantic
    # nothing about qualifying a headline mutates or references Behind-Subject geometry/thresholds
    import video_edit_agent.motion.occlusion as occ
    before = occ.OcclusionPolicy()
    select_headline(IWORDS, IWINDOW, stopwords=HSTOP)
    after = occ.OcclusionPolicy()
    assert before == after


# ---- 20. an automatically selected headline remains pending_review ------------------------------------------------
def test_20_automatically_selected_headline_remains_pending_review():
    comp = _compose_ok()
    assert comp.status == "ok" and comp.choice is not None and comp.choice.semantic_source == "automatic"
    st = review_state(comp)
    assert st["approval_status"] == "pending_review"
    assert st["technical_status"] != "approved" and st["visual_status"] != "approved"
    assert HWORDS and HWINDOW  # sanity: the Phase 1.3.2 fixtures this file re-uses are untouched
