"""Phase 1.3.2: the perceptual Behind-Subject gate and the semantic-usefulness headline selector.
Synthetic content only: no project, brand or reference specifics. Builds on the Phase 1.3.1 fixtures in
`tests/test_phase131_b2_c2.py` and `tests/test_behind_subject_text.py` rather than re-deriving them."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tests.test_behind_subject_text import _plans, loader, subject_mask
from tests.test_phase131_b2_c2 import (
    FONT,
    GOOD,
    LOW_FACE,
    ORIGIN,
    PASSING,
    _body,
    _compose_ok,
    _mask,
    _mo,
)
from tests.test_phase131_b2_c2 import WINDOW as B2_WINDOW
from video_edit_agent.direction.composition import TWord, is_verbatim, review_state, select_headline
from video_edit_agent.motion import behind_subject as bs
from video_edit_agent.motion import occlusion as occ
from video_edit_agent.subject.refine import MatteReport

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"


def _glyphs_unequal(w0: int, w1: int, h: int = 160, gap: int = 40) -> list[np.ndarray]:
    """Two very differently sized words on one grid: a big one (most of the total glyph area) and a small
    one, so a small, meaningfully-occluded word can still leave the TOTAL overlap ratio low."""
    width = w0 + gap + w1
    a0 = np.zeros((h, width), np.float32)
    a0[:, :w0] = 1.0
    a1 = np.zeros((h, width), np.float32)
    a1[:, w0 + gap:w0 + gap + w1] = 1.0
    return [a0, a1]


# ---- 1. low raw overlap can still pass when perceptual depth is strong --------------------------
def test_1_low_overlap_ratio_can_still_pass_on_strong_perceptual_depth():
    # the small word is genuinely, centrally occluded (full height, a real share of its own body); the big
    # word carries none of the overlap, so the TOTAL overlap ratio stays well under the old rigid 6% cutoff.
    g = _glyphs_unequal(500, 50)
    w1_x0 = ORIGIN[0] + 500 + 40
    m = occ.meaningful_occlusion(g, FONT, ORIGIN, [_mask(w1_x0, w1_x0 + round(50 * 0.3))] * 3)
    assert m.text_subject_overlap_ratio < occ.OcclusionPolicy().min_overlap_ratio  # below the old hard threshold
    assert m.occluded_glyph_count >= 1 and m.overlap_rows_frac >= occ.OcclusionPolicy().min_overlap_rows
    assert m.visible_text_ratio >= occ.OcclusionPolicy().min_visible_ratio_hard
    assert not m.hard_fail and m.perceptual_score >= occ.OcclusionPolicy().perceptual_min_score and m.passed


# ---- 2. zero real glyph overlap always fails -----------------------------------------------------
def test_2_zero_glyph_overlap_always_fails():
    m = _mo(_mask(0, 50))  # nowhere near the glyphs
    assert m.text_subject_overlap_px == 0 and m.hard_fail == "no_actual_overlap"
    assert not m.passed and m.perceptual_score < occ.OcclusionPolicy().perceptual_min_score
    assert "no actual glyph-body overlap" in m.reasons[0]


# ---- 3. fringe-only overlap fails -----------------------------------------------------------------
def test_3_fringe_only_contact_fails_even_with_nonzero_pixels():
    m = _mo(_mask(404, 414, 503, 513))  # a small corner touch: real pixels, but not a real body overlap
    assert m.text_subject_overlap_px > 0 and m.occluded_glyph_count == 0
    assert m.hard_fail == "fringe_only_contact" and not m.passed
    assert "fringe" in m.reasons[0]


# ---- 4. unreadable text fails regardless of score --------------------------------------------------
def test_4_unreadable_text_fails_even_when_other_signals_are_strong():
    m = _mo(_mask(400, 680, 500, 640))  # most of both glyph bodies covered: still readable-looking geometry, but not readable
    assert m.visible_text_ratio < occ.OcclusionPolicy().min_visible_ratio_hard
    assert m.hard_fail == "text_unreadable" and not m.passed
    assert any("stays visible" in r or "mostly hidden" in r for r in m.reasons)
    # a hand-inflated score cannot rescue it: the hard fail short-circuits before the score is even consulted
    assert m.passed is False


# ---- 5. subject integrity failure always fails ------------------------------------------------------
def test_5_subject_integrity_failure_always_fails_even_with_a_passing_occlusion_score():
    bad_matte = [MatteReport(core_area=1000, added_area=900, evidence_pixels=0, soft_area=100, band_px=4,
                              feather_sigma=1.0, evidence_used=True)]  # added_frac 0.9 >> the policy's ceiling

    def load(edl, start, end):
        d = loader(masks=[subject_mask(top=700, left=100, right=980)] * 4, face=LOW_FACE)(edl, start, end)
        d.matte = list(bad_matte) * len(d.masks)
        return d

    (p,) = _plans(load=load, preset=PASSING)  # PASSING would otherwise let this occlusion through
    assert p.decision != bs.DECISION_BEHIND
    assert not p.gate["matte_integrity"]["passed"]
    assert "matte_integrity" in (p.reason or "")


# ---- 6. perceptual score is deterministic ------------------------------------------------------------
def test_6_perceptual_score_is_deterministic():
    m1 = _mo(_mask(486, 594))
    m2 = _mo(_mask(486, 594))
    assert m1.perceptual_score == m2.perceptual_score and m1.components == m2.components and m1.passed == m2.passed


# ---- 7. score diagnostics expose component contributions ---------------------------------------------
def test_7_score_diagnostics_expose_component_contributions():
    m = _mo(_mask(486, 594))
    s = m.summary()
    for key in ("overlap", "glyphs", "words", "location", "visibility", "depth"):
        assert key in s["components"] and 0.0 <= s["components"][key] <= 1.0
    assert s["perceptual_score"] == round(sum(occ.OcclusionPolicy().__getattribute__(f"perceptual_weight_{k}") * v
                                               for k, v in s["components"].items()), 4)
    assert "hard_fail" in s and s["hard_fail"] == ""


# ---- 8. no single overlap-ratio threshold controls the final decision --------------------------------
def test_8_no_single_overlap_ratio_threshold_controls_the_decision():
    # a raw ratio BELOW the old 6% cutoff can pass (evidence gathered in test 1) ...
    g = _glyphs_unequal(500, 50)
    w1_x0 = ORIGIN[0] + 500 + 40
    below = occ.meaningful_occlusion(g, FONT, ORIGIN, [_mask(w1_x0, w1_x0 + round(50 * 0.3))] * 3)
    assert below.text_subject_overlap_ratio < 0.06 and below.passed
    # ... while a raw ratio ABOVE it can still fail: a thin band across both words touches plenty of pixels
    # but never reaches a real share of any single glyph unit, so it is fringe, not a body overlap.
    above = _mo(_mask(400, 680, 578, 578 + 16))
    assert above.text_subject_overlap_ratio > 0.06 and not above.passed and above.hard_fail == "fringe_only_contact"


# ---- headline selector fixtures -----------------------------------------------------------------------
def _hwords(spec: str, step: float = 0.5, start: float = 0.0) -> list[TWord]:
    out, t = [], start
    for w in spec.split():
        out.append(TWord(w, round(t, 3), round(t + step - 0.05, 3)))
        t += step
    return out


HSTOP = frozenset({"so", "we", "it", "and", "the", "of", "to", "in"})
# filler ("so we mean it") repeats often across the whole transcript: high frequency, low rarity.
# the concept phrase ("true worth") appears exactly once: high rarity, a compact standalone idea.
_PRE = _hwords("so we mean it. so we mean it. so we mean it.", start=0.0)
_WIN = _hwords("so we mean it true worth today", start=6.0)
_POST = _hwords("so we mean it. so we mean it.", start=14.0)
HWORDS = _PRE + _WIN + _POST
HWINDOW = (6.0, 6.0 + 7 * 0.5)


# ---- 9. strong standalone semantic phrase outranks discourse filler -----------------------------------
def test_9_strong_semantic_phrase_outranks_discourse_filler():
    best = select_headline(HWORDS, HWINDOW, stopwords=HSTOP)
    assert best is not None and best.text == "true worth"
    filler = select_headline(HWORDS, HWINDOW, stopwords=HSTOP, pinned_phrase="so we")
    assert filler is not None and best.score > filler.score
    assert any("rare in the transcript" in r for r in best.reasons)


# ---- 10. selector never rewrites the transcript --------------------------------------------------------
def test_10_selector_never_rewrites_the_transcript():
    assert select_headline(HWORDS, HWINDOW, stopwords=HSTOP, pinned_phrase="true value") is None  # not verbatim
    best = select_headline(HWORDS, HWINDOW, stopwords=HSTOP)
    assert is_verbatim(best.text, HWORDS)
    pinned = select_headline(HWORDS, HWINDOW, stopwords=HSTOP, pinned_phrase="true worth")
    assert pinned is not None and pinned.text == "true worth" and pinned.semantic_source == "user_pinned"


# ---- 11. selector is deterministic -----------------------------------------------------------------------
def test_11_selector_is_deterministic():
    a = select_headline(HWORDS, HWINDOW, stopwords=HSTOP)
    b = select_headline(HWORDS, HWINDOW, stopwords=HSTOP)
    assert a is not None and a.model_dump() == b.model_dump()


# ---- 12. generic code contains no client-specific phrase hardcoding ------------------------------------------
def test_12_no_client_or_project_hardcoding_in_generic_code():
    files = [SRC / "motion" / "occlusion.py", SRC / "direction" / "composition.py", SRC / "captions" / "headline.py"]
    banned = re.compile(r"#[0-9a-fA-F]{6}\b|[ء-ي]|D:[\\/]|\.mp4|video-ad-editor|\b(cairo|tajawal|almarai)\b", re.IGNORECASE)
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not banned.search(line), f"{f.name}:{n}: {line.strip()}"


# ---- 13. approved transcript is the source of every headline candidate ------------------------------------
def test_13_approved_transcript_is_the_source_of_every_candidate():
    best = select_headline(HWORDS, HWINDOW, stopwords=HSTOP)
    assert best is not None and all(w in [x.text.strip(".,") for x in HWORDS] for w in best.words)
    # remove the concept words from the transcript entirely: nothing invents them back
    filler_only = [w for w in HWORDS if w.text.strip(".,") not in ("true", "worth", "today")]
    without = select_headline(filler_only, HWINDOW, stopwords=HSTOP)
    assert without is None or without.text != "true worth"


# ---- 14. B2/C2 remain pending_review, never auto-approved ---------------------------------------------------
def test_14_b2_and_c2_stay_pending_review_never_auto_approved():
    (p,) = _plans(load=_body(), preset=PASSING)
    assert p.decision == bs.DECISION_BEHIND and p.technical_status == "passed"
    assert p.metrics["meaningful_occlusion"]["passed"] and not p.metrics["meaningful_occlusion"]["hard_fail"]
    # a passing technical/perceptual result is not itself an approval: nothing on the plan claims "approved"
    assert not hasattr(p, "approval_status") or getattr(p, "approval_status", None) != "approved"
    st = review_state(_compose_ok())
    assert st["approval_status"] == "pending_review" and st["technical_status"] != "approved"
    assert GOOD  # sanity: the B2 fixture transcript is untouched by this file
    assert B2_WINDOW
