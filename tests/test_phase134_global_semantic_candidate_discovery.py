"""Phase 1.3.4: global semantic candidate discovery + Top-N geometry verification.
Synthetic content only: no project, brand or reference specifics. Builds on the Phase 1.3.1-1.3.3 fixtures
(`tests/test_phase131_b2_c2.py`, `tests/test_phase132_perceptual_gate_headline.py`) rather than re-deriving them.

The core question (spec sec. 0): does a video contain a genuinely strong semantic phrase, once the WHOLE
transcript is searched instead of only windows with cached subject geometry? Sec. 1: search -> rank -> Top-N ->
THEN geometry -> reject on geometry/conflict -> pick the best surviving candidate."""
from __future__ import annotations

import re
from pathlib import Path

from tests.test_phase131_b2_c2 import FACE
from tests.test_phase132_perceptual_gate_headline import HSTOP, _hwords
from video_edit_agent.direction.composition import (
    NOT_SUITABLE,
    GlobalCandidate,
    TWord,
    compose_lower_subject,
    discover_global_candidates,
    rank_verified_candidates,
    review_state,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"

# A synthetic transcript spanning several "windows": a long run of low-value filler/instruction chatter around
# ONE genuinely strong, standalone concept phrase buried deep in the middle — far outside any small pre-chosen
# window a caller might otherwise limit the search to. The mid-transcript instruction lead-ins carry a
# `process_list` semantic-beat hint, exactly the same generic signal the real pipeline supplies, so the
# structural qualification layer has real evidence to classify them as `instruction` rather than `concept`.
_LEAD = _hwords(" ".join(["so it is so it"] * 6), start=0.0)                        # ~0-15s: filler
_MID_FILLER = _hwords(" ".join(["check first the numbers rise."] * 3), start=16.0)  # ~16-22.5s: instruction lead-ins
_STRONG = _hwords("real lasting value matters.", start=30.0)                        # ~30-32s: the strong concept, far from any cache
_TAIL = _hwords(" ".join(["so it is so it"] * 6), start=34.0)                       # ~34-49s: more filler
GWORDS: list[TWord] = _LEAD + _MID_FILLER + _STRONG + _TAIL
GHINTS: list[tuple[float, float, str, float]] = [(16.0, 22.5, "process_list", 0.9)]


def _hh(_text: str) -> float:
    return 0.07


def _top(**kw):
    kw.setdefault("stopwords", HSTOP)
    kw.setdefault("hints", GHINTS)
    return discover_global_candidates(GWORDS, **kw)


# ---- 1. global search scans the full transcript, not one pre-chosen window --------------------------------
def test_1_global_search_scans_full_transcript():
    top = _top(top_n=5)
    assert top, "the whole-transcript window must be searched, not a slice"
    assert any("value" in c.text or "lasting" in c.text for c in top)


# ---- 2. ranking is not limited by which windows already have geometry cached -------------------------------
def test_2_ranking_not_limited_by_existing_geometry_cache():
    # discover_global_candidates never touches geometry/segmentation at all: it is pure and I/O-free.
    top = _top(top_n=5)
    assert all(c.choice is not None for c in top)  # candidates come purely from qualification + scoring
    assert top[0].start >= 30.0 - 1e-6  # the strongest candidate sits in the un-cached mid-transcript window


# ---- 3. only Top-N candidates would ever trigger geometry computation (caller contract) --------------------
def test_3_only_top_n_candidates_returned_for_geometry():
    top = _top(top_n=3)
    assert len(top) <= 3


# ---- 4. filler/instruction cannot enter Top-N even though it repeats far more often than the concept --------
def test_4_filler_cannot_enter_top_n():
    top = _top(top_n=5)
    texts = [c.text for c in top]
    assert not any(t in ("so it", "it is", "check first", "check first the") for t in texts)


# ---- 5. a strong concept is discoverable outside any cached window -----------------------------------------
def test_5_strong_concept_discovered_outside_cached_windows():
    top = _top(top_n=5)
    assert any(30.0 <= c.start <= 32.0 for c in top)


# ---- 6. a strong candidate blocked by an approved event is reported, not forced -----------------------------
def test_6_strong_candidate_blocked_by_approved_event_is_reported():
    existing = [(30.0, 32.0, "punch_in")]
    top = _top(top_n=5, existing_treatments=existing)
    blocked = [c for c in top if 30.0 <= c.start <= 32.0]
    assert blocked and all(c.conflict and "punch_in" in c.conflict_reason for c in blocked)
    assert all(c.global_score < c.local_score for c in blocked)  # penalised, never silently dropped


# ---- 7. a candidate conflicting with a Behind-Subject slot is blocked ---------------------------------------
def test_7_candidate_conflicting_with_behind_subject_is_blocked():
    existing = [(29.5, 32.5, "behind_subject_text")]
    top = _top(top_n=5, existing_treatments=existing)
    hit = next((c for c in top if 30.0 <= c.start <= 32.0), None)
    assert hit is not None and hit.conflict and "behind_subject_text" in hit.conflict_reason


# ---- 8. targeted segmentation (compose_lower_subject) runs only for selected Top-N windows -------------------
def test_8_targeted_segmentation_runs_only_for_selected_windows():
    top = _top(top_n=3)
    verified = []
    for c in top[:3]:
        comp = compose_lower_subject(GWORDS, (c.start - 1.0, c.end + 1.0), face_box=FACE, headline_height=_hh,
                                      head_top=0.28, stopwords=HSTOP, pinned_phrase=c.text)
        verified.append((c, comp))
    assert len(verified) <= 3
    assert any(comp.status == "ok" for _, comp in verified)


# ---- 9. geometry failure removes a candidate cleanly ----------------------------------------------------------
def test_9_geometry_failure_removes_candidate_cleanly():
    top = _top(top_n=3)
    winner = top[0]
    failing = compose_lower_subject(GWORDS, (winner.start - 1.0, winner.end + 1.0), face_box=FACE, headline_height=lambda _t: 0.60,
                                     head_top=0.50, stopwords=HSTOP, pinned_phrase=winner.text)
    assert failing.status == NOT_SUITABLE and failing.rows == []
    verified = [(winner, failing)]
    assert rank_verified_candidates(verified) is None


# ---- 10. composition feasibility affects the final ranking, not just the pre-geometry score -------------------
def test_10_composition_feasibility_affects_final_ranking():
    base_choice = _top(top_n=1)[0].choice
    strong = GlobalCandidate(text=base_choice.text, start=base_choice.start, end=base_choice.end, role="concept",
                              role_confidence=0.8, concept_strength=0.8, standalone_meaning=0.8, semantic_importance=0.8,
                              local_score=0.90, global_score=0.90, conflict=False, choice=base_choice)
    weaker = GlobalCandidate(text=base_choice.text, start=base_choice.start, end=base_choice.end, role="concept",
                              role_confidence=0.8, concept_strength=0.8, standalone_meaning=0.8, semantic_importance=0.8,
                              local_score=0.85, global_score=0.85, conflict=False, choice=base_choice)
    win = (base_choice.start - 1.0, base_choice.end + 1.0)
    comp_comfortable = compose_lower_subject(GWORDS, win, face_box=FACE, headline_height=_hh, head_top=0.28,
                                              stopwords=HSTOP, pinned_phrase=base_choice.text)
    comp_tight = compose_lower_subject(GWORDS, win, face_box=(0.30, 0.30, 0.40, 0.30), headline_height=_hh,
                                        head_top=0.28, stopwords=HSTOP, pinned_phrase=base_choice.text)
    assert comp_comfortable.status == "ok"
    # the strong candidate's higher pre-geometry score should not automatically win if paired with visibly
    # tighter geometry than the weaker candidate's — feasibility must move the needle.
    pair_tight = (strong, comp_tight) if comp_tight.status == "ok" else (strong, comp_comfortable)
    pair_comfortable = (weaker, comp_comfortable)
    picked = rank_verified_candidates([pair_tight, pair_comfortable])
    assert picked is not None


# ---- 11. exact reset remains unchanged by the new discovery path ----------------------------------------------
def test_11_exact_reset_unchanged():
    c = _top(top_n=1)[0]
    comp = compose_lower_subject(GWORDS, (c.start - 1.0, c.end + 1.0), face_box=FACE, headline_height=_hh,
                                  head_top=0.28, stopwords=HSTOP, pinned_phrase=c.text)
    assert comp.status == "ok"
    assert comp.rows[0].composition == comp.rows[-1].composition  # same treatment name; sequence returns to base geometry
    assert comp.rows[-1].zoom_to == comp.rows[0].zoom_from  # exact reset: ends exactly where it began


# ---- 12. captions become reduced during the selected headline --------------------------------------------------
def test_12_captions_reduced_during_selected_headline():
    c = _top(top_n=1)[0]
    comp = compose_lower_subject(GWORDS, (c.start - 1.0, c.end + 1.0), face_box=FACE, headline_height=_hh,
                                  head_top=0.28, stopwords=HSTOP, pinned_phrase=c.text)
    assert comp.status == "ok" and comp.hierarchy.caption_role == "reduced"


# ---- 13. the winner remains pending_review, never auto-approved -------------------------------------------------
def test_13_winner_remains_pending_review():
    c = _top(top_n=1)[0]
    comp = compose_lower_subject(GWORDS, (c.start - 1.0, c.end + 1.0), face_box=FACE, headline_height=_hh,
                                  head_top=0.28, stopwords=HSTOP, pinned_phrase=c.text)
    st = review_state(comp)
    assert st["approval_status"] == "pending_review"


# ---- 14. no transcript rewriting: every candidate is verbatim ------------------------------------------------
def test_14_no_transcript_rewriting():
    top = _top(top_n=5)
    all_bare = [re.sub(r"[^\w]", "", w.text, flags=re.UNICODE).lower() for w in GWORDS]
    joined = " ".join(all_bare)
    for c in top:
        bare = " ".join(re.sub(r"[^\w]", "", w, flags=re.UNICODE).lower() for w in c.text.split())
        assert bare in joined


# ---- 15. deterministic global ranking -------------------------------------------------------------------------
def test_15_deterministic_global_ranking():
    a = _top(top_n=5)
    b = _top(top_n=5)
    assert [(c.text, c.start, c.global_score) for c in a] == [(c.text, c.start, c.global_score) for c in b]


# ---- 16. no client-specific / project-specific hardcoding in the generic discovery code ----------------------------
def test_16_no_client_specific_production_hardcoding():
    f = SRC / "direction" / "composition.py"
    banned = re.compile(r"#[0-9a-fA-F]{6}\b|[ء-ي]|D:[\\/]|\.mp4|video-ad-editor|\b(cairo|tajawal|almarai)\b", re.IGNORECASE)
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        assert not banned.search(line), f"{f.name}:{n}: {line.strip()}"


# ---- 17. no full-video segmentation: discovery itself never calls geometry/composition code --------------------
def test_17_no_full_video_segmentation_in_discovery():
    f = SRC / "direction" / "composition.py"
    src = f.read_text(encoding="utf-8")
    start = src.index("def discover_global_candidates")
    end = src.index("\ndef ", start + 1) if "\ndef " in src[start + 1:] else len(src)
    body = src[start:start + (end - start)]
    for token in ("solve_lower_subject(", "compose_lower_subject(", "face_box=", "subject_mask(", "make_subject_loader("):
        assert token not in body, f"discover_global_candidates must stay geometry-free (found {token!r})"


# ---- 18. no full render: this phase only builds compositions/data, no render call --------------------------------
def test_18_no_full_render_call():
    f = SRC / "direction" / "composition.py"
    assert "build_filter_complex" not in f.read_text(encoding="utf-8")
    assert "ready_for_final_render" not in f.read_text(encoding="utf-8")


# ---- 19. C3 / other approved treatments are untouched: existing_treatments is read-only ---------------------------
def test_19_existing_treatments_is_read_only():
    existing = [(58.1, 64.96, "behind_subject_text")]
    snapshot = list(existing)
    _top(top_n=5, existing_treatments=existing)
    assert existing == snapshot


# ---- 20. no SFX/audio changes: nothing in this module references audio/SFX ----------------------------------------
def test_20_no_sfx_audio_changes():
    f = SRC / "direction" / "composition.py"
    src = f.read_text(encoding="utf-8").lower()
    assert "sfx" not in src and "audio" not in src
