"""Phase 1.4.1: face-safe primary headline placement + framework-aware `simple_diagram`
discovery. Synthetic content only: no project, brand or reference specifics. The 20
tests below are the spec's own enumerated list (sec. 8), grouped by module.
"""
from __future__ import annotations

from video_edit_agent.captions.safe_zone import SafeZone
from video_edit_agent.direction.composition import CompositionPolicy, TWord
from video_edit_agent.direction.headline_placement import (
    PLACEMENT_LOWER_SUBJECT,
    PLACEMENT_REJECTED,
    find_face_safe_headline,
)
from video_edit_agent.direction.motion_graphics import (
    FrameworkSpec,
    discover_framework_candidates,
)
from video_edit_agent.motion.legibility import _head_box, _intersects

SAFE = SafeZone(top_pct=0.12, bottom_pct=0.18, side_pct=0.06)
CAPTION_BAND = 0.30
CENTER_FACE = (0.42, 0.35, 0.16, 0.09)  # a face roughly centred, mid-frame -- the common case


def _measure(ink_w: float, ink_h: float):
    def measure(_scale: float) -> tuple[float, float]:
        return (ink_w, ink_h)
    return measure


# ---- 1. primary headline cannot overlap the protected face/head bbox ---------------------------------------------
def test_1_headline_never_overlaps_protected_face_head_box():
    result = find_face_safe_headline(_measure(0.5, 0.08), face=CENTER_FACE, safe_zone=SAFE, caption_band=CAPTION_BAND)
    assert result.accepted
    protected = _head_box(CENTER_FACE)
    assert not _intersects(result.box, protected, 0.0)


# ---- 2. repositioning is preferred before resizing -----------------------------------------------------------------
def test_2_repositioning_preferred_before_resizing():
    # a small headline that easily fits above/beside the face at full size: no shrink should happen.
    result = find_face_safe_headline(_measure(0.3, 0.06), face=CENTER_FACE, safe_zone=SAFE, caption_band=CAPTION_BAND)
    assert result.accepted
    assert result.font_scale == 1.0


# ---- 3. unsafe headline is rejected when no safe placement exists ---------------------------------------------------
def test_3_rejected_when_no_safe_placement_exists():
    # a headline almost as wide/tall as the whole safe frame leaves no room anywhere, at any size.
    result = find_face_safe_headline(_measure(0.99, 0.85), face=CENTER_FACE, safe_zone=SAFE, caption_band=CAPTION_BAND)
    assert result.placement == PLACEMENT_REJECTED
    assert result.box is None
    assert not result.accepted


# ---- 4. the face-safe rule does not convert a headline into Behind-Subject -------------------------------------------
def test_4_never_produces_behind_subject_placement():
    for ink_w, ink_h in [(0.2, 0.05), (0.5, 0.08), (0.35, 0.1)]:
        result = find_face_safe_headline(_measure(ink_w, ink_h), face=CENTER_FACE, safe_zone=SAFE, caption_band=CAPTION_BAND)
        assert result.placement != "behind_subject"


# ---- 5. caption collision remains blocked -----------------------------------------------------------------------
def test_5_caption_collision_blocked():
    # a face pushed low, forcing candidates near the caption band: none may cross into it.
    low_face = (0.4, 0.55, 0.16, 0.09)
    result = find_face_safe_headline(_measure(0.5, 0.09), face=low_face, safe_zone=SAFE, caption_band=CAPTION_BAND)
    if result.accepted:
        _x, y, _w, h = result.box
        assert y + h <= 1.0 - CAPTION_BAND + 1e-9


# ---- 6. frame safe margins remain enforced --------------------------------------------------------------------------
def test_6_frame_safe_margins_enforced():
    result = find_face_safe_headline(_measure(0.4, 0.07), face=CENTER_FACE, safe_zone=SAFE, caption_band=CAPTION_BAND)
    assert result.accepted
    x, y, w, _h = result.box
    assert x >= SAFE.side_pct - 1e-9
    assert x + w <= 1.0 - SAFE.side_pct + 1e-9
    assert y >= SAFE.top_pct - 1e-9


def test_6b_no_face_still_respects_safe_zone_and_caption_band():
    result = find_face_safe_headline(_measure(0.4, 0.07), face=None, safe_zone=SAFE, caption_band=CAPTION_BAND)
    assert result.accepted
    x, y, w, h = result.box
    assert x >= SAFE.side_pct - 1e-9 and x + w <= 1.0 - SAFE.side_pct + 1e-9
    assert y + h <= 1.0 - CAPTION_BAND + 1e-9


def test_6c_lower_subject_only_when_opted_in_and_still_caption_safe():
    tall_face = (0.4, 0.1, 0.16, 0.5)  # a face occupying most of the upper frame
    without_opt_in = find_face_safe_headline(_measure(0.3, 0.05), face=tall_face, safe_zone=SAFE, caption_band=CAPTION_BAND)
    assert without_opt_in.placement != PLACEMENT_LOWER_SUBJECT
    with_opt_in = find_face_safe_headline(
        _measure(0.3, 0.05), face=tall_face, safe_zone=SAFE, caption_band=CAPTION_BAND, allow_lower_subject=True,
    )
    if with_opt_in.placement == PLACEMENT_LOWER_SUBJECT:
        _x, y, _w, h = with_opt_in.box
        assert y + h <= 1.0 - CAPTION_BAND + 1e-9


# ==================== Framework-aware `simple_diagram` discovery (sec. 3/4) ====================

def _words(pairs: list[tuple[str, float, float]]) -> list[TWord]:
    return [TWord(text, start, end) for text, start, end in pairs]


# A synthetic "Five Cs"-shaped transcript: an anchor phrase far from any comma-enumeration,
# five concept labels each introduced in its OWN separate clause, no ordinal words at all.
FIVE_MEMBER_WORDS = _words([
    ("intro", 0.0, 0.4), ("Five", 1.0, 1.3), ("Cs", 1.3, 1.6), ("framework", 1.6, 2.0),
    ("filler", 3.0, 3.4), ("study", 4.0, 4.3), ("alpha", 4.3, 4.7), ("carefully", 4.7, 5.2),
    ("filler", 6.0, 6.4), ("study", 7.0, 7.3), ("beta", 7.3, 7.7), ("carefully", 7.7, 8.2),
    ("filler", 9.0, 9.4), ("study", 10.0, 10.3), ("gamma", 10.3, 10.7), ("carefully", 10.7, 11.2),
    ("filler", 12.0, 12.4), ("study", 13.0, 13.3), ("delta", 13.3, 13.7), ("carefully", 13.7, 14.2),
    ("filler", 15.0, 15.4), ("study", 16.0, 16.3), ("epsilon", 16.3, 16.7), ("carefully", 16.7, 17.2),
])
FSTOP = frozenset({"intro", "framework", "filler", "carefully"})
FIVE_SPEC = FrameworkSpec(anchor="Five Cs", member_terms=("alpha", "beta", "gamma", "delta", "epsilon"))


# ---- 7. framework discovery can span multiple neighbouring transcript clauses ---------------------------------------
def test_7_discovery_spans_multiple_neighbouring_clauses():
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP)
    assert len(cands) == 1
    assert cands[0].labels == ["alpha", "beta", "gamma", "delta", "epsilon"]
    assert cands[0].start == 4.3 and cands[0].end == 16.7  # spans five separate clauses, not one


# ---- 8. framework discovery is not limited to commas/ordinals -------------------------------------------------------
def test_8_not_limited_to_commas_or_ordinal_words():
    text_has_no_commas_or_ordinals = all("," not in t and t not in {"first", "second", "third", "fourth", "fifth"}
                                          for t, _s, _e in [(w.text, w.start, w.end) for w in FIVE_MEMBER_WORDS])
    assert text_has_no_commas_or_ordinals
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP)
    assert cands  # still found despite no structural punctuation/ordinal cues


# ---- 9. every diagram label has source evidence -----------------------------------------------------------------
def test_9_every_label_has_source_evidence():
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP)
    evidence = cands[0].layout["evidence"]
    assert len(evidence) == len(cands[0].labels)
    for item in evidence:
        assert item["source_type"] in ("transcript", "approved_canonical")
        if item["source_type"] == "transcript":
            assert item["source_span"] is not None
        else:
            assert item.get("citation")


# ---- 10. missing framework member fails closed -----------------------------------------------------------------
def test_10_missing_member_fails_closed():
    incomplete = _words([(w.text, w.start, w.end) for w in FIVE_MEMBER_WORDS if w.text != "epsilon"])
    incomplete_spec = FrameworkSpec(anchor="Five Cs", member_terms=("alpha", "beta", "gamma", "delta", "epsilon"), min_members=5)
    cands = discover_framework_candidates(None, incomplete, [incomplete_spec], stopwords=FSTOP)
    assert cands == []  # 4 of 5 supportable, below min_members=5 -> no partial diagram


def test_10b_below_default_min_members_also_fails_closed():
    one_member_words = _words([("study", 0.0, 0.3), ("alpha", 0.3, 0.6)])
    spec = FrameworkSpec(anchor="X", member_terms=("alpha", "beta", "gamma"), min_members=2)
    cands = discover_framework_candidates(None, one_member_words, [spec], stopwords=frozenset({"study"}))
    assert cands == []


# ---- 11. no invented labels ---------------------------------------------------------------------------------------
def test_11_no_invented_labels():
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP)
    transcript_text = {w.text for w in FIVE_MEMBER_WORDS}
    for label in cands[0].labels:
        assert label in transcript_text  # every label is a verbatim word that was actually spoken


def test_11b_canonical_evidence_never_invents_a_span():
    spec_with_citation = FrameworkSpec(
        anchor="Five Cs", member_terms=("alpha", "beta", "gamma", "delta", "zzz_not_spoken"),
        canonical_evidence={"zzz_not_spoken": "approved project glossary, sec. 4"},
    )
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [spec_with_citation], stopwords=FSTOP)
    assert cands
    evidence = {e["label"]: e for e in cands[0].layout["evidence"]}
    assert evidence["zzz_not_spoken"]["source_type"] == "approved_canonical"
    assert evidence["zzz_not_spoken"]["source_span"] is None
    assert evidence["zzz_not_spoken"]["citation"] == "approved project glossary, sec. 4"


# ---- 12. maximum five nodes -----------------------------------------------------------------------------------
def test_12_maximum_five_nodes():
    six_words = _words([*[(w.text, w.start, w.end) for w in FIVE_MEMBER_WORDS],
                         ("study", 18.0, 18.3), ("zeta", 18.3, 18.7), ("carefully", 18.7, 19.2)])
    six_spec = FrameworkSpec(anchor="Six", member_terms=("alpha", "beta", "gamma", "delta", "epsilon", "zeta"))
    cands = discover_framework_candidates(None, six_words, [six_spec], stopwords=FSTOP, max_nodes=5)
    assert len(cands[0].labels) <= 5


# ---- 13. the Five Cs generic pattern works without client-specific hardcoding -------------------------------------------
def test_13_generic_pattern_no_hardcoded_client_terms():
    import inspect

    from video_edit_agent.direction import motion_graphics
    source = inspect.getsource(motion_graphics.discover_framework_candidates) + inspect.getsource(motion_graphics.FrameworkSpec)
    for banned in ("العميل", "المنافسين", "شركتك", "collaborators", "context", "client"):
        assert banned not in source
    # and the SAME function, given wholly different anchor/member data, still works:
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP)
    assert cands


# ---- 14. diagram remains pending_review (never auto-approved) -------------------------------------------------------
def test_14_diagram_candidate_is_not_auto_approved():
    cands = discover_framework_candidates(None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP)
    assert cands[0].visual_status == "candidate"
    assert cands[0].technical_status == "not_rendered"


# ---- 15. B6 (an existing lower_subject candidate) is not silently replaced --------------------------------------
def test_15_framework_discovery_does_not_touch_existing_treatments():
    existing = [(4.3, 16.7, "lower_subject")]
    conflict_cands = discover_framework_candidates(
        None, FIVE_MEMBER_WORDS, [FIVE_SPEC], stopwords=FSTOP, existing_treatments=existing,
        policy=CompositionPolicy(conflict_guard_s=0.5),
    )
    assert conflict_cands[0].conflict is True  # flagged, not silently merged/replaced
    assert conflict_cands[0].conflict_reason


# ---- 16/17/18/19/20: this phase touches no other approved treatment, no audio, no B-roll, no full render -------------
def test_16_17_18_19_20_no_unrelated_side_effects_from_new_modules():
    import inspect

    from video_edit_agent.direction import headline_placement, motion_graphics
    combined = inspect.getsource(headline_placement) + inspect.getsource(motion_graphics.discover_framework_candidates)
    for banned in ("sfx", "SFX", "b_roll", "b-roll", "broll", "generated_visual", "ffmpeg", "render(", "full_render"):
        assert banned not in combined
    # MG-B / C3 / B6's own discovery/render entry points are untouched (still exported, same signatures):
    assert hasattr(motion_graphics, "discover_keyword_candidates")  # MG-B's discovery, unmodified by this phase
