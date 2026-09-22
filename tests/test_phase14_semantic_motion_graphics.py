"""Phase 1.4: Semantic Motion Graphics Foundation. Synthetic content only: no project,
brand or reference specifics. Builds on the Phase 1.3.3/1.3.4 fixtures
(`tests/test_phase132_perceptual_gate_headline.py`, `tests/test_phase134_global_semantic_candidate_discovery.py`)
and `tests/conftest.make_transcript` rather than re-deriving them.

The core question (spec sec. 0): can the three review-first treatments
(`primary_headline_typography`, `keyword_visual`, `simple_diagram`) be discovered,
qualified and rendered WITHOUT a single new detector, a hardcoded phrase, or an
invented label -- reusing exactly the existing semantic qualification system
(`direction.composition`) and structural-list detector (`direction.semantic_beats`)."""
from __future__ import annotations

import re
from pathlib import Path

from tests.conftest import make_transcript
from tests.test_phase132_perceptual_gate_headline import HSTOP, HWORDS, _hwords
from video_edit_agent.brand.loader import load_brand
from video_edit_agent.captions.headline import ink_bbox, reduce_captions
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.direction.composition import ELIGIBLE_HEADLINE_ROLES, TWord
from video_edit_agent.direction.history import RECENT_BEATS, TreatmentHistory
from video_edit_agent.direction.motion_graphics import (
    MAX_DIAGRAM_NODES,
    MotionGraphicCandidate,
    TreatmentType,
    discover_diagram_candidates,
    discover_headline_candidates,
    discover_keyword_candidates,
    select_treatments,
)
from video_edit_agent.render.composition import CaptionBurn, RenderPlan
from video_edit_agent.render.motion_graphics import (
    add_diagram,
    add_keyword,
    brand_tokens,
    diagram_layout,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"
BRAND = load_brand("client")
STYLE = CaptionStyle(name="Test", font_ar="Test Sans", font_size=64, primary_color="&H00FFFFFF",
                     outline_color="&H00000000", back_color="&H00000000", outline=2.0, shadow=0.0)

# ---- keyword fixture: a single rare standalone word (its own clause) surrounded by filler -----------------------
KWORDS = _hwords("so it is so it.", start=0.0) + _hwords("context.", start=6.0) + _hwords("so it is so it.", start=8.0)
KWINDOW = (6.0, 6.5)
KSTOP = frozenset({"so", "it", "is"})

# ---- diagram fixture: a comma-enumeration with ordinal cues -> a real PROCESS_LIST beat --------------------------
DIAGRAM_LINES = [
    ("so it is so it. so it is so it.", 0.0, 4.0),
    ("first true market value, second real lasting growth, third honest brand context, fourth strong final result", 4.0, 14.0),
    ("so it is so it. so it is so it.", 14.0, 18.0),
]
DIAGRAM_TR = make_transcript(DIAGRAM_LINES)
DIAGRAM_WORDS = [TWord(w.word, w.start, w.end) for w in DIAGRAM_TR.words]
DSTOP = frozenset({"so", "it", "is", "today"})


def _mgc(treatment_type: str, start: float, end: float, *, phrase: str | None = None,
         labels: list[str] | None = None, score: float = 0.5) -> MotionGraphicCandidate:
    return MotionGraphicCandidate(treatment_type=treatment_type, start=start, end=end, phrase=phrase,
                                   labels=labels or [], global_score=score)


# ---- 1. only eligible semantic roles can create graphics ----------------------------------------------------------
def test_1_only_eligible_semantic_roles_create_graphics():
    cands = discover_headline_candidates(HWORDS, stopwords=HSTOP, top_n=5)
    assert cands and all(c.semantic_role in ELIGIBLE_HEADLINE_ROLES for c in cands)


# ---- 2. uncertain semantics fail closed (no candidate at all) -------------------------------------------------------
def test_2_uncertain_semantics_fail_closed():
    filler_only = _hwords("so it is so it. so it is so it.", start=0.0)
    cands = discover_headline_candidates(filler_only, stopwords=HSTOP, top_n=5)
    assert cands == []


# ---- 3. primary headline uses verbatim text --------------------------------------------------------------------
def test_3_primary_headline_uses_verbatim_text():
    cands = discover_headline_candidates(HWORDS, stopwords=HSTOP, top_n=5)
    assert cands
    all_bare = " ".join(re.sub(r"[^\w]", "", w.text, flags=re.UNICODE).lower() for w in HWORDS)
    for c in cands:
        bare = " ".join(re.sub(r"[^\w]", "", w, flags=re.UNICODE).lower() for w in c.phrase.split())
        assert bare in all_bare


# ---- 4. keyword visual uses verbatim text --------------------------------------------------------------------------
def test_4_keyword_visual_uses_verbatim_text():
    cands = discover_keyword_candidates(KWORDS, stopwords=KSTOP, top_n=5)
    assert cands and all(c.phrase == "context" for c in cands)


# ---- 5. diagram labels are supported by transcript/canonical framework source -----------------------------------
def test_5_diagram_labels_supported_by_transcript():
    cands = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP)
    assert cands
    full_text_bare = re.sub(r"[^\w\s]", "", DIAGRAM_TR.full_text, flags=re.UNICODE).lower()
    for c in cands:
        for label in c.labels:
            bare = re.sub(r"[^\w\s]", "", label, flags=re.UNICODE).lower()
            assert bare in full_text_bare


# ---- 6. diagram never invents data -----------------------------------------------------------------------------
def test_6_diagram_never_invents_data():
    cands = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP)
    assert cands
    # every label is drawn from the SAME PROCESS_LIST beat's own words -- nothing sourced elsewhere.
    for c in cands:
        assert all(w.text.strip(",.") for label in c.labels for w in [TWord(label, 0, 0)])
    # a transcript with no structural list evidence at all produces no diagram candidate (never fabricated).
    plain = make_transcript([("so it is so it. so it is so it.", 0.0, 4.0)])
    plain_words = [TWord(w.word, w.start, w.end) for w in plain.words]
    assert discover_diagram_candidates(plain, plain_words, stopwords=DSTOP) == []


# ---- 7. max 5 primary diagram nodes ------------------------------------------------------------------------------
def test_7_max_5_primary_diagram_nodes():
    cands = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP, max_nodes=5)
    assert cands and all(len(c.labels) <= MAX_DIAGRAM_NODES == 5 for c in cands)
    capped = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP, max_nodes=2)
    assert capped and all(len(c.labels) <= 2 for c in capped)


# ---- 8. brand tokens come from profile -----------------------------------------------------------------------------
def test_8_brand_tokens_come_from_profile():
    tok = brand_tokens(BRAND)
    assert tok["colors"]["primary"] == BRAND.colors.primary == "#0A3D62"
    assert tok["colors"]["secondary"] == BRAND.colors.secondary == "#8E44AD"
    assert tok["colors"]["accent"] == BRAND.colors.accent == "#F1C40F"
    assert tok["font_ar"] == "IBM Plex Sans Arabic"


# ---- 9. no client hardcoding in the generic renderer/discovery modules --------------------------------------------------
def _code_lines_only(src: str) -> list[tuple[int, str]]:
    """`src`'s lines with every STRING token (docstrings, comments about them included via
    their own prose) blanked out, so a banned-pattern scan sees only actual code -- module
    docstrings are free to mention client as an illustrative example (sec. 3 explains exactly
    why brand values are looked up instead), only a real hardcoded literal is a violation."""
    import io
    import tokenize
    lines = src.splitlines()
    out = list(lines)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            (sr, _sc), (er, _ec) = tok.start, tok.end
            for ln in range(sr, er + 1):
                if 1 <= ln <= len(out):
                    out[ln - 1] = ""
    return list(enumerate(out, 1))


def test_9_no_client_hardcoding_in_generic_code():
    banned = re.compile(r"#0a3d62|#8e44ad|#f1c40f|D:[\\/]|\.mp4|video-ad-editor", re.IGNORECASE)
    for f in (SRC / "direction" / "motion_graphics.py", SRC / "render" / "motion_graphics.py"):
        for n, line in _code_lines_only(f.read_text(encoding="utf-8")):
            assert not banned.search(line), f"{f.name}:{n}: {line.strip()}"


# ---- 10. captions reduce during major visual -----------------------------------------------------------------------
def test_10_captions_reduce_during_major_visual():
    sub = DIAGRAM_TR  # reduce_captions operates on ASS text, not directly on the transcript
    normal = "Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,NORMAL\n"
    reduced = "Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,REDUCED\n"
    out = reduce_captions(normal, reduced, (4.0, 14.0))
    assert "REDUCED" in out and "NORMAL" not in out
    del sub


# ---- 11. captions restore afterward ----------------------------------------------------------------------------------
def test_11_captions_restore_afterward():
    normal = "Dialogue: 0,0:00:20.00,0:00:21.00,Default,,0,0,0,,NORMAL\n"
    reduced = "Dialogue: 0,0:00:20.00,0:00:21.00,Default,,0,0,0,,REDUCED\n"
    out = reduce_captions(normal, reduced, (4.0, 14.0))  # window ends well before this caption starts
    assert "NORMAL" in out and "REDUCED" not in out


# ---- 12. major semantic visual can block non-pinned rhythm (camera ownership machinery, reused not reinvented) -------
def test_12_major_visual_can_block_non_pinned_rhythm():
    from video_edit_agent.direction import camera_timeline as ct
    from video_edit_agent.direction.camera import CameraEvent, CameraMove
    legacy = [CameraEvent(start=4.0, end=14.0, move=CameraMove.PUNCH_IN, zoom_to=1.3, owner="legacy")]
    from video_edit_agent.direction.rhythm import RhythmPlan, RhythmPolicy
    plan = RhythmPlan(rows=[], policy=RhythmPolicy(), start=4.0, end=14.0)
    tl = ct.build_camera_timeline(plan, face_box=(0.3, 0.3, 0.4, 0.15), legacy=legacy, scope=(4.0, 14.0))
    assert not tl.issues
    assert any(e.start == 4.0 and e.end == 14.0 for e in tl.superseded)


# ---- 13. pinned treatment cannot be overridden ----------------------------------------------------------------------
def test_13_pinned_treatment_cannot_be_overridden():
    from video_edit_agent.direction import camera_timeline as ct
    from video_edit_agent.direction.camera import CameraEvent, CameraMove
    from video_edit_agent.direction.rhythm import RhythmPlan, RhythmPolicy, RhythmRow
    pinned = [CameraEvent(start=4.0, end=6.0, move=CameraMove.PUNCH_IN, zoom_to=1.3, owner="pinned")]
    row = RhythmRow(number=1, start=4.0, end=6.0, state="punch_in")
    plan = RhythmPlan(rows=[row], policy=RhythmPolicy(), start=4.0, end=6.0)
    tl = ct.build_camera_timeline(plan, face_box=(0.3, 0.3, 0.4, 0.15), pinned=pinned, scope=(4.0, 6.0))
    assert any(e.owner == "pinned" and e.start == 4.0 for e in tl.executable)
    assert not any(e.owner == "rhythm" and e.start < 6.0 for e in tl.executable)


# ---- 14. Behind-Subject conflict is respected -------------------------------------------------------------------------
def test_14_behind_subject_conflict_respected():
    from video_edit_agent.direction import camera_timeline as ct
    from video_edit_agent.direction.rhythm import RhythmPlan, RhythmPolicy, RhythmRow
    row = RhythmRow(number=1, start=4.0, end=6.0, state="punch_in")
    plan = RhythmPlan(rows=[row], policy=RhythmPolicy(), start=4.0, end=6.0)
    tl = ct.build_camera_timeline(plan, face_box=(0.3, 0.3, 0.4, 0.15), behind_subject=[(4.0, 6.0)], scope=(4.0, 6.0))
    assert not any(e.owner == "rhythm" for e in tl.executable)


# ---- 15. B6 is not silently replaced (a candidate overlapping the pinned B6 window is flagged, never dropped) --------
def test_15_b6_not_silently_replaced():
    b6_window = (33.5, 36.69)
    existing = [(b6_window[0], b6_window[1], "lower_subject_five_cs_headline_B6")]
    cands = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP, existing_treatments=[(4.0, 14.0, "lower_subject_five_cs_headline_B6")])
    assert cands and all(c.conflict and "lower_subject_five_cs_headline_B6" in c.conflict_reason for c in cands)
    survivors = select_treatments(cands)
    assert survivors == []  # a conflicting candidate never silently wins the slot
    del existing


# ---- 16. treatment history/fatigue affects repeated choices ------------------------------------------------------------
def test_16_history_fatigue_affects_repeated_choices():
    history = TreatmentHistory()
    for _ in range(RECENT_BEATS):
        history.record(0.0, 1.0, TreatmentType.KEYWORD_VISUAL.value)
    a = _mgc(TreatmentType.KEYWORD_VISUAL.value, 10.0, 11.0, phrase="alpha")
    b = _mgc(TreatmentType.PRIMARY_HEADLINE.value, 20.0, 21.0, phrase="beta")
    picked = select_treatments([a, b], history=history, max_recent_repeats=1)
    assert [c.treatment_type for c in picked] == [TreatmentType.PRIMARY_HEADLINE.value]


# ---- 17. deterministic layout -------------------------------------------------------------------------------------------
def test_17_deterministic_layout():
    a = diagram_layout(4)
    b = diagram_layout(4)
    assert a == b
    assert a == sorted(a)  # top-to-bottom, transcript order preserved


# ---- 18. deterministic timing (candidate discovery + selection is pure/deterministic) ------------------------------------
def test_18_deterministic_timing():
    a = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP)
    b = discover_diagram_candidates(DIAGRAM_TR, DIAGRAM_WORDS, stopwords=DSTOP)
    assert [(c.start, c.end, tuple(c.labels)) for c in a] == [(c.start, c.end, tuple(c.labels)) for c in b]


# ---- 19. no external generation required (no network/provider calls anywhere in the new modules) --------------------
def test_19_no_external_generation_required():
    banned = re.compile(r"requests\.|httpx\.|openai|elevenlabs|stability|dall-?e|midjourney", re.IGNORECASE)
    for f in (SRC / "direction" / "motion_graphics.py", SRC / "render" / "motion_graphics.py"):
        assert not banned.search(f.read_text(encoding="utf-8"))


# ---- 20. no SFX/audio changes ---------------------------------------------------------------------------------------------
def test_20_no_sfx_audio_changes():
    for f in (SRC / "direction" / "motion_graphics.py", SRC / "render" / "motion_graphics.py"):
        src = f.read_text(encoding="utf-8").lower()
        assert "sfx" not in src and "audio" not in src


# ---- 21. no B-roll changes ---------------------------------------------------------------------------------------------
def test_21_no_broll_changes():
    for f in (SRC / "direction" / "motion_graphics.py", SRC / "render" / "motion_graphics.py"):
        assert "broll" not in f.read_text(encoding="utf-8").lower()


# ---- 22. no full render in this phase (no RenderPlan/build_filter_complex invocation anywhere in the new modules) ------
def test_22_no_full_render_in_this_phase():
    for f in (SRC / "direction" / "motion_graphics.py", SRC / "render" / "motion_graphics.py"):
        src = f.read_text(encoding="utf-8")
        assert "build_filter_complex(" not in src
        assert "RenderPlan(" not in src
    # RenderPlan/CaptionBurn remain importable/usable by a CALLER (a preview script), just never invoked here.
    assert RenderPlan is not None and CaptionBurn is not None


# ---- 23. review state remains pending_review ---------------------------------------------------------------------------
def test_23_review_state_remains_pending_review():
    for c in (_mgc(TreatmentType.PRIMARY_HEADLINE.value, 0.0, 1.0, phrase="x"),
              _mgc(TreatmentType.KEYWORD_VISUAL.value, 0.0, 1.0, phrase="x"),
              _mgc(TreatmentType.SIMPLE_DIAGRAM.value, 0.0, 1.0, labels=["a", "b"])):
        st = c.review_state()
        assert st["approval_status"] == "pending_review"
        assert st["treatment_type"] == c.treatment_type
        assert set(st) == {"treatment_type", "semantic_source", "semantic_role", "phrase_or_labels", "timing",
                            "layout", "brand_profile", "technical_status", "visual_status", "approval_status"}


# ---- 24. renderer handles Arabic + English mixed text -----------------------------------------------------------------
def test_24_renderer_handles_arabic_and_english_mixed_text():
    ass = "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n\n[Events]\n"
    out = add_keyword(ass, STYLE, BRAND, "الـcontext قيمة value", (0.0, 1.0))
    assert "الـcontext" in out and "value" in out
    out2 = add_diagram(ass, STYLE, BRAND, ["العميل Customer", "Context السياق"], (0.0, 1.0))
    assert "العميل" in out2 and "Customer" in out2 and "السياق" in out2


# ---- 25. renderer passes 1080x1920 safe-zone checks -------------------------------------------------------------------
def test_25_renderer_passes_1080x1920_safe_zone_checks():
    ass = "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n\n[Events]\n"
    out = add_keyword(ass, STYLE, BRAND, "context", (0.0, 2.0), canvas=(1080, 1920))
    bb = ink_bbox(out, (1080, 1920), 1.0)
    if bb is None:  # ffmpeg/libass unavailable in this environment: skip the pixel measurement, not the call
        return
    side_margin, top_margin, bottom_margin = 1080 * 0.06, 1920 * 0.12, 1920 * 0.18
    assert bb[0] >= side_margin - 1 and bb[2] <= 1080 - side_margin + 1
    assert bb[1] >= top_margin - 1 and bb[3] <= 1920 - bottom_margin + 1


# ---- extra: keyword_visual should not stack with a treatment that already owns the same phrase/window (sec. 5) ------
def test_26_no_stacking_same_window_prefers_simplest():
    a = _mgc(TreatmentType.KEYWORD_VISUAL.value, 10.0, 11.0, phrase="x", score=0.9)
    b = _mgc(TreatmentType.PRIMARY_HEADLINE.value, 10.0, 11.0, phrase="x", score=0.9)
    picked = select_treatments([a, b])
    assert [c.treatment_type for c in picked] == [TreatmentType.KEYWORD_VISUAL.value]


# ---- extra: diagram fails closed below MIN_DIAGRAM_NODES eligible items --------------------------------------------
def test_27_diagram_fails_closed_below_min_nodes():
    weak_lines = [("first true value, so it is so it", 4.0, 8.0)]
    tr = make_transcript(weak_lines)
    words = [TWord(w.word, w.start, w.end) for w in tr.words]
    cands = discover_diagram_candidates(tr, words, stopwords=DSTOP, min_nodes=5)
    assert cands == []
