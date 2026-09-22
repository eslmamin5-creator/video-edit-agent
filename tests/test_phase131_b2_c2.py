"""Phase 1.3.1: B2 (`lower_subject` as a semantic composition) and C2 (meaningful Behind-Subject occlusion).
Synthetic content only: no project, brand or reference specifics."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np

from tests.conftest import make_transcript
from tests.test_behind_subject_text import (
    _clip,
    _compose,
    _fakes,
    _plans,
    fake_probe,
    loader,
    subject_mask,
)
from tests.test_behind_subject_text import (
    _edl as _bs_edl,
)
from tests.test_behind_subject_text import (
    _plan as _bs_plan,
)
from video_edit_agent.captions.headline import (
    HEADLINE_STYLE,
    HeadlineSpec,
    headline_event,
    reduce_captions,
)
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.direction import BehindSubjectEvidence, RhythmPolicy, plan_rhythm
from video_edit_agent.direction.camera_timeline import (
    BASE_ANCHOR_X,
    BASE_ANCHOR_Y,
    BASE_ZOOM,
    apply_timeline,
    build_camera_timeline,
    sample_geometry,
)
from video_edit_agent.direction.composition import (
    NOT_SUITABLE,
    TREATMENT,
    CompositionPolicy,
    TWord,
    compose_lower_subject,
    hierarchy,
    is_verbatim,
    review_state,
    select_headline,
    solve_lower_subject,
    timeline_words,
)
from video_edit_agent.direction.rhythm import RhythmState, entries
from video_edit_agent.motion import behind_subject as bs
from video_edit_agent.motion import occlusion as occ
from video_edit_agent.render.composition import CaptionBurn, RenderPlan, build_filter_complex
from video_edit_agent.review.edit_plan import SlotStatus
from video_edit_agent.review.edit_plan_rhythm import attach_rhythm

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"
FACE = (0.30, 0.30, 0.40, 0.15)  # a measured face (normalised x, y, w, h)
STOP = frozenset({"and", "the", "of", "to", "in", "it", "so", "we"})
WINDOW = (0.0, 8.0)
FULL = BehindSubjectEvidence(
    mask_quality=True, readable_occlusion=True, phrase_timing=True, shot_composition=True, caption_hierarchy=True,
    visual_value=0.9, head_hair_integrity=True, meaningful_occlusion=True,
)


def _words(spec: str, step: float = 0.5, start: float = 1.0, gaps: dict[int, float] | None = None) -> list[TWord]:
    out, t = [], start
    for i, w in enumerate(spec.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append(TWord(w, round(t, 3), round(t + step - 0.05, 3)))
        t += step
    return out


GOOD = _words("we begin. check twice, then act now and keep going", gaps={1: 0.3})
WEAK = _words("and the of to in it and the of to in it")


def _headline_h(_text: str) -> float:
    return 0.07


def _compose_ok(**kw):
    kw.setdefault("stopwords", STOP)
    kw.setdefault("headline_height", _headline_h)
    kw.setdefault("head_top", 0.28)
    return compose_lower_subject(GOOD, WINDOW, face_box=FACE, **kw)


# ---- B2 -------------------------------------------------------------------------------------
def test_1_lower_subject_is_not_selected_by_basic_rhythm_without_a_semantic_treatment():
    t = make_transcript([("one two three four five six seven eight", i * 3.4, (i + 1) * 3.4) for i in range(40)])
    plan = plan_rhythm(t, face_box=FACE)
    assert plan.rows and not any(r.state == RhythmState.LOWER_SUBJECT.value for r in plan.rows)
    assert not any(r.composition for r in plan.rows)
    assert RhythmPolicy().plan_lower_subject is False  # opt-in only; a composition is the way in


def test_2_semantic_lower_subject_requires_a_meaningful_text_or_visual():
    weak = compose_lower_subject(WEAK, (2.0, 8.0), face_box=FACE, head_top=0.28, headline_height=_headline_h, stopwords=STOP)
    assert weak.status == NOT_SUITABLE and weak.rows == [] and weak.changes == []
    assert weak.technical_status == "not_applicable" and weak.visual_status == "not_suitable"
    assert review_state(weak)["treatment"] == NOT_SUITABLE and review_state(weak)["caption_role"] == "normal"
    ok = _compose_ok()
    assert ok.status == "ok" and ok.phrase and ok.rows
    assert all(r.composition == TREATMENT and r.composition_detail["phrase"] == ok.phrase for r in ok.rows)


def test_3_lower_subject_adapts_to_the_measured_geometry():
    face = (0.30, 0.40, 0.40, 0.15)
    hi = solve_lower_subject(face, 0.07, head_top=0.30)
    lo = solve_lower_subject(face, 0.07, head_top=0.36)
    assert hi.status == lo.status == "ok"
    assert hi.zoom > lo.zoom and hi.base_headroom == 0.30 and lo.base_headroom == 0.36  # driven by the measured head top
    assert abs(hi.delta - CompositionPolicy().target_gain) < 0.01 and abs(lo.delta - CompositionPolicy().target_gain) < 0.01
    small, big = solve_lower_subject(face, 0.05, head_top=0.30), solve_lower_subject(face, 0.10, head_top=0.30)
    assert small.headline_top != big.headline_top  # the headline is placed from its own rendered height
    assert solve_lower_subject(face, 0.25, head_top=0.30).status == "rejected"  # ... and a headline that does not fit is refused


def test_4_it_creates_usable_headline_space_when_safe():
    pol = CompositionPolicy()
    g = solve_lower_subject((0.30, 0.40, 0.40, 0.15), 0.07, head_top=0.30)
    assert g.status == "ok" and g.delta >= pol.min_gain
    assert g.headline_top >= pol.top_safe - 1e-9
    assert g.headline_bottom <= g.target_headroom - pol.head_gap + 1e-9
    assert g.headline_to_head_clearance >= pol.head_gap - 1e-9
    assert g.headline_bottom - g.headline_top == g.headline_h or abs((g.headline_bottom - g.headline_top) - g.headline_h) < 1e-3
    assert g.face_bottom_target <= pol.caption_safe_top + 1e-9 and g.anchor_y == 0.0 and g.zoom > BASE_ZOOM


def test_5_it_fails_or_falls_back_when_the_space_is_unsafe():
    assert solve_lower_subject(None, 0.07).status == "rejected"  # no measured face: never assumed safe
    low_face = (0.30, 0.55, 0.40, 0.30)  # would end up in the caption band
    assert solve_lower_subject(low_face, 0.07, head_top=0.50).status == "rejected"
    assert solve_lower_subject(FACE, 0.60, head_top=0.20).status == "rejected"  # the headline cannot fit
    refused = _compose_ok(head_top=0.50, headline_height=lambda _t: 0.60)
    assert refused.status == NOT_SUITABLE and refused.rows == [] and "no safe geometry" in refused.reason


def test_6_the_headline_comes_from_the_approved_transcript():
    c = _compose_ok()
    assert c.status == "ok" and is_verbatim(c.phrase, GOOD)
    assert WINDOW[0] <= c.phrase_start and c.phrase_end <= WINDOW[1]
    assert not is_verbatim("check thrice", GOOD)  # a paraphrase is not a transcript phrase
    t = make_transcript([("we begin. check twice then act", 0.0, 4.0)])
    edl = EDL(width=720, height=1280, clips=[EDLClip(source_file="a.mp4", source_in=0, source_out=4, timeline_in=0, timeline_out=4)])
    assert [w.text for w in timeline_words(t, edl)] == ["we", "begin.", "check", "twice", "then", "act"]


def test_7_the_headline_does_not_rewrite_the_transcript():
    before = copy.deepcopy(GOOD)
    made_up = compose_lower_subject(GOOD, WINDOW, face_box=FACE, head_top=0.28, headline_height=_headline_h, stopwords=STOP, pinned_phrase="check thrice")
    assert made_up.status == NOT_SUITABLE and made_up.rows == []
    pinned = compose_lower_subject(GOOD, WINDOW, face_box=FACE, head_top=0.28, headline_height=_headline_h, stopwords=STOP, pinned_phrase="check twice")
    assert pinned.status == "ok" and pinned.phrase == "check twice" and pinned.semantic_source == "user_pinned"
    assert select_headline(GOOD, WINDOW, stopwords=STOP, pinned_phrase="twice check") is None  # order is not rearranged either
    assert GOOD == before
    t = make_transcript([("we begin check twice then act now", 0.0, 6.0)])
    snap = t.model_dump()
    edl = EDL(width=720, height=1280, clips=[EDLClip(source_file="a.mp4", source_in=0, source_out=6, timeline_in=0, timeline_out=6)])
    attach_rhythm(_plan_for(t, edl), t, edl, start=0.0, face_box=FACE, compositions=[_compose_ok()])
    assert t.model_dump() == snap


def _plan_for(t, edl):
    from video_edit_agent.direction import Beat, BeatKind, direct
    from video_edit_agent.review import edit_plan_direction as ed

    beats = [Beat(start=0.0, end=t.duration, kind=BeatKind.PLAIN)]
    return ed.build_directed_plan(beats, direct(beats), t, edl)


def _long_edl_and_plan(seconds: float = 12.0):
    t = make_transcript([("we begin. check twice then act now and keep going", 0.0, seconds)])
    edl = EDL(width=720, height=1280, clips=[
        EDLClip(source_file="a.mp4", source_in=i * 2.0, source_out=(i + 1) * 2.0, timeline_in=i * 2.0, timeline_out=(i + 1) * 2.0)
        for i in range(int(seconds / 2))
    ])
    return t, edl


def test_8_captions_become_reduced_not_hidden_during_the_headline():
    h = hierarchy()
    assert h.caption_role == "reduced" and h.primary_layer == "headline" and h.speaker_visibility == "full"
    assert review_state(_compose_ok())["caption_role"] == "reduced"
    t, edl = _long_edl_and_plan()
    comp = compose_lower_subject(timeline_words(t, edl), (0.0, 12.0), face_box=FACE, head_top=0.28, headline_height=_headline_h, stopwords=STOP)
    assert comp.status == "ok"
    rhythm = attach_rhythm(_plan_for(t, edl), t, edl, start=0.0, face_box=FACE, compositions=[comp])
    rows = [r for r in rhythm.rows if r.composition]
    assert rows and all(r.caption_role == "reduced" and r.primary_layer == "headline" and r.approval_status == "pending_review" for r in rows)
    hdr = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    normal = hdr + "\n".join(f"Dialogue: 0,0:00:{s:02d}.00,0:00:{s + 1:02d}.00,Default,,0,0,0,,N{s}" for s in (1, 3, 5, 7))
    reduced = hdr + "\n".join(f"Dialogue: 0,0:00:{s:02d}.00,0:00:{s + 1:02d}.00,Default,,0,0,0,,R{s}" for s in (1, 3, 5, 7))
    out = reduce_captions(normal, reduced, (2.5, 6.0)).splitlines()
    texts = [ln.rsplit(",", 1)[1] for ln in out if ln.startswith("Dialogue:")]
    assert texts == ["N1", "N3", "R5", "N7"] or texts == ["N1", "R3", "R5", "N7"]  # only events starting in the window swap; none is dropped
    assert len(texts) == 4 and reduce_captions(normal, reduced + "\nDialogue: 0,0:00:09.00,0:00:10.00,Default,,0,0,0,,X", (0, 20)) == normal
    ev = headline_event(HeadlineSpec(text="x y", y_px=100, fade_in=(0.0, 0.3), fade_out=(2.0, 2.3), font_px=100), 720)
    assert HEADLINE_STYLE in ev and "\\fad(300,300)" in ev and "\\move(" in ev  # a fade, a subtle slide, a scale settle: nothing more
    assert "\\k" not in ev and "\\frz" not in ev  # no karaoke, no rotation


def _comp_timeline(seconds: float = 12.0):
    t, edl = _long_edl_and_plan(seconds)
    comp = compose_lower_subject(timeline_words(t, edl), (0.0, seconds), face_box=FACE, head_top=0.28, headline_height=_headline_h, stopwords=STOP)
    assert comp.status == "ok", comp.reason
    plan = plan_rhythm(t, face_box=FACE, compositions=[comp])
    return comp, edl, build_camera_timeline(plan, face_box=FACE)


def _times(lo: float, hi: float, step: float = 0.05) -> list[float]:
    return [round(lo + i * step, 3) for i in range(round((hi - lo) / step))]


def test_9_lower_subject_preserves_the_face_and_head():
    _, edl, tl = _comp_timeline()
    apply_timeline(edl, tl, face_box=FACE)
    pol = CompositionPolicy()
    rows = sample_geometry(edl, _times(0.0, 11.95, 0.05), 720, 1280, face_box=FACE)
    assert min(r["anchor_y"] for r in rows) == 0.0  # the composition really lowers the subject
    for r in rows:
        x0, y0, x1, y1 = r["window"]
        assert y0 >= -1e-6 and y1 <= 1280 + 1e-6 and x0 >= -1e-6 and x1 <= 720 + 1e-6
        assert r["face_top"] >= FACE[1] - 1e-6 and r["face_bottom"] <= pol.caption_safe_top + 1e-6
        assert 0.0 <= r["face_left"] and r["face_right"] <= 1.0


def test_10_the_reset_is_exact():
    comp, edl, tl = _comp_timeline()
    apply_timeline(edl, tl, face_box=FACE)
    settle = max(e.end for e in tl.executable)
    after = sample_geometry(edl, _times(settle + 0.05, 11.9, 0.25), face_box=FACE)
    assert after
    for r in after:
        assert (r["zoom"], r["anchor_x"], r["anchor_y"]) == (BASE_ZOOM, BASE_ANCHOR_X, BASE_ANCHOR_Y)
    ys = [r["anchor_y"] for r in sample_geometry(edl, _times(comp.start - 0.5, settle + 0.5), face_box=FACE)]
    assert ys[0] == BASE_ANCHOR_Y and ys[-1] == BASE_ANCHOR_Y and min(ys) < BASE_ANCHOR_Y  # base -> lowered -> exactly base
    assert entries(comp.rows)  # one composition, one excursion


# ---- C2 -------------------------------------------------------------------------------------
CW, CH = 1080, 1920
FONT = 160


def _glyphs(n_words: int = 2, w: int = 120, h: int = 160, gap: int = 40) -> list[np.ndarray]:
    """A stand-in for rendered words: filled bodies on one grid (the gate measures eroded cores, so bodies are what count)."""
    width = n_words * w + (n_words - 1) * gap
    out = []
    for i in range(n_words):
        a = np.zeros((h, width), np.float32)
        a[:, i * (w + gap):i * (w + gap) + w] = 1.0
        out.append(a)
    return out


def _mask(x0: int, x1: int, y0: int = 0, y1: int = CH) -> np.ndarray:
    m = np.zeros((CH, CW), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


ORIGIN = (400, 500)


def _mo(mask: np.ndarray, **kw):
    return occ.meaningful_occlusion(_glyphs(), FONT, ORIGIN, [mask] * 3, **kw)


def test_11_meaningful_occlusion_ignores_tiny_edge_contact():
    r = _mo(_mask(0, 402))  # the subject only grazes the first glyph's left edge
    assert not r.passed and r.text_subject_overlap_ratio < 0.06 and r.occluded_glyph_count == 0
    fringe = _mo(_mask(0, 405, 500, 505))  # a one-row hair-like contact inside the glyph
    assert not fringe.passed and fringe.overlap_rows_frac < 0.2


def test_12_meaningful_occlusion_requires_actual_glyph_body_overlap():
    between = _mo(_mask(524, 556))  # the subject stands only in the gap between the words
    assert between.text_subject_overlap_px == 0 and not between.passed
    over = _mo(_mask(486, 594))  # ... versus covering a real share of both glyph bodies
    assert over.passed, over.reasons
    assert over.occluded_glyph_count == 2 and over.text_subject_overlap_ratio >= occ.OcclusionPolicy().min_overlap_ratio


def test_13_visible_text_ratio_prevents_excessive_hiding():
    hidden = _mo(_mask(440, 640))  # covers most of both words
    assert hidden.text_subject_overlap_ratio > 0.3 and hidden.visible_text_ratio < occ.OcclusionPolicy().min_visible_ratio
    assert not hidden.passed and any("stays visible" in r or "mostly hidden" in r for r in hidden.reasons)


LOW_FACE = (0.42, 0.66, 0.16, 0.06)  # the face is well below the silhouette top: the text can dip behind the head/shoulders
PASSING = bs.TreatmentPreset(occlusion=occ.OcclusionPolicy(max_key_hidden=0.35))  # a key word may dip a third behind the subject


def _body(top: int = 700, left: int = 100, right: int = 980, face=LOW_FACE):
    return loader([subject_mask(top=top, left=left, right=right)] * 4, face=face)


def test_14_behind_subject_fails_when_the_overlap_is_trivial():
    # the flat-topped subject only just touches the text wherever the legal placements are: a sliver, not an occlusion
    (p,) = _plans(load=_body(), preset=bs.TreatmentPreset())
    assert p.decision != bs.DECISION_BEHIND
    assert p.technical_status == "failed_meaningful_occlusion" and p.recommendation.startswith("alternate treatment")
    detail = p.gate["meaningful_occlusion"]["detail"]
    assert not p.gate["meaningful_occlusion"]["passed"] and "not_suitable_for_this_shot" in detail and "composition candidates" in detail
    assert "not applied automatically" in p.recommendation  # reported, never swapped in behind the user's back


def test_15_behind_subject_passes_only_with_meaningful_overlap_and_readability():
    (p,) = _plans(load=_body(), preset=PASSING)
    mo = p.metrics["meaningful_occlusion"]
    assert p.decision == bs.DECISION_BEHIND and p.technical_status == "passed" and p.gate["meaningful_occlusion"]["passed"]
    # Phase 1.3.2: occluded-glyph count is soft evidence feeding the perceptual score, not a standalone >= 2 hard rule.
    assert mo["passed"] and mo["occluded_glyph_count"] >= 1 and mo["visible_text_ratio"] >= occ.OcclusionPolicy().min_visible_ratio_hard
    assert mo["perceptual_score"] >= occ.OcclusionPolicy().perceptual_min_score and not mo["hard_fail"]
    # the same shot under the default limits (a key word may hide at most a fifth) leaves no candidate that is meaningful AND readable
    (strict,) = _plans(load=_body(), preset=bs.TreatmentPreset())
    assert strict.decision != bs.DECISION_BEHIND and strict.technical_status == "failed_meaningful_occlusion"


def test_16_the_subject_rgb_stays_above_the_text(tmp_path: Path):
    edl = _bs_edl(_clip(0, 8))
    _, render_fn, cutout_fn = _fakes(tmp_path)
    (plan,) = _plans(edl, load=_body(), preset=PASSING)
    assert plan.spec is not None and plan.decision == bs.DECISION_BEHIND
    result = _compose(edl, plan.spec, tmp_path, render_fn, cutout_fn)
    assert [o.path.name for o in result.overlays] == ["graphic.webm", "cutout.mov"]  # background -> text -> subject RGB
    assert result.overlays[0].behind_subject and not result.overlays[1].behind_subject
    ass = tmp_path / "c.ass"
    ass.write_text("[Script Info]" + chr(10), encoding="utf-8")
    _, graph, _ = build_filter_complex(RenderPlan(edl=edl, overlays=result.overlays, captions=CaptionBurn(ass_path=ass, fonts_dir=None)))
    text_at, subject_at, captions_at = graph.index("[1:v]"), graph.index("[2:v]"), graph.rindex("subtitles")
    assert text_at < subject_at < captions_at  # captions are burned last: over the subject, over the text


def test_17_a_user_pinned_behind_subject_is_not_auto_approved():
    pending = _bs_plan(status=SlotStatus.PENDING_REVIEW)
    pending.slots[0].semantic_source = "user_pinned"
    assert _plans(plan=pending, load=_body(), preset=bs.TreatmentPreset()) == []  # nothing is even judged until approved
    assert pending.slots[0].status is SlotStatus.PENDING_REVIEW
    t, edl = _long_edl_and_plan()
    plan = _plan_for(t, edl)
    slot = plan.slots[0]
    slot.treatment, slot.text, slot.semantic_source, slot.status = "behind_subject_text", "check twice", "user_pinned", SlotStatus.PENDING_REVIEW
    rhythm = attach_rhythm(plan, t, edl, start=0.0, face_box=FACE, evidence=lambda _r: FULL)
    assert slot.status is SlotStatus.PENDING_REVIEW and slot.semantic_source == "user_pinned"
    assert all(r.approval_status != "approved" for r in rhythm.rows)
    st = review_state(_compose_ok())
    assert st["approval_status"] == "pending_review" and st["technical_status"] != "approved" and st["visual_status"] != "approved"


def test_18_no_client_or_project_hardcoding_in_the_generic_gate():
    files = [SRC / "motion" / "occlusion.py", SRC / "direction" / "composition.py", SRC / "captions" / "headline.py"]
    banned = re.compile(r"#[0-9a-fA-F]{6}\b|[ء-ي]|D:[\\/]|\.mp4|video-ad-editor|\b(cairo|tajawal|almarai)\b", re.IGNORECASE)
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not banned.search(line), f"{f.name}:{n}: {line.strip()}"


def test_19_composition_candidate_selection_is_deterministic():
    layouts = [fake_probe(["real value"]), fake_probe(["real", "value"])]
    masks = [subject_mask(top=700, left=100, right=980)] * 3

    def run():
        return occ.search_composition(layouts, masks, (1080, 1920), top_limit_px=230, zone=(1375, 1575), face=LOW_FACE, key=(1,), policy=PASSING.occlusion)

    a, b = run(), run()
    assert a.considered == b.considered > 0 and a.valid == b.valid and a.status == b.status
    assert dict(a.rejected) == dict(b.rejected)
    assert a.status == "ok" and a.best is not None and a.best.occlusion.passed and a.best.camera.static
    assert a.best.summary() == b.best.summary()  # the same winner every run
    assert a.near_miss is not None and a.near_miss.summary() == b.near_miss.summary() and not a.near_miss.occlusion.passed
    nothing = occ.search_composition(layouts, [subject_mask(top=1400, left=100, right=980)] * 3, (1080, 1920), top_limit_px=230, zone=(1375, 1575), face=LOW_FACE, policy=PASSING.occlusion)
    assert nothing.status == occ.NOT_SUITABLE_FOR_SHOT and nothing.best is None
