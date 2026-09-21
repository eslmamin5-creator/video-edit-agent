"""Behind-subject text: phrase-timed, occlusion-judged, gated, and executed by ONE
implementation shared by the final render and the micro-preview
(`motion/behind_subject.py`, `phrase_timing.py`, `occlusion.py`, `subject/refine.py`).

Nothing here is brand-, colour- or project-specific: the brand is synthetic, the
words arbitrary, the subject a rectangle and the glyphs filled boxes."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from video_edit_agent.brand.schema import Brand, BrandColors
from video_edit_agent.core import pipeline as pipeline_mod
from video_edit_agent.core.schemas import EDL, AnimationKind, EDLClip, MotionPlanItem, Word
from video_edit_agent.motion import behind_subject as bs
from video_edit_agent.motion import occlusion as occ
from video_edit_agent.motion import phrase_timing as pt
from video_edit_agent.render import micro_preview as mp
from video_edit_agent.render.composition import Overlay, RenderPlan, build_filter_complex
from video_edit_agent.review.edit_plan import EditPlan, EditPlanSlot, SlotStatus, save_plan
from video_edit_agent.subject import refine

BRAND = Brand(name="acme", colors=BrandColors(primary="#123456", secondary="#ABCDEF", accent="#FEDCBA"))
BEHIND = "behind_subject_text"
W, H = 1080, 1920
SLOT = (2.0, 8.0)  # a six second slot ...
WORDS = [  # ... whose phrase is spoken in well under a second of it
    Word(word="a", start=2.1, end=2.4),
    Word(word="big", start=4.9, end=5.2),
    Word(word="real", start=5.2, end=5.5),
    Word(word="value.", start=5.5, end=5.9),
    Word(word="today", start=6.0, end=6.4),
]
CAPTION = bs.CaptionInfo(zone=(1375, 1575), colours=frozenset({"#eeeeee", "#ff0000"}))
FACE = (0.40, 0.36, 0.20, 0.08)  # normalised x, y, w, h: head top at ~y 690


def _clip(tin: float, tout: float, **kw) -> EDLClip:
    return EDLClip(source_file="src.mp4", source_in=tin, source_out=tout, timeline_in=tin, timeline_out=tout, **kw)


def _edl(*clips: EDLClip) -> EDL:
    return EDL(fps=30.0, width=W, height=H, clips=list(clips))


def _slot(status=SlotStatus.APPROVED, text: str | None = "real value", treatment=BEHIND, number=1) -> EditPlanSlot:
    return EditPlanSlot(
        number=number, timeline_start=SLOT[0], timeline_end=SLOT[1], recommended=BEHIND, treatment=treatment,
        status=status, text=text,
    )


def _plan(**kw) -> EditPlan:
    return EditPlan(slots=[_slot(**kw)])


# -- synthetic glyphs and subject -------------------------------------------


def fake_probe(lines) -> occ.GlyphLayout:
    """Each word is a filled box, 55 ref-px per letter wide and 120 tall, words
    side by side in a line, lines stacked: a stand-in for the renderer's shapes."""
    rows = [line.split() for line in lines]

    def row_width(row):
        return sum(55 * len(w) for w in row) + 30 * (len(row) - 1)

    width = max(row_width(r) for r in rows)
    height = 140 * len(rows)
    layers = []
    for r, row in enumerate(rows):
        x = (width - row_width(row)) // 2
        for word in row:
            layer = np.zeros((height, width), np.float32)
            layer[r * 140 + 10:r * 140 + 130, x:x + 55 * len(word)] = 1.0
            layers.append(layer)
            x += 55 * len(word) + 30
    return occ.GlyphLayout(lines=tuple(lines), ref_px=200, words=tuple(layers))


def subject_mask(top: int = 690, left: int = 300, right: int = 780, bottom: int = H) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[top:bottom, left:right] = 1.0
    return m


def loader(masks=None, face=FACE, reason: str = "", seen: list | None = None):
    def load(edl, start, end):
        if seen is not None:
            seen.append((start, end))
        if masks is None and reason:
            return bs.SubjectData(reason=reason)
        ms = masks if masks is not None else [subject_mask()] * 4
        return bs.SubjectData(times=[start + i * 0.1 for i in range(len(ms))], masks=list(ms), face=face)

    return load


def _plans(edl=None, plan=None, words=WORDS, caption=CAPTION, probe=fake_probe, load=None, brand=BRAND, **kw):
    return bs.plan_behind_subject(
        plan or _plan(), edl or _edl(_clip(0, 12)), brand, words=words, caption=caption, probe=probe,
        load_subject=load or loader(), **kw,
    )


def _spec(edl=None, **kw):
    (p,) = _plans(edl, **kw)
    assert p.spec is not None, p.reason
    return p.spec


# --------------------------------------------------------------------------
# Only approved text, never invented
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    {"status": SlotStatus.PENDING_REVIEW},
    {"status": SlotStatus.REJECTED},
    {"text": None},
    {"text": "   "},
    {"number": 0},  # settled elsewhere, not a reviewable slot
    {"treatment": "punch_in"},
])
def test_only_an_approved_slot_with_approved_text_is_planned(kw):
    assert _plans(plan=_plan(**kw)) == []


def test_no_plan_plans_nothing():
    assert bs.plan_behind_subject_text(None, _edl(_clip(0, 12)), BRAND) == []


def test_text_is_the_approved_text_and_line_breaks_only_regroup_it():
    spec = _spec(plan=_plan(text="  real   value  "))
    assert spec.kind is AnimationKind.BEHIND_TEXT and spec.behind_subject is True
    assert spec.text == "real   value"
    assert " ".join(spec.extra["lines"]).split() == spec.text.split()


def test_a_brand_that_avoids_outlines_gets_none():
    brand = BRAND.model_copy(deep=True)
    brand.motion.avoid = ["text_outline"]
    assert _spec(brand=brand).extra["outline"] is None


def test_without_a_measurement_it_never_recommends_an_unchecked_behind_text():
    (p,) = bs.plan_behind_subject(_plan(), _edl(_clip(0, 12)), BRAND, words=WORDS, caption=CAPTION)
    assert p.decision == bs.DECISION_PUNCH_IN and p.spec is None and bs.NOT_RECOMMENDED in p.reason


# --------------------------------------------------------------------------
# 1-2. Phrase timing: from the approved word timings, not from the slot
# --------------------------------------------------------------------------


def test_phrase_timing_comes_from_the_approved_word_timings():
    (p,) = _plans()
    t = p.timing
    assert t.source == "words"
    assert (t.start, t.end) == (5.2, 5.9)  # "real" starts, "value." (punctuation ignored) ends
    assert [(w.word, w.start, w.end) for w in t.words] == [("real", 5.2, 5.5), ("value.", 5.5, 5.9)]
    preset = pt.DEFAULT_TIMING
    assert t.show_start == pytest.approx(t.start - preset.lead_in)
    assert t.readable_at == pytest.approx(t.start)  # fully in as the first word begins
    assert t.hold_end == pytest.approx(t.end + preset.tail)
    assert t.show_end == pytest.approx(t.hold_end + preset.exit)
    assert t.show_start < t.readable_at < t.hold_end < t.show_end


def test_the_spec_window_is_the_show_window_not_the_slot():
    spec = _spec()
    assert (spec.timeline_start, spec.timeline_end) == (pytest.approx(4.9), pytest.approx(6.55))
    assert spec.extra["enterSec"] == pytest.approx(0.3) and spec.extra["exitSec"] == pytest.approx(0.25)


def test_duration_is_not_the_full_slot():
    spec = _spec()
    shown = spec.timeline_end - spec.timeline_start
    assert shown < 0.5 * (SLOT[1] - SLOT[0])
    assert SLOT[0] <= spec.timeline_start and spec.timeline_end <= SLOT[1]


def test_the_planner_only_loads_the_subject_over_the_show_window():
    seen: list = []
    _plans(load=loader(seen=seen))
    assert seen == [(pytest.approx(4.9), pytest.approx(6.55))]


def test_a_repeated_phrase_is_picked_where_the_slot_is():
    words = [Word(word="real", start=0.5, end=0.8), Word(word="value", start=0.8, end=1.1), *WORDS]
    words.sort(key=lambda w: w.start)
    found = pt.locate_phrase(words, "real value", near=SLOT)
    assert found is not None and found[0].start == 5.2


def test_timing_is_clamped_to_the_slot_and_stretched_to_a_readable_minimum():
    preset = pt.TimingPreset()
    early = pt.plan_show_window((pt.WordTime("x", 2.05, 2.15),), SLOT, preset=preset)
    assert early.show_start == SLOT[0]
    assert early.duration >= preset.min_show


def test_a_phrase_that_is_not_spoken_gets_a_capped_slot_anchored_window():
    t = pt.plan_show_window(None, SLOT)
    assert t.source == "slot" and t.duration <= pt.DEFAULT_TIMING.max_unanchored + pt.DEFAULT_TIMING.exit + 1e-6
    (p,) = _plans(words=[])
    assert p.timing.source == "slot" and not p.gate["phrase_timing"]["passed"]


def test_word_times_map_through_the_clip_onto_the_timeline():
    clip = EDLClip(source_file="src.mp4", source_in=10.0, source_out=22.0, timeline_in=0.0, timeline_out=12.0)
    words = [Word(word="real", start=15.2, end=15.5), Word(word="value", start=15.5, end=15.9)]
    t = bs.phrase_timing_for_slot(_slot(), _edl(clip), words)
    assert (t.start, t.end) == (pytest.approx(5.2), pytest.approx(5.9))


# --------------------------------------------------------------------------
# 3-4. Occlusion inside bounds, key word readable
# --------------------------------------------------------------------------


def test_the_chosen_placement_stays_inside_the_occlusion_bounds():
    (p,) = _plans()
    m, pol = p.metrics, occ.DEFAULT_POLICY
    assert p.decision == bs.DECISION_BEHIND and p.recommended
    assert pol.min_hidden * 100 <= m["glyph_hidden_pct"] <= pol.max_hidden * 100
    assert max(m["word_hidden_pct"]) <= pol.max_word_hidden * 100
    assert m["face_hidden_pct"] <= pol.max_face_hidden * 100
    assert m["readable_glyph_pct"] == pytest.approx(100 - m["glyph_hidden_pct"])


def test_the_key_word_stays_readable():
    (p,) = _plans()
    assert p.metrics["key_hidden_pct"] <= occ.DEFAULT_POLICY.max_key_hidden * 100
    assert occ.key_word_indices(["real", "value"]) == (1,)  # the longest word carries the phrase
    # a bad spot (the phrase fully behind the body) is measured as bad
    bad = occ.evaluate(fake_probe(["real value"]), 200, (540.0, 1000.0), [subject_mask()], (W, H), key=(1,))
    assert bad is not None and bad.key_hidden > occ.DEFAULT_POLICY.max_key_hidden


def test_the_search_never_returns_a_placement_that_hides_the_face():
    face = (0.30, 0.10, 0.4, 0.5)  # the face box is where the subject is
    res = occ.search_placement(
        [fake_probe(["real value"])], [subject_mask(top=200)], (W, H), top_limit_px=100, zone=(1375, 1575),
        face=face, key=(1,),
    )
    if res.placement is not None:
        assert res.placement.face_hidden <= occ.DEFAULT_POLICY.max_face_hidden


# --------------------------------------------------------------------------
# 5. Placement avoids the caption safe zone
# --------------------------------------------------------------------------


def test_placement_keeps_clear_of_the_caption_zone():
    (p,) = _plans()
    y1 = p.metrics["bounds"][3]
    assert y1 <= CAPTION.zone[0] - occ.DEFAULT_POLICY.caption_clearance_px
    assert p.metrics["caption_clearance_px"] >= occ.DEFAULT_POLICY.caption_clearance_px
    assert p.gate["caption_zone_clear"]["passed"]


def test_the_caption_zone_is_derived_from_the_caption_style():
    top, bottom = occ.caption_zone(1920, 0.18, 48, 2, padding_px=20)
    assert bottom == round(1920 - 1920 * 0.18)
    assert top == bottom - round(2 * 48 * 1.25 + 40)


def test_ass_colours_convert_to_hex():
    assert bs.ass_to_hex("&H00FF0000") == "#0000ff"  # ASS is BGR
    assert bs.ass_to_hex("&H0032D7BF") == "#bfd732"
    assert bs.ass_to_hex("nonsense") is None


# --------------------------------------------------------------------------
# 6. One coordinated group
# --------------------------------------------------------------------------


def test_a_multi_word_phrase_is_one_locked_group_with_one_block_position():
    (p,) = _plans()
    extra = p.spec.extra
    assert " ".join(extra["lines"]).split() == ["real", "value"]
    assert isinstance(extra["centerX"], float) and isinstance(extra["centerY"], float)  # ONE position for the block
    x0, _y0, x1, _y1 = p.metrics["bounds"]
    # the words share one bounding box: the group is narrower than the subject-free canvas span,
    # never a word on each side of the body
    assert x1 - x0 < 0.85 * W
    assert len(p.metrics["word_hidden_pct"]) == 2


def test_line_options_only_regroup_the_same_words_in_order():
    for option in bs.line_options("one two three four", 3):
        assert " ".join(option).split() == ["one", "two", "three", "four"]
        assert 1 <= len(option) <= 3


def test_a_phrase_too_long_for_one_group_is_not_recommended():
    (p,) = _plans(plan=_plan(text="one two three four five six seven"))
    assert p.decision == bs.DECISION_PUNCH_IN and bs.NOT_RECOMMENDED in p.reason


# --------------------------------------------------------------------------
# 7. Fallback when there is no readable placement
# --------------------------------------------------------------------------


def test_no_readable_occlusion_falls_back_to_kinetic_typography_and_says_so():
    low = [subject_mask(top=1650, left=300, right=780)] * 4  # the subject never reaches where text can go
    (p,) = _plans(load=loader(low, face=None))
    assert p.decision == bs.DECISION_FOREGROUND and not p.recommended
    assert bs.NOT_RECOMMENDED in p.reason and "readable_occlusion" in p.reason
    assert p.spec is not None and p.spec.behind_subject is False
    assert p.spec.extra[bs.NOT_RECOMMENDED] is True
    assert p.metrics["glyph_hidden_pct"] <= occ.DEFAULT_POLICY.foreground_hidden * 100  # nothing hidden at all


def test_no_room_anywhere_falls_back_to_the_punch_in_alone():
    (p,) = _plans(load=loader([np.ones((H, W), np.float32)] * 4, face=None))
    assert p.decision == bs.DECISION_PUNCH_IN and p.spec is None and bs.NOT_RECOMMENDED in p.reason


def test_no_usable_mask_falls_back_and_names_the_reason():
    (p,) = _plans(load=loader(masks=None, reason="no usable subject mask"))
    assert p.decision == bs.DECISION_PUNCH_IN and "no usable subject mask" in p.reason
    assert not p.gate["subject_mask"]["passed"]


def test_an_unstable_mask_is_not_trusted():
    flicker = [subject_mask(), subject_mask(left=0, right=200), subject_mask(), subject_mask(left=0, right=200)]
    (p,) = _plans(load=loader(flicker))
    assert not p.gate["mask_stability"]["passed"] and not p.recommended


def test_the_text_specs_skip_slots_that_fall_back_to_the_punch_in_alone():
    specs = bs.plan_behind_subject_text(
        _plan(), _edl(_clip(0, 12)), BRAND, words=WORDS, caption=CAPTION, probe=fake_probe,
        load_subject=loader([np.ones((H, W), np.float32)] * 4, face=None),
    )
    assert specs == []


# --------------------------------------------------------------------------
# 7b. Caption hierarchy
# --------------------------------------------------------------------------


def test_the_phrase_never_reuses_a_colour_the_captions_use():
    used = frozenset({"#abcdef", "#fedcba"})
    for backdrop in [(0.9, 0.9, 0.9), (0.1, 0.1, 0.1), None]:
        assert bs.choose_colour(BRAND, backdrop, used)["color"].lower() not in used


def test_the_phrase_is_slightly_transparent_so_it_stays_below_the_captions():
    assert _spec().extra["opacity"] < 1.0


def test_styling_comes_from_the_brand_profile():
    spec = _spec()
    assert spec.extra["color"].lower() in {"#abcdef", "#fedcba", "#123456"}
    assert spec.extra["legibility"]["foreground_source"].startswith("brand.")
    assert spec.extra["gate"]["caption_hierarchy"]["passed"]


# --------------------------------------------------------------------------
# 8. Mask quality: deterministic stabilisation and clean edges
# --------------------------------------------------------------------------


def _noisy(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = np.zeros((96, 64), np.float32)
        m[20 + (i % 2):90, 12:52] = 1.0
        m += rng.normal(0, 0.05, m.shape).astype(np.float32)
        out.append(np.clip(m, 0, 1))
    return out


def test_temporal_stabilisation_is_deterministic():
    a = refine.stabilize_temporal(_noisy(1))
    b = refine.stabilize_temporal(_noisy(1))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_temporal_stabilisation_reduces_flicker():
    raw = _noisy(2)
    before = refine.mask_stability(raw)[0]
    after = refine.mask_stability(refine.stabilize_temporal(raw))[0]
    assert after >= before and 0.0 <= after <= 1.0


def test_refinement_is_deterministic_and_drops_specks_and_fills_holes():
    m = np.zeros((120, 80), np.float32)
    m[20:110, 15:65] = 1.0
    m[60:62, 30:32] = 0.0  # a small hole inside the subject
    m[3:5, 3:5] = 1.0  # a stray speck away from it
    a, b = refine.refine_alpha(m), refine.refine_alpha(m)
    assert np.array_equal(a, b)
    assert a[61, 31] > 0.9 and a[3:5, 3:5].max() < 0.1
    assert a.dtype == np.float32 and a.min() >= 0.0 and a.max() <= 1.0


def test_a_single_mask_is_trivially_stable():
    assert refine.mask_stability([np.zeros((4, 4), np.float32)]) == (1.0, 1.0)


def test_masks_are_brought_to_canvas_size():
    assert refine.to_canvas(np.ones((90, 160), np.float32), (100, 200)).shape == (200, 100)


# --------------------------------------------------------------------------
# Glyph measurement
# --------------------------------------------------------------------------


def test_the_probe_still_is_split_back_into_one_layer_per_word():
    from video_edit_agent.motion import glyph_probe as gp

    png = np.zeros((100, 200, 4), np.uint8)
    png[10:40, 10:90] = (255, 0, 0, 255)  # word 0
    png[60:90, 110:190] = (0, 255, 0, 255)  # word 1
    layout = gp.layout_from_probe_png(png, 2, (100.0, 50.0))
    assert layout is not None and len(layout.words) == 2
    assert layout.words[0].sum() > 0 and layout.words[1].sum() > 0
    assert (layout.words[0] * layout.words[1]).sum() == 0  # disjoint
    assert gp.layout_from_probe_png(np.zeros((10, 10, 4), np.uint8), 2, (5.0, 5.0)) is None


# --------------------------------------------------------------------------
# 10. Nothing brand- or video-specific in the generic modules
# --------------------------------------------------------------------------

_SRC = Path(pipeline_mod.__file__).parents[1]
GENERIC = [
    "motion/behind_subject.py", "motion/occlusion.py", "motion/phrase_timing.py", "motion/glyph_probe.py",
    "subject/refine.py", "subject/compositor.py", "subject/integrity.py",
    "motion/remotion/template/src/components/BehindText.tsx",
]
FORBIDDEN = [
    r"#0a3d62", r"#8e44ad", r"#f1c40f", "حقيقية", r"\b58\.1", r"\b63\.36\b", r"\b64\.16\b",
    r"video-ad-editor", ]


@pytest.mark.parametrize("rel", GENERIC)
def test_no_brand_or_video_specific_values_in_the_generic_modules(rel: str):
    src = (_SRC / rel).read_text(encoding="utf-8")
    hits = [pat for pat in FORBIDDEN if re.search(pat, src, flags=re.IGNORECASE)]
    assert hits == [], f"{rel} hardcodes {hits}"


# --------------------------------------------------------------------------
# Composition: text under, subject cutout on top
# --------------------------------------------------------------------------


def _fakes(tmp_path: Path, cutout: bool = True):
    calls: list = []

    def render_fn(spec, *_a, **_k):
        out = tmp_path / "graphic.webm"
        out.write_bytes(b"x")
        return MotionPlanItem(spec=spec, output_path=str(out))

    def cutout_fn(source, start, end, cache_dir, **kw):
        calls.append((str(source), start, end, kw))
        if not cutout:
            return None
        out = tmp_path / "cutout.mov"
        out.write_bytes(b"x")
        return out

    return calls, render_fn, cutout_fn


def _compose(edl, spec, tmp_path, render_fn, cutout_fn):
    return bs.compose_behind_subject(
        spec, edl, tmp_path, tmp_path / "motion", tmp_path / "cutouts", brand=BRAND, fps=30.0, slot_id="s",
        render_fn=render_fn, cutout_fn=cutout_fn,
    )


def test_graphic_is_drawn_first_and_the_subject_cutout_goes_on_top(tmp_path: Path):
    edl = _edl(_clip(0, 8))
    calls, render_fn, cutout_fn = _fakes(tmp_path)
    spec = _spec(edl)
    result = _compose(edl, spec, tmp_path, render_fn, cutout_fn)
    assert [o.path.name for o in result.overlays] == ["graphic.webm", "cutout.mov"]
    assert result.overlays[0].behind_subject is True
    assert all((o.start, o.end) == (spec.timeline_start, spec.timeline_end) for o in result.overlays)
    assert result.cutout_applied and not result.warnings
    (source, start, end, kw) = calls[0]
    assert (source, start, end) == ("src.mp4", spec.timeline_start, spec.timeline_end)
    assert kw["output_size"] == (1080, 1920)  # framed like the base timeline, not at source resolution


def test_no_usable_mask_keeps_a_plain_foreground_text_and_says_so(tmp_path: Path):
    edl = _edl(_clip(0, 8))
    _, render_fn, cutout_fn = _fakes(tmp_path, cutout=False)
    result = _compose(edl, _spec(edl), tmp_path, render_fn, cutout_fn)
    assert [o.path.name for o in result.overlays] == ["graphic.webm"]
    assert not result.cutout_applied
    assert "no usable subject mask" in result.warnings[0]


def test_a_window_spanning_a_cut_is_not_cut_out(tmp_path: Path):
    edl = _edl(_clip(0, 5.5), _clip(5.5, 12))
    calls, render_fn, cutout_fn = _fakes(tmp_path)
    result = _compose(edl, _spec(edl), tmp_path, render_fn, cutout_fn)
    assert calls == [] and len(result.overlays) == 1
    assert "spans a cut" in result.warnings[0]


def test_a_punched_in_clip_is_not_given_a_misaligned_cutout(tmp_path: Path):
    edl = _edl(_clip(0, 8, zoom=1.08))
    calls, render_fn, cutout_fn = _fakes(tmp_path)
    result = _compose(edl, _spec(edl), tmp_path, render_fn, cutout_fn)
    assert calls == [] and len(result.overlays) == 1
    assert "punch-in" in result.warnings[0]


def test_a_graphic_that_could_not_render_yields_no_layers(tmp_path: Path):
    edl = _edl(_clip(0, 8))
    _, _, cutout_fn = _fakes(tmp_path)
    result = _compose(edl, _spec(edl), tmp_path, lambda spec, *_a, **_k: MotionPlanItem(spec=spec), cutout_fn)
    assert result.overlays == []


# --------------------------------------------------------------------------
# Preview and final render share the implementation
# --------------------------------------------------------------------------


def test_final_render_and_preview_call_the_same_function():
    assert pipeline_mod.compose_behind_subject is bs.compose_behind_subject
    assert mp.compose_behind_subject is bs.compose_behind_subject
    assert mp.render_behind_subject_preview.__kwdefaults__["compose"] is bs.compose_behind_subject
    assert pipeline_mod.plan_behind_subject_for_project is bs.plan_behind_subject_for_project


def test_preview_renders_the_layers_the_composer_returns_rebased_to_the_window(tmp_path: Path, monkeypatch):
    edl = _edl(_clip(0, 8))
    spec = _spec(edl)
    _, render_fn, cutout_fn = _fakes(tmp_path)
    composite = _compose(edl, spec, tmp_path, render_fn, cutout_fn)
    seen: dict = {}

    def fake_compose(*_a, **_k):
        seen["composed"] = True
        return composite

    def fake_render(plan: RenderPlan, out_path, preset, *, crf=22):
        seen["plan"] = plan
        return out_path

    monkeypatch.setattr(mp, "render", fake_render)
    monkeypatch.setattr(mp, "_trim_to", lambda path, duration: path)
    mp.render_behind_subject_preview(
        edl, spec, tmp_path, tmp_path / "out.mp4", brand=BRAND, cutout_dir=tmp_path / "c", compose=fake_compose,
    )
    plan = seen["plan"]
    assert seen["composed"]
    assert [o.path for o in plan.overlays] == [o.path for o in composite.overlays]  # same layers, same order
    assert all((o.start, o.end) == (0.0, pytest.approx(1.65)) for o in plan.overlays)  # window-relative
    assert plan.captions is None and plan.edl.total_duration == pytest.approx(1.65)


def test_preview_refuses_a_spec_that_is_not_behind_subject(tmp_path: Path):
    edl = _edl(_clip(0, 8))
    spec = _spec(edl).model_copy(update={"behind_subject": False})
    with pytest.raises(mp.MicroPreviewError):
        mp.render_behind_subject_preview(edl, spec, tmp_path, tmp_path / "o.mp4", brand=BRAND, cutout_dir=tmp_path)


def test_timeline_window_maps_to_source_time():
    clip = _clip(2, 9).model_copy(update={"source_in": 10.0, "source_out": 17.0})
    assert mp.timeline_to_source_window(_edl(clip), 3.0, 5.0) == (11.0, 13.0)


def test_pipeline_reads_the_slot_text_from_the_saved_edit_plan(tmp_path: Path):
    edl = _edl(_clip(0, 8))
    save_plan(_plan(status=SlotStatus.PENDING_REVIEW), tmp_path)
    kw = {"words": WORDS, "caption": CAPTION, "probe": fake_probe, "load_subject": loader()}
    assert bs.plan_behind_subject_text(pipeline_mod.load_plan(tmp_path), edl, BRAND, **kw) == []
    save_plan(_plan(), tmp_path)
    (spec,) = bs.plan_behind_subject_text(pipeline_mod.load_plan(tmp_path), edl, BRAND, **kw)
    assert spec.text == "real value"


# --------------------------------------------------------------------------
# Composition: a late overlay plays from its first frame
# --------------------------------------------------------------------------


def test_overlay_clip_is_shifted_to_its_window_start():
    plan = RenderPlan(edl=_edl(_clip(0, 8)), overlays=[
        Overlay(path=Path("late.mov"), start=2.5, end=5.0),
        Overlay(path=Path("early.mov"), start=0.0, end=2.0),
    ])
    _, graph, _ = build_filter_complex(plan)
    assert "[1:v]setpts=PTS-STARTPTS+2.500/TB[ovshift0]" in graph
    assert "ovshift1" not in graph  # a window starting at 0 needs no shift
