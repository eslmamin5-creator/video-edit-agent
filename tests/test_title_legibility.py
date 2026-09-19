"""Generic motion-title legibility rule: contrast-aware colour, backing plate,
subject/face-aware placement inside the safe zone. Synthetic brands and frame
analyses only -- nothing here depends on a real brand or clip."""
from __future__ import annotations

import numpy as np

from tests.conftest import make_transcript
from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.motion.director import build_motion_plan
from video_edit_agent.motion.legibility import (
    MIN_CONTRAST,
    contrast_ratio,
    plan_title_treatment,
    relative_luminance,
)
from video_edit_agent.subject.framing import GRID_H, GRID_W, FrameAnalysis

W, H = 1080, 1920


def _brand(**colors) -> Brand:
    palette = {"primary": "#0A3D62", "secondary": "#8E44AD", "accent": "#F1C40F"}
    palette.update(colors)
    return Brand.model_validate({"name": "acme", "colors": palette})


def _analysis(backdrop_hex: str, *, person: tuple[float, float, float, float] | None = None,
              face: tuple[float, float, float, float] | None = None) -> FrameAnalysis:
    lum = relative_luminance(backdrop_hex)
    luma = np.full((GRID_H, GRID_W), lum, dtype=np.float32)
    occ = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    if person is not None:
        x, y, w, h = person
        occ[int(y * GRID_H):int((y + h) * GRID_H), int(x * GRID_W):int((x + w) * GRID_W)] = 1.0
    return FrameAnalysis(occupancy=occ, luma=luma, face=face)


def _overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def test_contrast_ratio_matches_wcag_extremes():
    assert round(contrast_ratio(relative_luminance("#FFFFFF"), relative_luminance("#000000")), 1) == 21.0
    assert contrast_ratio(0.3, 0.3) == 1.0


def test_brand_colour_on_a_similar_backdrop_is_rejected_for_a_contrasting_brand_colour():
    # secondary (purple) on a purple wall fails; the accent clears it
    brand = _brand(secondary="#8E44AD")
    t = plan_title_treatment(brand, "HOOK", W, H, analysis=_analysis("#5A30C0"))
    assert t.foreground != "#8E44AD"
    assert t.foreground_source == "brand.accent"
    assert t.contrast is not None and t.contrast >= MIN_CONTRAST
    assert t.plate_color is None


def test_when_no_brand_colour_clears_the_bare_backdrop_a_brand_plate_is_used():
    t = plan_title_treatment(_brand(), "HOOK", W, H, analysis=_analysis("#6B3FE0"))
    assert t.plate_color == "#0A3D62"
    assert t.contrast is not None and t.contrast >= MIN_CONTRAST


def test_every_chosen_pair_clears_the_ratio_and_uses_brand_or_neutral_colours_only():
    brand = _brand()
    for wall in ("#6B3FE0", "#FFFFFF", "#000000", "#808080", "#0A3D62"):
        t = plan_title_treatment(brand, "HOOK", W, H, analysis=_analysis(wall))
        allowed = {brand.colors.primary, brand.colors.secondary, brand.colors.accent, "#FFFFFF", "#000000"}
        assert t.foreground in allowed
        if t.plate_color:
            assert t.plate_color in allowed
        if t.contrast is not None:
            assert t.contrast >= MIN_CONTRAST


def test_title_overlapping_the_subject_gets_a_backing_plate_from_the_brand():
    brand = _brand()
    # the person fills the whole safe band so no clear position exists
    t = plan_title_treatment(brand, "HOOK", W, H, analysis=_analysis("#6B3FE0", person=(0.0, 0.0, 1.0, 1.0)))
    assert t.plate_color in {brand.colors.primary, brand.colors.secondary, brand.colors.accent}
    assert t.plate_opacity > 0
    assert t.contrast is not None and t.contrast >= MIN_CONTRAST
    assert t.foreground.lower() != t.plate_color.lower()


def test_title_avoids_the_face_and_stays_inside_the_safe_zone():
    brand = _brand()
    face = (0.30, 0.14, 0.40, 0.16)  # sits right where a top-anchored title would go
    t = plan_title_treatment(
        brand, "HOOK TEXT", W, H,
        analysis=_analysis("#6B3FE0", person=(0.2, 0.14, 0.6, 0.86), face=face),
    )
    assert not _overlap(t.box, face)
    x, y, w, h = t.box
    assert y >= 0.12 - 1e-6  # default top safe margin
    assert y + h <= 1.0 - 0.30 + 1e-6  # above the caption band
    assert 0.06 <= x and x + w <= 1.0 - 0.06 + 1e-6  # side margins


def test_placement_respects_brand_safe_zone_overrides():
    brand = Brand.model_validate({"name": "acme", "safe_zones": {"top": 0.25, "bottom": 0.35, "side": 0.1}})
    t = plan_title_treatment(brand, "HOOK", W, H, analysis=_analysis("#404040"))
    assert t.box[1] >= 0.25 - 1e-6
    assert t.box[1] + t.box[3] <= 1.0 - 0.35 + 1e-6
    assert t.max_width_px == round(W * 0.8)


def test_no_analysis_is_conservative_top_band_on_a_plate():
    t = plan_title_treatment(_brand(), "HOOK", W, H, analysis=None)
    assert t.plate_color is not None
    assert t.box[1] >= 0.12 - 1e-6
    assert any("no footage analysis" in r for r in t.reasons)


def test_outline_and_shadow_follow_brand_motion_avoid_list():
    brand = Brand.model_validate({
        "name": "acme", "colors": {"primary": "#0A3D62", "secondary": "#8E44AD", "accent": "#F1C40F"},
        "motion": {"avoid": ["text_outline", "text_shadow"]},
    })
    t = plan_title_treatment(brand, "HOOK", W, H, analysis=_analysis("#6B3FE0"))
    assert t.outline_color is None and t.outline_px == 0
    assert t.shadow == ""
    assert t.as_props()["outline"] is None


def test_brand_without_accent_never_invents_one():
    brand = Brand.model_validate({"name": "acme", "colors": {"primary": "#0A3D62", "secondary": "#8E44AD"}})
    t = plan_title_treatment(brand, "HOOK", W, H, analysis=_analysis("#6B3FE0"))
    assert t.foreground in {"#0A3D62", "#8E44AD", "#FFFFFF", "#000000"}
    assert t.foreground_source != "brand.accent"


def test_treatment_props_shape_for_the_renderer():
    t = plan_title_treatment(_brand(), "HOOK", W, H, analysis=None)
    props = t.as_props()
    assert set(props) == {"color", "centerY", "maxWidth", "plate", "outline", "shadow"}
    assert props["plate"] is not None and {"color", "opacity", "padding", "radius"} <= set(props["plate"])
    assert 0 < props["centerY"] < H


def test_director_attaches_the_treatment_to_the_hook_and_keeps_text():
    transcript = make_transcript([("HOOK TEXT", 0.0, 0.8)])
    edl = EDL(clips=[EDLClip(source_file="a.mp4", source_in=0.0, source_out=5.0, timeline_in=0.0, timeline_out=5.0)])
    specs = build_motion_plan(edl, transcript, brand=_brand(), analysis=_analysis("#6B3FE0"))
    hook = specs[0]
    assert hook.text == "HOOK TEXT"
    assert hook.extra["color"] != "#8E44AD"
    assert hook.extra["legibility"]["reasons"]
    # no brand -> renderer defaults, no invented treatment
    assert build_motion_plan(edl, transcript)[0].extra == {}
