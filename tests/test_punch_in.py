"""Baseline Recovery Milestone item 6: talking-head punch-ins/reframing,
ported from `majedphotos/video-ad-editor`'s `scripts/03_cut_zoom.py`."""
from __future__ import annotations

from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.editorial.punch_in import (
    ZOOM_LEVELS_CALM,
    ZOOM_LEVELS_NORMAL,
    plan_punch_ins,
)
from video_edit_agent.render.composition import RenderPlan, build_filter_complex


def _edl(n_clips: int) -> EDL:
    return EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[
            EDLClip(
                source_file=f"clip{i}.mp4", source_in=0.0, source_out=1.0,
                timeline_in=float(i), timeline_out=float(i + 1),
            )
            for i in range(n_clips)
        ],
    )


def test_plan_punch_ins_cycles_normal_zoom_levels_by_clip_index():
    edl = _edl(5)
    plan_punch_ins(edl, energy="medium")
    assert [c.zoom for c in edl.clips] == ZOOM_LEVELS_NORMAL[:5]


def test_plan_punch_ins_uses_calm_levels_for_low_energy_brand():
    edl = _edl(5)
    plan_punch_ins(edl, energy="low")
    assert [c.zoom for c in edl.clips] == ZOOM_LEVELS_CALM[:5]


def test_plan_punch_ins_leaves_a_single_clip_unzoomed():
    edl = _edl(1)
    plan_punch_ins(edl, energy="medium")
    assert edl.clips[0].zoom == 1.0


def test_zoomed_clip_gets_a_punch_in_crop_and_scale_stage_in_the_filter_graph():
    edl = _edl(2)
    edl.clips[1].zoom = 1.08
    plan = RenderPlan(edl=edl)
    _, filters, _ = build_filter_complex(plan)
    assert "scale=1080:1920:flags=lanczos,setsar=1" in filters
    crop_w = round(1080 / 1.08 / 2) * 2
    crop_h = round(1920 / 1.08 / 2) * 2
    assert f"crop={crop_w}:{crop_h}:" in filters


def test_unzoomed_clip_has_no_punch_in_scale_stage():
    edl = _edl(2)
    plan = RenderPlan(edl=edl)
    _, filters, _ = build_filter_complex(plan)
    assert "flags=lanczos" not in filters
