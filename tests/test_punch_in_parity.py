"""Punch-in parity: the final render and every preview play ONE reframe plan
through ONE implementation (`render/reframe.py`), and the plan is an eased zoom
that never steps at a cut."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.core.schemas import EDL, EDLClip, Reframe, ZoomRamp
from video_edit_agent.editorial.punch_in import RAMP_HALF_S, plan_punch_ins
from video_edit_agent.render import micro_preview as mp
from video_edit_agent.render.composition import RenderPlan, build_filter_complex
from video_edit_agent.render.export import resolve_preset
from video_edit_agent.render.ffmpeg import render
from video_edit_agent.render.reframe import (
    crop_window,
    reframe_filters,
    resolve_reframe,
    safe_anchor,
    zoom_at,
    zoom_expr,
)


def _edl(n: int, *, w: int = 1080, h: int = 1920, seconds: float = 2.0, source: str = "src.mp4") -> EDL:
    return EDL(width=w, height=h, fps=30.0, clips=[
        EDLClip(source_file=source, source_in=i * seconds, source_out=(i + 1) * seconds,
                timeline_in=i * seconds, timeline_out=(i + 1) * seconds, caption_refs=[f"s{i}"])
        for i in range(n)
    ])


def _eval_expr(expr: str, t: float) -> float:
    return eval(expr, {"clip": lambda x, lo, hi: min(max(x, lo), hi), "t": t})


# --- the plan: eased, centred on the cut, continuous --------------------------


def test_plan_is_continuous_at_every_cut_and_never_steps():
    edl = _edl(6)
    plan_punch_ins(edl)
    for prev, clip in zip(edl.clips, edl.clips[1:]):
        end_of_prev = zoom_at(resolve_reframe(prev) or Reframe(), prev.duration)
        start_of_next = zoom_at(resolve_reframe(clip) or Reframe(), 0.0)
        assert end_of_prev == pytest.approx(start_of_next, abs=1e-9)
        if prev.zoom != clip.zoom:
            assert min(prev.zoom, clip.zoom) < end_of_prev < max(prev.zoom, clip.zoom)  # mid-move at the join


def test_each_clip_settles_on_its_planned_zoom_and_moves_are_centred_on_the_cut():
    edl = _edl(3)
    plan_punch_ins(edl)
    second = edl.clips[1]
    rf = second.reframe
    assert rf is not None
    head = rf.ramps[0]
    assert (head.start_s, head.end_s) == (-RAMP_HALF_S, RAMP_HALF_S)
    assert zoom_at(rf, 1.0) == pytest.approx(second.zoom)  # mid-clip: settled
    tail = rf.ramps[-1]
    assert (tail.start_s, tail.end_s) == (second.duration - RAMP_HALF_S, second.duration + RAMP_HALF_S)


def test_easing_has_no_velocity_jump_at_either_end_of_a_move():
    rf = Reframe(zoom_start=1.0, ramps=[ZoomRamp(start_s=1.0, end_s=1.6, zoom_to=1.2)])
    eps = 1e-3
    assert zoom_at(rf, 1.0 + eps) - zoom_at(rf, 1.0) < 1e-5
    assert zoom_at(rf, 1.6) - zoom_at(rf, 1.6 - eps) < 1e-5
    assert zoom_at(rf, 0.0) == 1.0 and zoom_at(rf, 5.0) == pytest.approx(1.2)


def test_single_clip_edl_is_left_unzoomed():
    edl = _edl(1)
    plan_punch_ins(edl)
    assert edl.clips[0].reframe is None and resolve_reframe(edl.clips[0]) is None


def test_legacy_static_zoom_resolves_to_a_rampless_reframe():
    clip = _edl(1).clips[0].model_copy(update={"zoom": 1.08})
    rf = resolve_reframe(clip)
    assert rf is not None and rf.ramps == [] and rf.zoom_start == 1.08


# --- one transform: ffmpeg expression == Python twin ---------------------------


def test_ffmpeg_zoom_expression_matches_the_python_transform():
    edl = _edl(4)
    plan_punch_ins(edl)
    for clip in edl.clips:
        rf = resolve_reframe(clip)
        if rf is None:
            continue
        expr = zoom_expr(rf)
        for t in np.linspace(-0.2, clip.duration + 0.4, 25):
            assert _eval_expr(expr, float(t)) == pytest.approx(zoom_at(rf, float(t)), abs=1e-3)


def test_crop_window_keeps_the_planned_anchor():
    rf = Reframe(zoom_start=1.25)
    x, y, w, h = crop_window(rf, 0.0, 1000, 2000)
    assert (w, h) == pytest.approx((800.0, 1600.0))
    assert (x, y) == pytest.approx((100.0, 120.0))


# --- final render uses the shared eased path -----------------------------------


def test_final_render_graph_uses_the_shared_eased_zoom_for_a_moving_clip():
    edl = _edl(3)
    plan_punch_ins(edl)
    _, graph, _ = build_filter_complex(RenderPlan(edl=edl))
    rf = resolve_reframe(edl.clips[1])
    assert rf is not None and rf.ramps
    assert "eval=frame" in graph and "*(3-2*" in graph
    lines = reframe_filters(rf, "fr1", "v1", 1080, 1920, 30.0, edl.clips[1].duration, "1")
    for line in lines:  # the graph embeds exactly what the shared module emits
        assert line in graph


def test_static_reframe_still_renders_as_a_plain_crop():
    edl = _edl(2)
    edl.clips[1].zoom = 1.08
    _, graph, _ = build_filter_complex(RenderPlan(edl=edl))
    assert "eval=frame" not in graph
    assert f"crop={round(1080 / 1.08 / 2) * 2}:{round(1920 / 1.08 / 2) * 2}:" in graph


def test_preview_and_final_graphs_contain_identical_reframe_filters():
    """A preview window aligned to a clip start re-bases the plan by zero, so the
    reframe filters of that clip must equal the final render's, character for character."""
    edl = _edl(3)
    plan_punch_ins(edl)
    sub = mp.slice_edl(edl, 2.0, 6.0, zoom=None)  # clips 1 and 2, whole
    _, final_graph, _ = build_filter_complex(RenderPlan(edl=edl))
    _, preview_graph, _ = build_filter_complex(RenderPlan(edl=sub))
    for final_i, preview_i in ((1, 0), (2, 1)):
        rf_final, rf_prev = resolve_reframe(edl.clips[final_i]), resolve_reframe(sub.clips[preview_i])
        assert rf_final == rf_prev
        d = edl.clips[final_i].duration
        a = reframe_filters(rf_final, "fr", "v", 1080, 1920, 30.0, d, "x")
        b = reframe_filters(rf_prev, "fr", "v", 1080, 1920, 30.0, d, "x")
        assert a == b
    assert "eval=frame" in preview_graph and "eval=frame" in final_graph


def test_slicing_mid_clip_rebases_the_plan_without_changing_any_frame_zoom():
    edl = _edl(3)
    plan_punch_ins(edl)
    clip = edl.clips[1]
    shift = 0.8
    sub = mp.slice_edl(edl, clip.source_in + shift, clip.source_out, zoom=None)
    orig, rebased = resolve_reframe(clip), resolve_reframe(sub.clips[0])
    for t in np.linspace(0.0, clip.duration - shift, 20):
        assert zoom_at(rebased, float(t)) == pytest.approx(zoom_at(orig, float(t) + shift), abs=1e-9)


def test_neutral_preview_slice_drops_any_reframe():
    edl = _edl(3)
    plan_punch_ins(edl)
    sub = mp.slice_edl(edl, 0.0, 6.0)
    assert all(c.reframe is None and c.zoom == 1.0 for c in sub.clips)
    assert resolve_reframe(sub.clips[1]) is None


def test_punch_in_window_is_picked_around_the_biggest_zoom_change():
    edl = _edl(4)
    for clip, zoom in zip(edl.clips, (1.0, 1.04, 1.14, 1.10)):
        clip.zoom = zoom
    start, end, cut = mp.pick_punch_in_window(edl, before_s=1.0, after_s=3.0)
    assert cut == 4.0 and (start, end) == (3.0, 7.0)
    assert mp.pick_punch_in_window(_edl(3)) is None


# --- face-safe framing -----------------------------------------------------------


def test_face_safe_anchor_moves_only_as_far_as_needed():
    face = (0.30, 0.05, 0.40, 0.22)  # near the top of the frame
    rf = Reframe(zoom_start=1.5, anchor_y=0.9, face_box=face)
    ax, ay = safe_anchor(rf)
    assert ax == pytest.approx(0.5)  # already fine horizontally
    assert ay < 0.9  # pulled up so the face is not cropped away
    _, y, _w, h = crop_window(rf, 0.0, 1000, 2000)
    assert y / 2000 <= face[1] - rf.face_margin + 1e-9
    assert (y + h) / 2000 >= face[1] + face[3] + rf.face_margin - 1e-9


def test_face_safe_anchor_uses_the_tightest_zoom_of_the_move():
    face = (0.35, 0.05, 0.30, 0.15)
    tight = Reframe(zoom_start=1.0, ramps=[ZoomRamp(start_s=0, end_s=1, zoom_to=1.4)], anchor_y=0.9, face_box=face)
    loose = Reframe(zoom_start=1.0, ramps=[ZoomRamp(start_s=0, end_s=1, zoom_to=1.05)], anchor_y=0.9, face_box=face)
    assert safe_anchor(tight)[1] < safe_anchor(loose)[1] <= 0.9


def test_a_face_bigger_than_the_window_is_centred_and_no_face_changes_nothing():
    huge = Reframe(zoom_start=2.0, anchor_y=0.1, face_box=(0.1, 0.0, 0.8, 0.9))
    assert safe_anchor(huge)[1] == pytest.approx(0.45, abs=0.2)
    assert safe_anchor(Reframe(zoom_start=1.3, anchor_y=0.3)) == (0.5, 0.3)


def test_planner_passes_the_face_box_to_every_reframe():
    edl = _edl(3)
    face = (0.3, 0.1, 0.4, 0.2)
    plan_punch_ins(edl, face_box=face)
    assert all(c.reframe is not None and c.reframe.face_box == face for c in edl.clips)


# --- real pixels: the preview slice is the final render ---------------------------


def _frame(path: Path, t: float, w: int, h: int) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True, timeout=60).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w).astype(np.float32)


@requires_ffmpeg
def test_preview_frames_equal_final_render_frames(tmp_path: Path):
    w, h = 180, 320
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=s={w}x{h}:d=4:r=30",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=4", "-shortest", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-g", "1", "-crf", "12", "-c:a", "aac", str(src)],
        check=True, timeout=60)
    edl = EDL(width=w, height=h, fps=30.0, clips=[
        EDLClip(source_file=str(src), source_in=0.0, source_out=2.0, timeline_in=0.0, timeline_out=2.0),
        EDLClip(source_file=str(src), source_in=2.0, source_out=4.0, timeline_in=2.0, timeline_out=4.0),
    ])
    plan_punch_ins(edl)
    edl.clips[1].zoom = 1.3
    edl.clips[0].reframe = Reframe(zoom_start=1.0, ramps=[ZoomRamp(start_s=1.7, end_s=2.3, zoom_to=1.3)])
    edl.clips[1].reframe = Reframe(zoom_start=1.0, ramps=[ZoomRamp(start_s=-0.3, end_s=0.3, zoom_to=1.3)])

    final = render(RenderPlan(edl=edl), tmp_path / "final.mp4", resolve_preset("reel"), crf=12)
    preview = mp.render_punch_in(edl, (1.0, 3.5), tmp_path / "preview.mp4")

    for source_t in (1.2, 1.9, 2.0, 2.15, 2.6, 3.2):
        f = _frame(final, source_t, w, h)
        p = _frame(preview, source_t - 1.0, w, h)
        assert float(np.abs(f - p).mean()) < 6.0, f"preview differs from the final render at source t={source_t}"

    # and the zoom really is eased: the mid-move frame is neither the old nor the new framing
    before, mid, after = (_frame(final, t, w, h) for t in (1.5, 2.0, 2.6))
    assert float(np.abs(mid - before).mean()) > 1.0 and float(np.abs(mid - after).mean()) > 1.0
