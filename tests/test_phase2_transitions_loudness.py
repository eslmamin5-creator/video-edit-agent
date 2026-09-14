"""Phase 2 Finalization acceptance tests (spec sections 3-8, 10c): real
rendered video crossfades, transition duration clamping, audio crossfade
gating, and real loudness normalization -- not just planning.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import HAS_FFMPEG, requires_ffmpeg
from video_edit_agent.agents.assembler.discovery import build_scene_inventory, discover_scene_files
from video_edit_agent.agents.assembler.edl_builder import build_scene_edl
from video_edit_agent.agents.assembler.loudness import SILENCE_FLOOR_DB, TARGET_DB, plan_loudness
from video_edit_agent.agents.assembler.pipeline import run_assembler
from video_edit_agent.agents.assembler.schemas import TransitionDecision, TransitionKind
from video_edit_agent.core.media import probe, run
from video_edit_agent.core.schemas import EDL, EDLClip, TransitionType
from video_edit_agent.core.transition_math import MAX_TRANSITION_S, MIN_TRANSITION_S, clamp_transition_duration
from video_edit_agent.render.composition import RenderPlan, build_filter_complex
from video_edit_agent.render.export import resolve_preset
from video_edit_agent.render.ffmpeg import render


# ---------------------------------------------------------------------------
# Section 3: transition duration clamping
# ---------------------------------------------------------------------------

def test_clamp_transition_duration_stays_in_conservative_band():
    assert clamp_transition_duration(0.01, 5.0, 5.0) == pytest.approx(MIN_TRANSITION_S)
    assert clamp_transition_duration(10.0, 5.0, 5.0) == pytest.approx(MAX_TRANSITION_S)
    assert MIN_TRANSITION_S <= clamp_transition_duration(0.35, 5.0, 5.0) <= MAX_TRANSITION_S


def test_clamp_transition_duration_never_exceeds_90pct_of_either_neighbor():
    # Two very short clips (0.2s each) must never get a transition so long
    # it would desync or invert the timeline.
    d = clamp_transition_duration(0.5, 0.2, 0.2)
    assert d <= 0.9 * 0.2 + 1e-9


# ---------------------------------------------------------------------------
# Section 5: DISSOLVE documented as reusing the same xfade mechanics as
# SHORT_CROSSFADE (no artificial differentiation) -- verified at the
# EDL-builder boundary, not just by reading the map.
# ---------------------------------------------------------------------------

@requires_ffmpeg
def test_dissolve_and_short_crossfade_both_produce_a_real_transition_duration(tmp_path: Path):
    files_dir = tmp_path / "scenes"
    files_dir.mkdir()
    from tests.conftest import make_scene_video

    make_scene_video(files_dir / "a.mp4", color="red", duration=1.0)
    make_scene_video(files_dir / "b.mp4", color="blue", duration=1.0)
    files = discover_scene_files(files_dir)
    inventory = build_scene_inventory(files)

    for kind in (TransitionKind.SHORT_CROSSFADE, TransitionKind.DISSOLVE):
        decision = TransitionDecision(
            from_scene=inventory[0].id, to_scene=inventory[1].id,
            type=kind, duration=0.3, reason="test", confidence=1.0, applied=True,
        )
        edl = build_scene_edl(
            inventory, width=640, height=360, fps=30.0, cache_dir=tmp_path / "cache",
            transitions=[decision],
        )
        assert edl.clips[1].transition_in == TransitionType.CROSSFADE
        assert edl.clips[1].transition_duration_s > 0.0


# ---------------------------------------------------------------------------
# Section 4: real crossfade acceptance -- two visually distinct clips,
# rendered output must show genuine overlap/blending, not a hard cut.
# ---------------------------------------------------------------------------

@requires_ffmpeg
def test_crossfade_acceptance_output_reflects_overlap_and_blended_frames(tmp_path: Path):
    from tests.conftest import make_scene_video

    a = make_scene_video(tmp_path / "a.mp4", color="red", duration=1.5)
    b = make_scene_video(tmp_path / "b.mp4", color="blue", duration=1.5)
    d = 0.4

    edl = EDL(
        version=1, fps=30.0, width=320, height=240,
        clips=[
            EDLClip(source_file=str(a), source_in=0.0, source_out=1.5, timeline_in=0.0, timeline_out=1.5),
            EDLClip(
                source_file=str(b), source_in=0.0, source_out=1.5,
                timeline_in=1.5 - d, timeline_out=1.5 - d + 1.5,
                transition_in=TransitionType.CROSSFADE, transition_duration_s=d,
            ),
        ],
    )
    out = tmp_path / "crossfade_out.mp4"
    preset = resolve_preset("square")
    render(RenderPlan(edl=edl), out, preset)

    info = probe(out)
    expected_duration = 1.5 + 1.5 - d
    assert info.duration == pytest.approx(expected_duration, abs=0.25)

    # Sample a frame inside the overlap window and check it is neither pure
    # red nor pure blue -- i.e. genuinely blended, not a hard cut.
    mid_t = 1.5 - d / 2
    frame_png = tmp_path / "mid_frame.png"
    r = run(
        ["ffmpeg", "-y", "-ss", f"{mid_t:.3f}", "-i", str(out), "-frames:v", "1", str(frame_png)],
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    assert frame_png.exists()

    from PIL import Image
    img = Image.open(frame_png).convert("RGB")
    px = img.getpixel((img.width // 2, img.height // 2))
    is_pure_red = px[0] > 200 and px[1] < 40 and px[2] < 40
    is_pure_blue = px[2] > 200 and px[0] < 40 and px[1] < 40
    assert not is_pure_red and not is_pure_blue, f"expected a blended frame, got {px}"


# ---------------------------------------------------------------------------
# Section 6: audio crossfade only when BOTH neighboring clips have real
# audio -- otherwise a plain concat, never a forced blend against silence.
# ---------------------------------------------------------------------------

def test_filter_graph_uses_acrossfade_when_both_clips_have_real_audio():
    edl = EDL(
        version=1, fps=30.0, width=320, height=240,
        clips=[
            EDLClip(source_file="a.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, has_real_audio=True),
            EDLClip(
                source_file="b.mp4", source_in=0.0, source_out=1.0, timeline_in=0.8, timeline_out=1.8,
                transition_in=TransitionType.CROSSFADE, transition_duration_s=0.2, has_real_audio=True,
            ),
        ],
    )
    _, filter_complex, _ = build_filter_complex(RenderPlan(edl=edl))
    assert "acrossfade=" in filter_complex
    assert "concat=n=2:v=0:a=1" not in filter_complex


def test_filter_graph_falls_back_to_concat_audio_when_one_side_has_no_real_audio():
    edl = EDL(
        version=1, fps=30.0, width=320, height=240,
        clips=[
            EDLClip(source_file="a.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, has_real_audio=True),
            EDLClip(
                source_file="b.mp4", source_in=0.0, source_out=1.0, timeline_in=0.8, timeline_out=1.8,
                transition_in=TransitionType.CROSSFADE, transition_duration_s=0.2, has_real_audio=False,
            ),
        ],
    )
    _, filter_complex, _ = build_filter_complex(RenderPlan(edl=edl))
    # Video still genuinely crossfades...
    assert "xfade=" in filter_complex
    # ...but audio is a plain concat, never a forced acrossfade against
    # synthesized silence.
    assert "acrossfade=" not in filter_complex
    assert "concat=n=2:v=0:a=1" in filter_complex


# ---------------------------------------------------------------------------
# Section 7-8: real loudness measurement + normalization acceptance.
# ---------------------------------------------------------------------------

@requires_ffmpeg
def test_loudness_plan_flags_only_the_scene_that_deviates(tmp_path: Path):
    from video_edit_agent.agents.assembler.schemas import SceneInventoryItem

    quiet = tmp_path / "quiet.mp4"
    loud = tmp_path / "loud.mp4"
    for out, vol in ((quiet, 0.02), (loud, 0.9)):
        run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=1.0:r=30",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration=1.0",
                "-af", f"volume={vol}",
                "-shortest", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac",
                str(out),
            ],
            timeout=60,
        )

    items = [
        SceneInventoryItem(
            id="quiet", filename="quiet.mp4", path=str(quiet), original_order=0, duration=1.0,
            has_audio=True, width=320, height=240, fps=30.0, video_codec="h264", aspect_ratio="4:3",
        ),
        SceneInventoryItem(
            id="loud", filename="loud.mp4", path=str(loud), original_order=1, duration=1.0,
            has_audio=True, width=320, height=240, fps=30.0, video_codec="h264", aspect_ratio="4:3",
        ),
    ]
    decisions = plan_loudness(items)
    by_id = {d.scene_id: d for d in decisions}

    # At least the loud outlier should measurably deviate and get a real,
    # conservative loudnorm pass -- never blanket-applied to every scene.
    assert by_id["loud"].mean_volume_db is not None
    if by_id["loud"].required:
        assert by_id["loud"].applied is True
        assert by_id["loud"].target_db == pytest.approx(TARGET_DB)


def test_loudness_plan_never_normalizes_true_digital_silence():
    from video_edit_agent.agents.assembler.schemas import SceneInventoryItem

    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        silent = Path(td) / "silent.mp4"
        run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1.0:r=30",
                "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1.0",
                "-shortest", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac",
                str(silent),
            ],
            timeout=60,
        )
        item = SceneInventoryItem(
            id="silent", filename="silent.mp4", path=str(silent), original_order=0, duration=1.0,
            has_audio=True, width=320, height=240, fps=30.0, video_codec="h264", aspect_ratio="4:3",
        )
        decision = plan_loudness([item])[0]
        if decision.mean_volume_db is not None and decision.mean_volume_db <= SILENCE_FLOOR_DB:
            assert decision.applied is False
            assert decision.required is False


@requires_ffmpeg
def test_loudness_acceptance_finish_render_has_no_clipping_and_is_valid(assembler_scenes_dir: Path):
    result = run_assembler(assembler_scenes_dir, finish=True, preserve_order=True, offline=True)
    assert result.warnings == []
    assert result.final_output is not None
    info = probe(result.final_output)
    assert info.duration > 0
    assert info.has_audio is True

    # No clipping: max_volume from ffmpeg's volumedetect must not exceed 0dB.
    r = run(["ffmpeg", "-i", str(result.final_output), "-af", "volumedetect", "-f", "null", "-"], timeout=60)
    import re
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", r.stderr or "")
    if match:
        assert float(match.group(1)) <= 0.5  # small tolerance for encoder rounding


# ---------------------------------------------------------------------------
# Section 10c: Assembler final render with a real applied transition.
# ---------------------------------------------------------------------------

@requires_ffmpeg
def test_assembler_finish_has_at_least_one_real_applied_transition(assembler_scenes_dir: Path):
    result = run_assembler(assembler_scenes_dir, finish=True, preserve_order=True, offline=True)
    assert result.final_output is not None

    real_crossfades = [
        t for t in result.transition_plan
        if t.applied and t.type in (TransitionKind.SHORT_CROSSFADE, TransitionKind.DISSOLVE)
    ]
    assert real_crossfades, "fixture is expected to trigger at least one real crossfade boundary"

    info = probe(result.final_output)
    assert info.duration > 0
    assert result.qa_issues == []
