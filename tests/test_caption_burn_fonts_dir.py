"""Regression tests for brand-fonts-directory plumbing (Review-First Editing
Workflow spec section 9/10: "no silent system-wide installs") -- the ffmpeg
`subtitles` filter must gain a `fontsdir=` clause only when a fonts
directory is actually supplied, so libass can find brand fonts without
installing them system-wide."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.render.composition import CaptionBurn, RenderPlan, build_filter_complex


def _edl() -> EDL:
    return EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[EDLClip(source_file="a.mp4", source_in=0.0, source_out=2.0, timeline_in=0.0, timeline_out=2.0)],
    )


def test_no_fontsdir_clause_when_fonts_dir_is_none():
    plan = RenderPlan(edl=_edl(), captions=CaptionBurn(ass_path=Path("captions.ass")))
    _inputs, filter_complex, _labels = build_filter_complex(plan)
    assert "fontsdir=" not in filter_complex


def test_fontsdir_clause_present_when_fonts_dir_set():
    plan = RenderPlan(
        edl=_edl(), captions=CaptionBurn(ass_path=Path("captions.ass"), fonts_dir=Path("brands/acme/fonts"))
    )
    _inputs, filter_complex, _labels = build_filter_complex(plan)
    assert "fontsdir='brands/acme/fonts'" in filter_complex


def test_extract_frame_maps_the_unused_audio_output_to_the_null_muxer(monkeypatch, tmp_path):
    """Regression test: `build_filter_complex` always declares both a video
    AND an audio output pad ([vout][aout]), but `extract_frame` only wants a
    still image. ffmpeg refuses to build a filtergraph that leaves a
    declared output unmapped ("Error binding filtergraph inputs/outputs"),
    so the audio pad must be routed to the null muxer rather than dropped."""
    import subprocess

    from video_edit_agent.review.preview import extract_frame

    captured = {}

    image_path = tmp_path / "frame.jpg"

    def fake_run(cmd, timeout=None):
        captured["cmd"] = cmd
        image_path.touch()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("video_edit_agent.review.preview.run", fake_run)

    plan = RenderPlan(edl=_edl(), captions=CaptionBurn(ass_path=Path("captions.ass")))
    extract_frame(plan, 1.0, image_path)

    cmd = captured["cmd"]
    assert str(image_path) in cmd
    assert cmd[-5:] == ["-map", "[aout]", "-f", "null", "-"]
