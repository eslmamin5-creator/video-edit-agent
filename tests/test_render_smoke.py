"""End-to-end ffmpeg render smoke test (spec sections 12, 14, 47) — proves the
argument-array filter_complex actually produces a playable file, using a
synthetic lavfi source so the test needs no binary fixture."""
from __future__ import annotations

from pathlib import Path

from tests.conftest import requires_ffmpeg
from video_edit_agent.captions.engine import write_captions
from video_edit_agent.captions.styles import PRESETS
from video_edit_agent.core.media import probe
from video_edit_agent.render.composition import CaptionBurn, RenderPlan
from video_edit_agent.render.export import PRESETS as EXPORT_PRESETS
from video_edit_agent.render.ffmpeg import render


@requires_ffmpeg
def test_render_produces_a_valid_output_file(tmp_path: Path, sample_video, sample_edl, sample_transcript):
    ass_path = tmp_path / "captions.ass"
    srt_path = tmp_path / "captions.srt"
    write_captions(sample_transcript, sample_edl, PRESETS["minimal"], ass_path, srt_path)

    plan = RenderPlan(edl=sample_edl, overlays=[], captions=CaptionBurn(ass_path=ass_path))
    output = tmp_path / "final.mp4"
    result = render(plan, output, EXPORT_PRESETS["reel"])

    assert result == output
    assert output.exists() and output.stat().st_size > 0

    info = probe(output)
    assert info.has_audio
    assert info.width == sample_edl.width
    assert info.height == sample_edl.height
    assert abs(info.duration - sample_edl.total_duration) < 1.0


@requires_ffmpeg
def test_render_without_captions_still_produces_output(tmp_path: Path, sample_video, sample_edl):
    plan = RenderPlan(edl=sample_edl, overlays=[], captions=None)
    output = tmp_path / "final_no_captions.mp4"
    render(plan, output, EXPORT_PRESETS["reel"])
    assert output.exists() and output.stat().st_size > 0
