"""Baseline Recovery Milestone item 1 (Editor workflow): resolved B-roll from
`broll.planner.plan_broll()` must actually reach the `RenderPlan` handed to
the renderer, not just `broll_plan.json`. This test drives the real
`run_pipeline()` control flow but stubs out the heavy/network-ish stages
(probe, transcription, EDL building, motion, actual ffmpeg render, QA) so it
runs fast and fully offline while still proving the wiring fix end-to-end."""
from __future__ import annotations

from pathlib import Path

import pytest

from video_edit_agent.core import pipeline as pipeline_mod
from video_edit_agent.core.media import MediaInfo
from video_edit_agent.core.schemas import EDL, BrollPlanItem, BrollSourceKind
from video_edit_agent.qa.repair import RepairResult


@pytest.fixture
def stubbed_pipeline(monkeypatch, tmp_path: Path, sample_transcript, sample_edl: EDL):
    captured: dict = {}

    monkeypatch.setattr(pipeline_mod, "probe", lambda path, *_a, **_k: MediaInfo(
        path=Path(path), width=sample_edl.width, height=sample_edl.height, fps=sample_edl.fps,
        duration=sample_edl.total_duration, has_audio=True, video_codec="h264", audio_codec="aac",
    ))
    monkeypatch.setattr(pipeline_mod, "extract_audio", lambda *_a, **_k: None)

    class DummyRouter:
        def __init__(self, *_a, **_k):
            pass

        def transcribe(self, *_a, **_k):
            return sample_transcript

    monkeypatch.setattr(pipeline_mod, "TranscriptionRouter", DummyRouter)
    monkeypatch.setattr(pipeline_mod, "save_transcript", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "build_edl", lambda *_a, **_k: (sample_edl, []))
    monkeypatch.setattr(pipeline_mod, "write_takes_packed", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "validate_edl", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "write_captions", lambda *_a, **_k: None)

    resolved_item = BrollPlanItem(
        timeline_start=0.0,
        timeline_end=sample_edl.total_duration,
        purpose="context",
        spoken_concept="an office",
        recommended_visual="office establishing shot",
        source=BrollSourceKind.LOCAL_LIBRARY,
        asset_path=str(tmp_path / "office.mp4"),
    )
    monkeypatch.setattr(pipeline_mod, "plan_broll", lambda *_a, **_k: [resolved_item])
    monkeypatch.setattr(pipeline_mod, "build_motion_plan", lambda *_a, **_k: [])

    def fake_render(plan, output_path, _preset):
        captured["plan"] = plan
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        return output_path

    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", fake_render)
    monkeypatch.setattr(pipeline_mod, "run_technical_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "run_language_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "run_visual_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "run_brand_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(
        pipeline_mod, "run_repair_loop",
        lambda report, collect, repair_handlers: RepairResult(report=report, iterations_used=0, repairs_applied=[]),
    )
    return captured, resolved_item


def test_resolved_broll_reaches_the_render_plan_overlays(monkeypatch, stubbed_pipeline, tmp_path: Path):
    captured, resolved_item = stubbed_pipeline
    # Isolate this B-roll-focused test from the brand logo overlay (a
    # separate concern, covered by test_brand_logo_reaches_the_render_plan_
    # overlays / test_no_logo_file_contributes_no_overlay below) -- the
    # "default" brand now ships a real logos/logo.png test asset.
    monkeypatch.setattr(pipeline_mod, "resolve_brand_logo", lambda *_a, **_k: None)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-source")

    result = pipeline_mod.run_pipeline(
        source_video, enable_broll=True, enable_motion=False, offline=True, review=False,
    )

    assert result.final_output is not None
    assert result.broll_plan == [resolved_item]

    plan = captured["plan"]
    assert len(plan.overlays) == 1
    overlay = plan.overlays[0]
    assert overlay.path == Path(resolved_item.asset_path)
    assert overlay.start == resolved_item.timeline_start
    assert overlay.end == resolved_item.timeline_end
    assert overlay.scale_to_canvas is True


def test_unresolved_broll_contributes_no_overlay(monkeypatch, stubbed_pipeline, tmp_path: Path):
    captured, _ = stubbed_pipeline
    none_item = BrollPlanItem(
        timeline_start=0.0, timeline_end=1.0, purpose="context",
        spoken_concept="x", recommended_visual="x", source=BrollSourceKind.NONE,
    )
    monkeypatch.setattr(pipeline_mod, "plan_broll", lambda *_a, **_k: [none_item])
    monkeypatch.setattr(pipeline_mod, "resolve_brand_logo", lambda *_a, **_k: None)

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-source")
    pipeline_mod.run_pipeline(source_video, enable_broll=True, enable_motion=False, offline=True, review=False)

    assert captured["plan"].overlays == []


def test_brand_logo_reaches_the_render_plan_overlays(monkeypatch, stubbed_pipeline, tmp_path: Path, sample_edl: EDL):
    captured, _ = stubbed_pipeline
    monkeypatch.setattr(pipeline_mod, "plan_broll", lambda *_a, **_k: [])

    from video_edit_agent.brand import loader as brand_loader

    brands_root = tmp_path / "brands"
    brand_loader.init_brand("acme", root=brands_root)
    logo_path = brands_root / "acme" / "logos" / "logo.png"
    logo_path.write_bytes(b"fake-png")
    monkeypatch.setattr(pipeline_mod, "load_brand", lambda name, root=None: brand_loader.load_brand(name, root=brands_root))
    monkeypatch.setattr(pipeline_mod, "resolve_brand_logo", lambda brand, root=None: brand_loader.resolve_brand_logo(brand, root=brands_root))

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-source")
    result = pipeline_mod.run_pipeline(
        source_video, brand_name="acme", enable_broll=False, enable_motion=False, offline=True,
        review=False, logo_mode="persistent_bug",
    )

    assert result.final_output is not None
    plan = captured["plan"]
    assert len(plan.overlays) == 1
    overlay = plan.overlays[0]
    assert overlay.path == logo_path
    assert overlay.start == 0.0
    assert overlay.end == sample_edl.total_duration
    assert overlay.scale_width is not None


def test_no_logo_file_contributes_no_overlay(monkeypatch, stubbed_pipeline, tmp_path: Path):
    captured, _ = stubbed_pipeline
    monkeypatch.setattr(pipeline_mod, "plan_broll", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "resolve_brand_logo", lambda *_a, **_k: None)

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-source")
    pipeline_mod.run_pipeline(source_video, enable_broll=False, enable_motion=False, offline=True, review=False)

    assert captured["plan"].overlays == []
