"""Creator QA tests (Creator spec section 16): local, deterministic checks
only -- no cloud vision.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.agents.creator.qa import (
    check_aspect_ratio,
    check_brand_mismatch,
    check_broken_motion_assets,
    check_missing_assets,
    check_overlapping_items,
    check_render_output,
    check_scene_durations,
    check_scene_gaps,
    check_unsafe_text_zones,
    check_visual_coverage,
    run_creator_qa,
)
from video_edit_agent.agents.creator.schemas import Scene, ScenePurpose
from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import AnimationKind, AnimationSpec, MotionEngine, MotionPlanItem, QASeverity
from video_edit_agent.core.timeline import MasterTimeline, Provenance, ProviderKind, TimelineItem, TrackType


def _scene(id_="scene-01", duration=3.0, purpose=ScenePurpose.HOOK, text="hi") -> Scene:
    return Scene(
        id=id_, script_segment=text, purpose=purpose, estimated_duration=duration,
        on_screen_text=text, voice_over_text=text,
    )


def test_check_scene_durations_flags_zero_duration():
    scenes = [_scene(duration=0.0)]
    issues = check_scene_durations(scenes)
    assert len(issues) == 1
    assert issues[0].severity == QASeverity.ERROR


def test_check_scene_durations_passes_for_positive_duration():
    scenes = [_scene(duration=2.0)]
    assert check_scene_durations(scenes) == []


def test_check_scene_gaps_flags_zero_total_duration():
    scenes = [_scene(duration=0.0)]
    issues = check_scene_gaps(scenes)
    assert len(issues) == 1


def test_check_overlapping_items_flags_overlap_on_same_layer():
    timeline = MasterTimeline(items=[
        TimelineItem(id="a", type=TrackType.MOTION_GRAPHICS, start=0.0, duration=3.0, source="a.png", layer=1),
        TimelineItem(id="b", type=TrackType.MOTION_GRAPHICS, start=2.0, duration=3.0, source="b.png", layer=1),
    ])
    issues = check_overlapping_items(timeline)
    assert len(issues) == 1


def test_check_overlapping_items_ignores_different_layers():
    timeline = MasterTimeline(items=[
        TimelineItem(id="a", type=TrackType.BROLL, start=0.0, duration=3.0, source="a.mp4", layer=0),
        TimelineItem(id="b", type=TrackType.MOTION_GRAPHICS, start=1.0, duration=3.0, source="b.png", layer=1),
    ])
    assert check_overlapping_items(timeline) == []


def test_check_missing_assets_flags_failed_motion_item():
    spec = AnimationSpec(kind=AnimationKind.LABEL, timeline_start=0, timeline_end=1, text="x")
    item = MotionPlanItem(spec=spec, engine_used=None, output_path=None, fallback_log=[], error="engine unavailable")
    issues = check_missing_assets([item])
    assert len(issues) == 1


def test_check_broken_motion_assets_flags_missing_file(tmp_path: Path):
    timeline = MasterTimeline(items=[
        TimelineItem(
            id="a", type=TrackType.MOTION_GRAPHICS, start=0.0, duration=1.0,
            source=str(tmp_path / "does_not_exist.png"), layer=1,
        )
    ])
    issues = check_broken_motion_assets(timeline)
    assert len(issues) == 1


def test_check_visual_coverage_flags_scene_without_timeline_item():
    scenes = [_scene(id_="scene-01"), _scene(id_="scene-02")]
    timeline = MasterTimeline(items=[
        TimelineItem(id="scene-01-motion", type=TrackType.MOTION_GRAPHICS, start=0.0, duration=1.0, source="a.png")
    ])
    issues = check_visual_coverage(scenes, timeline)
    assert len(issues) == 1
    assert "scene-02" in issues[0].message


def test_check_unsafe_text_zones_flags_long_text():
    scenes = [_scene(text="x" * 200)]
    issues = check_unsafe_text_zones(scenes)
    assert len(issues) == 1


def test_check_brand_mismatch_flags_avoided_term():
    brand = Brand(name="t", avoid=["banned"])
    scenes = [_scene(text="this contains a banned word")]
    issues = check_brand_mismatch(scenes, brand)
    assert len(issues) == 1


def test_check_brand_mismatch_no_brand_returns_no_issues():
    scenes = [_scene(text="anything")]
    assert check_brand_mismatch(scenes, None) == []


def test_check_aspect_ratio_flags_mismatch():
    brand = Brand(name="t", preferred_aspect_ratio="9:16")
    timeline = MasterTimeline(width=1920, height=1080, items=[])
    issues = check_aspect_ratio(timeline, brand)
    assert len(issues) == 1


def test_check_aspect_ratio_passes_for_matching_ratio():
    brand = Brand(name="t", preferred_aspect_ratio="9:16")
    timeline = MasterTimeline(width=1080, height=1920, items=[])
    assert check_aspect_ratio(timeline, brand) == []


def test_check_render_output_flags_none_path():
    issues = check_render_output(None)
    assert len(issues) == 1
    assert issues[0].severity == QASeverity.ERROR


def test_check_render_output_flags_missing_file(tmp_path: Path):
    issues = check_render_output(tmp_path / "missing.mp4")
    assert len(issues) >= 1


def test_run_creator_qa_aggregates_all_checks():
    scenes = [_scene()]
    timeline = MasterTimeline(items=[])
    report = run_creator_qa(scenes, timeline, [], None, None)
    assert any(i.message == "No render output was produced" for i in report.issues)
