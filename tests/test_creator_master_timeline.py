"""Creator MasterTimeline generation tests (Creator spec section 14): Creator
must write only to the shared `core.timeline.MasterTimeline` schema, using
the shared motion router / B-roll provider -- no Creator-only timeline format.

`render_motion` is monkeypatched to the guaranteed-available `simple` engine
path so these tests stay fast and deterministic; the motion router's own
engine-priority/fallback behavior has its own dedicated test coverage.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from video_edit_agent.agents.creator import timeline as creator_timeline
from video_edit_agent.agents.creator.analysis import analyze_script
from video_edit_agent.agents.creator.asset_plan import build_asset_plan
from video_edit_agent.agents.creator.schemas import CreatorStyle
from video_edit_agent.agents.creator.scenes import build_scenes
from video_edit_agent.agents.creator.timeline import build_creator_timeline
from video_edit_agent.core.schemas import MotionEngine, MotionPlanItem
from video_edit_agent.core.timeline import MasterTimeline


@pytest.fixture(autouse=True)
def _fast_motion(monkeypatch, tmp_path: Path):
    def fake_render_motion(spec, project_root, output_dir, *, brand=None, fps=30, slot_id="x", offline=False):
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{slot_id}.png"
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        return MotionPlanItem(spec=spec, engine_used=MotionEngine.SIMPLE, output_path=str(out), fallback_log=[])

    monkeypatch.setattr(creator_timeline, "render_motion", fake_render_motion)


_SCRIPT = (
    "Ever feel overwhelmed by clutter? Too much stuff makes it hard to focus. "
    "A simple declutter routine clears mental space. Studies show people save "
    "30 percent more time when organized. Try a five minute daily tidy up. "
    "Subscribe for more simple productivity tips."
)


def test_build_creator_timeline_returns_shared_master_timeline_type(tmp_path: Path):
    analysis = analyze_script(_SCRIPT)
    scenes = build_scenes(_SCRIPT, analysis)
    asset_plan = build_asset_plan(scenes, CreatorStyle.MOTION)

    timeline, motion_items, broll_items = build_creator_timeline(
        scenes,
        asset_plan,
        project_root=tmp_path,
        motion_output_dir=tmp_path / "motion",
        broll_project_dir=tmp_path / "broll",
        broll_library_dir=None,
        brand=None,
        fps=30.0,
        width=1080,
        height=1920,
    )

    assert isinstance(timeline, MasterTimeline)
    assert timeline.workflow == "creator"
    assert timeline.fps == 30.0
    assert timeline.width == 1080 and timeline.height == 1920
    assert len(timeline.items) > 0


def test_build_creator_timeline_items_are_sequential_by_scene(tmp_path: Path):
    analysis = analyze_script(_SCRIPT)
    scenes = build_scenes(_SCRIPT, analysis)
    asset_plan = build_asset_plan(scenes, CreatorStyle.MOTION)

    timeline, _, _ = build_creator_timeline(
        scenes,
        asset_plan,
        project_root=tmp_path,
        motion_output_dir=tmp_path / "motion",
        broll_project_dir=tmp_path / "broll",
        broll_library_dir=None,
        brand=None,
        fps=30.0,
        width=1080,
        height=1920,
    )
    assert timeline.total_duration == sum(s.estimated_duration for s in scenes)
