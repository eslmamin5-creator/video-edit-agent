"""Storyboard generation + style validation tests (Creator spec sections 6-9)."""
from __future__ import annotations

import pytest

from video_edit_agent.agents.creator.analysis import analyze_script
from video_edit_agent.agents.creator.scenes import build_scenes
from video_edit_agent.agents.creator.schemas import CreatorStyle, Storyboard
from video_edit_agent.agents.creator.storyboard import build_storyboard, render_storyboard_markdown

_SCRIPT = (
    "Ever feel overwhelmed by clutter? Too much stuff makes it hard to focus. "
    "A simple declutter routine clears mental space. Studies show people save "
    "30 percent more time when organized. Try a five minute daily tidy up. "
    "Subscribe for more simple productivity tips."
)


def _scenes():
    analysis = analyze_script(_SCRIPT)
    return build_scenes(_SCRIPT, analysis)


def test_creator_style_enum_accepts_all_four_values():
    assert CreatorStyle("motion") == CreatorStyle.MOTION
    assert CreatorStyle("cinematic") == CreatorStyle.CINEMATIC
    assert CreatorStyle("infographic") == CreatorStyle.INFOGRAPHIC
    assert CreatorStyle("mixed") == CreatorStyle.MIXED


def test_creator_style_rejects_invalid_value():
    with pytest.raises(ValueError):
        CreatorStyle("invalid_style")


def test_build_storyboard_returns_one_frame_per_scene():
    scenes = _scenes()
    storyboard = build_storyboard(scenes, CreatorStyle.MIXED)
    assert isinstance(storyboard, Storyboard)
    assert len(storyboard.frames) == len(scenes)


def test_build_storyboard_mixed_style_varies_treatment_across_scenes():
    scenes = _scenes()
    storyboard = build_storyboard(scenes, CreatorStyle.MIXED)
    animation_concepts = {frame.animation_concept for frame in storyboard.frames}
    assert len(animation_concepts) > 1, "mixed style must not use the same treatment for every scene"


def test_build_storyboard_motion_style_favors_typography():
    scenes = _scenes()
    storyboard = build_storyboard(scenes, CreatorStyle.MOTION)
    assert all(frame.typography for frame in storyboard.frames)


def test_build_storyboard_does_not_touch_disk(tmp_path, monkeypatch):
    """Storyboard generation must not trigger expensive asset generation itself."""
    import os

    scenes = _scenes()
    before = set(os.listdir(tmp_path))
    monkeypatch.chdir(tmp_path)
    build_storyboard(scenes, CreatorStyle.MIXED)
    after = set(os.listdir(tmp_path))
    assert before == after


def test_render_storyboard_markdown_contains_each_scene_id():
    scenes = _scenes()
    storyboard = build_storyboard(scenes, CreatorStyle.MIXED)
    md = render_storyboard_markdown(storyboard, scenes)
    for scene in scenes:
        assert scene.id in md
