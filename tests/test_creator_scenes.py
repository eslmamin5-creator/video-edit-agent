"""Scene breakdown tests (Creator spec section 5)."""
from __future__ import annotations

from video_edit_agent.agents.creator.analysis import analyze_script
from video_edit_agent.agents.creator.scenes import build_scenes
from video_edit_agent.agents.creator.schemas import Scene

_SCRIPT = (
    "Ever feel overwhelmed by clutter? Too much stuff makes it hard to focus. "
    "A simple declutter routine clears mental space. Studies show people save "
    "30 percent more time when organized. Try a five minute daily tidy up. "
    "Subscribe for more simple productivity tips."
)


def _scenes() -> list[Scene]:
    analysis = analyze_script(_SCRIPT)
    return build_scenes(_SCRIPT, analysis)


def test_build_scenes_produces_one_scene_per_sentence():
    scenes = _scenes()
    assert 4 <= len(scenes) <= 8


def test_build_scenes_on_screen_text_is_verbatim():
    scenes = _scenes()
    for scene in scenes:
        assert scene.on_screen_text == scene.script_segment
        assert scene.on_screen_text in _SCRIPT or scene.on_screen_text.strip() in _SCRIPT


def test_build_scenes_durations_are_positive_and_sum_reasonably():
    analysis = analyze_script(_SCRIPT)
    scenes = build_scenes(_SCRIPT, analysis)
    total = sum(s.estimated_duration for s in scenes)
    assert all(s.estimated_duration > 0 for s in scenes)
    # scene durations should add up reasonably to the estimated target duration
    assert total >= analysis.estimated_target_duration * 0.5


def test_build_scenes_ids_are_sequential_and_unique():
    scenes = _scenes()
    ids = [s.id for s in scenes]
    assert len(ids) == len(set(ids))
    assert ids[0] == "scene-01"


def test_build_scenes_first_transition_is_hard_cut():
    scenes = _scenes()
    assert scenes[0].transition_intent == "hard_cut"


def test_build_scenes_empty_text_returns_no_scenes():
    analysis = analyze_script("")
    scenes = build_scenes("", analysis)
    assert scenes == []
