from __future__ import annotations

from video_edit_agent.agents.assembler.normalization import plan_normalization
from video_edit_agent.agents.assembler.schemas import AspectStrategy, SceneAnalysis


def _analysis(scene_id: str, *, aspect: str, brightness: float) -> SceneAnalysis:
    return SceneAnalysis(
        scene_id=scene_id, duration=1.0, dominant_color="#808080", brightness=brightness,
        contrast=0.4, motion_direction="static", has_audio=True, is_silent=False,
        aspect_ratio=aspect, width=640, height=360, fps=30.0,
    )


def test_mismatched_aspect_gets_letterbox_applied():
    analyses = [_analysis("scene_000", aspect="1:1", brightness=0.5)]
    plan = plan_normalization(analyses, target_aspect="16:9")
    assert plan[0].aspect_strategy == AspectStrategy.LETTERBOX
    assert plan[0].aspect_applied is True


def test_matching_aspect_is_not_flagged():
    analyses = [_analysis("scene_000", aspect="16:9", brightness=0.5)]
    plan = plan_normalization(analyses, target_aspect="16:9")
    assert plan[0].aspect_applied is False


def test_dark_scene_gets_conservative_brightness_recommendation():
    analyses = [_analysis("scene_000", aspect="16:9", brightness=0.05)]
    plan = plan_normalization(analyses, target_aspect="16:9")
    assert plan[0].recommended_brightness_delta > 0
    assert plan[0].recommended_brightness_delta <= 0.15


def test_color_normalization_is_never_applied_by_default():
    analyses = [_analysis("scene_000", aspect="16:9", brightness=0.9)]
    plan = plan_normalization(analyses, target_aspect="16:9")
    assert plan[0].color_applied is False
