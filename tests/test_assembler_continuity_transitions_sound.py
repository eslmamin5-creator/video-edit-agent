from __future__ import annotations

from video_edit_agent.agents.assembler.continuity import analyze_continuity
from video_edit_agent.agents.assembler.schemas import ContinuitySeverity, SceneAnalysis, TransitionKind
from video_edit_agent.agents.assembler.sound import plan_sound
from video_edit_agent.agents.assembler.transitions import plan_transitions


def _analysis(scene_id: str, **overrides) -> SceneAnalysis:
    base = dict(
        scene_id=scene_id, duration=1.0, dominant_color="#808080", brightness=0.5,
        contrast=0.4, motion_direction="static", has_audio=True, is_silent=False,
        aspect_ratio="16:9", width=640, height=360, fps=30.0,
    )
    base.update(overrides)
    return SceneAnalysis(**base)


def test_continuity_flags_large_brightness_jump():
    analyses = [_analysis("a", brightness=0.1), _analysis("b", brightness=0.8)]
    findings = analyze_continuity(analyses)
    brightness_findings = [f for f in findings if f.category == "brightness"]
    assert brightness_findings
    assert brightness_findings[0].severity == ContinuitySeverity.HIGH
    assert brightness_findings[0].from_scene == "a" and brightness_findings[0].to_scene == "b"


def test_continuity_flags_aspect_ratio_and_resolution_mismatch():
    analyses = [
        _analysis("a", aspect_ratio="16:9", width=640, height=360),
        _analysis("b", aspect_ratio="1:1", width=480, height=480),
    ]
    findings = analyze_continuity(analyses)
    categories = {f.category for f in findings}
    assert "aspect_ratio" in categories
    assert "resolution" in categories


def test_continuity_no_findings_for_identical_neighbors():
    analyses = [_analysis("a"), _analysis("b")]
    findings = analyze_continuity(analyses)
    assert findings == []


def test_transitions_default_to_hard_cut_when_no_issues():
    analyses = [_analysis("a"), _analysis("b")]
    findings = analyze_continuity(analyses)
    decisions = plan_transitions(["a", "b"], findings)
    assert len(decisions) == 1
    assert decisions[0].type == TransitionKind.HARD_CUT
    assert decisions[0].applied is True


def test_transitions_recommend_crossfade_on_high_severity_brightness_jump():
    analyses = [_analysis("a", brightness=0.05), _analysis("b", brightness=0.9)]
    findings = analyze_continuity(analyses)
    decisions = plan_transitions(["a", "b"], findings)
    assert decisions[0].type == TransitionKind.SHORT_CROSSFADE
    # Real video crossfade isn't wired into the renderer yet -- must be
    # honestly recorded as a recommendation, never silently faked.
    assert decisions[0].applied is False


def test_sound_plan_flags_silent_audio_track():
    analyses = [_analysis("a", has_audio=True, is_silent=True)]
    ops = plan_sound(analyses, [])
    assert any(op.scene_id == "a" and op.type.value == "silence_trim" for op in ops)


def test_sound_plan_never_introduces_generated_music():
    analyses = [_analysis("a"), _analysis("b")]
    findings = analyze_continuity(analyses)
    decisions = plan_transitions(["a", "b"], findings)
    ops = plan_sound(analyses, decisions)
    assert all(op.type.value != "music" for op in ops)
