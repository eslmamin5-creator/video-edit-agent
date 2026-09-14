"""Storyboard generation (Creator spec section 6-7): a lightweight
description of what each scene will look like. This step must never trigger
expensive asset generation itself -- it only plans concepts as text; actual
rendering happens later via the shared motion router / B-roll system.

Visual modes (spec section 7): `motion` favors typography/graphic systems,
`cinematic` favors B-roll-led visual storytelling, `infographic` favors
charts/diagrams, `mixed` combines all three based on each scene's own
purpose so not every scene gets the same treatment.
"""
from __future__ import annotations

from video_edit_agent.agents.creator.schemas import (
    CreatorStyle,
    Scene,
    ScenePurpose,
    Storyboard,
    StoryboardFrame,
)

_STAT_PURPOSES = {ScenePurpose.STATISTIC}
_BROLL_LED_PURPOSES = {ScenePurpose.PROBLEM, ScenePurpose.EXPLANATION, ScenePurpose.EXAMPLE, ScenePurpose.SOLUTION}


def _scene_treatment(scene: Scene, style: CreatorStyle) -> str:
    """Per-scene treatment, honoring both the requested style and the
    scene's own purpose so `mixed` genuinely mixes rather than repeating."""
    if style == CreatorStyle.INFOGRAPHIC:
        return "data_viz" if scene.purpose in _STAT_PURPOSES else "diagram"
    if style == CreatorStyle.MOTION:
        return "typography"
    if style == CreatorStyle.CINEMATIC:
        return "broll" if scene.purpose in _BROLL_LED_PURPOSES else "typography"
    # mixed: let each scene's own purpose decide, closest to how it would
    # naturally be shot -- statistics get charts, narrative beats get B-roll,
    # hook/cta stay bold typography.
    if scene.purpose in _STAT_PURPOSES:
        return "data_viz"
    if scene.purpose in _BROLL_LED_PURPOSES:
        return "broll"
    return "typography"


def _frame_for_scene(scene: Scene, style: CreatorStyle, brand_name: str | None) -> StoryboardFrame:
    treatment = _scene_treatment(scene, style)
    typography = f"Bold title card: \"{scene.on_screen_text}\"" if treatment == "typography" else ""
    broll_concept = scene.background_concept if treatment == "broll" else ""
    generated_visual_concept = scene.foreground_concept if treatment == "data_viz" else ""
    animation_concept = {
        "typography": "kinetic text reveal",
        "broll": "B-roll with lower-third caption",
        "diagram": "animated diagram",
        "data_viz": "animated stat counter / chart",
    }[treatment]

    return StoryboardFrame(
        scene_id=scene.id,
        on_screen=scene.on_screen_text,
        layout="full_bleed" if treatment == "broll" else "centered",
        framing="9:16" ,
        typography=typography,
        animation_concept=animation_concept,
        broll_concept=broll_concept,
        generated_visual_concept=generated_visual_concept,
        transition=scene.transition_intent,
        audio_intent="voice_over" if scene.voice_over_text else "silent",
    )


def build_storyboard(scenes: list[Scene], style: CreatorStyle, brand_name: str | None = None) -> Storyboard:
    return Storyboard(frames=[_frame_for_scene(s, style, brand_name) for s in scenes])


def render_storyboard_markdown(storyboard: Storyboard, scenes: list[Scene]) -> str:
    scenes_by_id = {s.id: s for s in scenes}
    lines = ["# Storyboard", ""]
    for frame in storyboard.frames:
        scene = scenes_by_id.get(frame.scene_id)
        lines.append(f"## {frame.scene_id}")
        if scene:
            lines.append(f"- Purpose: {scene.purpose.value}")
            lines.append(f"- Duration: {scene.estimated_duration}s")
        lines.append(f"- On screen: {frame.on_screen}")
        lines.append(f"- Layout: {frame.layout} ({frame.framing})")
        if frame.typography:
            lines.append(f"- Typography: {frame.typography}")
        lines.append(f"- Animation: {frame.animation_concept}")
        if frame.broll_concept:
            lines.append(f"- B-roll concept: {frame.broll_concept}")
        if frame.generated_visual_concept:
            lines.append(f"- Data/visual concept: {frame.generated_visual_concept}")
        lines.append(f"- Transition: {frame.transition}")
        lines.append(f"- Audio: {frame.audio_intent}")
        lines.append("")
    return "\n".join(lines)
