"""Normalization planning (spec sections 9-10): aspect-ratio and color
recommendations per scene. Conservative by design -- v1 wires only
letterbox aspect handling into real rendering (the shared render core's
`safe_format_filter` already scale+pads to the target box regardless of
source aspect ratio, so letterbox is "free" and honestly markable as
applied). Crop/fit/blur/brand-background strategies are recorded as
recommendations only (`aspect_applied=False`) until the shared render core
grows dedicated support -- never silently faked.

Color normalization is also planning-only in v1 (`color_applied=False`):
we do not want to risk destroying an intentionally stylized AI-generated
scene by auto-correcting it, and the shared render core has no color-grade
filter today. Recommendations are still computed and persisted so a future
Finish pass (or a human) can act on them.
"""
from __future__ import annotations

from video_edit_agent.agents.assembler.schemas import (
    AspectStrategy,
    NormalizationChoice,
    SceneAnalysis,
)

# A scene's brightness/contrast recommendation nudges it toward these
# targets only when it's a clear outlier vs. the rest of the sequence --
# never a blanket "everything must match" correction.
_TARGET_BRIGHTNESS_BAND = (0.25, 0.75)
_MAX_RECOMMENDED_DELTA = 0.15


def plan_normalization(
    analyses: list[SceneAnalysis],
    *,
    target_aspect: str = "16:9",
) -> list[NormalizationChoice]:
    if not analyses:
        return []
    brightness_values = [a.brightness for a in analyses if a.brightness is not None]
    avg_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else None

    choices: list[NormalizationChoice] = []
    for a in analyses:
        aspect_strategy = AspectStrategy.LETTERBOX
        aspect_applied = a.aspect_ratio != "" and a.aspect_ratio != target_aspect
        reasons = []
        if aspect_applied:
            reasons.append(f"source aspect {a.aspect_ratio} differs from target {target_aspect}: letterboxing")

        brightness_delta = 0.0
        if a.brightness is not None:
            lo, hi = _TARGET_BRIGHTNESS_BAND
            if a.brightness < lo:
                brightness_delta = min(_MAX_RECOMMENDED_DELTA, lo - a.brightness)
                reasons.append(f"scene is darker ({a.brightness:.2f}) than the safe band; conservative lift suggested")
            elif a.brightness > hi:
                brightness_delta = -min(_MAX_RECOMMENDED_DELTA, a.brightness - hi)
                reasons.append(f"scene is brighter ({a.brightness:.2f}) than the safe band; conservative reduction suggested")
            elif avg_brightness is not None and abs(a.brightness - avg_brightness) > 0.25:
                # Clear outlier vs. the rest of the sequence, even inside the safe band.
                direction = -1 if a.brightness > avg_brightness else 1
                brightness_delta = direction * min(_MAX_RECOMMENDED_DELTA, abs(a.brightness - avg_brightness) / 2)
                reasons.append("scene brightness is a sequence outlier vs. neighboring scenes")

        choices.append(
            NormalizationChoice(
                scene_id=a.scene_id,
                aspect_strategy=aspect_strategy,
                aspect_applied=aspect_applied,
                recommended_brightness_delta=round(brightness_delta, 3),
                recommended_contrast_delta=0.0,
                recommended_saturation_delta=0.0,
                color_applied=False,  # not wired to render in v1; see module docstring
                reason="; ".join(reasons) if reasons else "no adjustment needed",
            )
        )
    return choices
