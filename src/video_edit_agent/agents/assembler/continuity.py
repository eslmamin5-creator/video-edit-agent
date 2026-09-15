"""Continuity analysis (spec sections 15-16): compares adjacent scenes for
deterministic, locally-observable discontinuities. Never overclaims --
anything requiring true visual understanding (character identity, wardrobe,
props) is out of scope offline; cloud vision may add it later as pure
enrichment, never a requirement (see spec section 16).
"""
from __future__ import annotations

from itertools import pairwise

from video_edit_agent.agents.assembler.schemas import (
    ContinuityFinding,
    ContinuitySeverity,
    SceneAnalysis,
)

_BRIGHTNESS_JUMP_HIGH = 0.35
_BRIGHTNESS_JUMP_MED = 0.18


def analyze_continuity(analyses: list[SceneAnalysis]) -> list[ContinuityFinding]:
    findings: list[ContinuityFinding] = []
    for a, b in pairwise(analyses):
        findings.extend(_pair_findings(a, b))
    return findings


def _pair_findings(a: SceneAnalysis, b: SceneAnalysis) -> list[ContinuityFinding]:
    findings: list[ContinuityFinding] = []

    if a.brightness is not None and b.brightness is not None:
        delta = abs(b.brightness - a.brightness)
        if delta >= _BRIGHTNESS_JUMP_HIGH:
            findings.append(
                ContinuityFinding(
                    from_scene=a.scene_id,
                    to_scene=b.scene_id,
                    category="brightness",
                    description=f"large brightness jump ({a.brightness:.2f} -> {b.brightness:.2f})",
                    severity=ContinuitySeverity.HIGH,
                    confidence=0.9,
                    recommendation="consider a short crossfade or dip-to-black to soften the jump",
                )
            )
        elif delta >= _BRIGHTNESS_JUMP_MED:
            findings.append(
                ContinuityFinding(
                    from_scene=a.scene_id,
                    to_scene=b.scene_id,
                    category="brightness",
                    description=f"moderate brightness jump ({a.brightness:.2f} -> {b.brightness:.2f})",
                    severity=ContinuitySeverity.MEDIUM,
                    confidence=0.7,
                    recommendation="optional: brief crossfade if a hard cut feels abrupt",
                )
            )

    if a.dominant_color and b.dominant_color:
        temp_a = _warmth(a.dominant_color)
        temp_b = _warmth(b.dominant_color)
        if abs(temp_a - temp_b) >= 60:
            findings.append(
                ContinuityFinding(
                    from_scene=a.scene_id,
                    to_scene=b.scene_id,
                    category="color_temperature",
                    description="noticeable color-temperature shift between scenes (warm/cool mismatch)",
                    severity=ContinuitySeverity.MEDIUM,
                    confidence=0.55,
                    recommendation="review manually; automatic color correction is not applied by default",
                )
            )

    if (
        a.motion_direction and b.motion_direction
        and a.motion_direction != "unknown" and b.motion_direction != "unknown"
        and _opposing(a.motion_direction, b.motion_direction)
    ):
        findings.append(
            ContinuityFinding(
                from_scene=a.scene_id,
                to_scene=b.scene_id,
                category="motion_direction",
                description=f"motion direction conflict ({a.motion_direction} into {b.motion_direction})",
                severity=ContinuitySeverity.LOW,
                confidence=0.4,
                recommendation="heuristic only (brightness-centroid based); verify manually before acting",
            )
        )

    if a.aspect_ratio and b.aspect_ratio and a.aspect_ratio != b.aspect_ratio:
        findings.append(
            ContinuityFinding(
                from_scene=a.scene_id,
                to_scene=b.scene_id,
                category="aspect_ratio",
                description=f"aspect ratio mismatch ({a.aspect_ratio} vs {b.aspect_ratio})",
                severity=ContinuitySeverity.MEDIUM,
                confidence=1.0,
                recommendation="normalization plan will letterbox to the target aspect ratio",
            )
        )

    if (a.width, a.height) != (b.width, b.height) and a.width and b.width:
        findings.append(
            ContinuityFinding(
                from_scene=a.scene_id,
                to_scene=b.scene_id,
                category="resolution",
                description=f"resolution mismatch ({a.width}x{a.height} vs {b.width}x{b.height})",
                severity=ContinuitySeverity.LOW,
                confidence=1.0,
                recommendation="output will be normalized to the project's target resolution",
            )
        )

    return findings


def _warmth(hex_color: str) -> int:
    """Crude warm/cool proxy: red minus blue channel."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    b = int(hex_color[4:6], 16)
    return r - b


_OPPOSITES = {("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")}


def _opposing(dir_a: str, dir_b: str) -> bool:
    return (dir_a, dir_b) in _OPPOSITES
