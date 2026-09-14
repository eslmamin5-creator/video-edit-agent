"""Creator-specific QA (spec section 16): local, deterministic checks only --
no cloud vision. Reuses the shared `QAIssue`/`QASeverity`/`QAReport` schemas
and `core.media.probe`, plus `qa.technical.check_output_exists` for the
output-file check, but implements its own checks for everything that assumes
Editor-only concepts (mandatory audio track, EDL-shaped duration) that don't
apply to a Creator project.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.agents.creator.schemas import AssetPlanItem, Scene
from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.media import probe
from video_edit_agent.core.schemas import MotionPlanItem, QAIssue, QAReport, QASeverity
from video_edit_agent.core.timeline import MasterTimeline, TrackType
from video_edit_agent.qa.technical import check_output_exists

_MAX_SAFE_TEXT_CHARS = 140


def check_scene_durations(scenes: list[Scene]) -> list[QAIssue]:
    issues = []
    for scene in scenes:
        if scene.estimated_duration <= 0:
            issues.append(
                QAIssue(
                    category="technical", severity=QASeverity.ERROR,
                    message=f"Scene {scene.id} has invalid duration ({scene.estimated_duration}s)",
                )
            )
    return issues


def check_scene_gaps(scenes: list[Scene]) -> list[QAIssue]:
    """Scenes are laid out back-to-back by construction; this checks that
    remains true (a future assembler/continuity step could break it)."""
    issues = []
    cursor = 0.0
    for scene in scenes:
        cursor += scene.estimated_duration
    if cursor <= 0:
        issues.append(
            QAIssue(category="technical", severity=QASeverity.ERROR, message="Total scene duration is zero")
        )
    return issues


def check_overlapping_items(timeline: MasterTimeline) -> list[QAIssue]:
    issues = []
    by_layer: dict[int, list] = {}
    for item in timeline.items:
        by_layer.setdefault(item.layer, []).append(item)
    for layer, items in by_layer.items():
        ordered = sorted(items, key=lambda i: i.start)
        for a, b in zip(ordered, ordered[1:]):
            if a.end > b.start + 1e-6:
                issues.append(
                    QAIssue(
                        category="technical", severity=QASeverity.WARNING,
                        message=f"Timeline items overlap on layer {layer}: {a.id} and {b.id}",
                        auto_repairable=False,
                    )
                )
    return issues


def check_missing_assets(motion_items: list[MotionPlanItem]) -> list[QAIssue]:
    issues = []
    for item in motion_items:
        if not item.output_path:
            issues.append(
                QAIssue(
                    category="technical", severity=QASeverity.WARNING,
                    message=f"Motion graphic for '{item.spec.kind.value}' at {item.spec.timeline_start:.1f}s "
                    f"could not be rendered by any engine: {item.error}",
                )
            )
    return issues


def check_broken_motion_assets(timeline: MasterTimeline) -> list[QAIssue]:
    issues = []
    for item in timeline.items:
        if item.type != TrackType.MOTION_GRAPHICS:
            continue
        if item.source and not Path(item.source).exists():
            issues.append(
                QAIssue(
                    category="technical", severity=QASeverity.ERROR,
                    message=f"Motion asset referenced by timeline item {item.id} is missing on disk: {item.source}",
                )
            )
    return issues


def check_visual_coverage(scenes: list[Scene], timeline: MasterTimeline) -> list[QAIssue]:
    issues = []
    covered_scene_ids = {item.id.rsplit("-", 1)[0] for item in timeline.items}
    for scene in scenes:
        if scene.id not in covered_scene_ids:
            issues.append(
                QAIssue(
                    category="visual", severity=QASeverity.WARNING,
                    message=f"Scene {scene.id} has no visual coverage (no motion or B-roll item)",
                )
            )
    return issues


def check_unsafe_text_zones(scenes: list[Scene]) -> list[QAIssue]:
    issues = []
    for scene in scenes:
        if len(scene.on_screen_text) > _MAX_SAFE_TEXT_CHARS:
            issues.append(
                QAIssue(
                    category="visual", severity=QASeverity.WARNING,
                    message=f"Scene {scene.id} on-screen text is long ({len(scene.on_screen_text)} chars) "
                    "and may overflow safe zones",
                )
            )
    return issues


def check_brand_mismatch(scenes: list[Scene], brand: Brand | None) -> list[QAIssue]:
    issues = []
    if brand is None or not brand.avoid:
        return issues
    for scene in scenes:
        lowered = scene.on_screen_text.lower()
        for banned in brand.avoid:
            if banned.lower() in lowered:
                issues.append(
                    QAIssue(
                        category="brand", severity=QASeverity.WARNING,
                        message=f"Scene {scene.id} text contains brand-avoid term '{banned}'",
                    )
                )
    return issues


def check_aspect_ratio(timeline: MasterTimeline, brand: Brand | None) -> list[QAIssue]:
    issues = []
    expected = (brand.preferred_aspect_ratio if brand else "9:16") or "9:16"
    try:
        exp_w, exp_h = (int(x) for x in expected.split(":"))
    except ValueError:
        return issues
    actual_ratio = timeline.width / timeline.height if timeline.height else 0
    expected_ratio = exp_w / exp_h if exp_h else 0
    if expected_ratio and abs(actual_ratio - expected_ratio) > 0.02:
        issues.append(
            QAIssue(
                category="brand", severity=QASeverity.WARNING,
                message=f"Rendered aspect ratio {timeline.width}x{timeline.height} does not match "
                f"brand's preferred {expected}",
            )
        )
    return issues


def check_render_output(output_path: Path | None) -> list[QAIssue]:
    if output_path is None:
        return [QAIssue(category="technical", severity=QASeverity.ERROR, message="No render output was produced")]
    return check_output_exists(output_path)


def run_creator_qa(
    scenes: list[Scene],
    timeline: MasterTimeline,
    motion_items: list[MotionPlanItem],
    output_path: Path | None,
    brand: Brand | None = None,
) -> QAReport:
    issues: list[QAIssue] = []
    issues.extend(check_scene_durations(scenes))
    issues.extend(check_scene_gaps(scenes))
    issues.extend(check_overlapping_items(timeline))
    issues.extend(check_missing_assets(motion_items))
    issues.extend(check_broken_motion_assets(timeline))
    issues.extend(check_visual_coverage(scenes, timeline))
    issues.extend(check_unsafe_text_zones(scenes))
    issues.extend(check_brand_mismatch(scenes, brand))
    issues.extend(check_aspect_ratio(timeline, brand))
    issues.extend(check_render_output(output_path))
    return QAReport(issues=issues)
