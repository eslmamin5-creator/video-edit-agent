"""Creator MasterTimeline builder (spec section 14).

Writes ONLY to the shared `core.timeline.MasterTimeline` schema -- there is
no Creator-only timeline format. Motion is requested exclusively through the
shared `motion.router.render_motion` (never an engine directly); B-roll is
resolved exclusively through the shared `broll.providers.local.find_broll`.
If a scene was planned for B-roll but none is found, it falls back to a
motion/typography treatment instead of failing the whole workflow (spec
section 12).
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.agents.creator.schemas import (
    AssetPlanItem,
    AssetTreatment,
    Scene,
    ScenePurpose,
)
from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.providers.local import find_broll
from video_edit_agent.core.schemas import (
    AnimationKind,
    AnimationSpec,
    BrollPlanItem,
    BrollSourceKind,
    MotionEngine,
    MotionPlanItem,
)
from video_edit_agent.core.timeline import (
    BrandInfo,
    MasterTimeline,
    Provenance,
    ProviderKind,
    TimelineItem,
    TrackType,
)
from video_edit_agent.motion.router import render_motion

_PURPOSE_TO_KIND = {
    ScenePurpose.HOOK: AnimationKind.HOOK_TITLE,
    ScenePurpose.PROBLEM: AnimationKind.LABEL,
    ScenePurpose.EXPLANATION: AnimationKind.LOWER_THIRD,
    ScenePurpose.STATISTIC: AnimationKind.STAT_COUNTER,
    ScenePurpose.EXAMPLE: AnimationKind.LABEL,
    ScenePurpose.SOLUTION: AnimationKind.FEATURE_CARD,
    ScenePurpose.CTA: AnimationKind.CTA,
    ScenePurpose.OTHER: AnimationKind.LABEL,
}

_ENGINE_TO_PROVIDER = {
    MotionEngine.REMOTION: ProviderKind.REMOTION,
    MotionEngine.MANIM: ProviderKind.MANIM,
    MotionEngine.HYPERFRAMES: ProviderKind.HYPERFRAMES,
    MotionEngine.SIMPLE: ProviderKind.SIMPLE,
}

_BROLL_SOURCE_TO_PROVIDER = {
    BrollSourceKind.USER_ASSET: ProviderKind.USER,
    BrollSourceKind.PROJECT_ASSET: ProviderKind.USER,
    BrollSourceKind.LOCAL_LIBRARY: ProviderKind.LOCAL_LIBRARY,
    BrollSourceKind.GENERATED_IMAGE: ProviderKind.GEMINI,
    BrollSourceKind.GENERATED_VIDEO: ProviderKind.VEO,
}


def _brand_extra(brand: Brand | None) -> dict:
    if brand is None:
        return {}
    return {
        "primaryColor": brand.colors.primary,
        "secondaryColor": brand.colors.secondary,
        "accentColor": brand.colors.accent,
        "fontFamily": brand.fonts[0] if brand.fonts else None,
    }


def _spec_for_scene(scene: Scene, start: float, end: float, brand: Brand | None) -> AnimationSpec:
    kind = _PURPOSE_TO_KIND.get(scene.purpose, AnimationKind.LABEL)
    text = scene.on_screen_text
    value = None
    if kind == AnimationKind.STAT_COUNTER:
        import re

        match = re.search(r"\d+([.,]\d+)?%?", text)
        value = match.group(0) if match else text
    return AnimationSpec(
        kind=kind,
        timeline_start=start,
        timeline_end=end,
        text=text,
        value=value,
        extra={k: v for k, v in _brand_extra(brand).items() if v is not None},
    )


def build_creator_timeline(
    scenes: list[Scene],
    asset_plan: list[AssetPlanItem],
    *,
    project_root: Path,
    motion_output_dir: Path,
    broll_project_dir: Path,
    broll_library_dir: Path | None,
    brand: Brand | None,
    fps: float = 30.0,
    width: int = 1080,
    height: int = 1920,
    offline: bool = False,
) -> tuple[MasterTimeline, list[MotionPlanItem], list[BrollPlanItem]]:
    plan_by_scene = {item.scene_id: item for item in asset_plan}
    motion_items: list[MotionPlanItem] = []
    broll_items: list[BrollPlanItem] = []
    timeline_items: list[TimelineItem] = []

    cursor = 0.0
    for i, scene in enumerate(scenes):
        start, end = cursor, cursor + scene.estimated_duration
        cursor = end
        plan_item = plan_by_scene.get(scene.id)
        treatment = plan_item.treatment if plan_item else AssetTreatment.REMOTION_COMPOSITION

        used_broll = False
        if treatment == AssetTreatment.LOCAL_BROLL:
            broll_candidate = BrollPlanItem(
                timeline_start=start,
                timeline_end=end,
                purpose=scene.purpose.value,
                spoken_concept=scene.background_concept or scene.foreground_concept or scene.on_screen_text,
                recommended_visual=scene.visual_type,
                duration=end - start,
            )
            resolved = find_broll(broll_candidate, broll_project_dir, broll_library_dir)
            if resolved.source != BrollSourceKind.NONE:
                used_broll = True
                broll_items.append(resolved)
                if plan_item:
                    plan_item.asset_path = resolved.asset_path
                    plan_item.provenance = resolved.source.value
                timeline_items.append(
                    TimelineItem(
                        id=f"{scene.id}-broll",
                        type=TrackType.BROLL,
                        start=start,
                        duration=end - start,
                        source=resolved.asset_path or "",
                        layer=0,
                        provenance=Provenance(
                            kind=_BROLL_SOURCE_TO_PROVIDER.get(resolved.source, ProviderKind.OTHER),
                            detail=resolved.asset_path,
                        ),
                        brand=BrandInfo(brand_name=brand.name if brand else None, applied=brand is not None),
                        metadata={"scale_to_canvas": True},
                    )
                )
            else:
                # No matching B-roll: degrade to motion/typography rather than
                # failing the whole scene (spec section 12).
                if plan_item:
                    plan_item.treatment = AssetTreatment.REMOTION_COMPOSITION
                    plan_item.provenance = "broll_unavailable_fallback_to_motion"

        if not used_broll:
            spec = _spec_for_scene(scene, start, end, brand)
            item = render_motion(
                spec, project_root, motion_output_dir, brand=brand, fps=int(fps), slot_id=scene.id, offline=offline
            )
            motion_items.append(item)
            if plan_item:
                plan_item.provenance = item.engine_used.value if item.engine_used else "unavailable"
                plan_item.asset_path = item.output_path
                plan_item.offline_safe = item.engine_used != MotionEngine.HYPERFRAMES if item.engine_used else True
            if item.output_path:
                timeline_items.append(
                    TimelineItem(
                        id=f"{scene.id}-motion",
                        type=TrackType.MOTION_GRAPHICS,
                        start=start,
                        duration=end - start,
                        source=item.output_path,
                        layer=1,
                        provenance=Provenance(
                            kind=_ENGINE_TO_PROVIDER.get(item.engine_used, ProviderKind.OTHER),
                            detail=item.engine_used.value if item.engine_used else None,
                        ),
                        brand=BrandInfo(brand_name=brand.name if brand else None, applied=brand is not None),
                        metadata={"fallback_log": item.fallback_log},
                    )
                )

    timeline = MasterTimeline(
        version=1, workflow="creator", fps=fps, width=width, height=height, items=timeline_items
    )
    return timeline, motion_items, broll_items
