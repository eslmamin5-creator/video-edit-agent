"""Asset planning (Creator spec section 10): records, per scene, a coarse
treatment *category* only. It deliberately does NOT duplicate the shared
motion router's engine-selection logic (spec section 11) or the shared
B-roll provider's matching logic (spec section 12) -- those run for real at
timeline-build time and their actual result (`MotionPlanItem.engine_used`,
`BrollPlanItem.source`) is the authoritative provenance. This slice never
proposes cloud-generated assets: only local/typography/motion treatments are
planned, so the plan is offline-safe unconditionally.
"""
from __future__ import annotations

from video_edit_agent.agents.creator.schemas import AssetPlanItem, AssetTreatment, Scene
from video_edit_agent.agents.creator.storyboard import _scene_treatment
from video_edit_agent.agents.creator.schemas import CreatorStyle

_TREATMENT_TO_ASSET = {
    "typography": AssetTreatment.REMOTION_COMPOSITION,
    "broll": AssetTreatment.LOCAL_BROLL,
    "diagram": AssetTreatment.SIMPLE_GRAPHICS,
    "data_viz": AssetTreatment.SIMPLE_GRAPHICS,
}


def build_asset_plan(scenes: list[Scene], style: CreatorStyle) -> list[AssetPlanItem]:
    plan: list[AssetPlanItem] = []
    for scene in scenes:
        treatment_key = _scene_treatment(scene, style)
        plan.append(
            AssetPlanItem(
                scene_id=scene.id,
                treatment=_TREATMENT_TO_ASSET[treatment_key],
                asset_path=None,
                provenance="planned",
                offline_safe=True,
            )
        )
    return plan
