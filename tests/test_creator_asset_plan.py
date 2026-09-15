"""Asset planning tests (Creator spec section 10)."""
from __future__ import annotations

from video_edit_agent.agents.creator.analysis import analyze_script
from video_edit_agent.agents.creator.asset_plan import build_asset_plan
from video_edit_agent.agents.creator.scenes import build_scenes
from video_edit_agent.agents.creator.schemas import AssetTreatment, CreatorStyle

_SCRIPT = (
    "Ever feel overwhelmed by clutter? Too much stuff makes it hard to focus. "
    "A simple declutter routine clears mental space. Studies show people save "
    "30 percent more time when organized. Try a five minute daily tidy up. "
    "Subscribe for more simple productivity tips."
)


def _scenes():
    analysis = analyze_script(_SCRIPT)
    return build_scenes(_SCRIPT, analysis)


def test_build_asset_plan_one_item_per_scene():
    scenes = _scenes()
    plan = build_asset_plan(scenes, CreatorStyle.MIXED)
    assert len(plan) == len(scenes)
    assert [item.scene_id for item in plan] == [s.id for s in scenes]


def test_build_asset_plan_is_offline_safe_for_every_item():
    scenes = _scenes()
    for style in CreatorStyle:
        plan = build_asset_plan(scenes, style)
        assert all(item.offline_safe for item in plan)


def test_build_asset_plan_never_proposes_cloud_generation():
    scenes = _scenes()
    cloud_treatments = {AssetTreatment.GENERATED_IMAGE, AssetTreatment.GENERATED_VIDEO}
    for style in CreatorStyle:
        plan = build_asset_plan(scenes, style)
        assert not any(item.treatment in cloud_treatments for item in plan)


def test_build_asset_plan_initial_provenance_is_planned():
    scenes = _scenes()
    plan = build_asset_plan(scenes, CreatorStyle.MIXED)
    assert all(item.provenance == "planned" for item in plan)
