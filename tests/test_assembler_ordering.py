from __future__ import annotations

from video_edit_agent.agents.assembler.ordering import resolve_order
from video_edit_agent.agents.assembler.schemas import OrderPolicy, SceneInventoryItem, ScriptAlignmentItem


def _item(id_: str, order: int) -> SceneInventoryItem:
    return SceneInventoryItem(
        id=id_, filename=f"{id_}.mp4", path=f"/x/{id_}.mp4", original_order=order,
        duration=1.0, width=640, height=360, fps=30.0, video_codec="h264",
        aspect_ratio="16:9", has_audio=True,
    )


def test_default_preserve_policy_keeps_discovered_order():
    items = [_item("scene_000", 0), _item("scene_001", 1), _item("scene_002", 2)]
    resolution = resolve_order(items)
    assert resolution.policy_applied == OrderPolicy.PRESERVE
    assert [i.id for i in resolution.ordered_items] == ["scene_000", "scene_001", "scene_002"]


def test_filename_policy_matches_discovered_natural_order():
    items = [_item("scene_000", 0), _item("scene_001", 1)]
    resolution = resolve_order(items, policy=OrderPolicy.FILENAME)
    assert resolution.policy_applied == OrderPolicy.FILENAME
    assert [i.id for i in resolution.ordered_items] == ["scene_000", "scene_001"]


def test_script_order_never_applied_without_explicit_opt_in():
    items = [_item("scene_000", 0), _item("scene_001", 1)]
    alignment = [
        ScriptAlignmentItem(scene_id="scene_000", script_beat_index=1, confidence=0.9),
        ScriptAlignmentItem(scene_id="scene_001", script_beat_index=0, confidence=0.9),
    ]
    # Policy stays PRESERVE (the caller did not pass --order script) --
    # the AI's opinion must surface only as `recommendation`, never applied.
    resolution = resolve_order(items, policy=OrderPolicy.PRESERVE, script_alignment=alignment)
    assert resolution.policy_applied == OrderPolicy.PRESERVE
    assert [i.id for i in resolution.ordered_items] == ["scene_000", "scene_001"]
    assert resolution.recommendation == ["scene_001", "scene_000"]


def test_script_order_applies_only_with_high_confidence_full_alignment():
    items = [_item("scene_000", 0), _item("scene_001", 1)]
    alignment = [
        ScriptAlignmentItem(scene_id="scene_000", script_beat_index=1, confidence=0.9),
        ScriptAlignmentItem(scene_id="scene_001", script_beat_index=0, confidence=0.9),
    ]
    resolution = resolve_order(items, policy=OrderPolicy.SCRIPT, script_alignment=alignment)
    assert resolution.policy_applied == OrderPolicy.SCRIPT
    assert [i.id for i in resolution.ordered_items] == ["scene_001", "scene_000"]


def test_script_order_falls_back_when_confidence_insufficient():
    items = [_item("scene_000", 0), _item("scene_001", 1)]
    alignment = [
        ScriptAlignmentItem(scene_id="scene_000", script_beat_index=None, confidence=0.0),
        ScriptAlignmentItem(scene_id="scene_001", script_beat_index=None, confidence=0.0),
    ]
    resolution = resolve_order(items, policy=OrderPolicy.SCRIPT, script_alignment=alignment)
    assert resolution.policy_applied == OrderPolicy.PRESERVE
    assert [i.id for i in resolution.ordered_items] == ["scene_000", "scene_001"]
    assert resolution.warnings
