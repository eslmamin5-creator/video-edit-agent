"""Baseline Recovery Milestone item 1: resolved B-roll must convert into a
render `Overlay` and actually enter the render plan/final composition,
instead of only existing in `broll_plan.json` (the Reference-First Audit's
"Editor B-roll dangling" bug)."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.broll.overlay import broll_item_to_overlay, broll_items_to_overlays
from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind


def _item(source: BrollSourceKind, asset_path: str | None, start: float = 1.0, end: float = 3.0) -> BrollPlanItem:
    return BrollPlanItem(
        timeline_start=start,
        timeline_end=end,
        purpose="context",
        spoken_concept="a busy office",
        recommended_visual="office b-roll",
        source=source,
        asset_path=asset_path,
    )


def test_resolved_broll_item_converts_to_overlay_matching_its_timing():
    item = _item(BrollSourceKind.LOCAL_LIBRARY, "/assets/office.mp4", start=2.5, end=6.0)
    overlay = broll_item_to_overlay(item)
    assert overlay is not None
    assert overlay.path == Path("/assets/office.mp4")
    assert overlay.start == 2.5
    assert overlay.end == 6.0
    assert overlay.scale_to_canvas is True


def test_unresolved_broll_item_produces_no_overlay():
    assert broll_item_to_overlay(_item(BrollSourceKind.NONE, None)) is None


def test_resolved_item_with_missing_asset_path_produces_no_overlay():
    assert broll_item_to_overlay(_item(BrollSourceKind.LOCAL_LIBRARY, None)) is None


def test_items_to_overlays_skips_unresolved_and_keeps_order():
    items = [
        _item(BrollSourceKind.LOCAL_LIBRARY, "/a.mp4", start=0.0, end=2.0),
        _item(BrollSourceKind.NONE, None, start=2.0, end=4.0),
        _item(BrollSourceKind.GENERATED_IMAGE, "/b.png", start=4.0, end=6.0),
    ]
    overlays = broll_items_to_overlays(items)
    assert [o.path for o in overlays] == [Path("/a.mp4"), Path("/b.png")]
    assert all(o.scale_to_canvas for o in overlays)
