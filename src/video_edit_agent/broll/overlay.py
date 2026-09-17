"""Shared B-roll -> render `Overlay` conversion (Baseline Recovery Milestone
item 1). Reference behavior: `agents/creator/timeline.py::build_creator_timeline`
resolves a `BrollPlanItem` and (via `agents/creator/render.py::render_creator_timeline`,
line 77-84) turns it into an `Overlay(path=..., start=item.start, end=item.end,
scale_to_canvas=True)`. The Editor workflow (`core/pipeline.py`) resolved B-roll
through the same `BrollPlanItem` model via `broll.planner.plan_broll()` but never
performed this conversion, so resolved B-roll never reached the render plan
(`RenderPlan.overlays`) or `final.mp4` -- this module is the extracted, shared
version of that conversion so both workflows stay in sync.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind
from video_edit_agent.render.composition import Overlay


def broll_item_to_overlay(item: BrollPlanItem) -> Overlay | None:
    """Converts one resolved `BrollPlanItem` into a compositable `Overlay`,
    or `None` if the item has no usable asset (unresolved / `NONE` source)."""
    if item.source == BrollSourceKind.NONE or not item.asset_path:
        return None
    return Overlay(
        path=Path(item.asset_path),
        start=item.timeline_start,
        end=item.timeline_end,
        scale_to_canvas=True,
    )


def broll_items_to_overlays(items: list[BrollPlanItem]) -> list[Overlay]:
    """Converts every resolved item in a B-roll plan into overlays, in order,
    silently skipping unresolved (`BrollSourceKind.NONE`) items."""
    overlays: list[Overlay] = []
    for item in items:
        overlay = broll_item_to_overlay(item)
        if overlay is not None:
            overlays.append(overlay)
    return overlays
