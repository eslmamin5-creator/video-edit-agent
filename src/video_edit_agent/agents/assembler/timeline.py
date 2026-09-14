"""MasterTimeline construction for Assembler (spec section 8): adapts the
scene EDL via the existing shared `edl_to_master_timeline()` adapter --
no Assembler-only timeline format.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import EDL
from video_edit_agent.core.timeline import MasterTimeline, edl_to_master_timeline


def build_assembler_timeline(edl: EDL) -> MasterTimeline:
    return edl_to_master_timeline(edl, workflow="assembler")
