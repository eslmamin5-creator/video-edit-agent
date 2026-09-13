"""B-roll planner (spec section 20): orchestrates selection + the provider
fallback chain (user assets -> local library -> project assets -> generated
image -> generated video -> none), never failing the project if no B-roll
can be found -- BrollSourceKind.NONE is an accepted, normal outcome.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.library import default_library_dir
from video_edit_agent.broll.providers import gemini_image, local, veo
from video_edit_agent.broll.selector import select_broll_moments
from video_edit_agent.core.schemas import EDL, BrollPlanItem, BrollSourceKind, Transcript


def _apply_brand_aesthetic(item: BrollPlanItem, brand: Brand | None) -> BrollPlanItem:
    if brand and brand.broll_aesthetic:
        item.prompt = f"{item.recommended_visual}. Style: {brand.broll_aesthetic}."
    return item


def plan_broll(
    edl: EDL,
    transcript: Transcript,
    project_broll_dir: Path,
    generated_output_dir: Path,
    brand: Brand | None = None,
    allow_generation: bool = True,
) -> list[BrollPlanItem]:
    """Builds the full B-roll plan for a project, resolving each candidate
    slot through the provider priority chain until one succeeds or all are
    exhausted (leaving BrollSourceKind.NONE)."""
    candidates = select_broll_moments(edl, transcript)
    library_dir = default_library_dir()

    resolved: list[BrollPlanItem] = []
    for item in candidates:
        item = _apply_brand_aesthetic(item, brand)

        item = local.find_broll(item, project_broll_dir, library_dir)
        if item.source != BrollSourceKind.NONE:
            resolved.append(item)
            continue

        if allow_generation:
            item = gemini_image.generate_broll_image(item, generated_output_dir)
            if item.source != BrollSourceKind.NONE:
                resolved.append(item)
                continue

            item = veo.generate_broll_video(item, generated_output_dir)
            if item.source != BrollSourceKind.NONE:
                resolved.append(item)
                continue

        resolved.append(item)  # stays BrollSourceKind.NONE -- acceptable outcome

    return resolved
