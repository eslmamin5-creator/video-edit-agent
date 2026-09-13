"""Generated-image B-roll provider (spec section 20, priority tier 4: after
user assets, local library, project assets -- before generated video)."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind
from video_edit_agent.providers.gemini_client import GeminiRequestError, GeminiUnavailable
from video_edit_agent.providers.gemini_client import generate_image as _generate_image
from video_edit_agent.providers.gemini_client import is_available as gemini_available


def is_available() -> bool:
    return gemini_available()


def generate_broll_image(item: BrollPlanItem, output_dir: Path) -> BrollPlanItem:
    """Attempts to fill `item` with a generated still image. Leaves the item
    untouched (source stays whatever it was) on any failure so the caller's
    provider router can continue to the next tier."""
    if not is_available():
        return item

    prompt = item.prompt or f"{item.recommended_visual}. Cinematic, {item.aspect_ratio} aspect ratio."
    output_path = output_dir / f"generated_{abs(hash((item.timeline_start, item.timeline_end)))}.png"
    try:
        result_path = _generate_image(prompt, output_path)
    except (GeminiUnavailable, GeminiRequestError):
        return item

    item.source = BrollSourceKind.GENERATED_IMAGE
    item.asset_path = str(result_path)
    item.confidence = 0.5
    return item
