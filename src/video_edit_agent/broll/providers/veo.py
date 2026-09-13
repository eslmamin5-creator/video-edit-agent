"""Generated-video B-roll provider via Veo (spec section 20, the last and
most expensive priority tier before giving up entirely and leaving the B-roll
slot empty -- which is itself an acceptable outcome, never a hard failure).
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind
from video_edit_agent.providers.gemini_client import GeminiRequestError, GeminiUnavailable
from video_edit_agent.providers.gemini_client import generate_video as _generate_video
from video_edit_agent.providers.gemini_client import is_available as gemini_available


def is_available() -> bool:
    return gemini_available()


def generate_broll_video(item: BrollPlanItem, output_dir: Path) -> BrollPlanItem:
    """Attempts to fill `item` with a short generated video clip. Leaves the
    item untouched on any failure -- the caller should treat that as
    BrollSourceKind.NONE and proceed without B-roll for that slot."""
    if not is_available():
        return item

    prompt = item.prompt or f"{item.recommended_visual}. Cinematic, {item.aspect_ratio} aspect ratio."
    output_path = output_dir / f"generated_{abs(hash((item.timeline_start, item.timeline_end)))}.mp4"
    try:
        result_path = _generate_video(prompt, output_path)
    except (GeminiUnavailable, GeminiRequestError):
        return item

    item.source = BrollSourceKind.GENERATED_VIDEO
    item.asset_path = str(result_path)
    item.confidence = 0.45
    return item
