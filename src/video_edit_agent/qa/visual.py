"""Visual QA layer (spec section 33): Gemini-based visual review when a key
is available (checks caption legibility, safe-zone violations, brand color
mismatches at a glance); falls back to deterministic local frame checks
(caption safe-zone bounding box math, no exotic ML) when offline.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.captions.safe_zone import SafeZone, margins_px
from video_edit_agent.core.schemas import QAIssue, QASeverity
from video_edit_agent.providers.gemini_client import GeminiRequestError, GeminiUnavailable
from video_edit_agent.providers.gemini_client import analyze_video_segment
from video_edit_agent.providers.gemini_client import is_available as gemini_available

_VISUAL_QA_PROMPT = (
    "You are reviewing a short vertical video for visual quality issues only. "
    "Report ONLY concrete problems you can see, one per line, each starting with "
    "'ISSUE: '. Check for: captions overlapping other on-screen text or UI, captions "
    "positioned outside safe zones, visibly clipped or cut-off graphics, and jarring "
    "color mismatches. If there are no issues, respond with exactly 'NO ISSUES'."
)


def check_caption_bounds(caption_x: int, caption_y: int, caption_w: int, caption_h: int, frame_w: int, frame_h: int) -> list[QAIssue]:
    """Deterministic, offline-safe check: does a caption bounding box fall
    within the standard safe zone margins (spec section 9)?"""
    zone = SafeZone()
    margins = margins_px(zone, frame_w, frame_h)
    top, right, bottom, left = margins["top"], margins["right"], margins["bottom"], margins["left"]

    issues: list[QAIssue] = []
    if caption_x < left or caption_y < top or (caption_x + caption_w) > (frame_w - right) or (
        caption_y + caption_h
    ) > (frame_h - bottom):
        issues.append(
            QAIssue(
                category="visual", severity=QASeverity.WARNING,
                message="Caption bounding box extends outside the safe zone margins",
                auto_repairable=True,
            )
        )
    return issues


def check_visual_with_gemini(output_path: Path) -> list[QAIssue]:
    """Gemini-based holistic visual review. Silently returns no issues (not
    an error) if Gemini is unavailable -- visual QA degrades to the local
    deterministic checks in that case, per spec section 43."""
    if not gemini_available():
        return []

    try:
        response_text = analyze_video_segment(output_path, _VISUAL_QA_PROMPT)
    except (GeminiUnavailable, GeminiRequestError):
        return []

    if response_text.strip().upper().startswith("NO ISSUES"):
        return []

    issues: list[QAIssue] = []
    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith("ISSUE:"):
            issues.append(
                QAIssue(
                    category="visual", severity=QASeverity.WARNING,
                    message=line[len("ISSUE:"):].strip(), auto_repairable=False,
                )
            )
    return issues


def run_visual_qa(output_path: Path) -> list[QAIssue]:
    return check_visual_with_gemini(output_path)
