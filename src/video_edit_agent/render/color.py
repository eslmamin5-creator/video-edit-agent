"""Minimal, safe color handling. Keeps source color untouched by default —
this project does not attempt automatic color grading in V1; it only
normalizes pixel format so overlays/captions composite predictably."""
from __future__ import annotations


def safe_format_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"format=yuv420p"
    )
