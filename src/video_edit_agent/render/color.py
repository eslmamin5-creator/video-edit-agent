"""Minimal, safe color handling. Keeps source color untouched by default —
this project does not attempt automatic color grading in V1; it only
normalizes pixel format so overlays/captions composite predictably."""
from __future__ import annotations


def safe_format_filter(width: int, height: int) -> str:
    # setsar=1 guards against ffmpeg's scale filter leaving a rounding-induced
    # non-square SAR (e.g. 10240:10239) on some source aspect ratios, which
    # otherwise makes `concat` reject streams whose SAR doesn't exactly match.
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"setsar=1,"
        f"format=yuv420p"
    )
