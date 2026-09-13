"""Safe-zone margins so captions never collide with platform UI chrome
(spec sections 15, 38)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafeZone:
    top_pct: float = 0.12
    bottom_pct: float = 0.18  # room for TikTok/Reels caption + engagement bar
    side_pct: float = 0.06


def margins_px(zone: SafeZone, width: int, height: int) -> dict[str, int]:
    return {
        "top": int(height * zone.top_pct),
        "bottom": int(height * zone.bottom_pct),
        "left": int(width * zone.side_pct),
        "right": int(width * zone.side_pct),
    }
