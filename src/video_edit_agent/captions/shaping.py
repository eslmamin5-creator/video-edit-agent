"""Arabic shaping entry point for the caption engine — thin wrapper so
captions code never imports `language.arabic` directly (keeps the module
boundary clean per spec's separation-of-concerns guidance)."""
from __future__ import annotations

from video_edit_agent.language.arabic import shape_for_display

shape_line = shape_for_display
