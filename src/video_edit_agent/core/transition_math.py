"""Shared, conservative bounds for real rendered video transitions (Phase 2
Finalization spec section 3).

Both the Assembler's EDL builder (which must report an honest timeline
duration that accounts for crossfade overlap) and the shared render core
(which must render exactly the overlap it reports) import these from one
place so the two can never drift apart.
"""
from __future__ import annotations

MIN_TRANSITION_S = 0.15
MAX_TRANSITION_S = 0.5


def clamp_transition_duration(requested_s: float, dur_a: float, dur_b: float) -> float:
    """Clamps a requested crossfade overlap into the conservative
    0.15-0.5s band, and further caps it so the overlap never exceeds either
    neighboring clip's own duration -- which would desync the timeline."""
    if requested_s <= 0 or dur_a <= 0 or dur_b <= 0:
        return 0.0
    d = max(MIN_TRANSITION_S, min(MAX_TRANSITION_S, requested_s))
    d = min(d, dur_a * 0.9, dur_b * 0.9)
    return max(0.0, d)
