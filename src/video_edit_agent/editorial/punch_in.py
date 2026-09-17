"""Talking-head punch-in/reframing (Baseline Recovery Milestone item 6).

Reference: `majedphotos/video-ad-editor`'s `scripts/03_cut_zoom.py` assigns a
different zoom level to each successive kept segment (cycling through a fixed
list, `Z`), cropped centered horizontally and anchored toward the lower third
vertically (`ANCH=0.30`), then scaled back up to the output canvas -- this is
what keeps a static, single-camera talking-head take visually alive instead
of one unbroken flat shot. A "calm" pace variant (`CALM`, its own `Z` list,
fewer/smaller zoom changes) is also ported here, keyed off this project's own
`Brand.motion.energy` field (`"low"`) instead of the reference's
`theme.json`'s `"pace":"calm"` flag -- the closest existing equivalent in
this project's Brand Profile system (spec section 23).

**Classification: PORTED** (the zoom-level lists and the per-segment cycling
logic), see `THIRD_PARTY_NOTICES.md`. The actual cropping/scaling is applied
in `render/composition.py::build_filter_complex` on `EDLClip.zoom`, ported
into this project's own per-clip ffmpeg filter chain rather than the
reference's `filter_complex` string built directly.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import EDL

# Ported verbatim from 03_cut_zoom.py's `Z` (normal pace) and `Z` under
# `CALM` (calm pace) lists.
ZOOM_LEVELS_NORMAL: list[float] = [1.00, 1.08, 1.00, 1.06, 1.00, 1.12, 1.04, 1.14, 1.00, 1.08, 1.00, 1.05, 1.10, 1.00]
ZOOM_LEVELS_CALM: list[float] = [1.00, 1.00, 1.04, 1.00, 1.00, 1.06, 1.00, 1.03]

# Ported from 03_cut_zoom.py's `ANCH` -- fraction of the leftover (cropped-
# away) height placed above the crop window, i.e. how far down the frame the
# zoom is anchored (0.30 keeps a head-and-shoulders subject in frame when
# zoomed in past the top of a typical vertical talking-head shot).
ZOOM_ANCHOR_Y = 0.30


def plan_punch_ins(edl: EDL, *, energy: str = "medium") -> None:
    """Assigns a cycling per-clip punch-in `zoom` to every clip in `edl`, in
    place. `energy="low"` (from `Brand.motion.energy`) selects the calmer,
    smaller-and-less-frequent zoom-level list; anything else uses the normal
    list. A single-clip EDL is left at zoom 1.0 for every clip (nothing to
    punch in *from*)."""
    if len(edl.clips) < 2:
        return
    levels = ZOOM_LEVELS_CALM if energy == "low" else ZOOM_LEVELS_NORMAL
    for i, clip in enumerate(edl.clips):
        clip.zoom = levels[i % len(levels)]
