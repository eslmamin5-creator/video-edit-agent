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

from video_edit_agent.core.schemas import DEFAULT_ZOOM_ANCHOR_Y, EDL, Reframe, ZoomRamp

# Ported verbatim from 03_cut_zoom.py's `Z` (normal pace) and `Z` under
# `CALM` (calm pace) lists.
ZOOM_LEVELS_NORMAL: list[float] = [1.00, 1.08, 1.00, 1.06, 1.00, 1.12, 1.04, 1.14, 1.00, 1.08, 1.00, 1.05, 1.10, 1.00]
ZOOM_LEVELS_CALM: list[float] = [1.00, 1.00, 1.04, 1.00, 1.00, 1.06, 1.00, 1.03]

# Ported from 03_cut_zoom.py's `ANCH` -- fraction of the leftover (cropped-
# away) height placed above the crop window, i.e. how far down the frame the
# zoom is anchored (0.30 keeps a head-and-shoulders subject in frame when
# zoomed in past the top of a typical vertical talking-head shot).
ZOOM_ANCHOR_Y = DEFAULT_ZOOM_ANCHOR_Y

# Half-width in seconds of the eased zoom move centred on a cut: the outgoing
# clip starts easing this long before the cut and the incoming clip finishes
# this long after it, so the framing never steps at the join.
RAMP_HALF_S = 0.3


def plan_punch_ins(
    edl: EDL,
    *,
    energy: str = "medium",
    face_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Assigns a cycling per-clip punch-in `zoom` to every clip in `edl`, in
    place, plus the `Reframe` that makes each change of level an eased move
    centred on the cut (see `render/reframe.py`; the final render and the
    micro-previews both play this same plan). `energy="low"` (from
    `Brand.motion.energy`) selects the calmer, smaller-and-less-frequent
    zoom-level list; anything else uses the normal list. `face_box` (normalized
    x, y, w, h) makes every reframe face-safe. A single-clip EDL is left at zoom
    1.0 for every clip (nothing to punch in *from*)."""
    if len(edl.clips) < 2:
        return
    levels = ZOOM_LEVELS_CALM if energy == "low" else ZOOM_LEVELS_NORMAL
    zooms = [levels[i % len(levels)] for i in range(len(edl.clips))]
    clips = edl.clips
    # Half-width of the move at boundary i (between clip i-1 and clip i); 0 when
    # the level does not change there.
    half = [0.0] * len(clips)
    for i in range(1, len(clips)):
        if abs(zooms[i] - zooms[i - 1]) > 1e-6:
            half[i] = min(RAMP_HALF_S, clips[i - 1].duration / 2, clips[i].duration / 2)
    for i, clip in enumerate(clips):
        clip.zoom = zooms[i]
        ramps: list[ZoomRamp] = []
        zoom_start = zooms[i]
        if half[i] > 0:
            zoom_start = zooms[i - 1]
            ramps.append(ZoomRamp(start_s=-half[i], end_s=half[i], zoom_to=zooms[i]))
        if i + 1 < len(clips) and half[i + 1] > 0:
            ramps.append(ZoomRamp(
                start_s=clip.duration - half[i + 1], end_s=clip.duration + half[i + 1], zoom_to=zooms[i + 1],
            ))
        clip.reframe = Reframe(zoom_start=zoom_start, ramps=ramps, face_box=face_box)
