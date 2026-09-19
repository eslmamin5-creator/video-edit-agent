"""The one punch-in / reframe implementation shared by the final render and
every preview.

A clip's framing is a `Reframe` (see `core/schemas.py`): a start zoom, a list of
eased `ZoomRamp` moves, an anchor point and an optional face box. This module
turns it into (a) the ffmpeg filters `render/composition.py` puts in each clip's
chain and (b) the same numbers evaluated in Python (`zoom_at`, `crop_window`),
so a test can prove the preview and the final render use one transform.

Micro-previews do not have their own zoom code: they slice the real EDL and
render it through `build_filter_complex`.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import EDLClip, Reframe, ZoomRamp

EASING_SMOOTHSTEP = "smoothstep"
EASING_LINEAR = "linear"
_EPS = 1e-6


def resolve_reframe(clip: EDLClip) -> Reframe | None:
    """The clip's reframe, or None when it is shown unzoomed. A clip with only
    the legacy static `zoom` resolves to a ramp-less reframe."""
    if clip.reframe is not None:
        return clip.reframe if _is_active(clip.reframe) else None
    if clip.zoom and abs(clip.zoom - 1.0) > _EPS:
        return Reframe(zoom_start=clip.zoom)
    return None


def _is_active(reframe: Reframe) -> bool:
    return max_zoom(reframe) > 1.0 + _EPS or abs(reframe.zoom_start - 1.0) > _EPS


def is_static(reframe: Reframe) -> bool:
    return not reframe.ramps


def max_zoom(reframe: Reframe) -> float:
    level = reframe.zoom_start
    peak = level
    for ramp in reframe.ramps:
        level = ramp.zoom_to
        peak = max(peak, level)
    return peak


def zoom_end(reframe: Reframe) -> float:
    return reframe.ramps[-1].zoom_to if reframe.ramps else reframe.zoom_start


def _ease(fraction: float, easing: str) -> float:
    s = min(1.0, max(0.0, fraction))
    if easing == EASING_LINEAR:
        return s
    return s * s * (3.0 - 2.0 * s)


def zoom_at(reframe: Reframe, t: float) -> float:
    """Zoom factor at clip-local time `t` (the Python twin of `zoom_expr`)."""
    zoom = reframe.zoom_start
    level = reframe.zoom_start
    for ramp in reframe.ramps:
        span = max(ramp.end_s - ramp.start_s, 1e-3)
        zoom += (ramp.zoom_to - level) * _ease((t - ramp.start_s) / span, ramp.easing)
        level = ramp.zoom_to
    return zoom


def safe_anchor(reframe: Reframe) -> tuple[float, float]:
    """The anchor actually used: the requested one, nudged only as far as needed
    to keep `face_box` (with its margin) inside the crop window at the tightest
    zoom of the move. A face larger than the window is centred instead."""
    ax, ay = reframe.anchor_x, reframe.anchor_y
    if reframe.face_box is None:
        return ax, ay
    fx, fy, fw, fh = reframe.face_box
    frac = 1.0 / max_zoom(reframe)
    return (
        _clamp_axis(ax, fx, fw, frac, reframe.face_margin),
        _clamp_axis(ay, fy, fh, frac, reframe.face_margin),
    )


def _clamp_axis(anchor: float, start: float, size: float, frac: float, margin: float) -> float:
    free = 1.0 - frac
    if free <= _EPS:
        return anchor
    # crop window offset is free*anchor; it must satisfy
    #   offset + margin <= start  and  offset + frac >= start + size + margin
    hi = (start - margin) / free
    lo = (start + size + margin - frac) / free
    if lo > hi:
        return min(1.0, max(0.0, (lo + hi) / 2.0))
    return min(1.0, max(0.0, min(hi, max(lo, anchor))))


def crop_window(reframe: Reframe, t: float, width: int, height: int) -> tuple[float, float, float, float]:
    """(x, y, w, h) in source pixels of the region shown at clip-local time `t`."""
    z = zoom_at(reframe, t)
    ax, ay = safe_anchor(reframe)
    w, h = width / z, height / z
    return (width - w) * ax, (height - h) * ay, w, h


def _ease_expr(ramp: ZoomRamp) -> str:
    span = max(ramp.end_s - ramp.start_s, 1e-3)
    s = f"clip((t-{ramp.start_s:.3f})/{span:.3f},0,1)"
    if ramp.easing == EASING_LINEAR:
        return s
    return f"{s}*{s}*(3-2*{s})"


def zoom_expr(reframe: Reframe) -> str:
    """ffmpeg expression (variable `t` = clip-local seconds) for the zoom."""
    terms = [f"{reframe.zoom_start:.4f}"]
    level = reframe.zoom_start
    for ramp in reframe.ramps:
        terms.append(f"({ramp.zoom_to - level:.4f})*{_ease_expr(ramp)}")
        level = ramp.zoom_to
    return "(" + "+".join(terms) + ")"


def reframe_filters(
    reframe: Reframe, src_label: str, out_label: str, width: int, height: int,
    fps: float, duration: float, tag: str,
) -> list[str]:
    """Filter-graph lines that reframe `[src_label]` (already at width x height,
    clip-local timestamps) into `[out_label]`.

    A static reframe is a plain crop + scale. An eased one scales every frame by
    the zoom expression and overlays it on a black canvas, offset so the anchor
    point stays put -- the frame-accurate equivalent of the same crop window
    (`crop_window`)."""
    ax, ay = safe_anchor(reframe)
    if is_static(reframe):
        z = reframe.zoom_start
        crop_w = round(width / z / 2) * 2
        crop_h = round(height / z / 2) * 2
        crop_x = round((width - crop_w) * ax)
        crop_y = round((height - crop_h) * ay)
        return [(
            f"[{src_label}]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={width}:{height}:flags=lanczos,setsar=1[{out_label}]"
        )]
    z = zoom_expr(reframe)
    return [
        (
            f"[{src_label}]scale=w='trunc({width}*{z}/2)*2':h='trunc({height}*{z}/2)*2'"
            f":eval=frame:flags=bicubic[rfz{tag}]"
        ),
        f"color=c=black:s={width}x{height}:r={fps:g}:d={duration + 1.0:.3f}[rfbg{tag}]",
        (
            f"[rfbg{tag}][rfz{tag}]overlay=x='-(w-{width})*{ax:.4f}':y='-(h-{height})*{ay:.4f}'"
            f":eval=frame:shortest=1,format=yuv420p,setsar=1[{out_label}]"
        ),
    ]
