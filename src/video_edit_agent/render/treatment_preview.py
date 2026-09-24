"""Short previews of the treatments a user reviews (Phase 1.6): the camera motion and a headline lock.

Both render real footage through the same modules the final render uses (planned `Reframe`s, the caption ASS builder,
the locked-headline burner), for a few seconds around the treatment. Nothing is project-specific.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from video_edit_agent.captions.engine import build_ass
from video_edit_agent.captions.modes import REDUCED, apply_behavior
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.schemas import EDL, Transcript
from video_edit_agent.direction.production_profile import burn_locked_headline
from video_edit_agent.render import reframe as reframe_math
from video_edit_agent.render.composition import CaptionBurn, RenderPlan
from video_edit_agent.render.export import resolve_preset
from video_edit_agent.render.ffmpeg import RenderError, render
from video_edit_agent.render.micro_preview import (
    MicroPreviewError,
    render_punch_in,
    slice_edl,
    timeline_to_source_window,
)
from video_edit_agent.review.locked_treatments import LockedTreatment, shift_lock

_EPS = 1e-6
AFTER_HOOK_S = 4.0  # a motion preview avoids the opening seconds, where the hook title sits


def planned_peak_zoom(edl: EDL, window: tuple[float, float]) -> float:
    """The highest zoom the plan reaches inside the timeline `window`."""
    peak = 1.0
    for clip in edl.clips:
        if clip.timeline_out <= window[0] or clip.timeline_in >= window[1]:
            continue
        rf = reframe_math.resolve_reframe(clip)
        if rf is not None:
            peak = max(peak, reframe_math.max_zoom(rf))
    return round(peak, 4)


def pick_motion_window(edl: EDL, *, before_s: float = 1.0, after_s: float = 3.5) -> tuple[float, float] | None:
    """The timeline window around the strongest planned zoom move (preferring one after the hook), or None when the
    plan has no zoom at all."""
    best: tuple[tuple[bool, float, float], float] | None = None
    for clip in edl.clips:
        rf = reframe_math.resolve_reframe(clip)
        if rf is None:
            continue
        moves = [(clip.timeline_in + r.start_s, r.zoom_to) for r in rf.ramps if r.zoom_to > 1.0 + _EPS]
        if rf.zoom_start > 1.0 + _EPS:
            moves.append((clip.timeline_in, rf.zoom_start))
        for t, z in moves:
            key = (t >= AFTER_HOOK_S, z, -t)
            if best is None or key > best[0]:
                best = (key, t)
    if best is None:
        return None
    t = best[1]
    start = max(0.0, t - before_s)
    return round(start, 3), round(min(edl.total_duration, start + before_s + after_s), 3)


def fallback_window(edl: EDL, length: float = 4.5) -> tuple[float, float]:
    start = min(AFTER_HOOK_S, max(0.0, edl.total_duration - length))
    return round(start, 3), round(min(edl.total_duration, start + length), 3)


def render_camera_preview(edl: EDL, window: tuple[float, float], out_path: Path, *, preset: str = "reel") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render_punch_in(edl, timeline_to_source_window(edl, *window), out_path, preset=preset)


def locked_headline_ass(transcript: Transcript, edl: EDL, caption_style: CaptionStyle, lock: LockedTreatment) -> str:
    """The caption ASS of a headline lock: normal captions, reduced inside the locked window, headline burned in."""
    recipe = lock.headline
    normal = build_ass(transcript, edl, caption_style, safe_zone=None)
    reduced_style = (
        dataclasses.replace(caption_style, font_size=round(caption_style.font_size * recipe.caption_reduction_scale))
        if recipe.caption_reduction_scale else apply_behavior(caption_style, REDUCED)
    )
    reduced = build_ass(transcript, edl, reduced_style, safe_zone=None)
    return burn_locked_headline(normal, reduced, lock, caption_style, edl.width)


def headline_preview_path(review_dir: Path, lock: LockedTreatment) -> Path:
    """Deterministic per recipe, so an unchanged pending headline is never re-rendered."""
    import hashlib

    digest = hashlib.sha256(lock.model_dump_json().encode("utf-8")).hexdigest()[:10]
    return Path(review_dir) / "micro_previews" / f"headline_{lock.treatment_id}_{digest}.mp4"


def headline_preview_window(lock: LockedTreatment, edl: EDL, *, pad_s: float = 1.0) -> tuple[float, float]:
    r = lock.headline
    return round(max(0.0, r.caption_reduction_window[0] - pad_s), 3), round(min(edl.total_duration, r.fade_out[1] + pad_s), 3)


def render_headline_preview(
    lock: LockedTreatment, transcript: Transcript, edl: EDL, caption_style: CaptionStyle, out_path: Path, *,
    fonts_dir: Path | None = None, preset: str = "reel", workdir: Path | None = None,
) -> tuple[Path, tuple[float, float]]:
    """Real footage of the headline's window, planned camera included, with the locked headline and the reduced
    captions burned in exactly as the final render burns them. Returns (path, timeline window)."""
    window = headline_preview_window(lock, edl)
    sub = slice_edl(edl, *timeline_to_source_window(edl, *window), zoom=None)
    ass = locked_headline_ass(transcript, sub, caption_style, shift_lock(lock, window[0]))
    workdir = workdir or out_path.parent / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    ass_path = workdir / f"{out_path.stem}.ass"
    ass_path.write_text(ass, encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render(RenderPlan(edl=sub, captions=CaptionBurn(ass_path=ass_path, fonts_dir=fonts_dir)), out_path, resolve_preset(preset), crf=22)
    except RenderError as exc:
        raise MicroPreviewError(str(exc)) from exc
    return out_path, window
