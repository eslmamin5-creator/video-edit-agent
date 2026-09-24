"""Micro-previews: seconds-long real renders that let a reviewer judge motion
(caption sync, hook, punch-in, end card) without a full render.

Every preview cuts the real source footage + audio for one short window and
renders only the treatment under review through the same code the final
render uses (the caption ASS builder + libass burn-in, the end-card renderer,
the shared ffmpeg filter graph). Nothing here generates B-roll, renders other
motion assets, or touches the review gate; a preview is a throwaway artifact
under `<edit>/review/micro_previews/`.

Nothing is source-, brand- or project-specific: windows, styles, cards and
texts are all arguments.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.engine import build_ass
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.media import run
from video_edit_agent.core.schemas import EDL, AnimationSpec, EDLClip, Reframe, Transcript
from video_edit_agent.motion.behind_subject import BehindSubjectComposite, compose_behind_subject
from video_edit_agent.render.composition import CaptionBurn, Overlay, RenderPlan
from video_edit_agent.render.export import resolve_preset
from video_edit_agent.render.ffmpeg import RenderError, render

MIN_CAPTION_SYNC_S = 4.0
MAX_CAPTION_SYNC_S = 6.0


class MicroPreviewError(RuntimeError):
    pass


def _shift_reframe(reframe: Reframe | None, shift_s: float) -> Reframe | None:
    """`reframe` re-based to a clip that now starts `shift_s` later in the
    original clip. The zoom is a sum of eased moves, so moving every ramp
    earlier by `shift_s` reproduces the original zoom at every frame."""
    if reframe is None or shift_s <= 0:
        return reframe
    ramps = [r.model_copy(update={"start_s": r.start_s - shift_s, "end_s": r.end_s - shift_s})
             for r in reframe.ramps]
    return reframe.model_copy(update={"ramps": ramps})


def slice_edl(edl: EDL, start: float, end: float, *, zoom: float | None = 1.0) -> EDL:
    """The part of `edl` whose SOURCE time falls in [start, end], re-based to
    timeline 0. Clip ids for captions are kept so the real caption builder
    can map words. `zoom` overrides every clip's punch-in (default 1.0: a
    preview shows only the treatment under review; None keeps the planned
    reframe, re-based to the slice, so it plays exactly as the final render)."""
    if end <= start:
        raise MicroPreviewError(f"empty window [{start}, {end}]")
    clips: list[EDLClip] = []
    cursor = 0.0
    for clip in edl.clips:
        lo, hi = max(clip.source_in, start), min(clip.source_out, end)
        if hi - lo <= 1e-3:
            continue
        length = hi - lo
        update: dict = {
            "source_in": lo, "source_out": hi,
            "timeline_in": cursor, "timeline_out": cursor + length,
            "transition_duration_s": 0.0,
        }
        if zoom is None:
            speed = clip.speed if clip.speed and clip.speed > 0 else 1.0
            update["reframe"] = _shift_reframe(clip.reframe, (lo - clip.source_in) / speed)
        else:
            update["zoom"] = zoom
            update["reframe"] = None
        clips.append(clip.model_copy(update=update))
        cursor += length
    if not clips:
        raise MicroPreviewError(f"no EDL clip overlaps source window [{start}, {end}]")
    return edl.model_copy(update={"clips": clips})


@dataclass
class CaptionWindow:
    start: float
    end: float
    segment_id: str


def pick_caption_sync_window(
    transcript: Transcript,
    segment_ids: list[str],
    *,
    pad: float = 0.25,
    min_s: float = MIN_CAPTION_SYNC_S,
    max_s: float = MAX_CAPTION_SYNC_S,
) -> CaptionWindow | None:
    """Best `min_s`..`max_s` source window around one of `segment_ids`: the
    segment must fit (with `pad` either side) and the one with the most words
    wins -- more words means more highlight transitions to judge. When a
    segment is shorter than `min_s` the window is widened symmetrically."""
    best: tuple[int, CaptionWindow] | None = None
    for seg in transcript.segments:
        if seg.id not in segment_ids or not seg.words:
            continue
        span = (seg.end - seg.start) + 2 * pad
        if span > max_s:
            continue
        centre = (seg.start + seg.end) / 2
        half = max(span, min_s) / 2
        window = CaptionWindow(start=max(0.0, centre - half), end=centre + half, segment_id=seg.id)
        if best is None or len(seg.words) > best[0]:
            best = (len(seg.words), window)
    return best[1] if best else None


def render_caption_sync(
    transcript: Transcript,
    edl: EDL,
    style: CaptionStyle,
    window: tuple[float, float],
    out_path: Path,
    *,
    fonts_dir: Path | None = None,
    preset: str = "reel",
    workdir: Path | None = None,
    only_segments: list[str] | None = None,
) -> Path:
    """Real footage + audio for `window` with the actual caption renderer
    (ASS builder -> libass) burning word-level highlights. No zoom, no
    overlays. `only_segments` limits the captions to those transcript segments
    (the footage padding around them stays uncaptioned), so a preview never
    shows neighbouring text the reviewer has not confirmed."""
    sub = slice_edl(edl, *window, zoom=1.0)
    if only_segments is not None:
        keep = set(only_segments)
        sub = sub.model_copy(update={"clips": [
            c.model_copy(update={"caption_refs": [r for r in c.caption_refs if r in keep]}) for c in sub.clips
        ]})
    workdir = workdir or out_path.parent / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    ass_path = workdir / f"{out_path.stem}.ass"
    ass_path.write_text(build_ass(transcript, sub, style), encoding="utf-8")
    plan = RenderPlan(edl=sub, captions=CaptionBurn(ass_path=ass_path, fonts_dir=fonts_dir))
    try:
        return render(plan, out_path, resolve_preset(preset), crf=22)
    except RenderError as exc:
        raise MicroPreviewError(str(exc)) from exc


def render_overlay_preview(
    edl: EDL,
    window: tuple[float, float],
    overlay: Overlay,
    out_path: Path,
    *,
    preset: str = "reel",
) -> Path:
    """Real footage + audio for `window` with ONE pre-rendered overlay (e.g.
    the hook title). The overlay's own timing is relative to the window."""
    sub = slice_edl(edl, *window, zoom=1.0)
    plan = RenderPlan(edl=sub, overlays=[overlay])
    try:
        render(plan, out_path, resolve_preset(preset), crf=22)
    except RenderError as exc:
        raise MicroPreviewError(str(exc)) from exc
    return _trim_to(out_path, sub.total_duration)


def timeline_to_source_window(edl: EDL, start: float, end: float) -> tuple[float, float]:
    """The SOURCE window that plays on the timeline between `start` and `end`
    (first covered clip's start to last covered clip's end)."""
    lows, highs = [], []
    for clip in edl.clips:
        lo, hi = max(clip.timeline_in, start), min(clip.timeline_out, end)
        if hi - lo <= 1e-3:
            continue
        speed = clip.speed if clip.speed and clip.speed > 0 else 1.0
        lows.append(clip.source_in + (lo - clip.timeline_in) * speed)
        highs.append(clip.source_in + (hi - clip.timeline_in) * speed)
    if not lows:
        raise MicroPreviewError(f"no EDL clip covers timeline window [{start}, {end}]")
    return min(lows), max(highs)


@dataclass
class PreviewCaptions:
    """The caption renderer's inputs, so a preview shows the treatment under the real
    captions (the hierarchy between them is part of what is being judged)."""

    transcript: Transcript
    style: CaptionStyle
    fonts_dir: Path | None = None


def render_behind_subject_preview(
    edl: EDL,
    spec: AnimationSpec,
    project_root: Path,
    out_path: Path,
    *,
    brand: Brand | None,
    cutout_dir: Path,
    window: tuple[float, float] | None = None,
    captions: PreviewCaptions | None = None,
    workdir: Path | None = None,
    offline: bool = True,
    preset: str = "reel",
    compose: Callable[..., BehindSubjectComposite] = compose_behind_subject,
) -> BehindSubjectComposite:
    """Real footage + audio for the TIMELINE `window` (default: the spec's own show
    window) with the behind-subject text composed by `compose_behind_subject` -- the
    SAME function the final render calls, on the SAME spec (phrase timing, placement,
    colour, opacity, entrance/exit) -- so the preview is the cutout/text layering the
    final video will have, at the plan's own framing. `captions` burns the real
    captions over it. No other overlay."""
    if not spec.behind_subject:
        raise MicroPreviewError("the spec is not a behind-subject treatment")
    workdir = workdir or out_path.parent / "_work"
    workdir.mkdir(parents=True, exist_ok=True)
    window = window or (spec.timeline_start, spec.timeline_end)
    composite = compose(
        spec, edl, project_root, workdir / "motion", cutout_dir,
        brand=brand, fps=edl.fps, slot_id=f"{out_path.stem}", offline=offline,
    )
    if not composite.overlays:
        detail = composite.item.error or "; ".join(composite.item.fallback_log) or "no output"
        raise MicroPreviewError(f"the behind-subject graphic could not be rendered: {detail}")
    sub = slice_edl(edl, *timeline_to_source_window(edl, *window), zoom=None)
    shift = window[0]
    overlays = [replace(o, start=o.start - shift, end=o.end - shift) for o in composite.overlays]
    burn = None
    if captions is not None:
        ass_path = workdir / f"{out_path.stem}.ass"
        ass_path.write_text(build_ass(captions.transcript, sub, captions.style), encoding="utf-8")
        burn = CaptionBurn(ass_path=ass_path, fonts_dir=captions.fonts_dir)
    try:
        render(RenderPlan(edl=sub, overlays=overlays, captions=burn), out_path, resolve_preset(preset), crf=22)
    except RenderError as exc:
        raise MicroPreviewError(str(exc)) from exc
    _trim_to(out_path, sub.total_duration)
    return composite


def _trim_to(path: Path, duration: float) -> Path:
    """An overlay clip longer than the footage window (a motion composition
    has its own length) keeps the output running past the footage; cut the
    preview back to the window."""
    trimmed = path.with_name(f"{path.stem}.trim{path.suffix}")
    result = run(["ffmpeg", "-y", "-i", str(path), "-t", f"{duration:.3f}", "-c", "copy",
                  "-movflags", "+faststart", str(trimmed)], timeout=120)
    if result.returncode != 0 or not trimmed.exists():
        raise MicroPreviewError(f"could not trim preview:\n{result.stderr.strip()[-800:]}")
    trimmed.replace(path)
    return path


def render_punch_in(
    edl: EDL,
    window: tuple[float, float],
    out_path: Path,
    *,
    preset: str = "reel",
) -> Path:
    """Real footage + audio for the SOURCE `window`, rendered through the final
    render's own filter graph with the plan's own `Reframe`s (nothing is
    re-derived here), so the preview is the punch-in the final video will have."""
    sub = slice_edl(edl, *window, zoom=None)
    try:
        return render(RenderPlan(edl=sub), out_path, resolve_preset(preset), crf=22)
    except RenderError as exc:
        raise MicroPreviewError(str(exc)) from exc


def pick_punch_in_window(
    edl: EDL, *, before_s: float = 1.0, after_s: float = 3.0,
) -> tuple[float, float, float] | None:
    """(window_start, window_end, cut) in source seconds around the planned cut
    with the biggest zoom change (the punch-in most worth reviewing), or None
    when the plan has no zoom change between neighbouring clips."""
    best: tuple[float, float] | None = None
    for prev, clip in zip(edl.clips, edl.clips[1:]):
        delta = abs(clip.zoom - prev.zoom)
        if delta > 1e-6 and (best is None or delta > best[0]):
            best = (delta, clip.source_in)
    if best is None:
        return None
    cut = best[1]
    return max(0.0, cut - before_s), cut + after_s, cut
