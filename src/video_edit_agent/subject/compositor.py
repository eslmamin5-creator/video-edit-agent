"""Behind-subject compositing (spec section 16): places a motion graphic
or B-roll overlay BEHIND the on-camera subject rather than on top of it, by
cutting the subject out of the top layer with the segmentation mask.

Graceful degradation is central here: if segmentation is unavailable, or the
mask confidence is too low, `plan_overlay()` returns a plain foreground
overlay (the pre-existing, always-safe behavior) instead of failing.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_edit_agent.core.media import content_hash, run
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.subject import mask_cache
from video_edit_agent.subject.detect import SubjectDetectionUnavailable
from video_edit_agent.subject.segment import DEFAULT_SAMPLE_FPS, segment_clip
from video_edit_agent.subject.track import MaskTrack

MIN_USABLE_CONFIDENCE = 0.05


@dataclass
class CompositingPlan:
    behind_subject: bool
    mask_frames_path: Path | None
    reason: str


def plan_behind_subject_overlay(
    source_path: Path,
    start: float,
    end: float,
    cache_dir: Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> CompositingPlan:
    """Attempt to build a behind-subject compositing plan for a clip window.

    Returns `behind_subject=False` (safe fallback to a plain overlay) if:
    - mediapipe/segmentation isn't installed
    - ffmpeg frame extraction failed / clip window is degenerate
    - the resulting masks are essentially empty (no subject detected)
    """
    cached = mask_cache.load(cache_dir, source_path, start, end, sample_fps)
    if cached is not None and cached.masks:
        clip = cached
    else:
        try:
            clip = segment_clip(source_path, start, end, sample_fps)
        except SubjectDetectionUnavailable as exc:
            return CompositingPlan(behind_subject=False, mask_frames_path=None, reason=str(exc))

        if clip.masks:
            mask_cache.save(cache_dir, source_path, start, end, sample_fps, clip)

    if not clip.masks:
        return CompositingPlan(
            behind_subject=False, mask_frames_path=None, reason="no frames extracted for segmentation"
        )

    track = MaskTrack(clip)
    avg_confidence = sum(float(m.mean()) for m in clip.masks) / len(clip.masks)
    if avg_confidence < MIN_USABLE_CONFIDENCE:
        return CompositingPlan(
            behind_subject=False, mask_frames_path=None, reason="no confident subject mask found"
        )

    cache_path = mask_cache.cache_path(cache_dir, source_path, start, end, sample_fps)
    del track  # tracking is used by render-time filter graph construction, not needed further here
    return CompositingPlan(behind_subject=True, mask_frames_path=cache_path, reason="ok")


# --------------------------------------------------------------------------
# Real production compositing (spec Phase 2 section 25)
# --------------------------------------------------------------------------
#
# The base timeline (the concatenated EDL) already contains the on-camera
# subject baked into the full frame together with the background -- there is
# no separately rendered "background-only" plate. So "behind subject"
# compositing is achieved without any matte/keying magic: a graphic overlay
# is drawn on top of the full frame first (which incidentally covers the
# subject), and then a subject-only cutout (RGBA, background transparent via
# the segmentation mask as alpha) is drawn on top of THAT -- restoring the
# subject in front of the graphic. `render/composition.py::build_filter_complex`
# needs no changes for this: ordinary sequential `overlay` filter stacking
# (ffmpeg's overlay filter already respects an RGBA input's alpha channel)
# does the job, as long as the two overlays are appended to the plan in the
# right order (see `core/pipeline.py`).


def find_enclosing_clip(edl: EDL, start: float, end: float) -> EDLClip | None:
    """Finds the single EDL clip whose timeline window fully contains
    [start, end). Behind-subject compositing is only attempted when a motion
    window maps cleanly onto one source clip; if it spans a cut, callers
    should fall back to a plain overlay rather than trying to stitch cutouts
    from two different source clips together."""
    for clip in edl.clips:
        if clip.timeline_in <= start and end <= clip.timeline_out:
            return clip
    return None


def to_source_window(clip: EDLClip, timeline_start: float, timeline_end: float) -> tuple[float, float]:
    """Maps a [timeline_start, timeline_end) window back to source-file time
    for `clip`, accounting for the clip's playback speed."""
    speed = clip.speed if clip.speed else 1.0
    src_start = clip.source_in + (timeline_start - clip.timeline_in) * speed
    src_end = clip.source_in + (timeline_end - clip.timeline_in) * speed
    return src_start, src_end


def _cutout_cache_path(cache_dir: Path, source_path: Path, start: float, end: float, sample_fps: float) -> Path:
    file_hash = content_hash(source_path)
    return cache_dir / f"{file_hash}_{start:.3f}_{end:.3f}_{sample_fps:.2f}_cutout.mov"


def render_subject_cutout(
    source_path: Path,
    start: float,
    end: float,
    cache_dir: Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    output_fps: float = 30.0,
) -> Path | None:
    """Produces (and caches) an RGBA video of just the on-camera subject cut
    out of `source_path` between [start, end), using the segmentation mask
    (interpolated onto every output frame via `MaskTrack`) as the alpha
    channel. Encoded with `qtrle` (lossless, alpha-capable) so it can be
    layered back into the ffmpeg filter graph as a normal overlay input.

    Returns None (never raises) if segmentation is unavailable, extraction
    fails, or no usable mask was found -- callers must fall back to a plain
    foreground overlay (spec section 26) and log the fallback.
    """
    if end <= start:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _cutout_cache_path(cache_dir, source_path, start, end, sample_fps)
    if out_path.exists():
        return out_path

    clip = mask_cache.load(cache_dir, source_path, start, end, sample_fps)
    if clip is None or not clip.masks:
        try:
            clip = segment_clip(source_path, start, end, sample_fps)
        except SubjectDetectionUnavailable:
            return None
        if not clip.masks:
            return None
        mask_cache.save(cache_dir, source_path, start, end, sample_fps, clip)

    avg_confidence = sum(float(m.mean()) for m in clip.masks) / len(clip.masks)
    if avg_confidence < MIN_USABLE_CONFIDENCE:
        return None

    track = MaskTrack(clip)
    duration = end - start

    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frame_pattern = str(tmp_path / "src_%05d.png")
        extract = run(
            [
                "ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(source_path),
                "-vf", f"fps={output_fps}", frame_pattern,
            ],
            timeout=180,
        )
        if extract.returncode != 0:
            return None
        frame_files = sorted(tmp_path.glob("src_*.png"))
        if not frame_files:
            return None

        rgba_dir = tmp_path / "rgba"
        rgba_dir.mkdir()
        for i, frame_path in enumerate(frame_files):
            t = start + i / output_fps
            mask = track.mask_at(t)
            image = Image.open(frame_path).convert("RGB")
            if mask is None:
                alpha = np.zeros((image.height, image.width), dtype=np.uint8)
            elif mask.shape[:2] != (image.height, image.width):
                mask_img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).resize(
                    (image.width, image.height), Image.BILINEAR
                )
                alpha = np.array(mask_img)
            else:
                alpha = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
            rgba = np.dstack([np.array(image), alpha])
            Image.fromarray(rgba, mode="RGBA").save(rgba_dir / f"rgba_{i:05d}.png")

        encode = run(
            [
                "ffmpeg", "-y", "-framerate", str(output_fps), "-i", str(rgba_dir / "rgba_%05d.png"),
                "-c:v", "qtrle", str(out_path),
            ],
            timeout=300,
        )
        if encode.returncode != 0 or not out_path.exists():
            return None

    return out_path
