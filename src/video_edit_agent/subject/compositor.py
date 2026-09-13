"""Behind-subject compositing (spec section 16): places a motion graphic
or B-roll overlay BEHIND the on-camera subject rather than on top of it, by
cutting the subject out of the top layer with the segmentation mask.

Graceful degradation is central here: if segmentation is unavailable, or the
mask confidence is too low, `plan_overlay()` returns a plain foreground
overlay (the pre-existing, always-safe behavior) instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
