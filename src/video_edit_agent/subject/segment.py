"""Per-video-clip subject segmentation (spec section 16).

Wraps `detect.py`'s single-frame detector to process a source video clip and
produce a sequence of soft masks, sampling at a reduced rate (not every
frame) since segmentation is CPU/GPU expensive and adjacent frames' masks are
highly correlated -- the compositor interpolates between sampled masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_edit_agent.core.media import run
from video_edit_agent.subject.detect import SubjectDetectionUnavailable, detect_mask

DEFAULT_SAMPLE_FPS = 6.0


@dataclass
class SegmentedClip:
    source_path: str
    fps_sampled: float
    frame_times: list[float]
    masks: list[np.ndarray]


def _extract_sampled_frames(source_path: Path, start: float, end: float, sample_fps: float) -> list[tuple[float, np.ndarray]]:
    import tempfile

    duration = max(0.0, end - start)
    if duration <= 0:
        return []

    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "frame_%05d.png")
        result = run(
            [
                "ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(source_path),
                "-vf", f"fps={sample_fps}", pattern,
            ],
            timeout=120,
        )
        if result.returncode != 0:
            return []

        from PIL import Image

        frames: list[tuple[float, np.ndarray]] = []
        for i, frame_path in enumerate(sorted(Path(tmp).glob("frame_*.png"))):
            timestamp = start + i / sample_fps
            image = Image.open(frame_path).convert("RGB")
            frames.append((timestamp, np.array(image)))
        return frames


def segment_clip(
    source_path: Path, start: float, end: float, sample_fps: float = DEFAULT_SAMPLE_FPS
) -> SegmentedClip:
    """Segments a subject out of a clip window. Raises
    SubjectDetectionUnavailable if the underlying model isn't installed --
    callers should catch this and skip behind-subject compositing entirely."""
    frames = _extract_sampled_frames(source_path, start, end, sample_fps)
    if not frames:
        return SegmentedClip(source_path=str(source_path), fps_sampled=sample_fps, frame_times=[], masks=[])

    frame_times: list[float] = []
    masks: list[np.ndarray] = []
    for timestamp, frame_rgb in frames:
        try:
            result = detect_mask(frame_rgb)
        except SubjectDetectionUnavailable:
            raise
        frame_times.append(timestamp)
        masks.append(result.mask)

    return SegmentedClip(source_path=str(source_path), fps_sampled=sample_fps, frame_times=frame_times, masks=masks)
