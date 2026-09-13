"""Content-hash-keyed cache for subject masks (spec section 16 + the
project-wide caching strategy). Segmentation is one of the most expensive
per-clip operations, so masks are cached to `edit/.cache/masks/<hash>.npz`
keyed on the source file's content hash plus the requested time window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from video_edit_agent.core.media import content_hash
from video_edit_agent.subject.segment import SegmentedClip


def _cache_key(source_path: Path, start: float, end: float, sample_fps: float) -> str:
    file_hash = content_hash(source_path)
    return f"{file_hash}_{start:.3f}_{end:.3f}_{sample_fps:.2f}"


def cache_path(cache_dir: Path, source_path: Path, start: float, end: float, sample_fps: float) -> Path:
    key = _cache_key(source_path, start, end, sample_fps)
    return cache_dir / f"{key}.npz"


def load(cache_dir: Path, source_path: Path, start: float, end: float, sample_fps: float) -> SegmentedClip | None:
    path = cache_path(cache_dir, source_path, start, end, sample_fps)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        frame_times = list(data["frame_times"])
        masks = [data[f"mask_{i}"] for i in range(len(frame_times))]
        return SegmentedClip(
            source_path=str(source_path), fps_sampled=sample_fps, frame_times=frame_times, masks=masks
        )
    except (OSError, KeyError, ValueError):
        return None


def save(cache_dir: Path, source_path: Path, start: float, end: float, sample_fps: float, clip: SegmentedClip) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, source_path, start, end, sample_fps)
    arrays = {"frame_times": np.array(clip.frame_times, dtype=np.float64)}
    for i, mask in enumerate(clip.masks):
        arrays[f"mask_{i}"] = mask
    np.savez_compressed(path, **arrays)
    return path
