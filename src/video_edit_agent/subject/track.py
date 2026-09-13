"""Interpolates sparsely sampled subject masks onto a target frame rate
(spec section 16). Segmentation runs at a reduced sample rate for
performance; this module fills in the gaps with nearest/linear interpolation
so the compositor can request a mask for any timestamp on the clip.
"""
from __future__ import annotations

import bisect

import numpy as np

from video_edit_agent.subject.segment import SegmentedClip


class MaskTrack:
    """Answers `mask_at(t)` for a SegmentedClip, interpolating between the
    two nearest sampled masks (or returning the nearest one if only one
    sample exists / t is out of range)."""

    def __init__(self, clip: SegmentedClip):
        self._times = clip.frame_times
        self._masks = clip.masks

    @property
    def has_masks(self) -> bool:
        return bool(self._masks)

    def mask_at(self, t: float) -> np.ndarray | None:
        if not self._masks:
            return None
        if len(self._masks) == 1:
            return self._masks[0]

        idx = bisect.bisect_left(self._times, t)
        if idx <= 0:
            return self._masks[0]
        if idx >= len(self._times):
            return self._masks[-1]

        t0, t1 = self._times[idx - 1], self._times[idx]
        m0, m1 = self._masks[idx - 1], self._masks[idx]
        if t1 == t0:
            return m0
        alpha = (t - t0) / (t1 - t0)
        return (1.0 - alpha) * m0 + alpha * m1
