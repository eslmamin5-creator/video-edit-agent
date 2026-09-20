"""Deterministic post-processing of subject masks for behind-subject text.

The MediaPipe selfie segmenter returns a soft, slightly blobby mask. Behind a
big word that softness shows up as a halo around the speaker, islands of
background in the mask, and edges that breathe from one sampled frame to the
next. This module cleans the mask in a few fixed steps and nothing else:

    1. `to_canvas`          frame the mask exactly like the render frames the
                            footage (scale-to-cover + centre crop);
    2. `stabilize_temporal` a 3-tap weighted average over the neighbouring
                            samples, so an edge does not flicker between them;
    3. `refine_alpha`       tighten the soft edge with a contrast curve, drop
                            stray islands and fill small holes, then snap the
                            edge to the picture's own edges with a guided
                            filter (hair, collar, shoulders), and pull the edge
                            in a hair so no background rim survives (the halo).

Every step is pure numpy/OpenCV with fixed parameters: the same masks and
frames always produce the same alpha, which is what lets the micro-preview and
the final render agree.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

REFINE_VERSION = "r1"  # bump when the maths changes: it keys the cutout cache


@dataclass(frozen=True)
class RefineParams:
    edge_lo: float = 0.30  # soft mask values below this become background ...
    edge_hi: float = 0.70  # ... and above this become subject; the ramp between is the feather
    island_frac: float = 0.003  # subject islands smaller than this share of the frame are dropped
    hole_frac: float = 0.002  # background holes smaller than this share are filled
    guide_radius: int = 6  # guided-filter window (px at canvas size)
    guide_eps: float = 2e-3
    pull_in: float = 0.05  # alpha below this is zeroed and the rest re-stretched (kills the rim)
    temporal_weights: tuple[float, float, float] = (0.25, 0.5, 0.25)


DEFAULT_PARAMS = RefineParams()


def to_canvas(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`mask` (source-frame HxW, 0..1) framed like the render frames the footage:
    scaled to cover `size` (width, height) and centre-cropped."""
    import cv2

    width, height = size
    src_h, src_w = mask.shape[:2]
    if (src_w, src_h) == (width, height):
        return mask.astype(np.float32, copy=False)
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(width, round(src_w * scale)), max(height, round(src_h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(mask.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
    x0, y0 = (new_w - width) // 2, (new_h - height) // 2
    return np.ascontiguousarray(scaled[y0:y0 + height, x0:x0 + width])


def stabilize_temporal(
    masks: Sequence[np.ndarray], weights: tuple[float, float, float] = DEFAULT_PARAMS.temporal_weights,
) -> list[np.ndarray]:
    """Each mask blended with its neighbours (ends replicate themselves)."""
    n = len(masks)
    if n < 2:
        return [m.astype(np.float32, copy=True) for m in masks]
    w_prev, w_self, w_next = weights
    total = w_prev + w_self + w_next
    out: list[np.ndarray] = []
    for i, m in enumerate(masks):
        prev, nxt = masks[max(i - 1, 0)], masks[min(i + 1, n - 1)]
        out.append(((w_prev * prev + w_self * m + w_next * nxt) / total).astype(np.float32))
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def _clean_regions(binary: np.ndarray, params: RefineParams) -> tuple[np.ndarray, np.ndarray]:
    """(keep, fill) uint8 masks: subject pixels to keep (stray islands removed)
    and background holes to fill. The largest subject component always stays."""
    import cv2

    area = binary.shape[0] * binary.shape[1]
    keep = np.zeros_like(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for label in range(1, count):
            if label == largest or stats[label, cv2.CC_STAT_AREA] >= params.island_frac * area:
                keep[labels == label] = 1
    fill = np.zeros_like(binary)
    inverse = (1 - keep).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    h, w = binary.shape
    for label in range(1, count):
        x, y, bw, bh, a = (int(v) for v in stats[label])
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and a < params.hole_frac * area:
            fill[labels == label] = 1
    return keep, fill


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    import cv2

    size = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, size, borderType=cv2.BORDER_REPLICATE)

    mean_i, mean_p = box(guide), box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def refine_alpha(
    mask: np.ndarray, guide_rgb: np.ndarray | None = None, params: RefineParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Cleaned alpha (float32, 0..1, same shape as `mask`). `guide_rgb` is the
    frame the mask belongs to (uint8 HxWx3, same size); without it the edge is
    tightened and cleaned but not snapped to the picture."""
    import cv2

    m = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    m = _smoothstep(np.clip((m - params.edge_lo) / (params.edge_hi - params.edge_lo), 0.0, 1.0))
    binary = (m > 0.5).astype(np.uint8)
    keep, fill = _clean_regions(binary, params)
    if keep.any():
        # stray islands go (with their feathered rim), small holes are filled
        rim = cv2.dilate((binary & (1 - keep)).astype(np.uint8), np.ones((5, 5), np.uint8))
        m = np.where(rim > 0, 0.0, m)
        m = np.where(fill > 0, 1.0, m)
    if guide_rgb is not None and guide_rgb.shape[:2] == m.shape:
        gray = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        m = np.clip(_guided_filter(gray, m, params.guide_radius, params.guide_eps), 0.0, 1.0)
    if params.pull_in > 0:
        m = np.clip((m - params.pull_in) / (1.0 - params.pull_in), 0.0, 1.0)
    return m.astype(np.float32)


def mask_stability(masks: Sequence[np.ndarray]) -> tuple[float, float]:
    """(mean, worst) intersection-over-union of the subject silhouette between
    consecutive samples: 1.0 is a mask that does not move at all. A single mask
    is trivially stable."""
    if len(masks) < 2:
        return 1.0, 1.0
    ious: list[float] = []
    for a, b in itertools.pairwise(masks):
        sa, sb = a > 0.5, b > 0.5
        union = int(np.count_nonzero(sa | sb))
        ious.append(int(np.count_nonzero(sa & sb)) / union if union else 1.0)
    return float(np.mean(ious)), float(np.min(ious))
