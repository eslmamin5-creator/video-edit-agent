"""Deterministic post-processing of subject masks for behind-subject text.

The MediaPipe selfie segmenter returns a soft, blobby mask from a small model
input. Around a head it is both wide (a soft grey skirt) and, on dark hair
against a lighter backdrop, short: the crown and the fringe of hair sit below
the 0.5 line. Text drawn behind the speaker then shows *over* those hair
pixels, which reads as the head being eaten by the text.

The alpha built here is subject-favouring by construction: every pixel that is
subject in the ORIGINAL frame must end up opaque, and the edge only ever grows
outwards a few pixels, never inwards. The steps, all pure numpy/OpenCV with
fixed parameters (the same masks and frames always give the same alpha, which
is what lets the placement search, the micro-preview and the final render agree):

    1. `to_canvas`          frame the mask exactly like the render frames the
                            footage (scale-to-cover + centre crop);
    2. `stabilize_temporal` a 3-tap weighted average over the neighbouring
                            samples, so an edge does not flicker between them;
    3. `refine_matte`
       a. core              the mask above 0.5, stray islands dropped, small
                            holes filled (connected-component clean-up);
       b. colour evidence   inside a bounded band around the core, pixels that
                            differ from the local backdrop colour AND connect to
                            the core are subject (hair, fringe, an ear the model
                            missed). The backdrop is estimated from the pixels
                            outside the band, so a uniform backdrop yields no
                            growth and there is no halo;
       c. outward feather   a short Gaussian feather that can only add alpha
                            (`max(feathered, hard)`): the subject never gets
                            softer, or thinner, than its own core.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

REFINE_VERSION = "r2"  # bump when the maths changes: it keys the cutout cache


@dataclass(frozen=True)
class RefineParams:
    core_threshold: float = 0.5  # mask values above this are subject for certain
    ramp: float = 0.15  # a soft ramp below the threshold that keeps the model's own confidence
    island_frac: float = 0.003  # subject islands smaller than this share of the frame are dropped
    hole_frac: float = 0.002  # background holes smaller than this share are filled
    band_frac: float = 0.012  # of the frame height: how far hair may be missing from the core
    gap_frac: float = 0.003  # of the frame height: skipped between the band and the backdrop samples
    backdrop_sigma_frac: float = 0.010  # of the frame height: backdrop colour smoothing
    evidence_lo: float = 70.0  # colour distance (0..441 RGB) from the backdrop: below is backdrop ...
    evidence_hi: float = 120.0  # ... above is subject; the ramp between is the feather
    feather_frac: float = 0.0008  # of the frame height: outward Gaussian feather sigma
    temporal_weights: tuple[float, float, float] = (0.25, 0.5, 0.25)

    def band_px(self, height: int) -> int:
        return max(2, round(self.band_frac * height))

    def feather_sigma(self, height: int) -> float:
        return max(0.8, self.feather_frac * height)


DEFAULT_PARAMS = RefineParams()


@dataclass(frozen=True)
class MatteReport:
    """What `refine_matte` did to one mask (areas in pixels)."""

    core_area: int
    added_area: int  # subject pixels found by colour evidence, beyond the core
    evidence_pixels: int  # band pixels that looked like subject (before the connectivity test)
    soft_area: int  # pixels of the RAW mask in the uncertain 0.15..0.85 range
    band_px: int
    feather_sigma: float
    evidence_used: bool

    @property
    def added_frac(self) -> float:
        return self.added_area / self.core_area if self.core_area else 0.0

    @property
    def soft_frac(self) -> float:
        return self.soft_area / self.core_area if self.core_area else 1.0


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


def _ellipse(radius: int) -> np.ndarray:
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _backdrop_distance(rgb: np.ndarray, sample: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """(colour distance of every pixel from the smoothed backdrop, whether that
    backdrop estimate is defined there). `sample` marks the pixels the backdrop
    is read from; the estimate is a normalised blur of just those pixels."""
    import cv2

    weight = sample.astype(np.float32)
    img = rgb.astype(np.float32)
    num = cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    known = den > 1e-3
    backdrop = num / np.maximum(den, 1e-3)[..., None]
    dist = np.sqrt(((img - backdrop) ** 2).sum(axis=2))
    return dist, known


def refine_matte(
    mask: np.ndarray, guide_rgb: np.ndarray | None = None, params: RefineParams = DEFAULT_PARAMS,
) -> tuple[np.ndarray, MatteReport]:
    """(alpha float32 0..1, report). `guide_rgb` is the frame the mask belongs to
    (uint8 HxWx3, same size); without it the alpha is the cleaned core with a soft
    edge and no colour evidence."""
    import cv2

    m = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    height = m.shape[0]
    raw_soft = int(np.count_nonzero((m > 0.15) & (m < 0.85)))
    core_bin = (m > params.core_threshold).astype(np.uint8)
    keep, fill = _clean_regions(core_bin, params)
    if keep.any():
        core = ((keep > 0) | (fill > 0)).astype(np.uint8)
    else:
        core = core_bin
    core_area = int(core.sum())

    # the model's own confidence just below the threshold stays as a soft skirt (never > core)
    lo = params.core_threshold - params.ramp
    soft = _smoothstep(np.clip((m - lo) / max(params.ramp, 1e-6), 0.0, 1.0))
    soft = np.where(core > 0, 1.0, soft)
    # ... but only next to the core (an island's skirt is dropped with the island)
    near = cv2.dilate(core, _ellipse(max(1, params.band_px(height) // 4)))
    hard = np.where(near > 0, soft, 0.0).astype(np.float32)

    band_px = params.band_px(height)
    added = 0
    evidence_pixels = 0
    used = False
    if guide_rgb is not None and guide_rgb.shape[:2] == m.shape and core_area > 0:
        band = cv2.dilate(core, _ellipse(band_px))
        excluded = cv2.dilate(core, _ellipse(band_px + max(1, round(params.gap_frac * height))))
        dist, known = _backdrop_distance(guide_rgb, excluded == 0, max(2.0, params.backdrop_sigma_frac * height))
        evidence = _smoothstep(np.clip(
            (dist - params.evidence_lo) / max(params.evidence_hi - params.evidence_lo, 1e-6), 0.0, 1.0,
        ))
        evidence = np.where((band > 0) & known, evidence, 0.0).astype(np.float32)
        candidate = ((evidence > 0.5) | (core > 0)).astype(np.uint8)
        _, labels = cv2.connectedComponents(candidate, connectivity=8)
        touching = np.unique(labels[core > 0])
        connected = np.isin(labels, touching[touching > 0])
        evidence = np.where(connected, evidence, 0.0).astype(np.float32)
        evidence_pixels = int(np.count_nonzero(evidence > 0.5))
        hard = np.maximum(hard, evidence)
        added = int(np.count_nonzero((hard > 0.5) & (core == 0)))
        used = True

    sigma = params.feather_sigma(height)
    feathered = cv2.GaussianBlur(hard, (0, 0), sigma)
    alpha = np.clip(np.maximum(feathered, hard), 0.0, 1.0).astype(np.float32)
    return alpha, MatteReport(
        core_area=core_area, added_area=added, evidence_pixels=evidence_pixels, soft_area=raw_soft,
        band_px=band_px, feather_sigma=float(sigma), evidence_used=used,
    )


def refine_alpha(
    mask: np.ndarray, guide_rgb: np.ndarray | None = None, params: RefineParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Cleaned, hair-safe alpha (float32, 0..1, same shape as `mask`); see `refine_matte`."""
    return refine_matte(mask, guide_rgb, params)[0]


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
