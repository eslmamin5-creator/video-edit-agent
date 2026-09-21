"""The behind-subject compositing contract, and the checks that hold it.

Behind-subject means the text may disappear behind the speaker; the speaker must
never disappear behind the text. The layer stack, bottom to top:

    1. BACKGROUND     the original video frame
    2. GRAPHIC        the text layer (RGBA)
    3. SUBJECT        the cutout of the ORIGINAL frame: original RGB, alpha = the matte
    4. CAPTIONS       burned last, in front of everything

    final = (background * (1 - g) + graphic * g) * (1 - a) + source * a

`g` is the graphic's alpha and `a` the subject matte. Where `a` is 1 the final pixel
IS the source pixel, whatever the graphic does. `composite_behind` is that contract
in numpy; the renderers realise it with ffmpeg's `overlay` filter in the same order
(graphic overlay, then the cutout overlay), and the tests hold the two to each other.

Nothing here knows a brand, a colour or a video: it is pure array maths.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from video_edit_agent.subject.refine import MatteReport


def composite_behind(
    background: np.ndarray, graphic_rgb: np.ndarray, graphic_alpha: np.ndarray,
    source_rgb: np.ndarray, subject_alpha: np.ndarray,
) -> np.ndarray:
    """The behind-subject composite (uint8 HxWx3). `background`, `graphic_rgb` and
    `source_rgb` are uint8 HxWx3; the alphas are float HxW in 0..1. The subject is
    reconstructed from `source_rgb` (the original frame), never from `background`."""
    g = np.clip(graphic_alpha.astype(np.float32), 0.0, 1.0)[..., None]
    a = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)[..., None]
    under = background.astype(np.float32) * (1.0 - g) + graphic_rgb.astype(np.float32) * g
    out = under * (1.0 - a) + source_rgb.astype(np.float32) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def feather_width(alpha: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> float:
    """Mean width (px) of the partially transparent band around the subject:
    the band's area divided by the length of the silhouette outline."""
    import cv2

    band = int(np.count_nonzero((alpha > lo) & (alpha < hi)))
    contours, _ = cv2.findContours((alpha > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(len(c) for c in contours)
    return band / perimeter if perimeter else 0.0


def outward_growth_px(alpha: np.ndarray, reference: np.ndarray, threshold: float = 0.5) -> float:
    """How far (px, at most) the subject silhouette of `alpha` reaches beyond the
    `reference` silhouette: the halo. 0 when it stays inside."""
    import cv2

    outside = (alpha > threshold) & ~(reference > threshold)
    if not outside.any():
        return 0.0
    dist = cv2.distanceTransform((~(reference > threshold)).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[outside].max())


@dataclass(frozen=True)
class IntegrityReport:
    """How well a composite keeps the subject the matte says is there."""

    protected_px: int  # pixels the matte holds at (almost) full alpha
    graphic_px_in_protected: int  # of those, pixels where the graphic was drawn under the subject
    leaked_px: int  # protected pixels whose final RGB differs from the source RGB beyond tolerance
    rgb_max_error: float  # max |final - source| over the protected pixels (0..255)
    rgb_mean_error: float
    holes_px: int  # background-alpha pixels enclosed by the subject: a hole the graphic could show through
    feather_px: float

    @property
    def passed(self) -> bool:
        return self.protected_px > 0 and self.leaked_px == 0 and self.holes_px == 0


def check_integrity(
    source_rgb: np.ndarray, subject_alpha: np.ndarray, graphic_alpha: np.ndarray, final_rgb: np.ndarray,
    *, region: np.ndarray | None = None, protect: float = 0.98, tolerance: float | None = None,
) -> IntegrityReport:
    """The subject integrity invariant: where `subject_alpha` says the subject is
    present (>= `protect`), `final_rgb` must equal `source_rgb` within `tolerance`
    levels (codec/rounding slack), so the graphic can never delete, darken or punch a
    hole in the subject. `region` (bool HxW) limits the measurement, e.g. to the head.
    The default `tolerance` is the graphic a pixel at exactly `protect` alpha still lets
    through (`(1 - protect) * 255`) plus one level of rounding."""
    import cv2

    if tolerance is None:
        tolerance = (1.0 - protect) * 255.0 + 1.0

    where = subject_alpha >= protect
    if region is not None:
        where = where & region
    diff = np.abs(final_rgb.astype(np.float32) - source_rgb.astype(np.float32)).max(axis=2)
    bad = where & (diff > tolerance)
    silhouette = (subject_alpha > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(silhouette)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    holes = (filled > 0) & (silhouette == 0)
    if region is not None:
        holes = holes & region
    vals = diff[where]
    return IntegrityReport(
        protected_px=int(where.sum()),
        graphic_px_in_protected=int((where & (graphic_alpha > 0.01)).sum()),
        leaked_px=int(bad.sum()),
        rgb_max_error=float(vals.max()) if vals.size else 0.0,
        rgb_mean_error=float(vals.mean()) if vals.size else 0.0,
        holes_px=int(holes.sum()),
        feather_px=feather_width(subject_alpha),
    )


@dataclass(frozen=True)
class MatteGatePolicy:
    """When a subject matte is too unreliable to put text behind the speaker.
    Generic limits (shares of the subject's own area), not per-video values."""

    max_added_frac: float = 0.12  # colour evidence may add at most this share to the core (a missing body is not hair)
    max_soft_frac: float = 0.50  # the raw mask may be uncertain (0.15..0.85) on at most this share of the core
    require_guide: bool = True  # no picture to read the colour evidence from = cannot verify the hair

DEFAULT_MATTE_POLICY = MatteGatePolicy()


def assess_matte(reports: Sequence[MatteReport], policy: MatteGatePolicy | None = None) -> tuple[bool, str]:
    """(reliable, detail). A matte is rejected -- and behind-subject with it -- when
    a sample has no subject core, cannot be checked against the picture, needs more
    growth than hair can explain, or is too uncertain to trust."""
    policy = policy or DEFAULT_MATTE_POLICY
    if not reports:
        return False, "no matte"
    worst_added = max(r.added_frac for r in reports)
    worst_soft = max(r.soft_frac for r in reports)
    detail = f"added {worst_added:.3f} (max {policy.max_added_frac}), soft {worst_soft:.3f} (max {policy.max_soft_frac})"
    if any(r.core_area == 0 for r in reports):
        return False, "a sample has no subject core; " + detail
    if policy.require_guide and not all(r.evidence_used for r in reports):
        return False, "no picture to verify the matte against; " + detail
    if worst_added > policy.max_added_frac:
        return False, "the matte misses more of the subject than hair explains; " + detail
    if worst_soft > policy.max_soft_frac:
        return False, "the matte is too uncertain; " + detail
    return True, detail
