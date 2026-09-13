"""Subject (person) detection for behind-subject compositing (spec section 16).

Uses `mediapipe`'s selfie/person segmenter when available. This module only
answers "is there a usable subject mask model on this machine" and produces
a raw per-frame mask array; `segment.py` and `compositor.py` build the actual
compositing pipeline on top of it. If mediapipe is unavailable, callers must
fall back to plain overlay compositing (no behind-subject effect) rather than
failing the render (spec section 43).
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util

import numpy as np


class SubjectDetectionUnavailable(RuntimeError):
    pass


@dataclass
class SubjectMaskResult:
    mask: np.ndarray  # float32, HxW, values in [0, 1]; 1 = subject
    confidence: float


def is_available() -> bool:
    """Checks not just that `mediapipe` is importable, but that the specific
    legacy Solutions API this adapter needs is actually present. mediapipe
    1.0+ removed `mediapipe.solutions` in favor of the new Tasks API, so a
    bare import check alone would falsely report "available" and then crash
    (spec V1.1 hardening: never let a capability check pass on import alone)."""
    if importlib_util.find_spec("mediapipe") is None:
        return False
    try:
        import mediapipe as mp  # type: ignore

        return hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation")
    except ImportError:
        return False


def _segmenter():
    import mediapipe as mp  # type: ignore

    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


def detect_mask(frame_rgb: np.ndarray) -> SubjectMaskResult:
    """Runs person segmentation on a single RGB frame (HxWx3 uint8) and
    returns a soft mask. Raises SubjectDetectionUnavailable if mediapipe
    isn't installed, or if an installed version doesn't expose the expected
    API (e.g. mediapipe 1.0+ dropped `mediapipe.solutions`)."""
    if not is_available():
        raise SubjectDetectionUnavailable(
            "mediapipe is not installed, or the installed version doesn't provide the "
            "legacy `mediapipe.solutions.selfie_segmentation` API this adapter needs "
            "(mediapipe>=1.0 removed it in favor of the Tasks API; use mediapipe<1.0, "
            "e.g. mediapipe==0.10.21). Behind-subject compositing will fall back to a "
            "plain overlay."
        )

    try:
        with _segmenter() as seg:
            result = seg.process(frame_rgb)
            mask = result.segmentation_mask.astype(np.float32)
            confidence = float(mask.mean())
            return SubjectMaskResult(mask=mask, confidence=confidence)
    except AttributeError as e:  # noqa: BLE001 - defensive: is_available() should prevent this
        raise SubjectDetectionUnavailable(f"mediapipe API mismatch: {e}") from e
