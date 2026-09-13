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
    return importlib_util.find_spec("mediapipe") is not None


def _segmenter():
    import mediapipe as mp  # type: ignore

    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


def detect_mask(frame_rgb: np.ndarray) -> SubjectMaskResult:
    """Runs person segmentation on a single RGB frame (HxWx3 uint8) and
    returns a soft mask. Raises SubjectDetectionUnavailable if mediapipe
    isn't installed."""
    if not is_available():
        raise SubjectDetectionUnavailable(
            "mediapipe is not installed (pip install video-edit-agent[all]). "
            "Behind-subject compositing will fall back to a plain overlay."
        )

    with _segmenter() as seg:
        result = seg.process(frame_rgb)
        mask = result.segmentation_mask.astype(np.float32)
        confidence = float(mask.mean())
        return SubjectMaskResult(mask=mask, confidence=confidence)
