"""Where the subject is in the frame, and what the backdrop looks like.

Motion titles and punch-ins need two cheap facts about the footage: the face /
subject occupancy (so a title never lands on a face or chest, and a punch-in
keeps the face in view) and the backdrop luminance (so a title colour is chosen
for contrast, not by taste). `analyze_frames` reduces a handful of sampled
frames to a coarse grid of both; every consumer works on that grid, so it is
testable without a video and without mediapipe.

Detection is best-effort: when mediapipe or ffmpeg is unavailable the analysis
is simply missing (`None`) and callers fall back to conservative defaults.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Box = tuple[float, float, float, float]  # normalized x, y, w, h in the full frame

GRID_W, GRID_H = 54, 96  # analysis grid (9:16)


@dataclass
class FrameAnalysis:
    """`occupancy` and `luma` are GRID_H x GRID_W arrays over the full frame:
    occupancy is the subject mask (0..1, max over the sampled frames); luma is
    the backdrop's relative luminance (0..1, WCAG definition, mean over
    frames). `face` is the union face box over the frames, when one was found."""

    occupancy: np.ndarray
    luma: np.ndarray
    face: Box | None = None

    def region(self, box: Box) -> tuple[slice, slice]:
        x, y, w, h = box
        x0, x1 = int(np.clip(x, 0, 1) * GRID_W), int(np.ceil(np.clip(x + w, 0, 1) * GRID_W))
        y0, y1 = int(np.clip(y, 0, 1) * GRID_H), int(np.ceil(np.clip(y + h, 0, 1) * GRID_H))
        return slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1))

    def occupancy_in(self, box: Box) -> float:
        ys, xs = self.region(box)
        return float(self.occupancy[ys, xs].mean())

    def luma_stats(self, box: Box) -> tuple[float, float, float]:
        """(p10, mean, p90) backdrop luminance inside `box`."""
        ys, xs = self.region(box)
        cells = self.luma[ys, xs].ravel()
        return float(np.percentile(cells, 10)), float(cells.mean()), float(np.percentile(cells, 90))


def relative_luminance_grid(frame_rgb: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(frame_rgb, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lin = np.where(small <= 0.03928, small / 12.92, ((small + 0.055) / 1.055) ** 2.4)
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)


def _detect_face(frame_rgb: np.ndarray) -> Box | None:
    import mediapipe as mp  # type: ignore

    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        result = fd.process(frame_rgb)
    if not result.detections:
        return None
    best = max(result.detections, key=lambda d: d.score[0])
    b = best.location_data.relative_bounding_box
    x0, y0 = max(0.0, b.xmin), max(0.0, b.ymin)
    return (x0, y0, min(1.0 - x0, b.width), min(1.0 - y0, b.height))


def _detect_occupancy(frame_rgb: np.ndarray) -> np.ndarray:
    import cv2

    from video_edit_agent.subject.detect import detect_mask

    mask = detect_mask(frame_rgb).mask
    return cv2.resize(mask, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)


def analyze_frames(frames_rgb: list[np.ndarray], *, detect_face=_detect_face, detect_occupancy=_detect_occupancy) -> FrameAnalysis | None:
    """Reduce sampled RGB frames to a `FrameAnalysis`. Returns None when there
    are no frames or the subject detector is unavailable (never raises for a
    missing optional dependency). The detectors are injectable for tests."""
    if not frames_rgb:
        return None
    lumas, occs, faces = [], [], []
    try:
        for frame in frames_rgb:
            lumas.append(relative_luminance_grid(frame))
            occs.append(detect_occupancy(frame))
            face = detect_face(frame)
            if face is not None:
                faces.append(face)
    except Exception:  # noqa: BLE001 - optional dependency (mediapipe) missing or model failure
        return None
    face_union: Box | None = None
    if faces:
        x0 = min(f[0] for f in faces)
        y0 = min(f[1] for f in faces)
        x1 = max(f[0] + f[2] for f in faces)
        y1 = max(f[1] + f[3] for f in faces)
        face_union = (x0, y0, x1 - x0, y1 - y0)
    return FrameAnalysis(occupancy=np.max(occs, axis=0), luma=np.mean(lumas, axis=0), face=face_union)


def grab_frames(source: Path | str, times: list[float], *, width: int = 360, height: int = 640) -> list[np.ndarray]:
    """RGB frames of `source` at `times` (seconds), letterbox-free (scaled to
    cover then cropped, like the render). Frames ffmpeg cannot produce are skipped."""
    frames: list[np.ndarray] = []
    for t in times:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", str(source), "-frames:v", "1",
             "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True, timeout=60, check=False,
        )
        if result.returncode == 0 and len(result.stdout) == width * height * 3:
            frames.append(np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3))
    return frames


def analyze_clip(source: Path | str, start: float, end: float, *, samples: int = 4) -> FrameAnalysis | None:
    """Analysis of `source` between `start` and `end` (seconds), from `samples`
    evenly spread frames. None when frames cannot be read or detection is
    unavailable -- callers treat that as "unknown" and stay conservative."""
    if end <= start or samples < 1:
        return None
    span = end - start
    times = [start + span * (i + 0.5) / samples for i in range(samples)]
    try:
        return analyze_frames(grab_frames(source, times))
    except Exception:  # noqa: BLE001 - analysis is an optional refinement
        return None
