"""Local scene analysis (spec section 6) -- fully offline. Cloud vision may
enrich this later (spec section 16) but must never be required; every field
here is derived from ffmpeg/ffprobe/PIL only.

Motion direction is a coarse, honestly-labeled heuristic (comparing
brightness centroid between the first and last frame), not real optical
flow -- good enough to flag an obvious pan/drift for continuity checking,
nothing more.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from video_edit_agent.agents.assembler.schemas import SceneAnalysis, SceneInventoryItem
from video_edit_agent.core.media import MediaError, run
from video_edit_agent.editorial.silence import detect_silence

_MOTION_THRESHOLD = 0.06  # fraction of frame width/height the brightness centroid must shift


def analyze_scene(item: SceneInventoryItem, frames_dir: Path) -> SceneAnalysis:
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(item.path)

    first_frame_path = frames_dir / f"{item.id}_first.jpg"
    last_frame_path = frames_dir / f"{item.id}_last.jpg"
    _extract_frame(video_path, first_frame_path, at_seconds=0.0)
    last_ts = max(0.0, item.duration - 0.1)
    _extract_frame(video_path, last_frame_path, at_seconds=last_ts)

    dominant_color = brightness = contrast = None
    motion_direction = "unknown"
    first_img = _safe_open(first_frame_path)
    last_img = _safe_open(last_frame_path)
    if first_img is not None:
        dominant_color = _dominant_color(first_img)
        brightness = _brightness(first_img)
        contrast = _contrast(first_img)
    if first_img is not None and last_img is not None:
        motion_direction = _motion_direction(first_img, last_img)

    silence_spans = []
    is_silent = True
    if item.has_audio:
        try:
            silence_spans = detect_silence(video_path)
            total_silence = sum(s.duration for s in silence_spans)
            is_silent = item.duration > 0 and total_silence >= item.duration * 0.95
        except MediaError:
            is_silent = False

    return SceneAnalysis(
        scene_id=item.id,
        duration=item.duration,
        first_frame_path=str(first_frame_path) if first_img is not None else None,
        last_frame_path=str(last_frame_path) if last_img is not None else None,
        dominant_color=dominant_color,
        brightness=brightness,
        contrast=contrast,
        motion_direction=motion_direction,
        has_audio=item.has_audio,
        is_silent=is_silent,
        aspect_ratio=item.aspect_ratio,
        width=item.width,
        height=item.height,
        fps=item.fps,
    )


def analyze_scenes(items: list[SceneInventoryItem], frames_dir: Path) -> list[SceneAnalysis]:
    return [analyze_scene(item, frames_dir) for item in items]


def _extract_frame(video_path: Path, out_path: Path, at_seconds: float) -> None:
    result = run(
        [
            "ffmpeg", "-y", "-ss", f"{max(0.0, at_seconds):.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "3", str(out_path),
        ],
        timeout=30,
    )
    if result.returncode != 0 or not out_path.exists():
        # Non-fatal: analysis degrades gracefully without frame-derived fields.
        return


def _safe_open(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _dominant_color(img: Image.Image) -> str:
    small = img.resize((32, 32))
    pixels = list(small.getdata())
    r = sum(p[0] for p in pixels) // len(pixels)
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    return f"#{r:02x}{g:02x}{b:02x}"


def _brightness(img: Image.Image) -> float:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    return (sum(pixels) / len(pixels)) / 255.0


def _contrast(img: Image.Image) -> float:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return min(1.0, (variance ** 0.5) / 128.0)


def _brightness_centroid(img: Image.Image) -> tuple[float, float]:
    """Returns (x, y) in [0, 1] -- where the frame's "visual weight" sits."""
    small = img.convert("L").resize((16, 16))
    pixels = list(small.getdata())
    total = sum(pixels) or 1
    cx = sum((i % 16) * p for i, p in enumerate(pixels)) / total / 15
    cy = sum((i // 16) * p for i, p in enumerate(pixels)) / total / 15
    return cx, cy


def _motion_direction(first_img: Image.Image, last_img: Image.Image) -> str:
    x0, y0 = _brightness_centroid(first_img)
    x1, y1 = _brightness_centroid(last_img)
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) < _MOTION_THRESHOLD and abs(dy) < _MOTION_THRESHOLD:
        return "static"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
