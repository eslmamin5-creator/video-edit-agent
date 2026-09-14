"""Scene discovery (spec section 4): deterministically find and probe scene
video files in a directory. Never reorders anything -- ordering is a
separate, explicit decision (see `ordering.py`).
"""
from __future__ import annotations

import re
from pathlib import Path

from video_edit_agent.agents.assembler.schemas import SceneInventoryItem
from video_edit_agent.core.media import MediaError, probe

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(path: Path) -> list:
    """Splits a filename into text/number chunks so "scene_2" sorts before
    "scene_10" (plain alphabetical sort would put "scene_10" first)."""
    parts = _NUM_RE.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def discover_scene_files(scenes_dir: Path) -> list[Path]:
    if not scenes_dir.is_dir():
        raise NotADirectoryError(f"Scene directory not found: {scenes_dir}")
    files = [p for p in scenes_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    files.sort(key=natural_sort_key)
    return files


def build_scene_inventory(scene_files: list[Path]) -> list[SceneInventoryItem]:
    items: list[SceneInventoryItem] = []
    for i, path in enumerate(scene_files):
        try:
            info = probe(path)
        except MediaError as exc:
            raise MediaError(f"Could not probe scene file {path}: {exc}") from exc
        aspect = _aspect_ratio(info.width, info.height)
        items.append(
            SceneInventoryItem(
                id=f"scene_{i:03d}",
                filename=path.name,
                path=str(path),
                original_order=i,
                duration=info.duration,
                width=info.width,
                height=info.height,
                fps=info.fps,
                video_codec=info.video_codec,
                audio_codec=info.audio_codec,
                aspect_ratio=aspect,
                has_audio=info.has_audio,
            )
        )
    return items


def _aspect_ratio(width: int, height: int) -> str:
    if not width or not height:
        return "unknown"
    from math import gcd

    g = gcd(width, height)
    return f"{width // g}:{height // g}"
