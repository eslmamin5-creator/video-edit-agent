"""Local B-roll provider (spec section 20): looks for pre-existing footage
in the project's own `edit/broll/` assets folder and, failing that, a shared
local library directory. This is the cheapest, fastest, most-preferred
source and requires no network or API keys.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _keyword_score(filename: str, concept: str) -> int:
    tokens = [t for t in concept.lower().split() if len(t) > 2]
    name_lower = filename.lower()
    return sum(1 for t in tokens if t in name_lower)


def _best_match(directory: Path, concept: str) -> Path | None:
    if not directory.exists():
        return None
    candidates = [p for p in directory.rglob("*") if p.suffix.lower() in (_VIDEO_EXTS | _IMAGE_EXTS)]
    if not candidates:
        return None
    scored = sorted(candidates, key=lambda p: _keyword_score(p.stem, concept), reverse=True)
    top = scored[0]
    return top if _keyword_score(top.stem, concept) > 0 else None


def find_broll(item: BrollPlanItem, project_broll_dir: Path, library_dir: Path | None = None) -> BrollPlanItem:
    """Tries the project's own broll assets first (BrollSourceKind.PROJECT_ASSET),
    then a shared local library (BrollSourceKind.LOCAL_LIBRARY). Leaves the
    item untouched (source stays NONE) if nothing matches -- the caller's
    provider router will try generated sources next."""
    match = _best_match(project_broll_dir, item.spoken_concept)
    if match:
        item.source = BrollSourceKind.PROJECT_ASSET
        item.asset_path = str(match)
        item.confidence = 0.6
        return item

    if library_dir is not None:
        match = _best_match(library_dir, item.spoken_concept)
        if match:
            item.source = BrollSourceKind.LOCAL_LIBRARY
            item.asset_path = str(match)
            item.confidence = 0.4
            return item

    return item
