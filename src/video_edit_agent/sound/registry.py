"""The small SFX registry: metadata for a tiny curated pack, nothing more.

Layout (all optional; a project without any pack simply has no SFX):

    assets/sfx/registry.json      metadata for the pack
    assets/sfx/<file>.wav         the audio files it names
    brands/<brand>/sfx/           OPTIONAL future drop-in with the same layout;
                                  extension compatibility only, no manager.

The registry never lists sounds by brand and nobody asks the AI to pick a file:
`sound.resolver` maps `profile + intent + event type -> asset id`. An asset
whose file is missing is reported as unavailable and the plan falls back to no
sound. No sound is ever a dependency of a render.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from video_edit_agent.sound.intent import EventType, SoundIntent

REGISTRY_FILENAME = "registry.json"
SFX_DIR_ENV = "VIDEO_EDIT_AGENT_SFX_DIR"


class SfxCategory(str, Enum):
    SOFT_WHOOSH = "soft_whoosh"
    FAST_WHOOSH = "fast_whoosh"
    REVERSE_SOFT = "reverse_soft"
    TRANSITION_SWEEP = "transition_sweep"
    IMPACT_SOFT = "impact_soft"
    TEXT_TICK_SOFT = "text_tick_soft"


class SfxAsset(BaseModel):
    id: str
    category: SfxCategory
    file: str  # relative to the registry directory
    energy: str = "low"  # low | medium | high
    duration_s: float = Field(gt=0)
    style: str = "clean"  # clean | airy | soft | cartoon | glitch | aggressive_bass ...
    allowed_intents: list[SoundIntent] = Field(default_factory=list)
    recommended_events: list[EventType] = Field(default_factory=list)
    license: str | None = None  # where it may be distributed from, for the pack curator


class SfxRegistry(BaseModel):
    assets: list[SfxAsset] = Field(default_factory=list)
    root: str | None = None  # directory the asset files live in (not serialized by hand)

    def asset(self, asset_id: str | None) -> SfxAsset | None:
        return next((a for a in self.assets if a.id == asset_id), None)

    def path(self, asset_id: str) -> Path | None:
        asset = self.asset(asset_id)
        if asset is None or self.root is None:
            return None
        return Path(self.root) / asset.file

    def is_available(self, asset_id: str) -> bool:
        path = self.path(asset_id)
        return bool(path and path.is_file())

    @property
    def empty(self) -> bool:
        return not self.assets

    def merged(self, other: SfxRegistry) -> SfxRegistry:
        """`self` plus `other`'s assets (ids in `self` win); files keep their own roots."""
        known = {a.id for a in self.assets}
        extra = [a.model_copy(update={"file": str((Path(other.root or ".") / a.file).resolve())}) for a in other.assets if a.id not in known]
        return SfxRegistry(assets=[*self.assets, *extra], root=self.root)


def default_sfx_dir() -> Path:
    override = os.environ.get(SFX_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "assets" / "sfx"


def load_registry_dir(directory: Path) -> SfxRegistry:
    """The registry in `directory`; a missing or unreadable one is an empty registry (never an error)."""
    path = Path(directory) / REGISTRY_FILENAME
    if not path.is_file():
        return SfxRegistry(root=str(directory))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        assets = [SfxAsset.model_validate(a) for a in data.get("assets", [])]
    except (OSError, ValueError):
        return SfxRegistry(root=str(directory))
    return SfxRegistry(assets=assets, root=str(directory))


def load_registry(directory: Path | None = None, *, brand_dir: Path | None = None) -> SfxRegistry:
    """The pack registry, plus the optional `<brand_dir>/sfx` drop-in when present."""
    registry = load_registry_dir(directory or default_sfx_dir())
    if brand_dir is not None and (Path(brand_dir) / "sfx").is_dir():
        registry = registry.merged(load_registry_dir(Path(brand_dir) / "sfx"))
    return registry


__all__ = [
    "REGISTRY_FILENAME", "SFX_DIR_ENV", "SfxAsset", "SfxCategory", "SfxRegistry", "default_sfx_dir", "load_registry",
    "load_registry_dir",
]
