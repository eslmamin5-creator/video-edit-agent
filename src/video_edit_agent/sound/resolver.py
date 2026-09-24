"""Deterministic SFX selection: `profile + intent + event type -> asset id`.

The AI never names a file. Given the same inputs the resolver returns the same
asset every time (a total order on the candidates ends in the asset id), and
when nothing qualifies it says why and returns no asset: the plan then plays
no sound, which is always a valid result.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.sound.intent import EventType, SoundIntent
from video_edit_agent.sound.profile import ENERGY_ORDER, SoundProfile
from video_edit_agent.sound.registry import SfxAsset, SfxRegistry

AVAILABLE = "available"
UNAVAILABLE = "unavailable"  # a sound was wanted but nothing usable exists -> fallback to none
NOT_WANTED = "not_wanted"  # intent none / profile silent: nothing was asked for

REASON_INTENT_NONE = "intent_none"
REASON_PROFILE_SILENT = "profile_silent"
REASON_INTENT_NOT_ALLOWED = "intent_not_allowed_by_profile"
REASON_NO_REGISTRY = "no_sfx_pack_installed"
REASON_NO_MATCH = "no_asset_matches_profile_intent_event"
REASON_FILE_MISSING = "asset_file_missing"


@dataclass(frozen=True)
class Resolution:
    status: str  # available | unavailable | not_wanted
    asset_id: str | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status == AVAILABLE


def _fits(asset: SfxAsset, profile: SoundProfile, intent: SoundIntent) -> bool:
    return (
        intent in asset.allowed_intents
        and profile.accepts_energy(asset.energy)
        and asset.style not in profile.banned_styles
    )


def _rank(asset: SfxAsset, event: EventType) -> tuple:
    energy = ENERGY_ORDER.index(asset.energy) if asset.energy in ENERGY_ORDER else len(ENERGY_ORDER)
    return (0 if event in asset.recommended_events else 1, energy, asset.category.value, asset.id)


def resolve(profile: SoundProfile, intent: SoundIntent, event: EventType, registry: SfxRegistry) -> Resolution:
    if intent is SoundIntent.NONE:
        return Resolution(NOT_WANTED, None, REASON_INTENT_NONE)
    if profile.silent:
        return Resolution(NOT_WANTED, None, REASON_PROFILE_SILENT)
    if not profile.allows(intent):
        return Resolution(NOT_WANTED, None, REASON_INTENT_NOT_ALLOWED)
    if registry.empty:
        return Resolution(UNAVAILABLE, None, REASON_NO_REGISTRY)
    candidates = sorted((a for a in registry.assets if _fits(a, profile, intent)), key=lambda a: _rank(a, event))
    if not candidates:
        return Resolution(UNAVAILABLE, None, REASON_NO_MATCH)
    for asset in candidates:  # the best candidate whose file actually exists
        if registry.is_available(asset.id):
            return Resolution(AVAILABLE, asset.id)
    return Resolution(UNAVAILABLE, None, REASON_FILE_MISSING)


__all__ = ["AVAILABLE", "NOT_WANTED", "UNAVAILABLE", "Resolution", "resolve"]
