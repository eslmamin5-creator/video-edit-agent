"""Sound profiles: how much sound a project wants, as a choice, not a library.

A project/brand stores only a profile NAME (`minimal` | `dynamic` | `none`);
the profile turns that into density, spacing and an energy ceiling. There is no
per-brand SFX folder and no "one sound every N seconds": a sound must earn its
place through semantic importance, spacing and repeat-suppression.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from video_edit_agent.sound.intent import SoundIntent

ENERGY_ORDER = ("low", "medium", "high")


class Density(str, Enum):
    NONE = "none"  # zero sound
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SoundProfile(BaseModel):
    name: str
    density: Density
    max_energy: str = "low"  # loudest asset energy this profile accepts
    banned_styles: list[str] = Field(default_factory=list)
    allowed_intents: list[SoundIntent] = Field(default_factory=list)
    min_spacing_s: float = 6.0  # between two non-essential sounds
    same_event_cooldown_s: float = 12.0  # the same event type does not sound twice inside this
    importance_threshold: float = 0.6  # decorative sounds below this are dropped
    max_per_minute: int = 4
    stack_window_s: float = 0.6  # never two sounds on one beat, whatever the density
    gain_db: float = -18.0  # nominal SFX level relative to full scale voice

    @property
    def silent(self) -> bool:
        return self.density is Density.NONE or not self.allowed_intents

    def allows(self, intent: SoundIntent) -> bool:
        return intent is not SoundIntent.NONE and intent in self.allowed_intents

    def accepts_energy(self, energy: str) -> bool:
        if energy not in ENERGY_ORDER:
            return False
        return ENERGY_ORDER.index(energy) <= ENERGY_ORDER.index(self.max_energy)


_UNWANTED = ["cartoon", "pop", "aggressive_bass", "glitch"]

PROFILES: dict[str, SoundProfile] = {
    "none": SoundProfile(
        name="none", density=Density.NONE, allowed_intents=[], max_per_minute=0,
    ),
    # Low density, subtle: no cartoon pops, aggressive bass or glitch.
    "minimal": SoundProfile(
        name="minimal", density=Density.LOW, max_energy="low", banned_styles=list(_UNWANTED),
        allowed_intents=[SoundIntent.SUBTLE_MOTION, SoundIntent.TRANSITION, SoundIntent.ACCENT, SoundIntent.IMPACT],
        min_spacing_s=6.0, same_event_cooldown_s=12.0, importance_threshold=0.6, max_per_minute=4, gain_db=-20.0,
    ),
    "dynamic": SoundProfile(
        name="dynamic", density=Density.MEDIUM, max_energy="medium", banned_styles=["cartoon", "glitch"],
        allowed_intents=[SoundIntent.SUBTLE_MOTION, SoundIntent.TRANSITION, SoundIntent.ACCENT, SoundIntent.IMPACT],
        min_spacing_s=2.5, same_event_cooldown_s=6.0, importance_threshold=0.4, max_per_minute=10, gain_db=-16.0,
    ),
}

DEFAULT_PROFILE = "none"


def get_profile(name: str | None) -> SoundProfile:
    """The named profile; unknown or missing names mean `none` (silence is the safe default)."""
    return PROFILES.get((name or DEFAULT_PROFILE).strip().lower(), PROFILES[DEFAULT_PROFILE])


__all__ = ["DEFAULT_PROFILE", "ENERGY_ORDER", "PROFILES", "Density", "SoundProfile", "get_profile"]
