"""Logo policy: turns the active Brand Profile's `logo.behavior` into a
concrete plan (persistent bug and/or intro/end cards).

Generic by construction: the default mode is `end_card`; a persistent corner
bug only exists when the Brand Profile or project explicitly selects
`persistent_bug`. All colors, the logo asset, duration and reveal style are
read from the Brand Profile -- no client values live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.brand.schema import Brand, LogoMode

MIN_CARD_SECONDS = 0.5
MAX_CARD_SECONDS = 5.0
# Decorative end-card motifs a Brand Profile may explicitly opt into.
SUPPORTED_MOTIFS = frozenset({"underline"})
# Neutral card colors used only when a brand value is missing (reported as
# MISSING in the plan's warnings, never an invented accent).
_NEUTRAL_CARD_BG = "#111111"


@dataclass
class CardSpec:
    """One branded full-frame card (intro or end card)."""

    kind: str  # "intro" | "end_card"
    duration: float
    width: int
    height: int
    fps: float
    background_top: str
    background_bottom: str
    logo_path: Path | None
    logo_width_pct: float = 0.5
    reveal: str = "subtle"  # subtle | fade | none
    accent: str | None = None  # brand accent for the optional underline
    accent_style: str = "none"  # none | underline
    transition_s: float = 0.4  # dissolve between content and the card


@dataclass
class LogoPlan:
    mode: LogoMode
    logo_path: Path | None
    persistent_bug: bool = False
    intro: CardSpec | None = None
    end_card: CardSpec | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def cards(self) -> list[CardSpec]:
        return [c for c in (self.intro, self.end_card) if c is not None]


def _card(brand: Brand, kind: str, logo: Path, width: int, height: int, fps: float) -> CardSpec:
    b = brand.logo.behavior
    top = brand.colors.primary
    bottom = brand.colors.secondary if b.background == "gradient" else top
    # Decorative motifs come only from an explicit Brand Profile setting; the
    # palette alone (e.g. an accent color) never authorizes one.
    motif = b.accent_style if b.accent_style in SUPPORTED_MOTIFS and brand.colors.accent else "none"
    return CardSpec(
        kind=kind,
        duration=min(MAX_CARD_SECONDS, max(MIN_CARD_SECONDS, b.duration)),
        width=width,
        height=height,
        fps=fps,
        background_top=top,
        background_bottom=bottom,
        logo_path=logo,
        logo_width_pct=b.logo_width_pct,
        reveal=b.reveal,
        accent=brand.colors.accent if motif == "underline" else None,
        accent_style=motif,
        transition_s=b.transition,
    )


def plan_logo(
    brand: Brand, logo_path: Path | None, width: int, height: int, fps: float,
    mode_override: LogoMode | None = None,
) -> LogoPlan:
    mode = mode_override or brand.logo.behavior.mode
    plan = LogoPlan(mode=mode, logo_path=logo_path)
    if mode is LogoMode.NONE:
        return plan
    if logo_path is None:
        plan.warnings.append(
            f"MISSING logo asset: logo mode '{mode.value}' requested but brand '{brand.name}' has no logo file"
        )
        return plan
    if mode is LogoMode.PERSISTENT_BUG:
        plan.persistent_bug = True
        return plan
    if mode in (LogoMode.INTRO, LogoMode.INTRO_AND_END):
        plan.intro = _card(brand, "intro", logo_path, width, height, fps)
    if mode in (LogoMode.END_CARD, LogoMode.INTRO_AND_END):
        plan.end_card = _card(brand, "end_card", logo_path, width, height, fps)
    return plan
