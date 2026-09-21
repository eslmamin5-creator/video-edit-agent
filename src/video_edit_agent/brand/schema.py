"""Brand Profile schema (spec section 23)."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BrandLanguage(BaseModel):
    primary: str = "ar"
    tone: str = "simple"


class BrandCaptions(BaseModel):
    preset: str = "minimal"
    rtl: bool = True
    max_lines: int = 2
    word_highlight: bool = False
    font: str | None = None
    primary_color: str | None = None
    highlight_color: str | None = None
    # When true (default), caption colors not set explicitly above are derived
    # from the active Brand Profile palette (highlight <- accent, box tint <-
    # primary) instead of the caption preset's own generic colors.
    use_brand_colors: bool = True
    # Caption backing: "outline" (no box, stroked text), "box" (opaque black
    # box, legacy), "brand_box" (translucent box tinted with the brand
    # primary, hugging each line), or None to keep the preset default.
    background: str | None = None
    max_chars_per_line: int | None = None
    # Split long caption chunks into balanced lines instead of letting the
    # renderer wrap at the canvas edge.
    balance_lines: bool | None = None
    # `brand_box` weight: opacity of the tinted box (0 invisible .. 1 solid)
    # and its padding around the text in pixels (at the 1080-wide canvas).
    # Kept light and tight by default so the box supports the text instead of
    # dominating the frame.
    box_opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    box_padding: float = Field(default=8.0, ge=0.0, le=40.0)
    # Caption readability mode: "none" | "adaptive" | "plate" (see `captions.modes`).
    # None keeps the behaviour implied by `background` (existing brands are unchanged).
    mode: str | None = None


class BrandSound(BaseModel):
    """Sound direction is a PROFILE choice only (none | minimal | dynamic). There is no per-brand
    SFX library; sounds come from the shared registry (see `video_edit_agent.sound`)."""

    profile: str = "none"


class BrandMotion(BaseModel):
    energy: str = "medium"  # low | medium | high
    avoid: list[str] = Field(default_factory=list)
    preferred_engine: str | None = None  # hyperframes | remotion | manim | simple
    use_brand_palette: bool = True


class BrandColors(BaseModel):
    primary: str = "#111111"
    secondary: str = "#FFFFFF"
    # No default accent color (Review-First Editing Workflow spec section 3):
    # a brand that doesn't explicitly set an accent must not silently inherit
    # a strong color like the old hardcoded "#FFCC00" yellow. `None` means
    # "unset" -- consumers must fall back to a neutral color (usually
    # `secondary`) and report that they did so, never invent a strong accent.
    accent: str | None = None


class BrandTypography(BaseModel):
    """Per-script font families. Either may be unset: the loader then falls
    back to the legacy `Brand.fonts` list, and finally to a neutral system
    font (reported as such in the brand summary)."""

    arabic: str | None = None
    latin: str | None = None


class LogoMode(str, Enum):
    """Where the brand logo appears (project/brand overridable). The generic
    default is `END_CARD`; a persistent corner bug is opt-in only."""

    NONE = "none"
    INTRO = "intro"
    END_CARD = "end_card"
    INTRO_AND_END = "intro_and_end"
    PERSISTENT_BUG = "persistent_bug"


class BrandLogoBehavior(BaseModel):
    mode: LogoMode = LogoMode.END_CARD
    duration: float = 2.0  # seconds, per intro/end card
    reveal: str = "subtle"  # subtle | fade | none
    # Card background: "solid" (brand primary) or "gradient" (primary ->
    # secondary). Both are derived from the Brand Profile colors.
    background: str = "solid"
    # Fraction of the canvas width the centered logo occupies on a card.
    logo_width_pct: float = 0.5
    # Decorative motif beneath the logo ("none" | "underline"). A motif is
    # drawn ONLY when the Brand Profile names it here: having an accent color
    # in the palette never authorizes inventing one. "underline" additionally
    # needs the brand's accent color and is ignored without it.
    accent_style: str = "none"
    # Seconds of dissolve between the content and a card (0 = hard cut).
    transition: float = 0.4


class BrandLogo(BaseModel):
    # Path relative to `brands/<name>/` (or absolute). None -> the first
    # image in `brands/<name>/logos/`.
    asset: str | None = None
    behavior: BrandLogoBehavior = Field(default_factory=BrandLogoBehavior)


class BrandCTA(BaseModel):
    style: str = "minimal"
    text_default: str | None = None  # legacy name for `text`
    # CTA is NONE unless a text is provided (here, by the user or by an
    # approved edit plan). `enabled: false` forces NONE even if text is set.
    enabled: bool | None = None
    text: str | None = None


class Brand(BaseModel):
    name: str
    language: BrandLanguage = Field(default_factory=BrandLanguage)
    colors: BrandColors = Field(default_factory=BrandColors)
    fonts: list[str] = Field(default_factory=list)
    typography: BrandTypography = Field(default_factory=BrandTypography)
    logo: BrandLogo = Field(default_factory=BrandLogo)
    logo_rules: str | None = None
    captions: BrandCaptions = Field(default_factory=BrandCaptions)
    motion: BrandMotion = Field(default_factory=BrandMotion)
    sound: BrandSound = Field(default_factory=BrandSound)
    broll_aesthetic: str | None = None
    cta: BrandCTA = Field(default_factory=BrandCTA)
    safe_zones: dict[str, float] = Field(default_factory=dict)
    preferred_aspect_ratio: str = "9:16"
    avoid: list[str] = Field(default_factory=list)

    @property
    def arabic_font(self) -> str | None:
        return self.typography.arabic or (self.fonts[0] if self.fonts else None)

    @property
    def latin_font(self) -> str | None:
        return self.typography.latin or (self.fonts[0] if self.fonts else None) or self.typography.arabic

    @property
    def cta_text(self) -> str | None:
        """The brand's CTA text, or None (CTA is NONE by default)."""
        if self.cta.enabled is False:
            return None
        text = (self.cta.text or self.cta.text_default or "").strip()
        return text or None
