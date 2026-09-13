"""Brand Profile schema (spec section 23)."""
from __future__ import annotations

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


class BrandMotion(BaseModel):
    energy: str = "medium"  # low | medium | high
    avoid: list[str] = Field(default_factory=list)
    preferred_engine: str | None = None  # hyperframes | remotion | manim | simple


class BrandColors(BaseModel):
    primary: str = "#111111"
    secondary: str = "#FFFFFF"
    accent: str = "#FFCC00"


class BrandCTA(BaseModel):
    style: str = "minimal"
    text_default: str | None = None


class Brand(BaseModel):
    name: str
    language: BrandLanguage = Field(default_factory=BrandLanguage)
    colors: BrandColors = Field(default_factory=BrandColors)
    fonts: list[str] = Field(default_factory=list)
    logo_rules: str | None = None
    captions: BrandCaptions = Field(default_factory=BrandCaptions)
    motion: BrandMotion = Field(default_factory=BrandMotion)
    broll_aesthetic: str | None = None
    cta: BrandCTA = Field(default_factory=BrandCTA)
    safe_zones: dict[str, float] = Field(default_factory=dict)
    preferred_aspect_ratio: str = "9:16"
    avoid: list[str] = Field(default_factory=list)
