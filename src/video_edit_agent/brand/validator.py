"""Brand Profile validation (spec section 23) — used by `videoedit brand
validate` and by QA's brand-consistency checks."""
from __future__ import annotations

from dataclasses import dataclass, field

from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.styles import PRESETS as CAPTION_PRESETS

VALID_MOTION_ENGINES = {"hyperframes", "remotion", "manim", "simple", None}
HEX_COLOR_LEN = {4, 7}  # "#fff" or "#ffffff"


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_hex_color(value: str) -> bool:
    return isinstance(value, str) and value.startswith("#") and len(value) in HEX_COLOR_LEN


def validate_brand(brand: Brand) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not brand.name:
        errors.append("brand.name is required")

    for field_name, value in (
        ("colors.primary", brand.colors.primary),
        ("colors.secondary", brand.colors.secondary),
        ("colors.accent", brand.colors.accent),
    ):
        if not _is_hex_color(value):
            errors.append(f"{field_name} is not a valid hex color: {value!r}")

    if brand.captions.preset not in CAPTION_PRESETS:
        warnings.append(f"captions.preset '{brand.captions.preset}' is not a known preset; will fall back to 'minimal'")

    if brand.motion.preferred_engine not in VALID_MOTION_ENGINES:
        errors.append(f"motion.preferred_engine '{brand.motion.preferred_engine}' is not a known engine")

    if brand.preferred_aspect_ratio not in {"9:16", "1:1", "16:9", "4:5"}:
        warnings.append(f"preferred_aspect_ratio '{brand.preferred_aspect_ratio}' is unusual")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
