"""The setup answers a user gives once per project (Phase 1.6): editing style, brand, format.

They are saved next to the review files, so a rerun or a resume never asks again and never silently changes them.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

CHOICES_FILENAME = "setup_choices.json"
PROFILES = ("minimal", "balanced", "dynamic")


class SetupChoices(BaseModel):
    profile: str = "balanced"
    brand: str | None = None  # None = the neutral default look
    preset: str = "reel"
    caption_style: str = "word-highlight"
    offline: bool = False


def choices_path(review_dir: Path) -> Path:
    return Path(review_dir) / CHOICES_FILENAME


def load_choices(review_dir: Path) -> SetupChoices | None:
    p = choices_path(review_dir)
    return SetupChoices.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def resolve_choices(review_dir: Path, **given: object) -> SetupChoices:
    """Saved answers first, then anything given now (an explicit option changes the saved answer), then defaults."""
    base = load_choices(review_dir) or SetupChoices()
    merged = base.model_copy(update={k: v for k, v in given.items() if v is not None})
    if merged.profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
    return merged


def save_choices(review_dir: Path, choices: SetupChoices) -> Path:
    p = choices_path(review_dir)
    text = json.dumps(choices.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


__all__ = ["PROFILES", "SetupChoices", "load_choices", "resolve_choices", "save_choices"]
