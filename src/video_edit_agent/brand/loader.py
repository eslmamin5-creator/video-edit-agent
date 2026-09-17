"""Brand Profile loading (spec section 23) — users add brands as plain
folders under `brands/<name>/`, no core source edits required."""
from __future__ import annotations

from pathlib import Path

import yaml

from video_edit_agent.brand.defaults import DEFAULT_BRAND
from video_edit_agent.brand.schema import Brand

DEFAULT_BRANDS_ROOT = Path(__file__).resolve().parents[3] / "brands"


class BrandNotFoundError(RuntimeError):
    pass


def brands_root() -> Path:
    return DEFAULT_BRANDS_ROOT


def brand_dir(name: str, root: Path | None = None) -> Path:
    return (root or brands_root()) / name


def load_brand(name: str | None, root: Path | None = None) -> Brand:
    if not name:
        return DEFAULT_BRAND
    d = brand_dir(name, root)
    yaml_path = d / "brand.yaml"
    if not yaml_path.exists():
        raise BrandNotFoundError(f"Brand '{name}' not found at {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    data.setdefault("name", name)
    return Brand.model_validate(data)


_LOGO_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")


def resolve_logo_path(name: str | None, root: Path | None = None) -> Path | None:
    """Returns the brand's logo image file (spec section 23: users "drop a
    logo into brands/<name>/logos/", per `examples/README.md`), or None if
    the brand has no name, no `logos/` directory, or no image file in it.

    The first match (by sorted filename) among the supported extensions
    wins; brands are expected to keep exactly one logo file in this folder."""
    if not name:
        return None
    logos_dir = brand_dir(name, root) / "logos"
    if not logos_dir.is_dir():
        return None
    candidates = sorted(
        p for p in logos_dir.iterdir() if p.is_file() and p.suffix.lower() in _LOGO_EXTENSIONS
    )
    return candidates[0] if candidates else None


_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


def resolve_fonts_dir(name: str | None, root: Path | None = None) -> Path | None:
    """Returns the brand's `fonts/` directory (spec section 23: brands drop
    font files into `brands/<name>/fonts/`) so libass can be pointed at it
    via `fontsdir=` instead of requiring a system-wide font install (Review-
    First Editing Workflow spec section 9/10). Returns None if the brand has
    no name, no `fonts/` directory, or no font file in it -- captions then
    render with whatever's already on the system's font path, unchanged."""
    if not name:
        return None
    fonts_dir = brand_dir(name, root) / "fonts"
    if not fonts_dir.is_dir():
        return None
    has_font = any(p.is_file() and p.suffix.lower() in _FONT_EXTENSIONS for p in fonts_dir.iterdir())
    return fonts_dir if has_font else None


def init_brand(name: str, root: Path | None = None) -> Path:
    d = brand_dir(name, root)
    if d.exists():
        raise FileExistsError(f"Brand '{name}' already exists at {d}")
    for sub in ("logos", "fonts", "references", "broll", "motion", "captions"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    brand = Brand(name=name)
    (d / "brand.yaml").write_text(
        yaml.safe_dump(brand.model_dump(mode="json"), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return d
