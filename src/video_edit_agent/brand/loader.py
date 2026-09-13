"""Brand Profile loading (spec section 23) — users add brands as plain
folders under `brands/<name>/`, no core source edits required."""
from __future__ import annotations

from pathlib import Path

import yaml

from video_edit_agent.brand.defaults import DEFAULT_BRAND, default_brand_yaml
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
