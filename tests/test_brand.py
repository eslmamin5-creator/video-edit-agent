"""Brand Profile init/load/validate tests (spec section 23) — brands must be
addable as plain folders with zero core source changes."""
from __future__ import annotations

import pytest

from video_edit_agent.brand.loader import BrandNotFoundError, init_brand, load_brand
from video_edit_agent.brand.schema import Brand
from video_edit_agent.brand.validator import validate_brand


def test_init_brand_creates_expected_folder_layout(tmp_path):
    root = tmp_path / "brands"
    d = init_brand("acme", root=root)
    assert d == root / "acme"
    for sub in ("logos", "fonts", "references", "broll", "motion", "captions"):
        assert (d / sub).is_dir()
    assert (d / "brand.yaml").exists()


def test_init_brand_twice_raises(tmp_path):
    root = tmp_path / "brands"
    init_brand("acme", root=root)
    with pytest.raises(FileExistsError):
        init_brand("acme", root=root)


def test_load_brand_round_trips_after_init(tmp_path):
    root = tmp_path / "brands"
    init_brand("acme", root=root)
    brand = load_brand("acme", root=root)
    assert brand.name == "acme"
    assert brand.preferred_aspect_ratio == "9:16"


def test_load_brand_missing_raises_clear_error(tmp_path):
    root = tmp_path / "brands"
    with pytest.raises(BrandNotFoundError):
        load_brand("does_not_exist", root=root)


def test_load_brand_with_no_name_returns_default():
    brand = load_brand(None)
    assert brand.name


def test_validate_brand_accepts_default_brand():
    result = validate_brand(Brand(name="acme"))
    assert result.ok
    assert result.errors == []


def test_validate_brand_flags_invalid_hex_color():
    brand = Brand(name="acme")
    brand.colors.primary = "not-a-color"
    result = validate_brand(brand)
    assert not result.ok
    assert any("colors.primary" in e for e in result.errors)


def test_validate_brand_flags_unknown_motion_engine():
    brand = Brand(name="acme")
    brand.motion.preferred_engine = "not-a-real-engine"
    result = validate_brand(brand)
    assert not result.ok


def test_validate_brand_warns_on_unknown_caption_preset():
    brand = Brand(name="acme")
    brand.captions.preset = "not-a-real-preset"
    result = validate_brand(brand)
    assert result.ok  # warning only, not an error
    assert result.warnings
