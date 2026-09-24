"""Brand Profile init/load/validate tests (spec section 23) — brands must be
addable as plain folders with zero core source changes."""
from __future__ import annotations

from pathlib import Path

import pytest

from video_edit_agent.brand.loader import (
    BrandNotFoundError,
    brands_root,
    init_brand,
    load_brand,
    resolve_logo_path,
)
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


def test_resolve_logo_path_finds_dropped_in_image(tmp_path):
    root = tmp_path / "brands"
    init_brand("acme", root=root)
    logo = root / "acme" / "logos" / "logo.png"
    logo.write_bytes(b"fake-png")
    assert resolve_logo_path("acme", root=root) == logo


def test_resolve_logo_path_ignores_non_image_files(tmp_path):
    root = tmp_path / "brands"
    init_brand("acme", root=root)
    (root / "acme" / "logos" / "README.txt").write_text("drop a logo here")
    assert resolve_logo_path("acme", root=root) is None


def test_resolve_logo_path_returns_none_when_no_logos_dir(tmp_path):
    root = tmp_path / "brands"
    assert resolve_logo_path("does_not_exist", root=root) is None


def test_resolve_logo_path_returns_none_for_no_brand_name(tmp_path):
    assert resolve_logo_path(None, root=tmp_path / "brands") is None


def test_default_brand_ships_no_placeholder_logo():
    """Regression test (Review-First Editing Workflow spec section 4): the
    shipped 'default' brand must not ship any logo asset, placeholder or
    otherwise -- a real, un-brand-defined yellow circle previously shipped
    here and was silently composited onto every render using this brand."""
    assert resolve_logo_path("default", root=brands_root()) is None
    logos_dir = Path(brands_root()) / "default" / "logos"
    if logos_dir.is_dir():
        assert list(logos_dir.iterdir()) == []
