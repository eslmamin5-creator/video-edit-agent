"""Regression tests for the "no arbitrary default yellow" fix (Review-First
Editing Workflow spec section 3 / item B): `BrandColors.accent` is now
optional, and every consumer must fall back to the brand's own `secondary`
color -- never an invented strong color like the old hardcoded "#FFCC00"."""
from __future__ import annotations

from video_edit_agent.agents.creator.timeline import _brand_extra
from video_edit_agent.brand.schema import Brand
from video_edit_agent.motion.remotion.adapter import _theme_props
from video_edit_agent.motion.simple.engine import _DEFAULT_ACCENT, _hex_to_rgba, _Palette


def test_brand_with_no_accent_defaults_schema_field_to_none():
    brand = Brand(name="acme")
    assert brand.colors.accent is None


def test_simple_engine_palette_falls_back_to_secondary_when_accent_unset():
    brand = Brand(name="acme")
    brand.colors.secondary = "#00AAFF"
    palette = _Palette(brand)
    assert palette.accent == _hex_to_rgba("#00AAFF")


def test_simple_engine_palette_uses_explicit_accent_when_set():
    brand = Brand(name="acme")
    brand.colors.accent = "#123456"
    palette = _Palette(brand)
    assert palette.accent == _hex_to_rgba("#123456")


def test_simple_engine_no_brand_at_all_is_neutral_white_not_yellow():
    palette = _Palette(None)
    assert palette.accent == _DEFAULT_ACCENT
    assert palette.accent != (255, 204, 0, 255)


def test_creator_timeline_brand_extra_falls_back_to_secondary():
    brand = Brand(name="acme")
    brand.colors.secondary = "#00AAFF"
    extra = _brand_extra(brand)
    assert extra["accentColor"] == "#00AAFF"


def test_remotion_theme_props_falls_back_to_secondary():
    brand = Brand(name="acme")
    brand.colors.secondary = "#00AAFF"
    theme = _theme_props(brand)
    assert theme["accent"] == "#00AAFF"
    assert theme["accent"] != "#FFCC00"


def test_remotion_theme_props_none_when_no_brand():
    assert _theme_props(None) is None
