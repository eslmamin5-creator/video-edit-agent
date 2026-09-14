"""Brand Profile reuse tests (Creator spec section 9): Creator must use the
existing shared `brand.schema.Brand` / `brand.loader.load_brand` verbatim --
no separate Creator-only brand schema exists.
"""
from __future__ import annotations

import ast
from pathlib import Path

import video_edit_agent.agents.creator as creator_pkg
from video_edit_agent.agents.creator.timeline import _brand_extra
from video_edit_agent.brand.defaults import DEFAULT_BRAND
from video_edit_agent.brand.loader import load_brand
from video_edit_agent.brand.schema import Brand


def test_no_creator_only_brand_schema_defined():
    """No class named *Brand* other than re-exports/imports of the shared one
    is defined anywhere under agents/creator/."""
    root = Path(creator_pkg.__file__).parent
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and "brand" in node.name.lower():
                raise AssertionError(f"{path.name} defines a Creator-only brand class: {node.name}")


def test_timeline_brand_extra_uses_shared_brand_fields():
    brand = Brand(name="t", fonts=["Inter"])
    extra = _brand_extra(brand)
    assert extra["primaryColor"] == brand.colors.primary
    assert extra["fontFamily"] == "Inter"


def test_timeline_brand_extra_empty_for_no_brand():
    assert _brand_extra(None) == {}


def test_load_brand_none_returns_default_brand():
    assert load_brand(None) is DEFAULT_BRAND


def test_load_brand_acme_test_has_distinct_colors():
    brand = load_brand("acme_test")
    assert brand.colors.primary == "#0B1E3D"
    assert brand.colors.primary != DEFAULT_BRAND.colors.primary
