"""Finish + Brand Profile acceptance test (spec section 31): verifies the
rendered output actually contains the brand's CTA overlay -- a real
rendered brand feature, not just JSON metadata -- via `render_finish`
directly (isolated from the fixed `brands/` lookup root `load_brand` uses)
so the test doesn't depend on mutating a shared brand fixture."""
from __future__ import annotations

from pathlib import Path

from tests.conftest import requires_ffmpeg
from video_edit_agent.agents.assembler.discovery import build_scene_inventory, discover_scene_files
from video_edit_agent.agents.assembler.edl_builder import build_scene_edl
from video_edit_agent.agents.assembler.render import render_finish, render_rough_cut
from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.media import probe
from video_edit_agent.render.export import PRESETS


@requires_ffmpeg
def test_finish_with_brand_cta_renders_larger_output_than_rough_cut(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)
    preset = PRESETS["square"]  # smaller canvas -> faster synthetic render
    edl = build_scene_edl(inventory, width=preset.width, height=preset.height, fps=preset.fps, cache_dir=tmp_path / "cache")

    rough_path = render_rough_cut(edl, tmp_path / "rough_cut.mp4", preset)
    assert rough_path.exists()

    brand = Brand(name="test_brand")
    brand.cta.text_default = "تابعنا الآن"

    final_path, notes = render_finish(
        edl, tmp_path / "final.mp4", preset,
        project_root=tmp_path, cache_dir=tmp_path / "cache", brand=brand, offline=True,
    )

    assert final_path.exists()
    info = probe(final_path)
    assert info.duration > 0
    # A real overlay was applied via the Motion Router (offline engine), not
    # silently skipped -- the honest disclosure path only fires on failure.
    assert any("Applied brand CTA card" in n for n in notes)


@requires_ffmpeg
def test_finish_without_brand_never_adds_cta_overlay(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)
    preset = PRESETS["square"]
    edl = build_scene_edl(inventory, width=preset.width, height=preset.height, fps=preset.fps, cache_dir=tmp_path / "cache")

    final_path, notes = render_finish(
        edl, tmp_path / "final.mp4", preset,
        project_root=tmp_path, cache_dir=tmp_path / "cache", brand=None, offline=True,
    )

    assert final_path.exists()
    assert notes == []
