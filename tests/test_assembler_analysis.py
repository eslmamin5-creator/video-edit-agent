from __future__ import annotations

from pathlib import Path

from tests.conftest import requires_ffmpeg
from video_edit_agent.agents.assembler.analysis import analyze_scene, analyze_scenes
from video_edit_agent.agents.assembler.discovery import build_scene_inventory, discover_scene_files


@requires_ffmpeg
def test_analyze_scene_produces_full_schema(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)
    frames_dir = tmp_path / "frames"

    analysis = analyze_scene(inventory[0], frames_dir)

    assert analysis.scene_id == inventory[0].id
    assert analysis.duration > 0
    assert analysis.dominant_color is not None and analysis.dominant_color.startswith("#")
    assert analysis.brightness is not None and 0.0 <= analysis.brightness <= 1.0
    assert analysis.contrast is not None and 0.0 <= analysis.contrast <= 1.0
    assert analysis.motion_direction in {"left", "right", "up", "down", "static", "unknown"}
    assert analysis.has_audio is True
    assert analysis.width == 640 and analysis.height == 360


@requires_ffmpeg
def test_analyze_scene_detects_no_audio(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)
    action_scene = next(i for i in inventory if i.filename == "scene_03_action.mp4")

    analysis = analyze_scene(action_scene, tmp_path / "frames")

    assert analysis.has_audio is False
    assert analysis.is_silent is True


@requires_ffmpeg
def test_analyze_scenes_preserves_order(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)

    analyses = analyze_scenes(inventory, tmp_path / "frames")

    assert [a.scene_id for a in analyses] == [i.id for i in inventory]
