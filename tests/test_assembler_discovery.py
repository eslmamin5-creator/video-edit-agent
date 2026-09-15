from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.agents.assembler.discovery import (
    build_scene_inventory,
    discover_scene_files,
    natural_sort_key,
)


def test_natural_sort_key_orders_numbers_correctly():
    paths = [Path("clip10.mp4"), Path("clip2.mp4"), Path("clip1.mp4")]
    ordered = sorted(paths, key=natural_sort_key)
    assert [p.name for p in ordered] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]


def test_discover_scene_files_raises_for_missing_dir(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        discover_scene_files(tmp_path / "does_not_exist")


def test_discover_scene_files_filters_and_sorts(unordered_assembler_scenes_dir: Path):
    files = discover_scene_files(unordered_assembler_scenes_dir)
    assert [f.name for f in files] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]


def test_discover_scene_files_ignores_unsupported_suffixes(tmp_path: Path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "notes.txt").write_text("hello", encoding="utf-8")
    (d / "readme.md").write_text("hello", encoding="utf-8")
    assert discover_scene_files(d) == []


@requires_ffmpeg
def test_build_scene_inventory_probes_each_scene(assembler_scenes_dir: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)

    assert len(inventory) == 5
    assert [i.filename for i in inventory] == [
        "scene_01_intro.mp4", "scene_02_middle.mp4", "scene_03_action.mp4",
        "scene_04_closeup.mp4", "scene_05_outro.mp4",
    ]
    assert [i.id for i in inventory] == [f"scene_{n:03d}" for n in range(5)]
    assert [i.original_order for i in inventory] == [0, 1, 2, 3, 4]

    intro, middle, action, closeup, _outro = inventory
    assert intro.width == 640 and intro.height == 360
    assert intro.aspect_ratio == "16:9"
    assert closeup.width == 480 and closeup.height == 480
    assert closeup.aspect_ratio == "1:1"
    assert action.has_audio is False
    assert intro.has_audio is True
    assert intro.duration == pytest.approx(1.0, abs=0.15)
    assert middle.duration == pytest.approx(1.5, abs=0.15)
