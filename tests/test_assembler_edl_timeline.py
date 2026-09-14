from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.agents.assembler.discovery import build_scene_inventory, discover_scene_files
from video_edit_agent.agents.assembler.edl_builder import build_scene_edl
from video_edit_agent.agents.assembler.timeline import build_assembler_timeline
from video_edit_agent.core.media import probe
from video_edit_agent.core.timeline import TrackType


@requires_ffmpeg
def test_build_scene_edl_one_clip_per_scene_in_order(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)

    edl = build_scene_edl(inventory, width=640, height=360, fps=30.0, cache_dir=tmp_path / "cache")

    assert len(edl.clips) == len(inventory)
    # Direct cuts, contiguous, in the given order (no gaps/overlaps).
    cursor = 0.0
    for clip, item in zip(edl.clips, inventory):
        assert clip.timeline_in == pytest.approx(cursor)
        assert clip.duration == pytest.approx(item.duration)
        cursor = clip.timeline_out


@requires_ffmpeg
def test_build_scene_edl_synthesizes_silent_audio_for_muted_scene(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)
    action_scene = next(i for i in inventory if i.filename == "scene_03_action.mp4")
    assert action_scene.has_audio is False

    edl = build_scene_edl([action_scene], width=640, height=360, fps=30.0, cache_dir=tmp_path / "cache")

    referenced_path = Path(edl.clips[0].source_file)
    assert referenced_path != Path(action_scene.path)  # a muxed copy, not the original
    info = probe(referenced_path)
    assert info.has_audio is True
    # The original source file must never be modified.
    original_info = probe(Path(action_scene.path))
    assert original_info.has_audio is False


@requires_ffmpeg
def test_build_assembler_timeline_uses_shared_schema(assembler_scenes_dir: Path, tmp_path: Path):
    files = discover_scene_files(assembler_scenes_dir)
    inventory = build_scene_inventory(files)
    edl = build_scene_edl(inventory, width=640, height=360, fps=30.0, cache_dir=tmp_path / "cache")

    timeline = build_assembler_timeline(edl)

    assert timeline.workflow == "assembler"
    video_items = [i for i in timeline.items if i.type == TrackType.VIDEO]
    audio_items = [i for i in timeline.items if i.type == TrackType.AUDIO]
    assert len(video_items) == len(inventory)
    assert len(audio_items) == len(inventory)
    assert timeline.total_duration == pytest.approx(edl.total_duration)
