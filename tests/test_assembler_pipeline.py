"""Real end-to-end Assembler acceptance tests (spec sections 12, 13, 26,
29, 30): the "do not count JSON alone as success" tests -- these exercise
`run_assembler()` against real synthetic scene files, produce real rendered
video, and validate it with ffprobe, not just check that JSON files exist.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import requires_ffmpeg
from video_edit_agent.agents.assembler.pipeline import run_assembler
from video_edit_agent.agents.assembler.schemas import OrderPolicy
from video_edit_agent.core.media import probe


@requires_ffmpeg
def test_rough_cut_is_real_playable_video_in_preserved_order(assembler_scenes_dir: Path):
    result = run_assembler(assembler_scenes_dir, rough=True, preserve_order=True, offline=True)

    assert result.rough_cut_path is not None
    assert result.rough_cut_path.exists()
    assert result.order_policy == OrderPolicy.PRESERVE
    assert [i.filename for i in result.scene_inventory] == [
        "scene_01_intro.mp4", "scene_02_middle.mp4", "scene_03_action.mp4",
        "scene_04_closeup.mp4", "scene_05_outro.mp4",
    ]

    info = probe(result.rough_cut_path)
    assert info.duration > 0
    planned_duration = sum(i.duration for i in result.scene_inventory)
    assert abs(info.duration - planned_duration) < 0.5

    # edit/ project files exist alongside the source, not just in-memory.
    paths = result.project_dir
    assert (paths / "scene_inventory.json").exists()
    assert (paths / "master_timeline.json").exists()
    assert (paths / "rough_cut.mp4").exists()


@requires_ffmpeg
def test_preserve_order_regression_with_natural_sort_filenames(unordered_assembler_scenes_dir: Path):
    """Section 30: this must never regress -- clip1 < clip2 < clip10 by
    natural order, not alphabetical (which would put clip10 before clip2)."""
    result = run_assembler(unordered_assembler_scenes_dir, rough=True, preserve_order=True, offline=True)

    assert [i.filename for i in result.scene_inventory] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    assert result.order_policy == OrderPolicy.PRESERVE
    assert result.master_timeline is not None
    video_items = [i for i in result.master_timeline.items if i.type.value == "video"]
    assert [Path(i.source).name for i in video_items] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]


@requires_ffmpeg
def test_finish_reuses_rough_cut_plan_and_produces_real_video(assembler_scenes_dir: Path):
    rough_result = run_assembler(assembler_scenes_dir, rough=True, preserve_order=True, offline=True)
    assert rough_result.rough_cut_path is not None

    finish_result = run_assembler(assembler_scenes_dir, finish=True, preserve_order=True, offline=True)

    assert finish_result.final_output is not None
    assert finish_result.final_output.exists()
    info = probe(finish_result.final_output)
    assert info.duration > 0

    # The scene analysis computed during Rough Cut was reused, not
    # silently recomputed differently, for the same unchanged scene set.
    cached = json.loads((finish_result.project_dir / "scene_analysis.json").read_text(encoding="utf-8"))
    assert {c["scene_id"] for c in cached} == {i.id for i in rough_result.scene_inventory}

    memory_sidecar = finish_result.project_dir / ".project_memory.json"
    assert memory_sidecar.exists()
    memory = json.loads(memory_sidecar.read_text(encoding="utf-8"))
    assert memory["finish_status"] == "rendered"
    assert memory["workflow"] == "assembler"


@requires_ffmpeg
def test_project_resume_reuses_cached_scene_analysis_across_runs(assembler_scenes_dir: Path):
    first = run_assembler(assembler_scenes_dir, rough=True, preserve_order=True, offline=True)
    analysis_path = first.project_dir / "scene_analysis.json"
    first_mtime = analysis_path.stat().st_mtime_ns
    first_analysis_json = analysis_path.read_text(encoding="utf-8")

    second = run_assembler(assembler_scenes_dir, finish=True, preserve_order=True, offline=True)

    assert analysis_path.stat().st_mtime_ns == first_mtime  # not rewritten -- reused as-is
    assert analysis_path.read_text(encoding="utf-8") == first_analysis_json
    assert [a.scene_id for a in second.scene_analysis] == [a.scene_id for a in first.scene_analysis]


@requires_ffmpeg
def test_assembler_qa_runs_and_reports_no_issues_for_clean_scene_set(assembler_scenes_dir: Path):
    result = run_assembler(assembler_scenes_dir, rough=True, preserve_order=True, offline=True)
    assert result.rough_cut_path is not None
    # A clean, contiguous, fully-covered rough cut should raise no QA issues.
    assert result.qa_issues == []


@requires_ffmpeg
def test_assembler_never_reorders_scenes_by_default(assembler_scenes_dir: Path):
    """Section 5 non-negotiable, exercised end-to-end: default invocation
    (no --preserve-order, no --order flag) must still preserve discovery
    order -- the CLI/pipeline default is OrderPolicy.PRESERVE."""
    result = run_assembler(assembler_scenes_dir, rough=True, offline=True)
    assert result.order_policy == OrderPolicy.PRESERVE
    assert [i.filename for i in result.scene_inventory] == [
        "scene_01_intro.mp4", "scene_02_middle.mp4", "scene_03_action.mp4",
        "scene_04_closeup.mp4", "scene_05_outro.mp4",
    ]
