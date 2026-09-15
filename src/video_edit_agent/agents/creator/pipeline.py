"""Creator pipeline orchestration (spec sections 2-16): SCRIPT -> structured
analysis -> scenes -> storyboard -> visual treatment -> asset planning ->
MasterTimeline -> render -> QA -> playable video.

Works fully offline: this module never imports or calls any cloud provider
(Gemini, ElevenLabs, Veo). Every stage reuses shared core systems (Brand
Profile, motion router, B-roll provider, render pipeline, MasterTimeline
schema, QA primitives) rather than duplicating them -- see the individual
`agents/creator/*` modules for what each one reuses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.agents.creator.analysis import analyze_script
from video_edit_agent.agents.creator.asset_plan import build_asset_plan
from video_edit_agent.agents.creator.parser import parse_script
from video_edit_agent.agents.creator.qa import run_creator_qa
from video_edit_agent.agents.creator.render import render_creator_timeline
from video_edit_agent.agents.creator.scenes import build_scenes
from video_edit_agent.agents.creator.schemas import (
    AssetPlanItem,
    CreatorStyle,
    Scene,
    ScriptAnalysis,
    Storyboard,
)
from video_edit_agent.agents.creator.storyboard import build_storyboard, render_storyboard_markdown
from video_edit_agent.agents.creator.timeline import build_creator_timeline
from video_edit_agent.brand.loader import load_brand
from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.library import default_library_dir
from video_edit_agent.core.project import ProjectMemory, ProjectPaths
from video_edit_agent.core.schemas import MotionPlanItem, QAReport
from video_edit_agent.core.timeline import MasterTimeline
from video_edit_agent.core.timeline import save as save_master_timeline
from video_edit_agent.render.export import resolve_preset


@dataclass
class CreatorResult:
    project_dir: Path
    final_output: Path | None
    script_analysis: ScriptAnalysis
    scenes: list[Scene] = field(default_factory=list)
    storyboard: Storyboard | None = None
    asset_plan: list[AssetPlanItem] = field(default_factory=list)
    master_timeline: MasterTimeline | None = None
    motion_plan: list[MotionPlanItem] = field(default_factory=list)
    qa_report: QAReport | None = None
    warnings: list[str] = field(default_factory=list)


def run_creator(
    script_path: Path,
    *,
    brand_name: str | None = None,
    style: str = "mixed",
    offline: bool = False,
    preset_name: str = "reel",
) -> CreatorResult:
    style_enum = CreatorStyle(style)
    brand: Brand = load_brand(brand_name)

    paths = ProjectPaths.for_source(script_path)
    paths.ensure()
    memory = ProjectMemory.load_or_new(paths)
    memory.workflow = "creator"
    memory.brand = brand.name
    memory.style = style
    memory.source_inventory.append(str(script_path))
    warnings: list[str] = []

    # 1-4: parse + analyze (fully deterministic, offline, no cloud calls)
    text = parse_script(script_path)
    analysis = analyze_script(text, title=script_path.stem)
    paths.script_analysis_json.write_text(
        json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5: scene breakdown
    scenes = build_scenes(text, analysis)
    paths.scenes_json.write_text(
        json.dumps([s.model_dump(mode="json") for s in scenes], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    memory.scene_count = len(scenes)

    # 6: storyboard (never triggers asset generation itself)
    storyboard = build_storyboard(scenes, style_enum, brand.name)
    paths.storyboard_json.write_text(
        json.dumps(storyboard.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths.storyboard_md.write_text(render_storyboard_markdown(storyboard, scenes), encoding="utf-8")
    memory.storyboard_status = "generated"

    # 10: asset plan (coarse treatment category only; real provenance comes
    # from the actual motion/B-roll resolution at timeline-build time below)
    asset_plan = build_asset_plan(scenes, style_enum)
    memory.asset_plan_status = "planned"

    # 11-14: MasterTimeline -- motion via the shared router, B-roll via the
    # shared local provider, falling back scene-by-scene rather than failing.
    preset = resolve_preset(preset_name)
    broll_project_dir = paths.edit_dir / "broll_assets"
    # The local B-roll library is a plain folder on disk -- no network, no
    # API keys -- so it's available in both offline and online modes; only
    # cloud-generated B-roll (spec section 12 step 4) is gated by `offline`,
    # and this Creator slice never reaches that step at all.
    broll_library_dir = default_library_dir()

    timeline, motion_items, _broll_items = build_creator_timeline(
        scenes,
        asset_plan,
        project_root=paths.root,
        motion_output_dir=paths.cache_dir / "motion",
        broll_project_dir=broll_project_dir,
        broll_library_dir=broll_library_dir,
        brand=brand,
        fps=preset.fps,
        width=preset.width,
        height=preset.height,
        offline=offline,
    )
    save_master_timeline(timeline, paths.master_timeline_json)
    memory.master_timeline_path = str(paths.master_timeline_json)

    paths.asset_plan_json.write_text(
        json.dumps([a.model_dump(mode="json") for a in asset_plan], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 8/15: render -- always produces a playable video, degrading visuals
    # rather than failing when an optional engine/asset is unavailable.
    final_output = None
    try:
        final_output = render_creator_timeline(timeline, paths.root, paths.final_mp4, preset, paths.cache_dir)
        memory.render_history.append(f"Rendered Creator output {final_output} with preset '{preset_name}'")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Render failed: {exc}")
        memory.log_rejected(f"Creator render failed: {exc}")

    # 16: Creator QA
    qa_report = run_creator_qa(scenes, timeline, motion_items, final_output, brand)
    memory.qa_issues.extend(f"[{i.severity.value}] {i.category}: {i.message}" for i in qa_report.issues)
    memory.save()

    return CreatorResult(
        project_dir=paths.edit_dir,
        final_output=final_output,
        script_analysis=analysis,
        scenes=scenes,
        storyboard=storyboard,
        asset_plan=asset_plan,
        master_timeline=timeline,
        motion_plan=motion_items,
        qa_report=qa_report,
        warnings=warnings,
    )
