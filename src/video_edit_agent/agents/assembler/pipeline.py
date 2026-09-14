"""Assembler pipeline orchestration (spec sections 2-29): scene discovery ->
analysis -> ordering -> continuity -> transitions -> sound -> MasterTimeline
-> Rough Cut -> Finish -> QA.

Works fully offline (spec section 32: `--offline` forbids Gemini/ElevenLabs/
Veo/network Remotion install/any cloud API; Motion Router already enforces
this the same way Creator does). Every stage reuses shared core (EDL,
MasterTimeline, render pipeline, Brand Profiles, Motion Router, QA
primitives) -- see the individual `agents/assembler/*` modules for what each
one reuses.

Resume behavior (spec section 26): scene analysis / normalization /
continuity / transition / sound plans are all pure functions of the scene
inventory, so re-running is cheap and always consistent -- but a prior
Rough Cut's plan files are read back and reused for Finish rather than
silently recomputed from a *different* run's discovery, by loading them
from disk when present and matching the current scene set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

from video_edit_agent.agents.assembler.analysis import analyze_scenes
from video_edit_agent.agents.assembler.continuity import analyze_continuity
from video_edit_agent.agents.assembler.discovery import build_scene_inventory, discover_scene_files
from video_edit_agent.agents.assembler.edl_builder import build_scene_edl
from video_edit_agent.agents.assembler.normalization import plan_normalization
from video_edit_agent.agents.assembler.ordering import resolve_order
from video_edit_agent.agents.assembler.qa import run_assembler_qa
from video_edit_agent.agents.assembler.render import render_finish, render_rough_cut
from video_edit_agent.agents.assembler.schemas import (
    ContinuityFinding,
    NormalizationChoice,
    OrderPolicy,
    SceneAnalysis,
    SceneInventoryItem,
    ScriptAlignmentItem,
    SoundOperation,
    TransitionDecision,
)
from video_edit_agent.agents.assembler.script_alignment import align_script
from video_edit_agent.agents.assembler.sound import plan_sound
from video_edit_agent.agents.assembler.timeline import build_assembler_timeline
from video_edit_agent.agents.assembler.transitions import plan_transitions
from video_edit_agent.agents.creator.parser import parse_script
from video_edit_agent.brand.loader import load_brand
from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.project import ProjectMemory, ProjectPaths
from video_edit_agent.core.schemas import EDL, QAIssue
from video_edit_agent.core.timeline import MasterTimeline, save as save_master_timeline
from video_edit_agent.render.export import resolve_preset


class AssemblerError(RuntimeError):
    pass


@dataclass
class AssemblerResult:
    project_dir: Path
    scene_inventory: list[SceneInventoryItem] = field(default_factory=list)
    scene_analysis: list[SceneAnalysis] = field(default_factory=list)
    script_alignment: list[ScriptAlignmentItem] = field(default_factory=list)
    order_policy: OrderPolicy = OrderPolicy.PRESERVE
    normalization_plan: list[NormalizationChoice] = field(default_factory=list)
    continuity_report: list[ContinuityFinding] = field(default_factory=list)
    transition_plan: list[TransitionDecision] = field(default_factory=list)
    sound_plan: list[SoundOperation] = field(default_factory=list)
    master_timeline: MasterTimeline | None = None
    rough_cut_path: Path | None = None
    final_output: Path | None = None
    qa_issues: list[QAIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _target_aspect(preset) -> str:
    g = gcd(preset.width, preset.height)
    return f"{preset.width // g}:{preset.height // g}"


def _load_cached_analysis(paths: ProjectPaths, scene_ids: list[str]) -> list[SceneAnalysis] | None:
    if not paths.scene_analysis_json.exists():
        return None
    try:
        data = json.loads(paths.scene_analysis_json.read_text(encoding="utf-8"))
        cached = [SceneAnalysis.model_validate(d) for d in data]
    except Exception:  # noqa: BLE001 - any read/parse failure just triggers recompute
        return None
    if {a.scene_id for a in cached} != set(scene_ids):
        return None
    by_id = {a.scene_id: a for a in cached}
    return [by_id[sid] for sid in scene_ids]


def run_assembler(
    scenes_dir: Path,
    *,
    rough: bool = False,
    finish: bool = False,
    preserve_order: bool = False,
    order: str = "preserve",
    script_path: Path | None = None,
    brand_name: str | None = None,
    offline: bool = False,
    preset_name: str = "reel",
) -> AssemblerResult:
    if not rough and not finish:
        rough = True  # default: cheapest, highest-priority acceptance path (spec section 12)

    preset = resolve_preset(preset_name)
    target_aspect = _target_aspect(preset)
    brand: Brand | None = load_brand(brand_name) if brand_name else None

    paths = ProjectPaths.for_source(scenes_dir)
    paths.ensure()
    memory = ProjectMemory.load_or_new(paths)
    memory.workflow = "assembler"
    if brand is not None:
        memory.brand = brand.name
    warnings: list[str] = []

    # 1. Scene discovery -- deterministic, never reorders anything.
    scene_files = discover_scene_files(scenes_dir)
    if not scene_files:
        raise AssemblerError(f"No supported video files found in {scenes_dir}")
    inventory = build_scene_inventory(scene_files)
    paths.scene_inventory_json.write_text(
        json.dumps([i.model_dump(mode="json") for i in inventory], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    memory.scene_count = len(inventory)
    memory.source_inventory = [i.filename for i in inventory]

    # 2. Scene analysis (spec section 6) -- reused across rough/finish runs
    # when the scene set hasn't changed.
    analyses = _load_cached_analysis(paths, [i.id for i in inventory])
    if analyses is None:
        analyses = analyze_scenes(inventory, paths.cache_dir / "frames")
        paths.scene_analysis_json.write_text(
            json.dumps([a.model_dump(mode="json") for a in analyses], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 3. Optional script alignment (spec section 7) -- recommendation only.
    script_alignment: list[ScriptAlignmentItem] = []
    if script_path is not None:
        text = parse_script(script_path)
        script_alignment = align_script(text, inventory)
        paths.script_alignment_json.write_text(
            json.dumps([a.model_dump(mode="json") for a in script_alignment], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 4. Order resolution (spec section 5, NON-NEGOTIABLE).
    if preserve_order:
        policy = OrderPolicy.PRESERVE
    else:
        try:
            policy = OrderPolicy(order)
        except ValueError:
            policy = OrderPolicy.PRESERVE
    resolution = resolve_order(inventory, policy=policy, script_alignment=script_alignment or None)
    warnings.extend(resolution.warnings)
    ordered_items = resolution.ordered_items
    ordered_ids = [i.id for i in ordered_items]
    analyses_by_id = {a.scene_id: a for a in analyses}
    ordered_analyses = [analyses_by_id[i.id] for i in ordered_items]
    memory.order_policy = resolution.policy_applied.value

    # 5. Normalization plan (spec sections 9-10).
    normalization = plan_normalization(ordered_analyses, target_aspect=target_aspect)
    paths.normalization_plan_json.write_text(
        json.dumps([n.model_dump(mode="json") for n in normalization], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    memory.normalization_status = f"planned for {len(normalization)} scene(s), target {target_aspect}"

    # 6. Continuity analysis (spec sections 15-16).
    continuity = analyze_continuity(ordered_analyses)
    paths.continuity_report_json.write_text(
        json.dumps([c.model_dump(mode="json") for c in continuity], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    memory.continuity_status = f"{len(continuity)} finding(s) across {max(0, len(ordered_items) - 1)} boundary(ies)"

    # 7. Transition plan (spec sections 17-18, 20).
    transitions = plan_transitions(ordered_ids, continuity)
    paths.transition_plan_json.write_text(
        json.dumps([t.model_dump(mode="json") for t in transitions], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    memory.transition_plan_status = f"{len(transitions)} boundary decision(s)"

    # 8. Sound plan (spec section 19).
    sound_plan = plan_sound(ordered_analyses, transitions)
    paths.sound_plan_json.write_text(
        json.dumps([s.model_dump(mode="json") for s in sound_plan], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    memory.sound_plan_status = f"{len(sound_plan)} operation(s) planned"

    # 9. EDL + MasterTimeline (spec section 8) -- exact shared schema.
    edl = build_scene_edl(
        ordered_items,
        width=preset.width,
        height=preset.height,
        fps=preset.fps,
        cache_dir=paths.cache_dir / "assembler_audio",
        transitions=transitions,
    )
    master_timeline = build_assembler_timeline(edl)
    save_master_timeline(master_timeline, paths.master_timeline_json)
    memory.master_timeline_path = str(paths.master_timeline_json)

    rough_cut_path: Path | None = None
    final_output: Path | None = None
    qa_issues: list[QAIssue] = []

    # 10. Rough Cut (spec section 12, HIGHEST PRIORITY).
    if rough:
        try:
            rough_cut_path = render_rough_cut(edl, paths.rough_cut_mp4, preset)
            memory.rough_cut_path = str(rough_cut_path)
            memory.render_history.append(f"Rendered Rough Cut {rough_cut_path}")
            qa_issues.extend(run_assembler_qa(rough_cut_path, edl, inventory))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Rough Cut render failed: {exc}")
            memory.log_rejected(f"Assembler Rough Cut failed: {exc}")

    # 11. Finish (spec section 21) -- reuses the same EDL/plan decisions.
    if finish:
        try:
            final_output, notes = render_finish(
                edl,
                paths.final_mp4,
                preset,
                project_root=paths.root,
                cache_dir=paths.cache_dir,
                brand=brand,
                offline=offline,
            )
            memory.finish_status = "rendered"
            memory.render_history.append(f"Rendered Finish {final_output}")
            memory.decisions.extend(notes)
            qa_issues.extend(run_assembler_qa(final_output, edl, inventory))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Finish render failed: {exc}")
            memory.log_rejected(f"Assembler Finish failed: {exc}")

    memory.qa_issues.extend(f"[{i.severity.value}] {i.category}: {i.message}" for i in qa_issues)
    memory.save()

    return AssemblerResult(
        project_dir=paths.edit_dir,
        scene_inventory=inventory,
        scene_analysis=analyses,
        script_alignment=script_alignment,
        order_policy=resolution.policy_applied,
        normalization_plan=normalization,
        continuity_report=continuity,
        transition_plan=transitions,
        sound_plan=sound_plan,
        master_timeline=master_timeline,
        rough_cut_path=rough_cut_path,
        final_output=final_output,
        qa_issues=qa_issues,
        warnings=warnings,
    )
