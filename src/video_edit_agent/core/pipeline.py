"""Full pipeline orchestration (spec section 45): media inventory -> audio
extraction -> transcription -> take analysis -> EDL -> captions -> B-roll ->
motion -> behind-subject composition -> render -> QA -> auto-repair -> final
output. Every stage is wrapped so an optional capability's absence degrades
the feature, not the whole run (spec section 43): the pipeline can complete
with local Faster-Whisper only, zero API keys, and zero Node/Manim/mediapipe
installed, producing a valid (if visually simpler) final video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.brand.loader import load_brand
from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.planner import plan_broll
from video_edit_agent.captions.engine import write_captions
from video_edit_agent.captions.styles import resolve_style
from video_edit_agent.core.config import AppConfig
from video_edit_agent.core.media import MediaError, content_hash, extract_audio, probe
from video_edit_agent.core.project import ProjectMemory, ProjectPaths
from video_edit_agent.core.schemas import (
    EDL,
    BrollPlanItem,
    MotionPlanItem,
    QAIssue,
    QAReport,
    Transcript,
)
from video_edit_agent.editorial.edl import validate as validate_edl
from video_edit_agent.editorial.packer import write_takes_packed
from video_edit_agent.editorial.planner import build_edl
from video_edit_agent.motion.director import build_motion_plan
from video_edit_agent.motion.router import render_motion
from video_edit_agent.qa.brand import run_brand_qa
from video_edit_agent.qa.language import run_language_qa
from video_edit_agent.qa.repair import RepairResult, run_repair_loop
from video_edit_agent.qa.technical import run_technical_qa
from video_edit_agent.qa.visual import run_visual_qa
from video_edit_agent.render.composition import CaptionBurn, Overlay, RenderPlan
from video_edit_agent.render.export import resolve_preset
from video_edit_agent.render.ffmpeg import render as render_ffmpeg
from video_edit_agent.subject.compositor import (
    find_enclosing_clip,
    render_subject_cutout,
    to_source_window,
)
from video_edit_agent.transcription.router import TranscriptionRouter, save_transcript


@dataclass
class PipelineResult:
    project_dir: Path
    final_output: Path | None
    transcript: Transcript | None = None
    edl: EDL | None = None
    broll_plan: list[BrollPlanItem] = field(default_factory=list)
    motion_plan: list[MotionPlanItem] = field(default_factory=list)
    qa_report: QAReport | None = None
    repairs_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_pipeline(
    source_video: Path,
    *,
    config: AppConfig | None = None,
    brand_name: str | None = None,
    offline: bool = False,
    preset_name: str = "reel",
    caption_style_name: str = "word-highlight",
    enable_broll: bool = True,
    enable_motion: bool = True,
    on_progress=None,
) -> PipelineResult:
    """Runs the full editing pipeline for a single source video and writes
    all intermediate + final artifacts under `<source_video parent>/edit/`."""

    def progress(stage: str) -> None:
        if on_progress:
            on_progress(stage)

    cfg = config or AppConfig.load(project_dir=source_video.parent)
    offline = offline or cfg.offline
    brand: Brand = load_brand(brand_name or cfg.brand)

    paths = ProjectPaths.for_source(source_video)
    paths.ensure()
    memory = ProjectMemory.load_or_new(paths)
    memory.source_inventory.append(str(source_video))
    memory.brand = brand.name
    warnings: list[str] = []

    # 1. Media inventory
    progress("probe")
    try:
        probe(source_video)
    except MediaError as exc:
        memory.log_rejected(f"Could not probe source media: {exc}")
        memory.save()
        return PipelineResult(project_dir=paths.edit_dir, final_output=None, warnings=[str(exc)])

    # 2. Audio extraction
    progress("extract_audio")
    audio_path = paths.cache_dir / f"{content_hash(source_video)}.wav"
    if not audio_path.exists():
        extract_audio(source_video, audio_path)

    # 3. Transcription (auto no-key fallback per spec section 5/6)
    progress("transcribe")
    router = TranscriptionRouter(cfg.transcription, offline=offline, gemini_model=cfg.gemini.transcription_model)
    transcript = router.transcribe(audio_path)
    save_transcript(transcript, paths.transcript_unified)
    paths.transcript_txt.write_text(transcript.full_text, encoding="utf-8")
    memory.log_decision(f"Transcribed with provider: {transcript.provider}")

    # 4. Take analysis + EDL (word-boundary precise cuts, spec section 11/12)
    progress("edl")
    edl, verdicts = build_edl(
        transcript, source_video, crossfade_ms=cfg.editorial.crossfade_ms,
        width=resolve_preset(preset_name).width, height=resolve_preset(preset_name).height,
        fps=resolve_preset(preset_name).fps, audio_path=audio_path,
    )
    write_takes_packed(transcript, verdicts, paths.takes_packed)
    try:
        validate_edl(edl)
    except Exception as exc:  # noqa: BLE001
        memory.log_rejected(f"EDL validation failed: {exc}")
        memory.save()
        return PipelineResult(project_dir=paths.edit_dir, final_output=None, transcript=transcript, warnings=[str(exc)])
    from video_edit_agent.editorial.edl import save as save_edl

    save_edl(edl, paths.edl_json)
    from video_edit_agent.core.timeline import edl_to_master_timeline
    from video_edit_agent.core.timeline import save as save_master_timeline

    save_master_timeline(edl_to_master_timeline(edl, workflow="editor"), paths.master_timeline_json)

    # 5. Captions (Arabic-capable, RTL, word-highlight per spec section 15)
    progress("captions")
    caption_style = resolve_style(caption_style_name, brand.captions.model_dump())
    write_captions(transcript, edl, caption_style, paths.edit_dir / "captions.ass", paths.master_srt)

    # 6. B-roll planning (graceful: never blocks the pipeline)
    broll_items: list[BrollPlanItem] = []
    if enable_broll:
        progress("broll")
        try:
            broll_items = plan_broll(
                edl, transcript, paths.edit_dir / "broll_assets", paths.cache_dir / "broll_generated",
                brand=brand, allow_generation=not offline,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"B-roll planning skipped: {exc}")
        import json

        paths.broll_plan.write_text(
            json.dumps([item.model_dump(mode="json") for item in broll_items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 7. Motion graphics (graceful: falls back to `simple` engine, spec section 17/43)
    motion_items: list[MotionPlanItem] = []
    overlays: list[Overlay] = []
    if enable_motion:
        progress("motion")
        try:
            specs = build_motion_plan(edl, transcript)
        except Exception as exc:  # noqa: BLE001
            specs = []
            warnings.append(f"Motion planning skipped: {exc}")

        motion_output_dir = paths.cache_dir / "motion"
        for i, spec in enumerate(specs):
            item = render_motion(
                spec, paths.root, motion_output_dir, brand=brand, fps=edl.fps, slot_id=f"motion{i}", offline=offline
            )
            motion_items.append(item)
            if not item.output_path:
                continue

            graphic_overlay = Overlay(path=Path(item.output_path), start=spec.timeline_start, end=spec.timeline_end)

            if not spec.behind_subject:
                overlays.append(graphic_overlay)
                continue

            # Behind-subject compositing (spec Phase 2 section 25): draw the
            # graphic first (it will cover the subject baked into the base
            # frame), then draw a subject-only cutout on top to restore the
            # subject in front of it. Falls back to a plain foreground
            # overlay -- never drops the overlay -- if the window doesn't map
            # onto a single source clip or segmentation can't produce a
            # usable mask (spec section 26).
            graphic_overlay.behind_subject = True
            overlays.append(graphic_overlay)

            enclosing = find_enclosing_clip(edl, spec.timeline_start, spec.timeline_end)
            cutout_path = None
            if enclosing is not None:
                src_start, src_end = to_source_window(enclosing, spec.timeline_start, spec.timeline_end)
                cutout_path = render_subject_cutout(
                    Path(enclosing.source_file), src_start, src_end, paths.cache_dir / "subject_cutouts",
                )

            if cutout_path is not None:
                overlays.append(
                    Overlay(path=cutout_path, start=spec.timeline_start, end=spec.timeline_end)
                )
                memory.log_decision(
                    f"Behind-subject compositing applied for motion slot {i} ({spec.kind.value})"
                )
            else:
                reason = "motion window spans a cut" if enclosing is None else "no usable subject mask"
                warnings.append(
                    f"Behind-subject requested for motion slot {i} but unavailable ({reason}); "
                    "used plain foreground overlay instead"
                )
                memory.log_decision(
                    f"Behind-subject fallback for motion slot {i}: {reason} -> plain overlay"
                )

        import json

        paths.motion_plan.write_text(
            json.dumps([item.model_dump(mode="json") for item in motion_items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 8. Render (spec sections 12, 14, 47 — argument-array ffmpeg only)
    progress("render")
    plan = RenderPlan(edl=edl, overlays=overlays, captions=CaptionBurn(ass_path=paths.edit_dir / "captions.ass"))
    preset = resolve_preset(preset_name)
    final_output = render_ffmpeg(plan, paths.final_mp4, preset)
    memory.render_history.append(f"Rendered {final_output} with preset '{preset_name}'")

    # 9. Multi-layer QA + bounded auto-repair (spec section 33)
    progress("qa")

    def collect_qa() -> QAReport:
        issues: list[QAIssue] = []
        issues.extend(run_technical_qa(final_output, edl))
        issues.extend(run_language_qa(transcript, transcript))
        issues.extend(run_visual_qa(final_output))
        issues.extend(run_brand_qa(edl, transcript, brand, has_cta_slot=any(
            m.spec.kind.value == "cta" for m in motion_items if m.output_path
        )))
        return QAReport(issues=issues)

    initial_report = collect_qa()
    repair_result: RepairResult = run_repair_loop(initial_report, collect_qa, repair_handlers={})
    memory.qa_issues.extend(f"[{i.severity.value}] {i.category}: {i.message}" for i in repair_result.report.issues)
    memory.save()

    return PipelineResult(
        project_dir=paths.edit_dir,
        final_output=final_output,
        transcript=transcript,
        edl=edl,
        broll_plan=broll_items,
        motion_plan=motion_items,
        qa_report=repair_result.report,
        repairs_applied=repair_result.repairs_applied,
        warnings=warnings,
    )
