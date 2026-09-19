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

from video_edit_agent.brand.loader import load_brand, resolve_brand_logo, resolve_fonts_dir
from video_edit_agent.brand.logo_policy import LogoPlan, plan_logo
from video_edit_agent.brand.schema import Brand, LogoMode
from video_edit_agent.broll.overlay import broll_items_to_overlays
from video_edit_agent.broll.planner import plan_broll
from video_edit_agent.broll.treatment import load_decisions
from video_edit_agent.captions.brand_style import resolve_brand_caption_style
from video_edit_agent.captions.engine import caption_chunks, write_captions
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
from video_edit_agent.editorial.punch_in import plan_punch_ins
from video_edit_agent.motion.director import build_motion_plan, plan_hook
from video_edit_agent.motion.router import render_motion
from video_edit_agent.qa.brand import run_brand_qa
from video_edit_agent.qa.language import run_language_qa
from video_edit_agent.qa.repair import RepairResult, run_repair_loop
from video_edit_agent.qa.technical import run_technical_qa
from video_edit_agent.qa.visual import run_visual_qa
from video_edit_agent.render.composition import CaptionBurn, Overlay, RenderPlan
from video_edit_agent.render.end_card import compose_with_cards, render_card_frame
from video_edit_agent.render.export import resolve_preset
from video_edit_agent.render.ffmpeg import render as render_ffmpeg
from video_edit_agent.review import state as review_state
from video_edit_agent.review.builder import (
    build_brand_summary,
    build_broll_review,
    build_caption_preview,
    build_timeline_review,
    build_transcript_review,
    render_timeline_markdown,
)
from video_edit_agent.review.corrections import apply_corrections, load_corrections
from video_edit_agent.review.preview import (
    build_contact_sheet,
    generate_preview_frames,
    pick_caption_timestamps,
    pick_representative_timestamps,
)
from video_edit_agent.review.schemas import (
    PreviewFrame,
    ReviewApprovalState,
    ReviewStage,
    TextTreatmentReview,
)
from video_edit_agent.subject.compositor import (
    find_enclosing_clip,
    render_subject_cutout,
    to_source_window,
)
from video_edit_agent.subject.framing import analyze_clip
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
    # Review-First Editing Workflow (spec section 8): True once the pipeline
    # has either rendered (legacy `review=False` callers) or the user has
    # explicitly approved a review (`review.state.approve`/`bypass`). When
    # `review=True` was requested and no approval has happened yet, this is
    # False and `final_output` is None -- no expensive render has occurred.
    ready_for_final_render: bool = True
    review_dir: Path | None = None


def _merged_text_treatments(approval: ReviewApprovalState, hook_review: TextTreatmentReview | None) -> list[TextTreatmentReview]:
    """The state's text treatments with this run's hook record swapped in (its
    visual approval and user-supplied copy were carried over by `plan_hook`)."""
    kept = [t for t in approval.text_treatments if hook_review is None or t.treatment != hook_review.treatment]
    return kept + ([hook_review] if hook_review is not None else [])


def _planned_motion_labels(
    motion_items: list[MotionPlanItem], timestamps: dict[str, float], broll_items: list[BrollPlanItem] | None = None
) -> dict[str, str]:
    """Maps each motion-derived preview label to a human-readable "planned"
    banner, using the same kind -> label rules as `pick_representative_timestamps`."""
    labels: dict[str, str] = {}
    for m in motion_items:
        kind = m.spec.kind.value
        label = (
            "cta" if kind == "cta"
            else "logo" if kind == "logo_reveal"
            else "behind_subject" if m.spec.behind_subject
            else "motion_graphic"
        )
        if label in timestamps and label not in labels:
            behind = " | behind subject" if m.spec.behind_subject else ""
            labels[label] = (
                f"PLANNED (not rendered): {kind} {m.spec.timeline_start:.1f}-{m.spec.timeline_end:.1f}s{behind}"
            )
    at = timestamps.get("broll")
    for b in broll_items or []:
        if at is None or "broll" in labels or b.asset_path or not (b.timeline_start <= at <= b.timeline_end):
            continue
        if b.treatment == "generated_broll":
            labels["broll"] = "PLANNED (not generated): generated B-roll"
        elif b.treatment == "local_broll":
            labels["broll"] = "PLANNED (footage needed): local B-roll"
    return labels


def _caption_sample_lines(chunks: list, picks: dict[str, float]) -> tuple[str, str | None, str | None]:
    """Real caption text (from the user's own transcript) at the picked preview moments."""

    def text_at(label: str) -> str | None:
        t = picks.get(label)
        if t is None:
            return None
        return next((c.text for c in chunks if c.start <= t <= c.end), None)

    def is_arabic(ch: str) -> bool:
        return "\u0600" <= ch <= "\u06ff"

    normal = text_at("caption_normal") or (chunks[0].text if chunks else "")
    mixed = next(
        (
            c.text for c in chunks
            if any("a" <= ch.lower() <= "z" for ch in c.text) and any(is_arabic(ch) for ch in c.text)
        ),
        None,
    )
    return normal, text_at("caption_multiline"), mixed


def _brand_logo_overlay(logo_path: Path, brand: Brand, edl: EDL) -> Overlay:
    """Builds a persistent, top-right brand-logo watermark overlay (spec
    section 23; item 5 of the Baseline Recovery Milestone). Sized to 15% of
    the canvas width (aspect-preserved) and inset from the top-right corner
    by the brand's `safe_zones["top"]` fraction (default 5%) plus a fixed 4%
    right margin, matching `logo_rules` guidance such as "Logo stays in the
    top-right safe zone, never over captions" (examples/brand_profile)."""
    top_margin = round(brand.safe_zones.get("top", 0.05) * edl.height)
    right_margin = round(0.04 * edl.width)
    scale_width = round(0.15 * edl.width)
    return Overlay(
        path=logo_path,
        start=0.0,
        end=edl.total_duration,
        x=f"W-w-{right_margin}",
        y=str(top_margin),
        scale_width=scale_width,
    )


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
    review: bool | None = None,
    transcript_override: Transcript | None = None,
    logo_mode: str | None = None,
) -> PipelineResult:
    """Runs the editing pipeline for a single source video and writes all
    intermediate + final artifacts under `<source_video parent>/edit/`.

    Review-first is the default. `review=None` consults the persisted review
    state under `edit/review/`: without an approved state the run plans only
    (transcript/caption/brand/timeline/B-roll review artifacts, preview
    frames, the gate state) and stops before any expensive work -- no B-roll
    generation, no motion or subject-cutout rendering, no final render. Once
    the user has approved (`review.state.approve`), the same call renders.
    `review=True` always plans only. `review=False` is the explicit,
    recorded bypass (`--yes` / `--no-review`).

    User transcript corrections saved under `edit/review/` are applied on
    every run, so re-running regenerates only cheap review artifacts.
    """

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

    review_dir = paths.edit_dir / "review"
    if review is False and not review_state.is_ready_for_final_render(review_dir):
        review_state.bypass(review_dir, "review bypassed explicitly (--yes / --no-review)")
    approval = review_state.load_review_state(review_dir)
    plan_only = review is True or not approval.ready_for_final_render

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
    if transcript_override is not None:
        transcript = transcript_override
        memory.log_decision("Using corrected transcript override; skipped re-transcription")
    else:
        router = TranscriptionRouter(
            cfg.transcription, offline=offline, gemini_model=cfg.gemini.transcription_model
        )
        transcript = router.transcribe(audio_path)
    save_transcript(transcript, paths.transcript_unified)  # raw ASR; corrections stay separate
    corrections = load_corrections(review_dir)
    if corrections:
        transcript = apply_corrections(transcript, corrections)
        memory.log_decision(f"Applied {len(corrections)} persisted transcript correction(s); timing preserved")
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
    # Where the subject/face sit and how bright the backdrop is, sampled from
    # the opening seconds. Best-effort (None when unavailable): it keeps the
    # punch-in face-safe and lets the hook title be placed and coloured legibly.
    lead_clip = edl.clips[0] if edl.clips else None
    footage = analyze_clip(
        lead_clip.source_file, lead_clip.source_in, min(lead_clip.source_out, lead_clip.source_in + 4.0)
    ) if lead_clip else None
    plan_punch_ins(edl, energy=brand.motion.energy, face_box=footage.face if footage else None)
    from video_edit_agent.editorial.edl import save as save_edl

    save_edl(edl, paths.edl_json)
    from video_edit_agent.core.timeline import edl_to_master_timeline
    from video_edit_agent.core.timeline import save as save_master_timeline

    save_master_timeline(edl_to_master_timeline(edl, workflow="editor"), paths.master_timeline_json)

    # 5. Captions (Arabic-capable, RTL, word-highlight per spec section 15)
    progress("captions")
    resolved_style = resolve_brand_caption_style(caption_style_name, brand)
    caption_style = resolved_style.style
    write_captions(transcript, edl, caption_style, paths.edit_dir / "captions.ass", paths.master_srt)

    # 6. B-roll planning (graceful: never blocks the pipeline)
    broll_items: list[BrollPlanItem] = []
    if enable_broll:
        progress("broll")
        try:
            broll_items = plan_broll(
                edl, transcript, paths.edit_dir / "broll_assets", paths.cache_dir / "broll_generated",
                brand=brand, decisions=load_decisions(review_dir),
                allow_generation=not offline and not plan_only and approval.broll_generation_approved,
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
    hook_review = None
    # B-roll overlays go in first so motion graphics / behind-subject cutouts
    # draw on top of them, matching the Creator workflow's layer ordering
    # (agents/creator/render.py: "B-roll drawn first ... motion graphics
    # drawn after"). This is what makes resolved B-roll actually reach
    # final.mp4 instead of only `broll_plan.json` (Baseline Recovery
    # Milestone item 1).
    overlays: list[Overlay] = broll_items_to_overlays(broll_items)
    if enable_motion:
        progress("motion")
        try:
            # Hook copy is gated by the copy-approval rule: unresolved ASR never
            # becomes hook copy (placeholder in review, no copy layer in a final render).
            hook_plan = plan_hook(edl, transcript, brand, footage, approval, for_final_render=not plan_only)
            specs = build_motion_plan(edl, transcript, brand=brand, analysis=footage, hook=hook_plan)
            hook_review = hook_plan.review
            if hook_plan.spec is None and hook_plan.review is not None:
                warnings.append(
                    f"Hook title omitted from the final render: copy is {hook_plan.review.copy_status.value} "
                    f"({hook_plan.review.blocking_reason})"
                )
        except Exception as exc:  # noqa: BLE001
            specs = []
            warnings.append(f"Motion planning skipped: {exc}")

        motion_output_dir = paths.cache_dir / "motion"
        for i, spec in enumerate(specs):
            if plan_only:
                # Review-First: plan only. No motion or subject-cutout rendering
                # happens before the user approves the plan.
                motion_items.append(MotionPlanItem(spec=spec, engine_used=None, output_path=None))
                continue
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

        if not plan_only:
            paths.motion_plan.write_text(
                json.dumps([item.model_dump(mode="json") for item in motion_items], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # 7b. Logo policy: the Brand Profile (or an explicit override) decides
    # whether the logo appears as a persistent bug, an intro/end card, or not
    # at all. Only `persistent_bug` adds an overlay; cards wrap the content.
    logo_path = resolve_brand_logo(brand)
    logo_plan: LogoPlan = plan_logo(
        brand, logo_path, edl.width, edl.height, edl.fps,
        mode_override=LogoMode(logo_mode) if logo_mode else None,
    )
    warnings.extend(logo_plan.warnings)
    if logo_plan.persistent_bug and logo_path is not None:
        overlays.append(_brand_logo_overlay(logo_path, brand, edl))
        memory.log_decision(f"Applied persistent brand logo bug from {logo_path} (logo mode: persistent_bug)")
    else:
        memory.log_decision(f"Logo mode '{logo_plan.mode.value}': no persistent logo bug")

    fonts_dir = resolve_fonts_dir(brand.name)
    plan = RenderPlan(
        edl=edl,
        overlays=overlays,
        captions=CaptionBurn(ass_path=paths.edit_dir / "captions.ass", fonts_dir=fonts_dir),
    )

    # Review gate (Review-First Editing Workflow spec sections 1-8): stop
    # here, before the expensive final render, and hand the user everything
    # needed to inspect/correct the plan. Never reached when `review=False`
    # (the default), so this cannot regress any existing caller.
    if plan_only:
        progress("review")
        review_dir.mkdir(parents=True, exist_ok=True)

        cta_text = next(
            (m.spec.text for m in motion_items if m.spec.kind.value == "cta" and m.spec.text),
            brand.cta_text,
        )
        chunks = caption_chunks(transcript, edl, caption_style)
        caption_picks = pick_caption_timestamps(chunks, caption_style.line_break_chars, caption_style.word_highlight)
        sample_lines = _caption_sample_lines(chunks, caption_picks)

        (review_dir / "transcript_review.json").write_text(
            build_transcript_review(transcript, caption_style).model_dump_json(indent=2), encoding="utf-8"
        )
        (review_dir / "caption_preview.json").write_text(
            build_caption_preview(
                caption_style, brand, sample_lines=sample_lines, notes=resolved_style.notes
            ).model_dump_json(indent=2), encoding="utf-8"
        )
        (review_dir / "brand_summary.json").write_text(
            build_brand_summary(
                brand, str(logo_path) if logo_path else None, cta_text, caption_style,
                logo_plan=logo_plan, style_notes=resolved_style.notes,
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        end_card_frame: PreviewFrame | None = None
        if logo_plan.end_card is not None:
            try:
                spec = logo_plan.end_card
                frame_path = render_card_frame(
                    spec, review_dir / "frames" / "frame_end_card.jpg", paths.cache_dir / "end_card_preview"
                )
                end_card_frame = PreviewFrame(
                    label="end_card", timeline_at=edl.total_duration + spec.duration * 0.8, image_path=str(frame_path)
                )
            except Exception as exc:  # noqa: BLE001 - preview only
                warnings.append(f"End-card preview skipped: {exc}")
        timeline = build_timeline_review(
            edl, transcript, broll_items, motion_items,
            logo_plan=logo_plan, cta_text=cta_text,
            caption_note="brand-driven (font, colors and highlight from the Brand Profile)",
            end_card_preview=end_card_frame.image_path if end_card_frame else None,
        )
        (review_dir / "timeline_review.json").write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        (review_dir / "timeline_review.md").write_text(render_timeline_markdown(timeline), encoding="utf-8")
        (review_dir / "broll_review.json").write_text(
            build_broll_review(broll_items).model_dump_json(indent=2), encoding="utf-8"
        )

        try:
            timestamps = pick_representative_timestamps(
                edl, motion_items, broll_items, transcript, caption_picks=caption_picks
            )
            # Motion is planned-only in review mode, so its frames get a
            # "planned treatment" banner instead of a real Remotion render.
            planned_labels = _planned_motion_labels(motion_items, timestamps, broll_items)
            frame_set = generate_preview_frames(
                plan, timestamps, review_dir / "frames", planned_labels=planned_labels
            )
            if end_card_frame is not None:
                frame_set.frames.append(end_card_frame)
            build_contact_sheet(frame_set, review_dir / "contact_sheet.jpg")
            (review_dir / "preview_frames.json").write_text(frame_set.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - preview frames are a convenience, never a hard failure
            warnings.append(f"Preview frame generation skipped: {exc}")

        # Carry open transcript items forward: re-running the review must
        # never silently forget what the user has not confirmed yet.
        review_state.save_review_state(
            ReviewApprovalState(
                stage=ReviewStage.REVIEW_VISUALS,
                ready_for_final_render=False,
                unresolved_transcript=approval.unresolved_transcript,
                text_treatments=_merged_text_treatments(approval, hook_review),
            ),
            review_dir,
        )
        memory.log_decision("Stopped at review gate before final render (approval required)")
        memory.save()

        return PipelineResult(
            project_dir=paths.edit_dir,
            final_output=None,
            transcript=transcript,
            edl=edl,
            broll_plan=broll_items,
            motion_plan=motion_items,
            qa_report=None,
            repairs_applied=[],
            warnings=warnings,
            ready_for_final_render=False,
            review_dir=review_dir,
        )

    # 8. Render (spec sections 12, 14, 47 — argument-array ffmpeg only)
    progress("render")
    preset = resolve_preset(preset_name)
    cards = logo_plan.cards
    content_output = render_ffmpeg(plan, paths.cache_dir / "main_content.mp4" if cards else paths.final_mp4, preset)
    final_output = content_output
    if cards:
        final_output = compose_with_cards(
            content_output, paths.final_mp4, paths.cache_dir / "cards", logo_plan.intro, logo_plan.end_card
        )
    memory.render_history.append(f"Rendered {final_output} with preset '{preset_name}'")

    # 9. Multi-layer QA + bounded auto-repair (spec section 33)
    progress("qa")

    def collect_qa() -> QAReport:
        issues: list[QAIssue] = []
        issues.extend(run_technical_qa(content_output, edl))
        issues.extend(run_language_qa(transcript, transcript))
        issues.extend(run_visual_qa(content_output))
        issues.extend(run_brand_qa(edl, transcript, brand, has_cta_slot=any(
            m.spec.kind.value == "cta" for m in motion_items if m.output_path
        )))
        return QAReport(issues=issues)

    initial_report = collect_qa()
    repair_result: RepairResult = run_repair_loop(initial_report, collect_qa, repair_handlers={})
    memory.qa_issues.extend(f"[{i.severity.value}] {i.category}: {i.message}" for i in repair_result.report.issues)
    memory.save()
    review_state.mark_rendered(review_dir, "final render completed")

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
