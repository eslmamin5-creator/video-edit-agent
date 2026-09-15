"""Rough Cut / Finish rendering for Assembler (spec sections 12, 21-24).

Rough Cut is intentionally the fast path: direct EDL render via the shared
`render.ffmpeg.render_preview`, no overlays. Finish reuses the exact same
EDL/render core at full quality, optionally adding one Brand Profile
overlay (a CTA/title card, via the shared Motion Router -- the same engine
Creator already uses) when a brand was explicitly requested. Assembler
never invents B-roll or Behind-Subject overlays on its own initiative
(spec section 25) -- those are only wired in when the caller's plan
explicitly asks for them, which the current planning layers do not do
by default.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import EDL, AnimationKind, AnimationSpec
from video_edit_agent.motion.router import render_motion
from video_edit_agent.render.composition import Overlay, RenderPlan
from video_edit_agent.render.export import ExportPreset
from video_edit_agent.render.ffmpeg import render as render_ffmpeg
from video_edit_agent.render.ffmpeg import render_preview as render_preview_ffmpeg

CTA_CARD_DURATION = 2.0


def render_rough_cut(edl: EDL, output_path: Path, preset: ExportPreset) -> Path:
    plan = RenderPlan(edl=edl)
    return render_preview_ffmpeg(plan, output_path, preset)


def render_finish(
    edl: EDL,
    output_path: Path,
    preset: ExportPreset,
    *,
    project_root: Path,
    cache_dir: Path,
    brand: Brand | None,
    offline: bool,
) -> tuple[Path, list[str]]:
    overlays: list[Overlay] = []
    notes: list[str] = []

    if brand is not None and brand.cta.text_default:
        total_duration = edl.total_duration
        cta_start = max(0.0, total_duration - CTA_CARD_DURATION)
        spec = AnimationSpec(
            kind=AnimationKind.CTA,
            timeline_start=cta_start,
            timeline_end=total_duration,
            text=brand.cta.text_default,
        )
        motion_dir = cache_dir / "motion"
        motion_dir.mkdir(parents=True, exist_ok=True)
        result = render_motion(
            spec, project_root, motion_dir, brand=brand, fps=int(preset.fps), slot_id="assembler_cta", offline=offline
        )
        if result.output_path:
            overlays.append(
                Overlay(
                    path=Path(result.output_path),
                    start=cta_start,
                    end=total_duration,
                    scale_to_canvas=False,
                )
            )
            notes.append(f"Applied brand CTA card '{brand.cta.text_default}' via {result.engine_used}")
        else:
            notes.append(f"Brand CTA card requested but motion render failed: {result.error}")

    plan = RenderPlan(edl=edl, overlays=overlays)
    out = render_ffmpeg(plan, output_path, preset)
    return out, notes
