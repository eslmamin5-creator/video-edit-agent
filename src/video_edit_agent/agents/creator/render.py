"""Creator rendering (spec sections 8, 14): synthesizes a single solid-color
+ silent-audio background clip covering the full MasterTimeline duration,
then reuses the existing, unmodified shared render pipeline
(`render.composition.build_filter_complex` + `render.ffmpeg.render`) --
motion graphics and B-roll become `Overlay`s on top of that background,
exactly like B-roll/motion overlays already work for the Editor workflow.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.media import MediaError, run
from video_edit_agent.core.schemas import EDL, CutReason, EDLClip
from video_edit_agent.core.timeline import MasterTimeline, TrackType
from video_edit_agent.render.composition import Overlay, RenderPlan
from video_edit_agent.render.export import ExportPreset
from video_edit_agent.render.ffmpeg import render as render_ffmpeg


class CreatorRenderError(RuntimeError):
    pass


def synthesize_background(
    output_path: Path, *, duration: float, width: int, height: int, fps: float, color: str = "black"
) -> Path:
    """Solid-color, silent-audio clip via ffmpeg's `lavfi` virtual devices --
    no external asset required, so this always works fully offline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(duration, 0.1)
    result = run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}:d={duration:.3f}:r={fps}",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_path),
        ],
        timeout=300,
    )
    if result.returncode != 0:
        raise CreatorRenderError(f"Background synthesis failed: {result.stderr.strip()[-2000:]}")
    return output_path


def render_creator_timeline(
    timeline: MasterTimeline, project_root: Path, output_path: Path, preset: ExportPreset, cache_dir: Path
) -> Path:
    duration = max(timeline.total_duration, 0.1)
    background_path = synthesize_background(
        cache_dir / "creator_background.mp4",
        duration=duration,
        width=timeline.width,
        height=timeline.height,
        fps=timeline.fps,
    )

    bg_clip = EDLClip(
        source_file=str(background_path),
        source_in=0.0,
        source_out=duration,
        timeline_in=0.0,
        timeline_out=duration,
        reason=CutReason.MANUAL,
    )
    edl = EDL(version=1, fps=timeline.fps, width=timeline.width, height=timeline.height, clips=[bg_clip])

    # B-roll drawn first (covers the background), motion graphics drawn after
    # (on top of B-roll), matching the Editor's existing layer ordering.
    overlays: list[Overlay] = []
    for item in sorted(timeline.items, key=lambda i: i.layer):
        if item.type not in (TrackType.BROLL, TrackType.MOTION_GRAPHICS):
            continue
        if not item.source:
            continue
        overlays.append(
            Overlay(
                path=Path(item.source),
                start=item.start,
                end=item.end,
                scale_to_canvas=item.type == TrackType.BROLL,
            )
        )

    plan = RenderPlan(edl=edl, overlays=overlays, captions=None)
    return render_ffmpeg(plan, output_path, preset)
