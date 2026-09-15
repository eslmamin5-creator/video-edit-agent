"""FFmpeg render engine (spec sections 12, 14, 47).

Always invokes ffmpeg via an argument array (never a shell string) so paths
with spaces/special characters can never cause shell injection.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.media import run
from video_edit_agent.render.composition import RenderPlan, build_filter_complex
from video_edit_agent.render.export import ExportPreset


class RenderError(RuntimeError):
    pass


def render(plan: RenderPlan, output_path: Path, preset: ExportPreset, *, crf: int = 20) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs, filter_complex, map_labels = build_filter_complex(plan)
    v_map, a_map = map_labels.strip("[]").split("][")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{v_map}]", "-map", f"[{a_map}]",
        "-r", str(preset.fps),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-b:v", preset.video_bitrate,
        "-c:a", "aac", "-b:a", preset.audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = run(cmd, timeout=3600)
    if result.returncode != 0:
        raise RenderError(f"ffmpeg render failed:\n{result.stderr.strip()[-4000:]}")
    return output_path


def render_preview(plan: RenderPlan, output_path: Path, preset: ExportPreset) -> Path:
    """Fast, lower-quality render for quick iteration."""
    return render(plan, output_path, preset, crf=30)
