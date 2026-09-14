"""Builds the ffmpeg filter_complex graph for one EDL (spec sections 11, 12).

V1 approach: word-boundary trims per clip, hard-cut concat for video, and
short `afade` fades on audio at every cut to avoid clicks/pops (spec section
11's explicit fallback when a full crossfade dissolve isn't required). Caption
burn-in and overlay compositing (B-roll / motion / behind-subject layers) are
applied after the concat, in the layer order from spec section 16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.core.schemas import EDL
from video_edit_agent.render.color import safe_format_filter


@dataclass
class Overlay:
    """One compositable layer: an image or short video clip placed on top of
    the base timeline between `start`/`end` seconds."""

    path: Path
    start: float
    end: float
    x: str = "(W-w)/2"
    y: str = "(H-h)/2"
    behind_subject: bool = False  # reserved for spec section 16 layering
    scale_to_canvas: bool = False  # scale+crop this overlay to cover the full canvas (Creator B-roll of arbitrary aspect ratio)


@dataclass
class CaptionBurn:
    ass_path: Path


@dataclass
class RenderPlan:
    edl: EDL
    overlays: list[Overlay] = field(default_factory=list)
    captions: CaptionBurn | None = None


def build_filter_complex(plan: RenderPlan) -> tuple[list[str], str, str]:
    """Returns (input_args, filter_complex, [vout,aout] labels usable for -map)."""
    edl = plan.edl
    inputs: list[str] = []
    input_index_by_file: dict[str, int] = {}
    for clip in edl.clips:
        if clip.source_file not in input_index_by_file:
            input_index_by_file[clip.source_file] = len(input_index_by_file)
            inputs += ["-i", clip.source_file]

    filters: list[str] = []
    v_labels, a_labels = [], []
    fmt = safe_format_filter(edl.width, edl.height)

    for i, clip in enumerate(edl.clips):
        src_idx = input_index_by_file[clip.source_file]
        v_label, a_label = f"v{i}", f"a{i}"
        speed = clip.speed if clip.speed and clip.speed != 1.0 else 1.0

        vf = (
            f"[{src_idx}:v]trim=start={clip.source_in:.3f}:end={clip.source_out:.3f},"
            f"setpts=PTS-STARTPTS"
        )
        if speed != 1.0:
            vf += f",setpts={1 / speed:.6f}*PTS"
        vf += f",{fmt}[{v_label}]"
        filters.append(vf)

        af = (
            f"[{src_idx}:a]atrim=start={clip.source_in:.3f}:end={clip.source_out:.3f},"
            f"asetpts=PTS-STARTPTS"
        )
        if speed != 1.0:
            af += f",atempo={max(0.5, min(2.0, speed)):.3f}"
        fade_parts = []
        if clip.audio_fade_in_ms > 0:
            fade_parts.append(f"afade=t=in:st=0:d={clip.audio_fade_in_ms / 1000:.3f}")
        if clip.audio_fade_out_ms > 0:
            start = max(0.0, clip.duration - clip.audio_fade_out_ms / 1000)
            fade_parts.append(f"afade=t=out:st={start:.3f}:d={clip.audio_fade_out_ms / 1000:.3f}")
        if fade_parts:
            af += "," + ",".join(fade_parts)
        af += f"[{a_label}]"
        filters.append(af)

        v_labels.append(v_label)
        a_labels.append(a_label)

    concat_inputs = "".join(f"[{v}][{a}]" for v, a in zip(v_labels, a_labels))
    filters.append(f"{concat_inputs}concat=n={len(edl.clips)}:v=1:a=1[vconcat][aout]")

    video_out = "vconcat"
    overlay_idx_offset = len(input_index_by_file)
    for j, ov in enumerate(plan.overlays):
        inputs += ["-i", str(ov.path)]
        ov_input = overlay_idx_offset + j
        ov_label = f"{ov_input}:v"
        if ov.scale_to_canvas:
            scaled_label = f"ovscaled{j}"
            filters.append(
                f"[{ov_input}:v]scale={edl.width}:{edl.height}:force_original_aspect_ratio=increase,"
                f"crop={edl.width}:{edl.height}[{scaled_label}]"
            )
            ov_label = scaled_label
        new_label = f"vov{j}"
        enable = f"between(t,{ov.start:.3f},{ov.end:.3f})"
        filters.append(
            f"[{video_out}][{ov_label}]overlay=x={ov.x}:y={ov.y}:enable='{enable}'[{new_label}]"
        )
        video_out = new_label

    if plan.captions is not None:
        ass_escaped = str(plan.captions.ass_path).replace("\\", "/").replace(":", "\\:")
        filters.append(f"[{video_out}]subtitles='{ass_escaped}'[vout]")
        video_out = "vout"
    else:
        filters.append(f"[{video_out}]null[vout]")
        video_out = "vout"

    return inputs, ";".join(filters), f"[{video_out}][aout]"
