"""Builds the ffmpeg filter_complex graph for one EDL (spec sections 11, 12;
Phase 2 Finalization spec sections 2-7 for real transitions/loudness).

Base approach: word-boundary trims per clip, hard-cut concat for video, and
short `afade` fades on audio at every cut to avoid clicks/pops (spec section
11's explicit fallback when a full crossfade dissolve isn't required).

A clip boundary gets a REAL rendered transition only when the incoming clip
carries a positive `transition_duration_s` -- today only the Assembler's
Transition Director ever sets that field, so Editor's and Creator's existing
flat hard-cut concat output is completely unchanged. When it is set, that
boundary is rendered as a genuine `xfade` (video) chained pairwise with the
accumulated output so far, paired with a real `acrossfade` (audio) only when
both neighboring clips have a genuine (non-synthesized) audio stream --
otherwise the audio side stays a plain concat, never a forced crossfade
(spec section 6). Per-clip `loudnorm_target_db` similarly triggers a real,
conservative `loudnorm` pass on that clip's own audio only, never blanket
across every clip.

Caption burn-in and overlay compositing (B-roll / motion / behind-subject
layers) are applied after the video chain, in the layer order from spec
section 16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.core.schemas import EDL, TransitionType
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

    # A boundary (between clip i-1 and clip i) gets a real rendered xfade
    # only when clip i explicitly asks for one -- see module docstring.
    xfade_at: list[bool] = [
        clip.transition_in == TransitionType.CROSSFADE and clip.transition_duration_s > 0
        for clip in edl.clips
    ]

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
            f"asetpts=PTS-STARTPTS,"
            f"aformat=sample_rates=44100:channel_layouts=stereo"
        )
        if speed != 1.0:
            af += f",atempo={max(0.5, min(2.0, speed)):.3f}"
        if clip.loudnorm_target_db is not None:
            af += f",loudnorm=I={clip.loudnorm_target_db:.1f}:LRA=11:TP=-1.5"
        fade_parts = []
        # Suppress the plain edge afade on any side that a real acrossfade
        # will already handle -- otherwise the signal gets faded twice.
        suppress_in = xfade_at[i]
        suppress_out = xfade_at[i + 1] if i + 1 < len(edl.clips) else False
        if clip.audio_fade_in_ms > 0 and not suppress_in:
            fade_parts.append(f"afade=t=in:st=0:d={clip.audio_fade_in_ms / 1000:.3f}")
        if clip.audio_fade_out_ms > 0 and not suppress_out:
            start = max(0.0, clip.duration - clip.audio_fade_out_ms / 1000)
            fade_parts.append(f"afade=t=out:st={start:.3f}:d={clip.audio_fade_out_ms / 1000:.3f}")
        if fade_parts:
            af += "," + ",".join(fade_parts)
        af += f"[{a_label}]"
        filters.append(af)

        v_labels.append(v_label)
        a_labels.append(a_label)

    durations = [c.duration for c in edl.clips]
    acc_v, acc_a = v_labels[0], a_labels[0]
    acc_dur = durations[0]
    for i in range(1, len(edl.clips)):
        clip = edl.clips[i]
        prev_clip = edl.clips[i - 1]
        if xfade_at[i]:
            d = min(clip.transition_duration_s, acc_dur, durations[i])
            offset = max(0.0, acc_dur - d)
            # `xfade`/`acrossfade` require both operands to share one
            # timebase; a freshly-trimmed clip's native tbn otherwise
            # mismatches an already-concatenated accumulator's timebase.
            acc_v_tb, v_tb = f"vtb{i}a", f"vtb{i}b"
            filters.append(f"[{acc_v}]settb=AVTB[{acc_v_tb}]")
            filters.append(f"[{v_labels[i]}]settb=AVTB[{v_tb}]")
            new_v = f"vxf{i}"
            filters.append(
                f"[{acc_v_tb}][{v_tb}]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[{new_v}]"
            )
            acc_v = new_v
            if prev_clip.has_real_audio and clip.has_real_audio:
                acc_a_tb, a_tb = f"atb{i}a", f"atb{i}b"
                filters.append(f"[{acc_a}]asettb=AVTB[{acc_a_tb}]")
                filters.append(f"[{a_labels[i]}]asettb=AVTB[{a_tb}]")
                new_a = f"axf{i}"
                filters.append(f"[{acc_a_tb}][{a_tb}]acrossfade=d={d:.3f}[{new_a}]")
                acc_a = new_a
                acc_dur = acc_dur + durations[i] - d
            else:
                # Only one side has real audio -- a forced acrossfade would
                # blend real audio against synthesized silence, which is not
                # a real audio crossfade. Concatenate audio as a hard cut
                # while the video still genuinely crossfades (spec section 6).
                new_a = f"aconcat{i}"
                filters.append(f"[{acc_a}][{a_labels[i]}]concat=n=2:v=0:a=1[{new_a}]")
                acc_a = new_a
                acc_dur = acc_dur + durations[i] - d
        else:
            new_v, new_a = f"vc{i}", f"ac{i}"
            filters.append(
                f"[{acc_v}][{acc_a}][{v_labels[i]}][{a_labels[i]}]concat=n=2:v=1:a=1[{new_v}][{new_a}]"
            )
            acc_v, acc_a = new_v, new_a
            acc_dur += durations[i]

    filters.append(f"[{acc_v}]null[vconcat]")
    filters.append(f"[{acc_a}]anull[aout]")

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
