"""Builds a shared-core `EDL` from ordered whole-scene files (spec section
8): one direct-cut `EDLClip` per scene, full duration, in the caller's
chosen order. No Assembler-only timeline format -- this feeds straight into
the same `render.composition.build_filter_complex` / `render.ffmpeg.render`
the Editor and Creator already use.

The one wrinkle the shared render core doesn't handle on its own: every
`EDLClip`'s audio branch references `[src_idx:a]` unconditionally, so a
source file with no audio stream would break the filter graph. Scenes
without audio are muxed with a synthesized silent track (same `anullsrc`
technique Creator's background synthesis already uses) into a cached copy
before being referenced by the EDL -- the original source file is never
modified.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.agents.assembler.schemas import SceneInventoryItem, TransitionDecision, TransitionKind
from video_edit_agent.core.media import MediaError, run
from video_edit_agent.core.schemas import EDL, CutReason, EDLClip, TransitionType
from video_edit_agent.core.transition_math import clamp_transition_duration

_TRANSITION_MAP = {
    TransitionKind.SHORT_CROSSFADE: TransitionType.CROSSFADE,
    TransitionKind.DISSOLVE: TransitionType.CROSSFADE,
    TransitionKind.ACTION_CUT: TransitionType.HARD_CUT,
    TransitionKind.MATCH_CUT: TransitionType.HARD_CUT,
    TransitionKind.J_CUT: TransitionType.HARD_CUT,
    TransitionKind.L_CUT: TransitionType.HARD_CUT,
    TransitionKind.AUDIO_BRIDGE: TransitionType.HARD_CUT,
    TransitionKind.BRANDED: TransitionType.CROSSFADE,
    TransitionKind.HARD_CUT: TransitionType.HARD_CUT,
}
# Kinds that get a REAL rendered video crossfade (`xfade`/`acrossfade` in the
# shared render core, spec Phase 2 Finalization section 2) -- distinct from
# `_AUDIO_BRIDGE_KINDS` below, which only ever gets the older audio-only
# `afade` de-click treatment.
_VIDEO_CROSSFADE_KINDS = {TransitionKind.SHORT_CROSSFADE, TransitionKind.DISSOLVE}
_AUDIO_BRIDGE_KINDS = {TransitionKind.AUDIO_BRIDGE}


class EDLBuildError(RuntimeError):
    pass


def _ensure_has_audio(item: SceneInventoryItem, cache_dir: Path) -> str:
    """Returns a source path guaranteed to have an audio stream."""
    if item.has_audio:
        return item.path
    cache_dir.mkdir(parents=True, exist_ok=True)
    muxed = cache_dir / f"{item.id}_with_silent_audio.mp4"
    if muxed.exists():
        return str(muxed)
    result = run(
        [
            "ffmpeg", "-y", "-i", str(item.path),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            "-map", "0:v:0", "-map", "1:a:0",
            str(muxed),
        ],
        timeout=300,
    )
    if result.returncode != 0:
        raise EDLBuildError(f"Failed to synthesize silent audio for {item.path}: {result.stderr.strip()[-2000:]}")
    return str(muxed)


def build_scene_edl(
    ordered_items: list[SceneInventoryItem],
    *,
    width: int,
    height: int,
    fps: float,
    cache_dir: Path,
    transitions: list[TransitionDecision] | None = None,
    loudness_targets: dict[str, float] | None = None,
) -> EDL:
    if not ordered_items:
        raise EDLBuildError("Cannot build an EDL from zero scenes.")

    transitions_by_from = {t.from_scene: t for t in (transitions or [])}
    loudness_targets = loudness_targets or {}

    clips: list[EDLClip] = []
    cursor = 0.0
    prev_duration = 0.0
    for i, item in enumerate(ordered_items):
        source_path = _ensure_has_audio(item, cache_dir)
        duration = max(0.0, item.duration)
        if duration <= 0:
            continue

        outgoing = transitions_by_from.get(item.id)
        transition_out = TransitionType.HARD_CUT
        audio_fade_out_ms = 0
        if outgoing is not None:
            transition_out = _TRANSITION_MAP.get(outgoing.type, TransitionType.HARD_CUT)
            if outgoing.applied and outgoing.type in _AUDIO_BRIDGE_KINDS:
                audio_fade_out_ms = int(outgoing.duration * 1000)

        incoming = transitions_by_from.get(ordered_items[i - 1].id) if i > 0 else None
        transition_in = TransitionType.HARD_CUT
        audio_fade_in_ms = 0
        transition_duration_s = 0.0
        if incoming is not None:
            transition_in = _TRANSITION_MAP.get(incoming.type, TransitionType.HARD_CUT)
            if incoming.applied and incoming.type in _VIDEO_CROSSFADE_KINDS:
                # A real, rendered crossfade -- the timeline overlaps by `d`
                # seconds, so both this clip's `timeline_in` and the running
                # cursor must account for it (spec Phase 2 Finalization
                # section 3: "do not introduce A/V desynchronization").
                transition_duration_s = clamp_transition_duration(incoming.duration, prev_duration, duration)
            elif incoming.applied and incoming.type in _AUDIO_BRIDGE_KINDS:
                audio_fade_in_ms = int(incoming.duration * 1000)

        timeline_in = cursor - transition_duration_s if transition_duration_s > 0 else cursor
        timeline_out = timeline_in + duration

        clips.append(
            EDLClip(
                source_file=source_path,
                source_in=0.0,
                source_out=duration,
                timeline_in=timeline_in,
                timeline_out=timeline_out,
                reason=CutReason.MANUAL,
                transition_in=transition_in,
                transition_out=transition_out,
                transition_duration_s=transition_duration_s,
                has_real_audio=item.has_audio,
                loudnorm_target_db=loudness_targets.get(item.id),
                audio_fade_in_ms=audio_fade_in_ms,
                audio_fade_out_ms=audio_fade_out_ms,
            )
        )
        cursor = timeline_out
        prev_duration = duration

    if not clips:
        raise EDLBuildError("All scenes had zero duration; nothing to assemble.")

    return EDL(version=1, fps=fps, width=width, height=height, clips=clips)
