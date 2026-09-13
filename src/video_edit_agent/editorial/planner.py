"""Editorial strategy -> EDL (spec sections 11, 12).

Combines take verdicts (repetition/false-start cuts) with silence trimming to
build a re-renderable EDL. Word-boundary precision: cuts always land on a
word boundary from the transcript, never mid-word.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import CutReason, EDL, EDLClip, Segment, Transcript, TransitionType
from video_edit_agent.editorial.silence import SilenceSpan, detect_silence
from video_edit_agent.editorial.takes import TakeVerdict, analyze_takes


def _trim_leading_trailing_silence(seg: Segment, silences: list[SilenceSpan], pad: float = 0.05) -> tuple[float, float]:
    start, end = seg.start, seg.end
    for s in silences:
        # trim a silence that overlaps the very start of the segment
        if s.start <= start <= s.end and s.end < end:
            start = max(start, s.end - pad)
        # trim a silence that overlaps the very end of the segment
        if s.start < end <= s.end and s.start > start:
            end = min(end, s.start + pad)
    return start, end


def build_edl(
    transcript: Transcript,
    source_file: Path,
    *,
    crossfade_ms: int = 30,
    width: int = 1080,
    height: int = 1920,
    fps: float = 30.0,
    audio_path: Path | None = None,
) -> tuple[EDL, list[TakeVerdict]]:
    verdicts = analyze_takes(transcript)
    keep_ids = {v.segment_id for v in verdicts if v.keep}

    silences: list[SilenceSpan] = []
    if audio_path is not None and audio_path.exists():
        try:
            silences = detect_silence(audio_path)
        except Exception:  # noqa: BLE001 - silence detection is a nice-to-have, never fatal
            silences = []

    clips: list[EDLClip] = []
    timeline_cursor = 0.0
    fade_s = crossfade_ms / 1000.0

    kept_segments = [s for s in transcript.segments if s.id in keep_ids and s.text.strip()]
    for i, seg in enumerate(kept_segments):
        src_in, src_out = _trim_leading_trailing_silence(seg, silences)
        if src_out <= src_in:
            continue
        duration = src_out - src_in
        clip = EDLClip(
            source_file=str(source_file),
            source_in=src_in,
            source_out=src_out,
            timeline_in=timeline_cursor,
            timeline_out=timeline_cursor + duration,
            speaker=seg.speaker,
            reason=CutReason.SILENCE if (src_in, src_out) != (seg.start, seg.end) else CutReason.MANUAL,
            transition_in=TransitionType.CROSSFADE if i > 0 else TransitionType.HARD_CUT,
            transition_out=TransitionType.HARD_CUT,
            audio_fade_in_ms=crossfade_ms if i > 0 else 0,
            audio_fade_out_ms=crossfade_ms if i < len(kept_segments) - 1 else 0,
            caption_refs=[seg.id],
        )
        clips.append(clip)
        timeline_cursor = clip.timeline_out

    return EDL(clips=clips, width=width, height=height, fps=fps), verdicts
