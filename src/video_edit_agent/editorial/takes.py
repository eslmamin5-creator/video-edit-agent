"""Take analysis: combine repetition groups + false-start signals into a
per-segment editorial verdict (spec section 11)."""
from __future__ import annotations

from dataclasses import dataclass, field

from video_edit_agent.core.schemas import Segment, Transcript
from video_edit_agent.editorial.false_starts import FalseStart, analyze_segment
from video_edit_agent.editorial.repetition import find_repetition_groups, pick_best_take


@dataclass
class TakeVerdict:
    segment_id: str
    keep: bool
    reason: str
    false_starts: list[FalseStart] = field(default_factory=list)


def analyze_takes(transcript: Transcript) -> list[TakeVerdict]:
    segments_by_id = {s.id: s for s in transcript.segments}
    verdicts: dict[str, TakeVerdict] = {
        s.id: TakeVerdict(segment_id=s.id, keep=True, reason="kept") for s in transcript.segments
    }

    for group in find_repetition_groups(transcript.segments):
        best_id = pick_best_take(group, segments_by_id)
        for sid in group.segment_ids:
            if sid != best_id:
                verdicts[sid] = TakeVerdict(segment_id=sid, keep=False, reason=f"repetition: superseded by {best_id}")

    for seg in transcript.segments:
        if not verdicts[seg.id].keep:
            continue
        fs = analyze_segment(seg)
        verdicts[seg.id].false_starts = fs
        # A segment that is *almost entirely* fillers/self-correction with no
        # other content is a false start and should be dropped outright.
        if fs and len(" ".join(f.word for f in fs)) > 0.7 * len(seg.text):
            verdicts[seg.id] = TakeVerdict(segment_id=seg.id, keep=False, reason="false_start", false_starts=fs)

    return list(verdicts.values())
