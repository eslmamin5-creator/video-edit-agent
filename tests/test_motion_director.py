"""Baseline Recovery Milestone item 7: behind-subject treatment must actually
be SELECTED by the motion director, not just correctly handled downstream.

`core/pipeline.py`'s behind-subject compositing logic (graphic drawn first,
subject cutout layered on top, real mediapipe-backed mask generation) already
existed and is exercised by `tests/test_behind_subject.py` -- but it was
never reached in practice because `build_motion_plan()` never set
`AnimationSpec.behind_subject = True` on anything it built. A `stat_counter`
is the one motion-graphic kind that reads as a background element the
speaker is meant to stand in front of (a documentary/social-video stat
overlay), unlike `lower_third`/`cta`, which are clean foreground chyron/UI
elements."""
from __future__ import annotations

from video_edit_agent.core.schemas import EDL, AnimationKind, EDLClip, Segment, Transcript, Word
from video_edit_agent.motion.director import build_motion_plan


def _transcript(words: list[tuple[str, float, float]]) -> Transcript:
    word_objs = [Word(word=w, start=s, end=e) for w, s, e in words]
    segment = Segment(
        id="seg0", start=word_objs[0].start, end=word_objs[-1].end,
        text=" ".join(w.word for w in word_objs), words=word_objs,
    )
    return Transcript(provider="test", language="ar", segments=[segment])


def test_stat_counter_specs_are_marked_behind_subject():
    edl = EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[
            EDLClip(source_file="a.mp4", source_in=0.0, source_out=5.0, timeline_in=0.0, timeline_out=5.0),
        ],
    )
    transcript = _transcript([("increased", 0.5, 1.0), ("50%", 1.0, 1.5), ("this", 1.5, 2.0), ("year", 2.0, 2.5)])

    specs = build_motion_plan(edl, transcript)
    stat_specs = [s for s in specs if s.kind == AnimationKind.STAT_COUNTER]

    assert stat_specs, "expected at least one stat_counter slot for a spoken number"
    assert all(s.behind_subject is True for s in stat_specs)


def test_lower_third_and_cta_specs_stay_foreground():
    edl = EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[
            EDLClip(
                source_file="a.mp4", source_in=0.0, source_out=5.0, timeline_in=0.0, timeline_out=5.0, speaker="Host",
            ),
            EDLClip(
                source_file="b.mp4", source_in=0.0, source_out=5.0, timeline_in=5.0, timeline_out=10.0, speaker="Guest",
            ),
        ],
    )
    transcript = _transcript([("hello", 0.5, 1.0), ("world", 1.0, 1.5)])

    specs = build_motion_plan(edl, transcript)
    non_stat_specs = [s for s in specs if s.kind != AnimationKind.STAT_COUNTER]

    assert non_stat_specs, "expected hook/lower-third/cta slots to be proposed"
    assert all(s.behind_subject is False for s in non_stat_specs)
