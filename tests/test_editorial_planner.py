"""Editorial pipeline tests: repetition / false-start detection -> EDL
(spec section 11/12) — word-boundary precise, re-renderable output."""
from __future__ import annotations

from video_edit_agent.core.schemas import CutReason, Segment, Transcript, Word
from video_edit_agent.editorial.false_starts import detect_fillers
from video_edit_agent.editorial.planner import build_edl
from video_edit_agent.editorial.repetition import find_repetition_groups, pick_best_take
from video_edit_agent.editorial.takes import analyze_takes


def _w(word: str, start: float, end: float) -> Word:
    return Word(word=word, start=start, end=end)


def test_repetition_group_detected_for_retaken_line():
    segments = [
        Segment(id="s0", start=0.0, end=2.0, text="this is my line"),
        Segment(id="s1", start=2.0, end=4.0, text="this is my line take two"),
        Segment(id="s2", start=4.0, end=6.0, text="a completely different sentence"),
    ]
    groups = find_repetition_groups(segments)
    assert len(groups) == 1
    assert set(groups[0].segment_ids) == {"s0", "s1"}


def test_pick_best_take_prefers_last_full_length_take():
    segments = [
        Segment(id="s0", start=0.0, end=2.0, text="this is my line"),
        Segment(id="s1", start=2.0, end=4.5, text="this is my line take two"),
    ]
    from video_edit_agent.editorial.repetition import RepetitionGroup

    group = RepetitionGroup(segment_ids=["s0", "s1"], normalized_text="this is my line")
    best = pick_best_take(group, {s.id: s for s in segments})
    assert best == "s1"


def test_analyze_takes_drops_superseded_repetition():
    segments = [
        Segment(id="s0", start=0.0, end=2.0, text="hello everyone welcome"),
        Segment(id="s1", start=2.0, end=4.0, text="hello everyone welcome again"),
    ]
    transcript = Transcript(provider="test", segments=segments)
    verdicts = analyze_takes(transcript)
    by_id = {v.segment_id: v for v in verdicts}
    assert by_id["s1"].keep is True
    assert by_id["s0"].keep is False
    assert "repetition" in by_id["s0"].reason


def test_detect_fillers_flags_arabic_and_english_fillers():
    words = [_w("يعني", 0.0, 0.3), _w("hello", 0.3, 0.6), _w("um", 0.6, 0.9), _w("world", 0.9, 1.2)]
    fillers = detect_fillers(words, "s0")
    flagged = {f.word for f in fillers}
    assert "يعني" in flagged
    assert "um" in flagged
    assert "hello" not in flagged


def test_build_edl_produces_word_boundary_clips(sample_video, sample_transcript):
    edl, verdicts = build_edl(sample_transcript, sample_video)
    assert len(edl.clips) == len(sample_transcript.segments)
    assert all(v.keep for v in verdicts)
    # Timeline must be contiguous starting at 0.
    assert edl.clips[0].timeline_in == 0.0
    for a, b in zip(edl.clips, edl.clips[1:]):
        assert b.timeline_in == a.timeline_out


def test_build_edl_drops_fully_superseded_repeated_segment(sample_video):
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=1.5, text="hello everyone welcome"),
        Segment(id="s1", start=1.5, end=3.0, text="hello everyone welcome again"),
    ])
    edl, verdicts = build_edl(transcript, sample_video)
    kept_ids = {c.caption_refs[0] for c in edl.clips}
    assert kept_ids == {"s1"}
    assert edl.clips[0].reason in (CutReason.MANUAL, CutReason.SILENCE)
