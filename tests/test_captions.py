"""Caption timing / RTL / word-highlight tests (spec section 15)."""
from __future__ import annotations

from video_edit_agent.captions.engine import _remap_words_to_timeline, build_ass, build_srt
from video_edit_agent.captions.styles import PRESETS, resolve_style
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript


def _edl_for(transcript: Transcript, *, timeline_in: float = 0.0) -> EDL:
    seg = transcript.segments[0]
    offset = timeline_in - seg.start
    return EDL(clips=[
        EDLClip(
            source_file="dummy.mp4",
            source_in=seg.start, source_out=seg.end,
            timeline_in=timeline_in, timeline_out=seg.end + offset,
            caption_refs=[seg.id],
        )
    ])


def test_words_remap_to_timeline_when_clip_starts_later_than_source(sample_transcript):
    edl = _edl_for(sample_transcript.model_copy(update={"segments": sample_transcript.segments[:1]}), timeline_in=10.0)
    remapped = _remap_words_to_timeline(sample_transcript, edl)
    assert remapped, "expected at least one remapped word"
    # Words from segment s0 (0.0-1.5s in source) should land at 10.0-11.5s on the timeline.
    assert all(w.start >= 10.0 - 1e-6 for w in remapped)
    assert all(w.end <= 11.5 + 1e-6 for w in remapped)


def test_words_outside_clip_source_range_are_dropped():
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=2.0, text="hello world", words=[
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 5.0, "end": 5.5},  # outside the clip's source range
        ]),
    ])
    edl = EDL(clips=[
        EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"]),
    ])
    remapped = _remap_words_to_timeline(transcript, edl)
    assert [w.word for w in remapped] == ["hello"]


def test_build_ass_contains_rtl_override_for_arabic_text():
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=1.0, text="مرحبا", words=[
            {"word": "مرحبا", "start": 0.0, "end": 1.0},
        ]),
    ])
    edl = EDL(clips=[
        EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"]),
    ])
    ass = build_ass(transcript, edl, PRESETS["minimal"])
    assert "\\rtl" in ass
    assert "[Events]" in ass


def test_build_ass_word_highlight_style_emits_karaoke_tags():
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=1.0, text="hello world", words=[
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]),
    ])
    edl = EDL(clips=[
        EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"]),
    ])
    ass = build_ass(transcript, edl, PRESETS["word-highlight"])
    assert "\\k" in ass and "\\kf" in ass


def test_build_ass_minimal_style_has_no_karaoke_tags():
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=1.0, text="hello world", words=[
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]),
    ])
    edl = EDL(clips=[
        EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"]),
    ])
    ass = build_ass(transcript, edl, PRESETS["minimal"])
    assert "\\k" not in ass


def test_build_srt_produces_sequential_numbered_cues():
    transcript = Transcript(provider="test", segments=[
        Segment(id="s0", start=0.0, end=1.0, text="hello world there", words=[
            {"word": "hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.3, "end": 0.6},
            {"word": "there", "start": 0.6, "end": 1.0},
        ]),
    ])
    edl = EDL(clips=[
        EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"]),
    ])
    srt = build_srt(transcript, edl)
    assert srt.startswith("1\n")
    assert "-->" in srt


def test_resolve_style_applies_brand_overrides_without_mutating_preset():
    original_size = PRESETS["minimal"].font_size
    styled = resolve_style("minimal", {"font_size": 99})
    assert styled.font_size == 99
    assert PRESETS["minimal"].font_size == original_size  # preset dict itself untouched
