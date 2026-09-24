"""Baseline Recovery Milestone item 3: focused regression tests proving the
caption engine no longer pre-shapes/pre-reorders Arabic text before handing
it to libass.

Background: this project's ffmpeg/libass build is compiled with HarfBuzz +
FriBidi (see `captions/rtl.py`), so libass performs its own complex-script
shaping and BiDi reordering given plain logical-order Unicode text. The
caption engine used to also run `arabic_reshaper.reshape()` +
`bidi.algorithm.get_display()` before handing text to libass, which
double-processes it -- confirmed via an empirical rendered-frame comparison
(see test_caption_shaping_visual_regression.py) to produce garbled,
disjointed glyphs. These tests assert the fix at the text level: caption
output must always contain the exact, untouched logical-order source words.
"""
from __future__ import annotations

from video_edit_agent.captions.chunking import CaptionChunk
from video_edit_agent.captions.engine import build_ass, build_srt
from video_edit_agent.captions.rtl import rtl_override_tags
from video_edit_agent.captions.styles import PRESETS
from video_edit_agent.captions.word_highlight import build_karaoke_text
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript, Word

ARABIC_WORDS = ["مرحبا", "بكم", "في", "هذا", "الفيديو"]


def _chunk(words: list[str]) -> CaptionChunk:
    ws = [Word(word=w, start=i * 0.4, end=(i + 1) * 0.4, confidence=0.9) for i, w in enumerate(words)]
    return CaptionChunk(words=ws, start=ws[0].start, end=ws[-1].end)


def test_build_karaoke_text_keeps_raw_logical_order_arabic_words():
    chunk = _chunk(ARABIC_WORDS)
    text = build_karaoke_text(chunk, PRESETS["word-highlight"])
    # Every original word must appear byte-for-byte -- reshaping replaces
    # each letter with a presentation-form codepoint, so a reshaped word
    # would NOT match the original substring.
    for w in ARABIC_WORDS:
        assert w in text, f"expected raw word {w!r} in karaoke text: {text!r}"
    # Words must still be in logical (original) order, not BiDi-reversed.
    positions = [text.index(w) for w in ARABIC_WORDS]
    assert positions == sorted(positions)


def test_build_karaoke_text_does_not_reorder_or_reshape_mixed_language():
    words = ["عايز", "أعمل", "launch", "بكرة"]
    chunk = _chunk(words)
    text = build_karaoke_text(chunk, PRESETS["word-highlight"])
    for w in words:
        assert w in text
    positions = [text.index(w) for w in words]
    assert positions == sorted(positions)


def _edl_and_transcript_for(text: str, words: list[str]) -> tuple[Transcript, EDL]:
    step = 1.0 / max(1, len(words))
    seg_words = [{"word": w, "start": i * step, "end": (i + 1) * step} for i, w in enumerate(words)]
    transcript = Transcript(provider="test", segments=[Segment(id="s0", start=0.0, end=1.0, text=text, words=seg_words)])
    edl = EDL(clips=[
        EDLClip(source_file="dummy.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0, caption_refs=["s0"]),
    ])
    return transcript, edl


def test_build_ass_non_karaoke_path_keeps_raw_logical_text():
    text = " ".join(ARABIC_WORDS)
    transcript, edl = _edl_and_transcript_for(text, ARABIC_WORDS)
    ass = build_ass(transcript, edl, PRESETS["minimal"])
    for w in ARABIC_WORDS:
        assert w in ass, f"expected raw word {w!r} in ASS output"
    assert "\\rtl" in ass  # RTL paragraph direction still forced for pure-Arabic lines


def test_build_srt_keeps_raw_logical_text():
    text = " ".join(ARABIC_WORDS)
    transcript, edl = _edl_and_transcript_for(text, ARABIC_WORDS)
    srt = build_srt(transcript, edl)
    for w in ARABIC_WORDS:
        assert w in srt


def test_rtl_override_tag_present_for_pure_arabic_absent_for_english():
    assert rtl_override_tags("مرحبا بكم") == "{\\rtl}"
    assert rtl_override_tags("hello there") == ""
