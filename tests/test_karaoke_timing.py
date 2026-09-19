"""Word-highlight karaoke timing: every ASS karaoke tag consumes time, so a word may carry exactly
one duration tag (`\\kf`), and an event's karaoke total must equal its timed span -- never ~2x.
Synthetic transcripts only; nothing here depends on a brand, a language or a real timestamp."""
from __future__ import annotations

import re

from tests.conftest import make_word
from video_edit_agent.captions.chunking import CaptionChunk, chunk_words
from video_edit_agent.captions.engine import build_ass
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.captions.word_highlight import (
    build_karaoke_text,
    check_karaoke_event,
    karaoke_cs,
    word_timing_tags,
)
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript, Word

_STYLE = CaptionStyle(name="t", word_highlight=True, max_chars_per_line=26)
_TAG = re.compile(r"\\k[fo]?\d+|\\K\d+")


def _chunk(spec: list[tuple[str, float, float]]) -> CaptionChunk:
    words = [make_word(w, s, e) for w, s, e in spec]
    return CaptionChunk(words=words, start=words[0].start, end=words[-1].end)


def _ass_events(words: list[Word], style: CaptionStyle = _STYLE) -> list[tuple[str, str, str]]:
    seg = Segment(id="s0", start=words[0].start, end=words[-1].end, text=" ".join(w.word for w in words), words=words)
    end = words[-1].end + 0.5
    edl = EDL(clips=[EDLClip(source_file="x.mp4", source_in=0.0, source_out=end, timeline_in=0.0,
                             timeline_out=end, caption_refs=["s0"])])
    ass = build_ass(Transcript(language="ar", segments=[seg], provider="test"), edl, style)
    return [tuple(ln.split(",", 9)[1:3]) + (ln.split(",", 9)[9],) for ln in ass.splitlines() if ln.startswith("Dialogue:")]


def _ts(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _plain(text: str) -> list[str]:
    text = re.sub(r"\{[^}]*\}", "", text).replace("\\N", " ")
    return re.sub(r"[\u202a-\u202e\u2066-\u2069\u200e\u200f]", "", text).split()


def test_one_word_of_038s_consumes_38_cs_not_76():
    text = build_karaoke_text(_chunk([("ولا", 1.0, 1.38)]), _STYLE)
    assert text == "{\\kf38}ولا"
    assert karaoke_cs(text) == 38


def test_multi_word_event_total_equals_its_word_span():
    chunk = _chunk([("ولا", 0.25, 0.63), ("هي", 0.63, 1.21), ("كانت", 1.21, 1.59), ("عبارة", 1.59, 2.05)])
    text = build_karaoke_text(chunk, _STYLE)
    assert karaoke_cs(text) == round((chunk.end - chunk.start) * 100) == 180
    assert re.findall(r"\\kf(\d+)", text) == ["38", "58", "38", "46"]  # each word's own duration


def test_pauses_are_a_silent_gap_tag_so_words_stay_on_their_real_start():
    # 0.20 s of silence before the second word: the sweep must not begin early or drift later.
    chunk = _chunk([("alpha", 0.0, 0.30), ("beta", 0.50, 0.90)])
    text = build_karaoke_text(chunk, _STYLE)
    assert text == "{\\kf30}alpha {\\k20}{\\kf40}beta"
    assert karaoke_cs(text) == 90  # 30 + 20 + 40 == first start .. last end


def test_rounding_is_cumulative_and_never_accumulates_drift():
    # 0.333 s words: a per-word round() would gain/lose a centisecond per word.
    spec = [(f"w{i}", round(i * 0.333, 3), round((i + 1) * 0.333, 3)) for i in range(12)]
    chunk = _chunk(spec)
    text = build_karaoke_text(chunk, _STYLE)
    assert abs(karaoke_cs(text) - round((chunk.end - chunk.start) * 100)) <= 1


def test_no_word_carries_a_stacked_timing_pair():
    words = [make_word(w, round(i * 0.4, 2), round(i * 0.4 + 0.35, 2)) for i, w in enumerate(
        ["ما", "تكونش", "آراء", "شخصية", "أو", "وجهة", "نظر", "من", "الـbusiness", "owner"])]
    for start, end, text in _ass_events(words):
        assert not check_karaoke_event(text, _ts(start), _ts(end)), text
        for tags in word_timing_tags(text):
            sweeps = [t for t in tags if not re.fullmatch(r"\\k\d+", t)]
            assert len(sweeps) == 1, tags  # exactly one duration-bearing sweep per word
        assert not re.search(r"\\k\d+\\kf", text)  # the old `\k38\kf38` doubling


def test_checker_flags_the_old_doubling():
    old = "{\\k38\\kf38}ولا {\\k58\\kf58}هي"
    problems = check_karaoke_event(old, 0.0, 0.96)
    assert problems and any("karaoke total" in p for p in problems)
    assert karaoke_cs(old) == 192  # 2x the 96 cs of speech


def test_ass_event_totals_match_event_spans_for_the_whole_run():
    tokens = ["ولا", "هي", "كانت", "عبارة", "عن", "وجهة", "نظر", "من", "الـbusiness", "owner", "أو", "من", "الـbrand", "owner"]
    durs = [0.38, 0.58, 0.38, 0.46, 0.12, 0.40, 0.36, 0.18, 0.54, 0.32, 0.16, 0.16, 0.42, 0.36]
    words, t = [], 0.25
    for tok, d in zip(tokens, durs, strict=True):
        words.append(make_word(tok, round(t, 3), round(t + d, 3)))
        t += d
    events = _ass_events(words)
    assert len(events) >= 3
    for start, end, text in events:
        span = round(_ts(end) * 100) - round(_ts(start) * 100)
        assert abs(karaoke_cs(text) - span) <= 2, (start, end, text)  # never ~2x
        assert not check_karaoke_event(text, _ts(start), _ts(end)), text


def test_word_order_and_text_are_unchanged_for_rtl_and_code_switching():
    tokens = ["كانت", "عبارة", "عن", "وجهة", "نظر", "من", "الـbusiness", "owner", "أو", "من", "الـbrand", "owner"]
    words = [make_word(w, round(i * 0.4, 2), round(i * 0.4 + 0.4, 2)) for i, w in enumerate(tokens)]
    events = _ass_events(words)
    assert [w for _, _, text in events for w in _plain(text)] == tokens  # logical order, Latin runs in place
    for _, _, text in events:
        assert text.startswith("{\\rtl}")  # RTL paragraph handling is untouched by the karaoke tags


def test_semantic_units_stay_in_one_event_with_karaoke_tags():
    tokens = ["ولا", "هي", "كانت", "عبارة", "عن", "وجهة", "نظر", "من", "الـbusiness", "owner", "أو", "من", "الـbrand", "owner"]
    durs = [0.38, 0.58, 0.38, 0.46, 0.12, 0.40, 0.36, 0.18, 0.54, 0.32, 0.16, 0.16, 0.42, 0.36]
    words, t = [], 0.25
    for tok, d in zip(tokens, durs, strict=True):
        words.append(make_word(tok, round(t, 3), round(t + d, 3)))
        t += d
    joined = [" ".join(_plain(text)) for _, _, text in _ass_events(words)]
    for unit in ("وجهة نظر", "الـbusiness owner", "الـbrand owner"):
        assert any(unit in e for e in joined), (unit, joined)
    chunks = chunk_words(words, max_chars=26, max_duration=3.2)
    assert [c.text for c in chunks] == joined  # tagging never changes event boundaries


def test_style_colors_still_drive_the_karaoke_paint_not_the_tags():
    # Highlight/base colours come from the Style line (Primary = highlight, Secondary = base text);
    # the per-word tags carry timing only, so any brand palette works unchanged.
    style = CaptionStyle(name="t", word_highlight=True, primary_color="&H00112233", highlight_color="&H00445566")
    words = [make_word("alpha", 0.0, 0.4), make_word("beta", 0.4, 0.8)]
    seg = Segment(id="s0", start=0.0, end=0.8, text="alpha beta", words=words)
    edl = EDL(clips=[EDLClip(source_file="x.mp4", source_in=0.0, source_out=1.0, timeline_in=0.0,
                             timeline_out=1.0, caption_refs=["s0"])])
    ass = build_ass(Transcript(segments=[seg], provider="test"), edl, style)
    style_line = next(ln for ln in ass.splitlines() if ln.startswith("Style:"))
    fields = style_line.split(",")
    assert fields[3] == "&H00445566" and fields[4] == "&H00112233"  # primary=highlight, secondary=base
    body = "\n".join(ln for ln in ass.splitlines() if ln.startswith("Dialogue:"))
    assert not re.search(r"\\[1-4]c", body)  # no colour hardcoded in the events


def test_non_highlight_styles_emit_no_karaoke_tags():
    words = [make_word("alpha", 0.0, 0.4), make_word("beta", 0.4, 0.8)]
    style = CaptionStyle(name="t", word_highlight=False)
    for _, _, text in _ass_events(words, style):
        assert not _TAG.search(text)
