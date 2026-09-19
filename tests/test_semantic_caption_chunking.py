"""Phrase-aware caption chunking: events must not cut a phrase in two, must not
strand a lone continuation word, and must leave every word's timing untouched.
Synthetic transcripts only -- the heuristics are language-level."""
from __future__ import annotations

from itertools import pairwise

from tests.conftest import make_word
from video_edit_agent.captions.chunking import MIN_WORDS_PER_EVENT, CaptionChunk, chunk_words
from video_edit_agent.captions.engine import build_ass
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript, Word


def _words(tokens: list[str], start: float = 0.0, step: float = 0.35, gaps: dict[int, float] | None = None) -> list[Word]:
    out, t = [], start
    for i, tok in enumerate(tokens):
        t += (gaps or {}).get(i, 0.0)
        out.append(make_word(tok, round(t, 3), round(t + step - 0.05, 3)))
        t += step
    return out


def _texts(chunks: list[CaptionChunk]) -> list[str]:
    return [c.text for c in chunks]


def _never_splits(chunks: list[CaptionChunk], first: str, second: str) -> bool:
    for a, b in pairwise(chunks):
        if a.words[-1].word == first and b.words[0].word == second:
            return False
    return True


def test_wugha_nazar_is_kept_together_at_a_budget_boundary():
    # The greedy budget alone would close the event right after "وجهة".
    tokens = ["ما", "تكونش", "آراء", "شخصية", "أو", "وجهة", "نظر", "شخصية", "من", "صاحب"]
    words = _words(tokens)
    chunks = chunk_words(words, max_chars=24)
    assert _never_splits(chunks, "وجهة", "نظر")
    assert any("وجهة نظر" in t for t in _texts(chunks))


def test_wugha_nazar_stays_whole_for_every_budget():
    tokens = ["كانت", "عبارة", "عن", "وجهة", "نظر", "من", "صاحب", "البراند", "نفسه", "وده", "مهم"]
    for budget in range(14, 40):
        chunks = chunk_words(_words(tokens), max_chars=budget)
        assert _never_splits(chunks, "وجهة", "نظر"), budget


def test_arabic_english_code_switching_is_not_split():
    tokens = ["لازم", "تتأكد", "من", "الـ", "business", "owner", "أو", "من", "الـ", "brand", "owner", "وده", "مهم"]
    for budget in range(20, 34):  # tighter than one phrase unit + its article is not a real budget
        chunks = chunk_words(_words(tokens), max_chars=budget)
        assert _never_splits(chunks, "business", "owner"), budget
        assert _never_splits(chunks, "brand", "owner"), budget
        assert _never_splits(chunks, "الـ", "business"), budget
        assert _never_splits(chunks, "الـ", "brand"), budget


def test_events_do_not_end_on_a_function_word():
    tokens = ["فاهم", "من", "البزنس", "فاليوز", "اللي", "تتقلق", "ديها", "تحاجات", "حقيقة", "سواء", "في", "المنطقة", "وفي", "الخدمة"]
    chunks = chunk_words(_words(tokens), max_chars=22)
    for c in chunks[:-1]:
        assert c.words[-1].word not in {"من", "في", "اللي", "وفي", "على", "و"}, _texts(chunks)


def test_short_phrase_near_an_event_boundary_is_not_orphaned():
    # "نظر" is the last word before the end of the run: it must not sit alone.
    tokens = ["دي", "مجرد", "وجهة", "نظر"]
    chunks = chunk_words(_words(tokens), max_chars=14)
    assert all(len(c.words) >= 2 for c in chunks), _texts(chunks)
    assert _never_splits(chunks, "وجهة", "نظر")


def test_lone_trailing_word_is_folded_into_the_previous_event():
    tokens = ["اللي", "إنت", "عايز", "تعمله", "الماركتنج"]
    chunks = chunk_words(_words(tokens), max_chars=18)
    assert min(len(c.words) for c in chunks) >= MIN_WORDS_PER_EVENT, _texts(chunks)


def test_a_real_pause_still_allows_a_break_between_bound_words():
    # Nobody says "في" then pauses for a second and continues the phrase.
    tokens = ["واحد", "اتنين", "تلاتة", "في", "أربعة", "خمسة", "ستة"]
    words = _words(tokens, gaps={4: 1.0})
    chunks = chunk_words(words, max_chars=20)
    assert any(c.words[-1].word == "في" for c in chunks[:-1])


def test_sentence_punctuation_still_closes_an_event():
    words = _words(["أول", "جملة.", "تاني", "جملة", "هنا"])
    chunks = chunk_words(words, max_chars=40)
    assert _texts(chunks)[0] == "أول جملة."


def test_word_timings_and_order_are_preserved_exactly():
    tokens = ["كانت", "عبارة", "عن", "وجهة", "نظر", "من", "الـ", "business", "owner", "أو", "من", "الـ", "brand", "owner"]
    words = _words(tokens)
    chunks = chunk_words(words, max_chars=22)
    flat = [w for c in chunks for w in c.words]
    assert [(w.word, w.start, w.end) for w in flat] == [(w.word, w.start, w.end) for w in words]
    for c in chunks:
        assert (c.start, c.end) == (c.words[0].start, c.words[-1].end)


def test_density_stays_reasonable_for_9x16():
    tokens = ["كلمة"] * 3 + ["وجهة", "نظر"] * 10
    chunks = chunk_words(_words(tokens), max_chars=26)
    for c in chunks:
        assert len(c.text) <= int(26 * 1.3), c.text
        assert (c.end - c.start) <= 3.2 * 1.25


def test_ass_events_carry_the_same_phrase_units():
    tokens = ["ما", "تكونش", "آراء", "شخصية", "أو", "وجهة", "نظر", "شخصية", "من", "صاحب", "البراند"]
    words = _words(tokens)
    seg = Segment(id="s1", start=words[0].start, end=words[-1].end, text=" ".join(tokens), words=words)
    transcript = Transcript(language="ar", segments=[seg], provider="test")
    edl = EDL(clips=[EDLClip(source_file="x.mp4", source_in=0.0, source_out=words[-1].end + 0.5,
                             timeline_in=0.0, timeline_out=words[-1].end + 0.5, caption_refs=["s1"])])
    ass = build_ass(transcript, edl, CaptionStyle(name="t", word_highlight=True, max_chars_per_line=24))
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert events
    joined = "\n".join(events)
    assert "وجهة" in joined and "نظر" in joined
    # each event that contains "وجهة" also contains "نظر" (the pair is one unit)
    for ev in events:
        assert ("وجهة" in ev) == ("نظر" in ev), ev


def test_line_wrap_keeps_a_code_switched_phrase_on_one_slightly_long_line():
    from video_edit_agent.captions.phrasing import phrase_break_index
    words = _words(["من", "الـbusiness", "owner"])  # 20 chars, no phrase-safe break exists
    assert phrase_break_index(words, 18) is None
    # far too long for one line: still wrapped somehow
    long_run = _words(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    assert phrase_break_index(long_run, 12) is not None
