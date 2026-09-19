"""Confirmed-corrections pass: timing-preserving corrections that change word
count, phrase-aware caption wrapping, lighter brand box, the end-card
decorative-motif policy, unresolved-transcript review state and the
micro-preview rendering path.

Everything uses synthetic brands/transcripts -- these rules are generic.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import make_transcript, make_word, requires_ffmpeg
from video_edit_agent.brand.logo_policy import plan_logo
from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.brand_style import resolve_brand_caption_style
from video_edit_agent.captions.chunking import CaptionChunk
from video_edit_agent.captions.engine import balanced_break_index, build_ass
from video_edit_agent.captions.phrasing import phrase_break_index, width_break_index
from video_edit_agent.captions.styles import CaptionStyle, hex_to_ass
from video_edit_agent.captions.word_highlight import build_karaoke_text
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.render import micro_preview as mp
from video_edit_agent.review import state as review_state
from video_edit_agent.review.corrections import apply_corrections
from video_edit_agent.review.schemas import TranscriptCorrection, UnresolvedTranscriptItem


def _brand(**overrides) -> Brand:
    data = {
        "name": "acme",
        "colors": {"primary": "#0A3D62", "secondary": "#F1F2F6", "accent": "#E58E26"},
        "typography": {"arabic": "Acme Arabic Sans", "latin": "Acme Sans"},
    }
    data.update(overrides)
    return Brand.model_validate(data)


def _words(texts: list[str], step: float = 0.3, gap_after: dict[int, float] | None = None):
    out, t = [], 0.0
    for i, text in enumerate(texts):
        out.append(make_word(text, t, t + step))
        t += step + (gap_after or {}).get(i, 0.0)
    return out


# --- corrections with a different word count keep unchanged timings ----------


def test_correction_keeps_unchanged_word_timings_and_segment_span():
    tr = make_transcript([("one two three four five", 0.0, 5.0)])
    seg = tr.segments[0]
    fixed = apply_corrections(
        tr, [TranscriptCorrection(segment_id=seg.id, corrected_text="one two THREE and a half four five")]
    ).segments[0]
    by_text = {w.word: w for w in fixed.words}
    for kept in ("one", "two", "five"):
        old = next(w for w in seg.words if w.word == kept)
        assert (by_text[kept].start, by_text[kept].end) == (old.start, old.end)
    assert (fixed.words[0].start, fixed.words[-1].end) == (seg.start, seg.end)
    starts = [w.start for w in fixed.words]
    assert starts == sorted(starts)
    assert all(w.end >= w.start for w in fixed.words)


def test_correction_replacing_words_with_fewer_words_spreads_only_the_replaced_span():
    tr = make_transcript([("keep a1 a2 a3 tail", 0.0, 5.0)])
    seg = tr.segments[0]
    fixed = apply_corrections(tr, [TranscriptCorrection(segment_id=seg.id, corrected_text="keep X tail")]).segments[0]
    assert [w.word for w in fixed.words] == ["keep", "X", "tail"]
    assert (fixed.words[0].start, fixed.words[0].end) == (seg.words[0].start, seg.words[0].end)
    assert (fixed.words[2].start, fixed.words[2].end) == (seg.words[4].start, seg.words[4].end)
    assert fixed.words[1].start == pytest.approx(seg.words[1].start)
    assert fixed.words[1].end == pytest.approx(seg.words[3].end)


def test_dropped_leading_word_hands_its_span_to_the_next_word():
    tr = make_transcript([("um hello world", 0.0, 3.0)])
    seg = tr.segments[0]
    fixed = apply_corrections(tr, [TranscriptCorrection(segment_id=seg.id, corrected_text="hello world")]).segments[0]
    assert (fixed.words[0].start, fixed.words[-1].end) == (seg.start, seg.end)


# --- phrase-aware wrapping ----------------------------------------------------


def _split(words, max_chars=18):
    i = phrase_break_index(words, max_chars)
    return None if i is None else (" ".join(w.word for w in words[:i]), " ".join(w.word for w in words[i:]))


def test_short_chunk_is_not_split():
    assert phrase_break_index(_words(["hello", "world"]), 18) is None


def test_never_ends_a_line_on_a_preposition_or_conjunction():
    words = _words(["جودة", "المنتج", "في", "كل", "مرحلة", "من", "مراحل", "التصنيع"])
    first, _second = _split(words, 20)
    assert first.split()[-1] not in {"في", "من", "على", "و", "أو"}
    assert width_break_index(words, 20) is not None


def test_keeps_a_code_switched_latin_phrase_together():
    words = _words(["ولا", "هي", "وجهة", "نظر", "من", "الـbusiness", "owner", "أو", "الـbrand", "owner"])
    first, second = _split(words, 28)  # two lines of <= ~37 chars must hold the 52-char phrase
    for phrase in ("الـbusiness owner", "الـbrand owner"):
        assert phrase in first or phrase in second


def test_prefers_a_real_pause_over_a_slightly_better_balance():
    texts = ["بببب", "تتتت", "ثثثث", "جج", "حححح", "خخ"]  # non-Latin (no Latin-run rule); widths 4,4,4,2,4,2
    assert phrase_break_index(_words(texts), 16) == 3  # balance alone: 14|10 beats 9|15
    words = _words(texts, gap_after={1: 0.6})
    assert phrase_break_index(words, 16) == 2  # a real pause after word two outweighs the balance gap


def test_prefers_punctuation():
    words = _words(["one", "two", "three,", "four", "five", "six", "seven"])
    assert phrase_break_index(words, 14) == 3


def test_falls_back_to_width_balance_when_no_phrase_safe_break_exists():
    # Every boundary sits between two bound Latin words.
    words = _words(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
    assert phrase_break_index(words, 12) == width_break_index(words, 12)


def test_engine_wrapper_is_phrase_aware():
    words = _words(["ما", "تكونش", "آراء", "شخصية", "من", "صاحب", "البراند", "نفسه"])
    i = balanced_break_index(words, 18)
    assert i is not None
    assert words[i - 1].word != "من"  # never strands the preposition


def test_wrapping_never_changes_timing_and_caps_at_two_lines():
    words = _words(["لازم", "تتأكد", "من", "القيمة", "الحقيقية", "في", "كل", "منتج", "أو", "service"])
    chunk = CaptionChunk(words=words, start=words[0].start, end=words[-1].end)
    style = CaptionStyle(name="t", word_highlight=True, line_break_chars=18)
    text = build_karaoke_text(chunk, style, break_before=balanced_break_index(words, 18))
    assert text.count("\\N") == 1  # a two-line cap
    durations = [int(m) for m in re.findall(r"\\k(\d+)", text)]
    assert durations == [round((w.end - w.start) * 100) for w in words]  # per-word timing untouched


# --- caption highlight timing path (ASS) --------------------------------------


def test_karaoke_durations_track_word_timing_in_the_ass_dialogue():
    tr = make_transcript([("alpha beta gamma delta", 1.0, 5.0)])
    seg = tr.segments[0]
    edl = EDL(clips=[EDLClip(
        source_file="x.mp4", source_in=0.0, source_out=6.0, timeline_in=0.0, timeline_out=6.0,
        caption_refs=[seg.id],
    )])
    style = CaptionStyle(name="t", word_highlight=True, primary_color=hex_to_ass("#FFFFFF"),
                         highlight_color=hex_to_ass("#00AA33"))
    ass = build_ass(tr, edl, style)
    fields = [ln.split(",", 9) for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert fields[0][1] == "0:00:01.00" and fields[-1][2] == "0:00:05.00"  # first word in / last word out
    ks = [int(m) for f in fields for m in re.findall(r"\\k(\d+)", f[9])]
    assert ks == [round((w.end - w.start) * 100) for w in seg.words]  # one sweep per word, real durations
    assert sum(f[9].count("\\kf") for f in fields) == 4  # sweep highlight per word


# --- brand box is light, compact and brand-configurable -----------------------


def test_brand_box_is_translucent_and_tight_by_default():
    style = resolve_brand_caption_style("word-highlight", _brand()).style
    assert style.background == "brand_box"
    alpha = int(style.outline_color[2:4], 16)
    assert alpha >= 0x60  # at most ~60% opaque, never a heavy solid strip
    assert style.outline <= 10


def test_brand_box_weight_comes_from_the_profile():
    heavy = _brand(captions={"box_opacity": 0.9, "box_padding": 20})
    style = resolve_brand_caption_style("word-highlight", heavy).style
    assert style.outline == 20
    assert int(style.outline_color[2:4], 16) == round((1.0 - 0.9) * 255)


# --- end-card decorative motif policy ------------------------------------------


def test_accent_color_alone_never_draws_a_motif(tmp_path: Path):
    brand = _brand()  # has an accent color, no explicit motif
    assert brand.colors.accent
    logo = tmp_path / "logo.png"
    _make_logo(logo)
    card = plan_logo(brand, logo, 1080, 1920, 30).end_card
    assert card.accent_style == "none" and card.accent is None


def test_explicit_motif_is_drawn_and_unknown_or_accentless_ones_are_ignored(tmp_path: Path):
    logo = tmp_path / "logo.png"
    _make_logo(logo)

    def card(brand):
        return plan_logo(brand, logo, 1080, 1920, 30).end_card

    explicit = card(_brand(logo={"behavior": {"accent_style": "underline"}}))
    assert (explicit.accent_style, explicit.accent) == ("underline", "#E58E26")
    assert card(_brand(logo={"behavior": {"accent_style": "sparkles"}})).accent_style == "none"
    no_accent = card(_brand(colors={"primary": "#0A3D62", "secondary": "#F1F2F6"},
                            logo={"behavior": {"accent_style": "underline"}}))
    assert no_accent.accent_style == "none"


def test_shipped_client_profile_defines_no_end_card_motif():
    root = Path(__file__).resolve().parent.parent / "brands" / "client" / "brand.yaml"
    if not root.exists():
        pytest.skip("client brand not present")
    import yaml

    behavior = (yaml.safe_load(root.read_text(encoding="utf-8")).get("logo") or {}).get("behavior") or {}
    assert behavior.get("accent_style", "none") == "none"


def _make_logo(path: Path) -> None:
    from PIL import Image

    Image.new("RGBA", (200, 80), (255, 255, 255, 255)).save(path)


# --- unresolved transcript items in review state --------------------------------


def test_unresolved_items_block_approval_and_survive_until_resolved(tmp_path: Path):
    rd = tmp_path / "review"
    review_state.flag_unresolved(rd, UnresolvedTranscriptItem(segment_id="s0", segment=1, reason="unclear"))
    state = review_state.load_review_state(rd)
    assert [i.segment_id for i in state.unresolved_transcript] == ["s0"]
    assert state.ready_for_final_render is False
    with pytest.raises(review_state.UnresolvedReviewItems):
        review_state.approve(rd)
    assert review_state.is_ready_for_final_render(rd) is False
    review_state.resolve_unresolved(rd, "s0")
    assert review_state.approve(rd).ready_for_final_render is True


# --- micro-preview rendering path -------------------------------------------------


def _edl(source: str) -> EDL:
    return EDL(clips=[
        EDLClip(source_file=source, source_in=0.0, source_out=3.0, timeline_in=0.0, timeline_out=3.0,
                zoom=1.08, caption_refs=["s0"]),
        EDLClip(source_file=source, source_in=3.0, source_out=6.0, timeline_in=3.0, timeline_out=6.0,
                zoom=1.14, caption_refs=["s1"]),
    ])


def test_slice_edl_rebases_to_zero_and_neutralizes_zoom():
    sub = mp.slice_edl(_edl("x.mp4"), 2.0, 4.0)
    assert [(c.source_in, c.source_out, c.timeline_in, c.timeline_out) for c in sub.clips] == [
        (2.0, 3.0, 0.0, 1.0), (3.0, 4.0, 1.0, 2.0)]
    assert {c.zoom for c in sub.clips} == {1.0}
    assert mp.slice_edl(_edl("x.mp4"), 2.0, 4.0, zoom=None).clips[1].zoom == 1.14
    with pytest.raises(mp.MicroPreviewError):
        mp.slice_edl(_edl("x.mp4"), 10.0, 12.0)


def test_caption_sync_window_is_4_to_6_seconds_around_a_chosen_segment():
    tr = make_transcript([
        ("a b c", 0.0, 2.0), ("one two three four five six", 2.0, 6.6), ("x y", 6.6, 20.0),
    ])
    window = mp.pick_caption_sync_window(tr, [tr.segments[0].id, tr.segments[1].id, tr.segments[2].id])
    assert window is not None
    assert window.segment_id == tr.segments[1].id  # most words that fit
    assert 4.0 <= window.end - window.start <= 6.0
    assert window.start <= tr.segments[1].start and window.end >= tr.segments[1].end
    assert mp.pick_caption_sync_window(tr, ["missing"]) is None


@requires_ffmpeg
def test_caption_sync_micro_preview_renders_real_audio_and_captions(sample_video: Path, tmp_path: Path):
    from video_edit_agent.core.media import probe

    info = probe(sample_video)
    tr = make_transcript([("alpha beta gamma delta epsilon", 0.0, min(info.duration, 2.0))])
    edl = EDL(clips=[EDLClip(source_file=str(sample_video), source_in=0.0, source_out=info.duration,
                             timeline_in=0.0, timeline_out=info.duration, caption_refs=[tr.segments[0].id])])
    style = CaptionStyle(name="t", word_highlight=True)
    out = mp.render_caption_sync(tr, edl, style, (0.0, min(info.duration, 2.0)), tmp_path / "cap.mp4")
    got = probe(out)
    assert got.has_audio and got.duration == pytest.approx(min(info.duration, 2.0), abs=0.25)
    assert (tmp_path / "_work" / "cap.ass").read_text(encoding="utf-8").count("\\kf") == 5


@requires_ffmpeg
def test_punch_in_micro_preview_renders_the_plan_with_audio(sample_video: Path, tmp_path: Path):
    from video_edit_agent.core.media import probe
    from video_edit_agent.editorial.punch_in import plan_punch_ins

    edl = EDL(width=360, height=640, fps=30.0, clips=[
        EDLClip(source_file=str(sample_video), source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0),
        EDLClip(source_file=str(sample_video), source_in=1.0, source_out=2.0, timeline_in=1.0, timeline_out=2.0),
    ])
    plan_punch_ins(edl)
    out = mp.render_punch_in(edl, (0.0, 2.0), tmp_path / "punch.mp4")
    got = probe(out)
    assert (got.width, got.height) == (360, 640)
    assert got.has_audio and got.duration == pytest.approx(2.0, abs=0.25)


@requires_ffmpeg
def test_overlay_preview_is_trimmed_to_the_footage_window(sample_video: Path, tmp_path: Path):
    """A motion overlay longer than the footage must not extend the preview
    (regression: the trim used a non-existent EDL attribute)."""
    import subprocess

    from video_edit_agent.core.media import probe
    from video_edit_agent.render.composition import Overlay

    long_overlay = tmp_path / "ov.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=6:r=30",
                    "-pix_fmt", "yuv420p", str(long_overlay)], check=True)
    edl = EDL(width=360, height=640, fps=30.0, clips=[
        EDLClip(source_file=str(sample_video), source_in=0.0, source_out=2.0, timeline_in=0.0, timeline_out=2.0)])
    out = mp.render_overlay_preview(edl, (0.0, 1.5), Overlay(path=long_overlay, start=0.0, end=1.5), tmp_path / "h.mp4")
    assert probe(out).duration == pytest.approx(1.5, abs=0.2)


@requires_ffmpeg
def test_end_card_micro_preview_is_only_the_card(tmp_path: Path):
    from video_edit_agent.core.media import probe
    from video_edit_agent.render.end_card import render_card_clip

    logo = tmp_path / "logo.png"
    _make_logo(logo)
    spec = plan_logo(_brand(), logo, 360, 640, 30).end_card
    out = render_card_clip(spec, tmp_path / "end_card.mp4", tmp_path / "w")
    got = probe(out)
    assert got.duration == pytest.approx(spec.duration, abs=0.2)
    assert spec.accent_style == "none"


def test_micro_previews_never_call_generators_or_the_full_renderer():
    """The module must not even import the B-roll generators, the motion
    router or the pipeline -- a preview can only use the shared render core."""
    src = Path(mp.__file__).read_text(encoding="utf-8")
    for forbidden in ("gemini", "veo", "motion.router", "core.pipeline", "run_pipeline", "remotion"):
        assert forbidden not in src.lower()


# --- code-switched chunks keep Arabic reading direction -----------------------


def test_code_switched_chunk_with_mostly_arabic_words_is_rtl_even_when_latin_holds_most_letters():
    from video_edit_agent.captions.rtl import rtl_override_tags

    assert rtl_override_tags("owner أو من الـbrand owner") == "{\\rtl}"
    assert rtl_override_tags("just some english words") == ""


def test_caption_sync_only_segments_filters_neighbour_captions(tmp_path: Path, monkeypatch):
    seen = {}

    def fake_render(plan, out, preset, crf=22):
        seen["refs"] = [c.caption_refs for c in plan.edl.clips]
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(mp, "render", fake_render)
    tr = make_transcript([("alpha beta", 0.0, 3.0), ("gamma delta", 3.0, 6.0)])
    edl = _edl("x.mp4")
    mp.render_caption_sync(tr, edl, CaptionStyle(name="t"), (1.0, 5.0), tmp_path / "c.mp4",
                           only_segments=["s1"])
    assert seen["refs"] == [[], ["s1"]]


def test_rtl_chunks_are_wrapped_in_an_rtl_embedding_and_latin_chunks_are_not():
    from video_edit_agent.captions.rtl import wrap_rtl

    rle, pdf, bs = chr(0x202B), chr(0x202C), chr(92)
    wrapped = wrap_rtl(f'{{{bs}k10{bs}kf10}}من{bs}N{{{bs}k10{bs}kf10}}الـbrand', 'من الـbrand')
    assert wrapped.startswith(f'{{{bs}rtl}}{rle}') and wrapped.endswith(pdf)
    # libass runs BiDi per hard line: the embedding is re-opened after every break
    assert wrapped.count(rle) == 2 and wrapped.count(pdf) == 2
    assert f'{pdf}{bs}N{rle}' in wrapped
    assert wrap_rtl('hello there', 'hello there') == 'hello there'


def test_ass_style_uses_auto_direction_encoding():
    # libass takes the BiDi base direction from the style Encoding; 1 forces LTR
    tr = make_transcript([('alpha beta', 0.0, 2.0)])
    edl = EDL(clips=[EDLClip(source_file='x.mp4', source_in=0.0, source_out=3.0, timeline_in=0.0,
                             timeline_out=3.0, caption_refs=[tr.segments[0].id])])
    ass = build_ass(tr, edl, CaptionStyle(name='t'))
    style_line = next(ln for ln in ass.splitlines() if ln.startswith('Style:'))
    assert style_line.rsplit(',', 1)[1] == '-1'


@requires_ffmpeg
def test_trim_to_cuts_an_overlong_preview_back_to_the_footage_window(tmp_path):
    from video_edit_agent.core.media import probe

    src = tmp_path / 'long.mp4'
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=160x90:d=5:r=30',
                    '-pix_fmt', 'yuv420p', str(src)], check=True)
    out = mp._trim_to(src, 2.0)
    assert out == src and not (tmp_path / 'long.trim.mp4').exists()
    assert probe(out).duration == pytest.approx(2.0, abs=0.15)
