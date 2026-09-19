r"""The caption backing is a separate plate, never a per-word box.

libass draws a BorderStyle=3 box once per text run, and a karaoke line is one run
per `\kf` word, so translucent boxes drawn behind karaoke text overlap in their
padding and show as vertical tone bands. The plate is its own set of un-tagged
events under the text: one plate per visual line, derived by libass from the real
rendered glyphs, with the box colour, opacity and padding taken from the style
(the Brand Profile). These tests pin that structure on the generated ASS and, with
the real libass build, on rendered pixels.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions import engine
from video_edit_agent.captions.brand_style import resolve_brand_caption_style
from video_edit_agent.captions.engine import PLATE_STYLE, build_ass, uses_plate
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.captions.word_highlight import check_karaoke_event
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript, Word

_BIDI = re.compile("[\u202a-\u202e\u2066-\u2069\u200e\u200f]")
_BLOCK = re.compile(r"\{[^}]*\}")
_FONTS = Path(__file__).resolve().parents[1] / "brands" / "client" / "fonts"

_SEG16 = ["ولا", "هي", "كانت", "عبارة", "عن", "وجهة", "نظر", "من", "الـbusiness", "owner", "أو", "من", "الـbrand", "owner"]
_SEG16_DUR = [0.38, 0.58, 0.38, 0.46, 0.12, 0.40, 0.36, 0.18, 0.54, 0.32, 0.16, 0.16, 0.42, 0.36]


def _style(**kw) -> CaptionStyle:
    base = {
        "name": "t", "font_ar": "IBM Plex Sans Arabic", "font_size": 64, "word_highlight": True,
        "background": "brand_box", "outline": 8.0, "outline_color": "&H737C0014", "back_color": "&HFF000000",
        "primary_color": "&H00FFFFFF", "highlight_color": "&H0030D7BF", "max_chars_per_line": 26,
        "line_break_chars": 18,
    }
    base.update(kw)
    return CaptionStyle(**base)


def _timed(tokens: list[str], durations: list[float] | None = None, t0: float = 0.25) -> list[Word]:
    out, t = [], t0
    for i, tok in enumerate(tokens):
        d = durations[i] if durations else 0.4
        out.append(Word(word=tok, start=round(t, 3), end=round(t + d, 3), confidence=0.9))
        t += d
    return out


def _ass(words: list[Word], style: CaptionStyle, *, width: int = 1080, height: int = 1920) -> str:
    seg = Segment(id="s0", start=words[0].start, end=words[-1].end, text=" ".join(w.word for w in words), words=words)
    end = words[-1].end + 0.5
    edl = EDL(width=width, height=height, clips=[EDLClip(
        source_file="x.mp4", source_in=0.0, source_out=end, timeline_in=0.0, timeline_out=end, caption_refs=["s0"])])
    return build_ass(Transcript(language="ar", segments=[seg], provider="t"), edl, style)


def _events(ass: str, kind: str) -> list[list[str]]:
    """Dialogue fields of the plate ("Plate") or karaoke text ("Default") events."""
    style = PLATE_STYLE if kind == "plate" else "Default"
    rows = [ln.split(",", 9) for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    return [r for r in rows if r[3] == style]


def _plain(text: str) -> str:
    return " ".join(_BIDI.sub("", _BLOCK.sub("", text)).replace("\\N", " ").split())


def _lines(text: str) -> list[str]:
    return [" ".join(_BIDI.sub("", _BLOCK.sub("", part)).split()) for part in text.split("\\N")]


def _brand(**captions) -> Brand:
    return Brand.model_validate({
        "name": "acme", "colors": {"primary": "#0A3D62", "secondary": "#F1F2F6", "accent": "#E58E26"},
        "typography": {"arabic": "Acme Arabic Sans", "latin": "Acme Sans"}, "captions": captions,
    })


# --- 1. karaoke timing tags never create per-word background boxes ------------------------


def test_karaoke_text_style_carries_no_box_and_no_border():
    ass = _ass(_timed(_SEG16, _SEG16_DUR), _style())
    default = next(ln for ln in ass.splitlines() if ln.startswith("Style: Default,")).split(",")
    # BorderStyle, Outline, Shadow (fields 15-17 once "Style: " is part of the name)
    assert default[15] == "1" and float(default[16]) == 0.0 and float(default[17]) == 0.0


def test_karaoke_events_contain_no_background_or_drawing_overrides():
    events = _events(_ass(_timed(_SEG16, _SEG16_DUR), _style()), "text")
    assert events
    for f in events:
        assert not re.search(r"\\(bord|xbord|ybord|3c|3a|4c|4a|p\d|shad)", f[9]), f[9]
        assert f[9].count("\\kf") >= 2  # still a karaoke event, only the backing moved out of it


def test_plate_events_carry_no_karaoke_tags_and_no_timing():
    plates = _events(_ass(_timed(_SEG16, _SEG16_DUR), _style()), "plate")
    assert plates
    for f in plates:
        assert not re.search(r"\\k[fo]?\d|\\K\d", f[9]), f[9]
        # a plate is a single un-split run: no colour/alpha override blocks in the text either
        blocks = _BLOCK.findall(_BIDI.sub("", f[9]))  # one position/clip block, then only the direction mark
        assert blocks[0].startswith("{\\q2\\an2\\pos(") and set(blocks[1:]) <= {"{\\rtl}"}, f[9]


def test_no_plate_for_non_karaoke_or_outline_styles():
    words = _timed(["alpha", "beta", "gamma"])
    assert not uses_plate(_style(word_highlight=False))
    assert not uses_plate(_style(background="outline"))
    for style in (_style(word_highlight=False), _style(background="outline")):
        ass = _ass(words, style)
        assert not _events(ass, "plate")
        assert f"Style: {PLATE_STYLE}," not in ass
    # an un-tagged box event is one libass run: it keeps its own single box
    plain = next(ln for ln in _ass(words, _style(word_highlight=False)).splitlines() if ln.startswith("Style: Default,"))
    assert plain.split(",")[15] == "3"


# --- 2. one caption event -> one coherent plate per chosen (per-line) strategy ------------


def test_one_line_event_gets_exactly_one_unclipped_plate():
    words = _timed(["عن", "وجهة", "نظر"], [0.12, 0.40, 0.36])
    ass = _ass(words, _style())
    plates, texts = _events(ass, "plate"), _events(ass, "text")
    assert len(texts) == 1 and len(plates) == 1
    assert "\\clip" not in plates[0][9]
    assert plates[0][1:3] == texts[0][1:3]  # same on/off times as the karaoke event


def test_two_line_event_gets_one_plate_per_line_never_per_word():
    words = _timed(["ولا", "هي", "كانت", "عبارة", "عن", "وجهة"], [0.38, 0.58, 0.38, 0.46, 0.12, 0.40])
    ass = _ass(words, _style(line_break_chars=12))
    plates, texts = _events(ass, "plate"), _events(ass, "text")
    assert len(texts) == 1 and "\\N" in texts[0][9]
    assert len(plates) == 2  # two visual lines, six words
    assert [_plain(p[9]) for p in plates] == [" ".join(_lines(texts[0][9])[i].split()) for i in range(2)]
    assert all(p[1:3] == texts[0][1:3] for p in plates)


def test_line_plates_abut_on_one_row_so_they_can_not_overlap():
    words = _timed(["ولا", "هي", "كانت", "عبارة", "عن", "وجهة"])
    plates = _events(_ass(words, _style(line_break_chars=12)), "plate")
    clips = [tuple(int(v) for v in re.search(r"\\clip\((\d+),(\d+),(\d+),(\d+)\)", p[9]).groups()) for p in plates]
    (_, top1, _, bottom1), (_, top2, _, bottom2) = clips
    assert bottom1 == top2  # the band boundary: no overlap, no gap
    pad, pitch = 8, 64
    assert bottom2 - top1 == 2 * pitch + 2 * pad  # the whole block, padding only on the outer edges
    ys = [int(re.search(r"\\pos\((\d+),(\d+)\)", p[9]).group(2)) for p in plates]
    assert ys[1] - ys[0] == pitch  # the same line pitch libass gives the text lines


def test_plate_is_placed_like_the_text_inside_the_safe_zone():
    words = _timed(["عن", "وجهة", "نظر"])
    ass = _ass(words, _style())
    x, y = map(int, re.search(r"\\pos\((\d+),(\d+)\)", _events(ass, "plate")[0][9]).groups())
    margin_l = int(1080 * 0.06)
    assert x == (margin_l + 1080 - margin_l) // 2  # centred between the safe-zone side margins
    assert y == 1920 - int(1920 * 0.18)  # bottom margin of the safe zone, same anchor as the text
    plate_style = next(ln for ln in ass.splitlines() if ln.startswith(f"Style: {PLATE_STYLE},")).split(",")
    assert plate_style[18:22] == ["2", str(margin_l), str(margin_l), str(int(1920 * 0.18))]  # alignment + margins as the text


# --- 3. plate bounds include the complete rendered event text --------------------------------


def test_plate_text_is_the_complete_event_text_in_order():
    for tokens, dur, brk in (
        (_SEG16[:4], _SEG16_DUR[:4], 18), (_SEG16[7:10], _SEG16_DUR[7:10], 18), (_SEG16[10:], _SEG16_DUR[10:], 12),
        (_SEG16, _SEG16_DUR, 18),
    ):
        ass = _ass(_timed(tokens, dur), _style(line_break_chars=brk))
        texts, plates = _events(ass, "text"), _events(ass, "plate")
        by_time: dict[tuple[str, str], list[str]] = {}
        for p in plates:
            by_time.setdefault((p[1], p[2]), []).append(_plain(p[9]))
        for t in texts:
            assert " ".join(by_time[(t[1], t[2])]) == _plain(t[9]), (t[9], by_time)


def _render(ass: str, tmp_path: Path, name: str, width: int, height: int, at: float = 0.3):
    from PIL import Image

    fonts = tmp_path / "fonts"
    if not fonts.exists():
        shutil.copytree(_FONTS, fonts)
    (tmp_path / f"{name}.ass").write_text(ass, encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x3a2a8a:s={width}x{height}:d=3",
         "-vf", f"subtitles={name}.ass:fontsdir=fonts", "-ss", str(at), "-frames:v", "1", f"{name}.png"],
        cwd=tmp_path, capture_output=True, check=True, timeout=120,
    )
    return Image.open(tmp_path / f"{name}.png").convert("RGB")


_BG = (0x3A, 0x2A, 0x8A)


def _bbox(img, keep) -> tuple[int, int, int, int]:
    px, w, h = img.load(), *img.size
    pts = [(x, y) for y in range(h) for x in range(w) if keep(px[x, y])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _spread(img, box) -> int:
    """Largest per-channel tone difference across a rectangle."""
    px = img.load()
    chans = [[px[x, y][c] for y in range(box[1], box[3] + 1) for x in range(box[0], box[2] + 1)] for c in range(3)]
    return max(max(c) - min(c) for c in chans)


@requires_ffmpeg
@pytest.mark.skipif(not _FONTS.is_dir(), reason="brand font not available")
def test_plate_bounds_contain_every_rendered_glyph(tmp_path: Path):
    for name, tokens, dur, brk in (
        ("one", ["عن", "وجهة", "نظر"], [0.12, 0.40, 0.36], 18),
        ("mixed", ["من", "الـbusiness", "owner"], [0.18, 0.54, 0.32], 18),
        ("two", ["ولا", "هي", "كانت", "عبارة", "عن", "وجهة"], [0.38, 0.58, 0.38, 0.46, 0.12, 0.40], 12),
    ):
        ass = _ass(_timed(tokens, dur), _style(line_break_chars=brk))
        plate_only = "\n".join(ln for ln in ass.splitlines() if not ln.startswith("Dialogue: 1"))
        plate = _bbox(_render(plate_only, tmp_path, f"{name}_plate", 1080, 1920), lambda c: c != _BG)
        full = _render(ass, tmp_path, f"{name}_full", 1080, 1920)
        glyphs = _bbox(full, lambda c: sum(c) > 500)  # white / brand-highlight text is far brighter than any plate tone
        assert plate[0] < glyphs[0] and plate[1] < glyphs[1] and glyphs[2] < plate[2] and glyphs[3] < plate[3], (name, plate, glyphs)
        assert glyphs[0] - plate[0] >= 4 and plate[2] - glyphs[2] >= 4  # padding on the sides
        assert glyphs[1] - plate[1] >= 4 and plate[3] - glyphs[3] >= 4  # and above / below
        assert plate[2] - plate[0] < 1080 * 0.7  # compact: never a full-width strip


@requires_ffmpeg
@pytest.mark.skipif(not _FONTS.is_dir(), reason="brand font not available")
def test_rendered_plate_is_one_uniform_tone_with_no_word_bands(tmp_path: Path, monkeypatch):
    words = _timed(["من", "الـbusiness", "owner"], [0.18, 0.54, 0.32])
    style = _style()
    ass = _ass(words, style)
    plate_only = "\n".join(ln for ln in ass.splitlines() if not ln.startswith("Dialogue: 1"))
    img = _render(plate_only, tmp_path, "plate", 1080, 1920)
    x0, y0, x1, y1 = _bbox(img, lambda c: c != _BG)
    interior = (x0 + 3, y0 + 3, x1 - 3, y1 - 3)
    assert _spread(img, interior) <= 4  # a single tone across the whole plate

    # the old layout (box drawn by the karaoke text runs) fails the same measurement, on the
    # padding strip above the glyphs where only the backing is visible
    monkeypatch.setattr(engine, "uses_plate", lambda _s: False)
    old = _render(_ass(words, style), tmp_path, "old", 1080, 1920)
    ox0, oy0, ox1, _ = _bbox(old, lambda c: c != _BG)
    assert _spread(old, (ox0 + 3, oy0 + 2, ox1 - 3, oy0 + 5)) > 4, "the check no longer sees the old per-word bands"
    assert _spread(img, (x0 + 3, y0 + 2, x1 - 3, y0 + 5)) <= 4


@requires_ffmpeg
@pytest.mark.skipif(not _FONTS.is_dir(), reason="brand font not available")
def test_rendered_two_line_plate_has_no_overlap_stripe(tmp_path: Path):
    words = _timed(["ولا", "هي", "كانت", "عبارة", "عن", "وجهة"], [0.38, 0.58, 0.38, 0.46, 0.12, 0.40])
    ass = _ass(words, _style(line_break_chars=12))
    plate_only = "\n".join(ln for ln in ass.splitlines() if not ln.startswith("Dialogue: 1"))
    img = _render(plate_only, tmp_path, "two_plate", 1080, 1920)
    x0, y0, x1, y1 = _bbox(img, lambda c: c != _BG)
    assert (y1 - y0) >= 2 * 64 + 2 * 8 - 2  # the block is both lines tall
    # the columns both lines share (the middle) are one tone top to bottom, including across the line seam
    mid = (x0 + (x1 - x0) // 2 - 20, y0 + 3, x0 + (x1 - x0) // 2 + 20, y1 - 3)
    assert _spread(img, mid) <= 4


# --- 4. padding, opacity and colour come from the Brand Profile ------------------------------


def _plate_style(ass: str) -> list[str]:
    return next(ln for ln in ass.splitlines() if ln.startswith(f"Style: {PLATE_STYLE},")).split(",")


def test_plate_padding_opacity_and_colour_follow_the_brand_profile():
    words = _timed(["عن", "وجهة", "نظر"])
    light = resolve_brand_caption_style("word-highlight", _brand(box_opacity=0.3, box_padding=6)).style
    heavy = resolve_brand_caption_style("word-highlight", _brand(box_opacity=0.9, box_padding=20)).style
    a, b = _plate_style(_ass(words, light)), _plate_style(_ass(words, heavy))
    # Format: Name, Fontname, Fontsize, Primary, Secondary, Outline(colour), Back, ..., BorderStyle, Outline, Shadow
    assert float(a[16]) == 6 and float(b[16]) == 20  # padding = the box width
    assert int(a[5][2:4], 16) == round((1 - 0.3) * 255) and int(b[5][2:4], 16) == round((1 - 0.9) * 255)  # alpha
    assert a[5].endswith("623D0A") and b[5].endswith("623D0A")  # the brand's primary #0A3D62, as ASS BGR
    assert a[15] == "3"


def test_plate_uses_whatever_the_style_says_not_a_fixed_palette():
    words = _timed(["alpha", "beta"])
    one = _plate_style(_ass(words, _style(outline_color="&H40112233", outline=11.0)))
    two = _plate_style(_ass(words, _style(outline_color="&HC0AABBCC", outline=3.0)))
    assert one[5] == "&H40112233" and float(one[16]) == 11
    assert two[5] == "&HC0AABBCC" and float(two[16]) == 3


def test_generic_caption_modules_hardcode_no_brand_values():
    src = Path(engine.__file__).parent
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in src.glob("*.py"))
    for value in ("f1c40f", "0a3d62", "8e44ad", "ibm plex"):
        assert value not in text.replace("neutral", ""), value


def test_plate_clip_padding_follows_the_configured_padding():
    words = _timed(["ولا", "هي", "كانت", "عبارة", "عن", "وجهة"])
    for pad in (4.0, 8.0, 14.0):
        plates = _events(_ass(words, _style(line_break_chars=12, outline=pad)), "plate")
        clips = [tuple(int(v) for v in re.search(r"\\clip\((\d+),(\d+),(\d+),(\d+)\)", p[9]).groups()) for p in plates]
        assert clips[1][3] - clips[0][1] == 2 * 64 + 2 * int(pad)


# --- 5. RTL and Arabic/English code-switching are unchanged ----------------------------------


def test_plate_and_text_share_direction_handling():
    for tokens, rtl in ((["من", "الـbusiness", "owner"], True), (["alpha", "beta", "gamma"], False)):
        ass = _ass(_timed(tokens), _style())
        for kind in ("plate", "text"):
            body = _events(ass, kind)[0][9]
            assert ("{\\rtl}" in body) is rtl, (kind, body)
            assert ("\u202b" in body) is rtl and body.count("\u202b") == body.count("\u202c"), (kind, body)


def test_code_switched_words_stay_in_logical_order_in_both_layers():
    ass = _ass(_timed(["من", "الـbusiness", "owner"], [0.18, 0.54, 0.32]), _style())
    assert _plain(_events(ass, "plate")[0][9]) == "من الـbusiness owner"
    assert _plain(_events(ass, "text")[0][9]) == "من الـbusiness owner"
    assert "\u200f" not in _events(ass, "plate")[0][9]  # nothing invented into the text


def test_text_style_keeps_paragraph_direction_detection():
    ass = _ass(_timed(["من", "الـbusiness", "owner"]), _style())
    for prefix in ("Style: Default,", f"Style: {PLATE_STYLE},"):
        assert next(ln for ln in ass.splitlines() if ln.startswith(prefix)).endswith(",-1")  # Encoding -1


# --- 6. karaoke timing math is unchanged ------------------------------------------------------


def test_karaoke_tags_are_exactly_the_committed_single_kf_math():
    words = _timed(_SEG16, _SEG16_DUR)
    texts = _events(_ass(words, _style()), "text")
    flat = [int(m) for t in texts for m in re.findall(r"\\kf(\d+)", t[9])]
    assert flat == [round(d * 100) for d in _SEG16_DUR]  # one sweep per word, its real duration
    for t in texts:
        h, m, s = t[1].split(":")
        start = int(h) * 3600 + int(m) * 60 + float(s)
        h, m, s = t[2].split(":")
        end = int(h) * 3600 + int(m) * 60 + float(s)
        assert not check_karaoke_event(t[9], start, end), t[9]


def test_text_events_are_identical_with_or_without_the_plate():
    words = _timed(_SEG16, _SEG16_DUR)
    with_plate = _events(_ass(words, _style()), "text")
    without = [r for r in (ln.split(",", 9) for ln in _ass(words, _style(word_highlight=True, background="outline"))
                           .splitlines() if ln.startswith("Dialogue:"))]
    assert [(r[1], r[2], r[9]) for r in with_plate] == [(r[1], r[2], r[9]) for r in without]


# --- 7. semantic grouping is unchanged --------------------------------------------------------


def test_semantic_pairs_stay_together_in_text_and_plate_events():
    ass = _ass(_timed(_SEG16, _SEG16_DUR), _style())
    for kind in ("text", "plate"):
        joined = [_plain(f[9]) for f in _events(ass, kind)]
        for pair in ("وجهة نظر", "الـbusiness owner", "الـbrand owner"):
            assert any(pair in j for j in joined), (kind, pair, joined)
    # the event boundaries are the chunking's, plate or not
    plain_bg = _events(_ass(_timed(_SEG16, _SEG16_DUR), _style(background="outline")), "text")
    assert [(f[1], f[2], _plain(f[9])) for f in _events(ass, "text")] == [(f[1], f[2], _plain(f[9])) for f in plain_bg]
