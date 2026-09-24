"""Arabic-capable caption engine (spec section 15).

Produces an ASS file (for burn-in via the `subtitles` ffmpeg filter, with
full RTL/shaping/word-highlight support) and a plain SRT (spec section 14's
`master.srt`, for editors/platforms that want a separate subtitle track).

Caption timing is computed on the TIMELINE (post-cut) axis, not the original
recording axis: each EDL clip carries `caption_refs` pointing back at the
transcript segment it came from, and word times are remapped by the clip's
source->timeline offset.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.captions.chunking import CaptionChunk, chunk_words
from video_edit_agent.captions.phrasing import phrase_break_index
from video_edit_agent.captions.rtl import wrap_rtl
from video_edit_agent.captions.safe_zone import SafeZone, margins_px
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.captions.word_highlight import build_karaoke_text
from video_edit_agent.core.schemas import EDL, Transcript, Word


def _remap_words_to_timeline(transcript: Transcript, edl: EDL) -> list[Word]:
    segments_by_id = {s.id: s for s in transcript.segments}
    remapped: list[Word] = []
    for clip in edl.clips:
        for ref in clip.caption_refs:
            seg = segments_by_id.get(ref)
            if seg is None:
                continue
            for w in seg.words:
                if w.end < clip.source_in or w.start > clip.source_out:
                    continue
                offset = clip.timeline_in - clip.source_in
                start = max(clip.timeline_in, w.start + offset)
                end = min(clip.timeline_out, w.end + offset)
                if end > start:
                    remapped.append(Word(word=w.word, start=start, end=end, confidence=w.confidence))
    remapped.sort(key=lambda w: w.start)
    return remapped


def _fmt_ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = round((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# Style `Encoding` is -1 (libass: detect the paragraph direction from the text):
# libass takes the BiDi base direction from it, and the usual 1 forces
# left-to-right, which lays a code-switched Arabic line out in the wrong order
# as soon as the line carries karaoke override tags.
ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},{highlight},{outline_color},{back},{bold},0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,{margin_l},{margin_r},{margin_v},-1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


_BOX_BACKGROUNDS = ("box", "brand_box")

# Background plate (see `plate_lines`). A karaoke line is several libass text
# runs (one per `\kf` word), and libass draws BorderStyle=3 (opaque box) per run,
# so neighbouring translucent boxes overlap in their padding and stack into
# visible vertical tone bands. The plate is therefore its own set of events on
# a lower layer, made of un-tagged text (one run per line -> one box), and the
# karaoke text above it carries no box at all.
_CLIP_RIGHT = 100000  # wider than any canvas: the clip only bounds the rows
PLATE_STYLE = "Plate"
_PLATE_LAYER, _TEXT_LAYER = 0, 1
_TRANSPARENT = "&HFF000000"


def uses_plate(style: CaptionStyle) -> bool:
    """True when the caption backing must be drawn as a separate plate: a box
    behind karaoke-tagged text (an un-tagged event is one run, so its own box
    is already uniform)."""
    return style.word_highlight and style.background in _BOX_BACKGROUNDS


def _plate_style_line(style: CaptionStyle, margins: dict[str, int]) -> str:
    """The plate's ASS style: same font, size and layout as the text style (so
    libass gives it the text's exact bounds), glyphs fully transparent, and the
    opaque-box border in the caption box colour with the box padding."""
    return (
        f"Style: {PLATE_STYLE},{style.font_ar},{style.font_size},{_TRANSPARENT},{_TRANSPARENT},"
        f"{style.outline_color},{_TRANSPARENT},{-1 if style.bold else 0},0,0,0,100,100,0,0,3,{style.outline},0,2,"
        f"{margins['left']},{margins['right']},{margins['bottom']},-1"
    )


def plate_lines(
    line_texts: list[str],
    chunk_text: str,
    style: CaptionStyle,
    *,
    width: int,
    height: int,
    margins: dict[str, int],
) -> list[str]:
    """Event texts of the plate: one per visual line.

    Each line is its own single-line, un-tagged, no-wrap event, so libass sizes
    its box from the real rendered glyph run (any script, any bidi mix) and no
    box can ever be split per word. Lines are placed with an explicit position
    (libass would otherwise shift same-layer events apart to avoid a collision).
    Adjacent lines' boxes would overlap by the padding and darken where they do,
    so each line is clipped to its own band. That is exact because libass lays
    lines out `font_size` apart (ascent + descent is normalised to the font
    size) up from the bottom margin (bottom-centre alignment). Bands meet on an
    integer row, so the plate is one uniform shape and only its outer top and
    bottom edges carry the padding."""
    n = len(line_texts)
    pad = round(style.outline)
    pitch = round(style.font_size)
    x = (margins["left"] + width - margins["right"]) // 2
    out = []
    for i, line in enumerate(line_texts):
        j = n - 1 - i  # position counted from the bottom line
        bottom = height - margins["bottom"] - j * pitch
        tags = f"\\q2\\an2\\pos({x},{bottom})"
        if n > 1:
            top = bottom - pitch - (pad if i == 0 else 0)
            tags += f"\\clip(0,{top},{_CLIP_RIGHT},{bottom + (pad if j == 0 else 0)})"
        out.append(f"{{{tags}}}{wrap_rtl(line, chunk_text)}")
    return out


def balanced_break_index(words: list, max_line_chars: int) -> int | None:
    """Index of the word that should start the second line when a chunk is
    longer than `max_line_chars`, or None when no split is needed/possible.
    Phrase-aware (see `captions/phrasing.py`): it avoids splitting tightly
    connected phrases and falls back to a width-balanced split otherwise."""
    return phrase_break_index(words, max_line_chars)


def chunk_line_texts(chunk: CaptionChunk, break_before: int | None) -> list[str]:
    """The plain text of each visual line of a chunk (words in logical order)."""
    words = [w.word for w in chunk.words]
    if break_before is None:
        return [" ".join(words)]
    return [" ".join(words[:break_before]), " ".join(words[break_before:])]


def caption_chunks(transcript: Transcript, edl: EDL, style: CaptionStyle) -> list[CaptionChunk]:
    """The timeline-axis caption chunks the ASS file is built from (also used
    by the review preview to pick real caption moments)."""
    words = _remap_words_to_timeline(transcript, edl)
    return chunk_words(words, max_chars=style.max_chars_per_line, max_duration=3.2)


def build_ass(
    transcript: Transcript,
    edl: EDL,
    style: CaptionStyle,
    *,
    safe_zone: SafeZone | None = None,
    max_lines: int | None = None,
) -> str:
    safe_zone = safe_zone or SafeZone()
    margins = margins_px(safe_zone, edl.width, edl.height)
    chunks = caption_chunks(transcript, edl, style)

    # ASS karaoke paints a word with SecondaryColour until it is spoken and
    # PrimaryColour afterwards. Upcoming words must stay in the text color and
    # turn to the highlight as they are spoken, so a karaoke style swaps the two.
    primary, secondary = (
        (style.highlight_color, style.primary_color) if style.word_highlight
        else (style.primary_color, style.highlight_color)
    )
    plate = uses_plate(style)
    header = ASS_HEADER_TEMPLATE.format(
        width=edl.width, height=edl.height, font=style.font_ar, size=style.font_size,
        primary=primary, highlight=secondary, outline_color=style.outline_color,
        back=style.back_color, bold=-1 if style.bold else 0,
        # With a plate the text style carries no box (and no border): the plate
        # style below draws the backing.
        border_style=1 if plate or style.background not in _BOX_BACKGROUNDS else 3,
        outline=0 if plate else style.outline, shadow=style.shadow,
        margin_l=margins["left"], margin_r=margins["right"], margin_v=margins["bottom"],
    )
    if plate:
        header = header.replace("\n\n[Events]", f"\n{_plate_style_line(style, margins)}\n\n[Events]", 1)

    events = []
    for chunk in chunks:
        if chunk.end <= chunk.start:
            continue
        break_at = balanced_break_index(chunk.words, style.line_break_chars)
        if style.word_highlight:
            text = build_karaoke_text(chunk, style, break_before=break_at)
        elif break_at is not None:
            text = (
                " ".join(w.word for w in chunk.words[:break_at])
                + "\\N"
                + " ".join(w.word for w in chunk.words[break_at:])
            )
        else:
            # Raw logical-order text: this project's libass build (HarfBuzz +
            # FriBidi) shapes and reorders it itself -- see captions/rtl.py.
            text = chunk.text
        text = wrap_rtl(text, chunk.text)
        span = f"{_fmt_ass_time(chunk.start)},{_fmt_ass_time(chunk.end)}"
        if plate:
            lines = chunk_line_texts(chunk, break_at)
            for plate_text in plate_lines(
                lines, chunk.text, style, width=edl.width, height=edl.height, margins=margins,
            ):
                events.append(f"Dialogue: {_PLATE_LAYER},{span},{PLATE_STYLE},,0,0,0,,{plate_text}")
        events.append(f"Dialogue: {_TEXT_LAYER if plate else 0},{span},Default,,0,0,0,,{text}")

    return header + "\n".join(events) + "\n"


def build_srt(transcript: Transcript, edl: EDL) -> str:
    words = _remap_words_to_timeline(transcript, edl)
    chunks = chunk_words(words, max_chars=42, max_duration=4.0)
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(chunk.start)} --> {_fmt_srt_time(chunk.end)}")
        # Plain-text SRT sidecar: raw logical-order text. SRT has no shaping
        # engine of its own -- players/editors that open it do their own
        # Arabic shaping/BiDi, same as any other subtitle text they load.
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


def write_captions(transcript: Transcript, edl: EDL, style: CaptionStyle, ass_path: Path, srt_path: Path) -> None:
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(build_ass(transcript, edl, style), encoding="utf-8")
    srt_path.write_text(build_srt(transcript, edl), encoding="utf-8")
