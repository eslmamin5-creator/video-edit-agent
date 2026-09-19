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


def balanced_break_index(words: list, max_line_chars: int) -> int | None:
    """Index of the word that should start the second line when a chunk is
    longer than `max_line_chars`, or None when no split is needed/possible.
    Phrase-aware (see `captions/phrasing.py`): it avoids splitting tightly
    connected phrases and falls back to a width-balanced split otherwise."""
    return phrase_break_index(words, max_line_chars)


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
    header = ASS_HEADER_TEMPLATE.format(
        width=edl.width, height=edl.height, font=style.font_ar, size=style.font_size,
        primary=primary, highlight=secondary, outline_color=style.outline_color,
        back=style.back_color, bold=-1 if style.bold else 0,
        border_style=3 if style.background in _BOX_BACKGROUNDS else 1, outline=style.outline, shadow=style.shadow,
        margin_l=margins["left"], margin_r=margins["right"], margin_v=margins["bottom"],
    )

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
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(chunk.start)},{_fmt_ass_time(chunk.end)},Default,,0,0,0,,{text}"
        )

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
