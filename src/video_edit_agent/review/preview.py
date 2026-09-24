"""Lightweight preview frames + contact sheet (Review-First Editing Workflow
spec section 6, marked "critical"): lets the user see representative frames
of the ACTUAL composited timeline (captions, brand colors, logo, motion
graphics, B-roll, CTA) before paying for a full render.

Each frame reuses the exact same `RenderPlan`/`build_filter_complex` graph
the real render uses -- so what the user approves here is what they get, not
a separate lower-fidelity mock.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.captions.engine import balanced_break_index
from video_edit_agent.core.media import run
from video_edit_agent.core.schemas import EDL, MotionPlanItem
from video_edit_agent.render.composition import RenderPlan, build_filter_complex
from video_edit_agent.review.schemas import PreviewFrame, PreviewFrameSet


class PreviewFrameError(RuntimeError):
    pass


def pick_caption_timestamps(chunks: list, line_break_chars: int, word_highlight: bool) -> dict[str, float]:
    """Picks real caption moments from the actual chunks: a single-line
    caption, a caption the line balancer splits over two lines, and (when
    word highlighting is on) one late in a chunk so several words are already
    highlighted. Labels are skipped when no chunk qualifies."""
    picks: dict[str, float] = {}
    usable = [c for c in chunks if c.end > c.start and len(c.words) >= 2]
    multi = next((c for c in usable if len(c.words) >= 4 and balanced_break_index(c.words, line_break_chars) is not None), None)
    if multi is not None:
        picks["caption_multiline"] = multi.start + 0.6 * (multi.end - multi.start)
    # Prefer a chunk that fits on one line; if the line balancer splits every
    # chunk, fall back to the shortest one so a "normal" frame still exists.
    rest = [c for c in usable if c is not multi]
    singles = [c for c in rest if balanced_break_index(c.words, line_break_chars) is None]
    normal = singles[len(singles) // 3] if singles else min(rest, key=lambda c: len(c.text), default=None)
    if normal is not None:
        picks["caption_normal"] = normal.start + 0.15 * (normal.end - normal.start)
    if word_highlight:
        pool = [c for c in rest if c is not normal and len(c.words) >= 3]
        highlight = pool[len(pool) // 2] if pool else None
        if highlight is not None:
            picks["caption_highlight"] = highlight.start + 0.65 * (highlight.end - highlight.start)
    return picks


def pick_representative_timestamps(
    edl: EDL, motion_plan: list[MotionPlanItem], broll_plan: list, transcript, caption_picks: dict[str, float] | None = None
) -> dict[str, float]:
    """Chooses one timestamp per interesting moment (spec section 6's list:
    hook, normal caption, mixed-language caption, logo, punch-in, B-roll,
    motion graphic, behind-subject, CTA). Silently skips a label when the
    timeline has nothing matching it -- never invents a fake timestamp."""
    total = edl.total_duration
    picks: dict[str, float] = {}
    if total <= 0:
        return picks

    picks["hook"] = min(1.0, total / 2)

    for clip in edl.clips:
        if clip.zoom and clip.zoom != 1.0:
            picks.setdefault("punch_in", (clip.timeline_in + clip.timeline_out) / 2)
            break

    for item in broll_plan:
        # Only slots that will actually cut away to footage get a B-roll frame.
        if getattr(item, "treatment", None) in (None, "local_broll", "generated_broll"):
            picks.setdefault("broll", (item.timeline_start + item.timeline_end) / 2)
            break

    for m in motion_plan:
        kind = m.spec.kind.value if hasattr(m.spec.kind, "value") else str(m.spec.kind)
        mid = (m.spec.timeline_start + m.spec.timeline_end) / 2
        if kind == "cta":
            picks.setdefault("cta", mid)
        elif kind == "logo_reveal":
            picks.setdefault("logo", mid)
        elif m.spec.behind_subject:
            picks.setdefault("behind_subject", mid)
        else:
            picks.setdefault("motion_graphic", mid)

    words = transcript.words if transcript is not None else []
    has_arabic = any(any("؀" <= ch <= "ۿ" for ch in w.word) for w in words)
    has_latin = any(any("a" <= ch.lower() <= "z" for ch in w.word) for w in words)
    if caption_picks:
        # Real caption moments replace the generic proportional guess.
        picks.update(caption_picks)
    elif has_arabic:
        picks.setdefault("caption_ar", min(total - 0.1, total * 0.3) if total > 0.2 else 0.0)
    if has_arabic and has_latin and not caption_picks:
        picks.setdefault("caption_mixed", min(total - 0.1, total * 0.6) if total > 0.2 else 0.0)

    return picks


def extract_frame(plan: RenderPlan, timestamp_s: float, output_path: Path) -> Path:
    """Extracts a single JPEG frame from the exact composited timeline at
    `timestamp_s`, using the real `build_filter_complex` graph (video branch
    only) so caption/brand/motion/B-roll compositing is identical to what a
    full render would produce at that instant."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs, filter_complex, map_labels = build_filter_complex(plan)
    v_map, a_map = map_labels.strip("[]").split("][")

    # The filter graph also declares an audio output pad ([aout]); ffmpeg
    # refuses to build the graph unless every declared output is mapped
    # somewhere, so route it to the null muxer instead of dropping it.
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{v_map}]",
        "-ss", f"{max(0.0, timestamp_s):.3f}",
        "-frames:v", "1",
        "-q:v", "3",
        str(output_path),
        "-map", f"[{a_map}]",
        "-f", "null",
        "-",
    ]
    result = run(cmd, timeout=300)
    if result.returncode != 0 or not output_path.exists():
        raise PreviewFrameError(f"frame extraction failed at t={timestamp_s}:\n{result.stderr.strip()[-2000:]}")
    return output_path


def annotate_planned_treatment(frame_path: Path, text: str) -> None:
    """Stamps a "planned, not rendered" banner onto a preview frame. Used in
    review mode, where motion graphics are not rendered before approval, so a
    motion frame would otherwise look identical to a plain talking-head frame.
    Best-effort: a missing Pillow install leaves the frame untouched."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    with Image.open(frame_path) as opened:
        img = opened.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(18, img.width // 32))
    _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
    pad = 10
    y = img.height - (bottom - top) - 3 * pad
    draw.rectangle([0, y - pad, img.width, y + (bottom - top) + pad], fill=(0, 0, 0))
    draw.text((pad, y - top), text, fill=(255, 220, 0), font=font)
    img.save(frame_path, quality=90)


def generate_preview_frames(
    plan: RenderPlan,
    timestamps: dict[str, float],
    out_dir: Path,
    planned_labels: dict[str, str] | None = None,
) -> PreviewFrameSet:
    """Generates one frame per label in `timestamps`. Only the affected
    labels need to be regenerated when the user requests a change (spec
    section 6: "only affected frames regenerate") -- callers can pass a
    filtered `timestamps` dict on a re-run."""
    frames: list[PreviewFrame] = []
    for label, t in timestamps.items():
        frame_path = out_dir / f"frame_{label}.jpg"
        extract_frame(plan, t, frame_path)
        if planned_labels and label in planned_labels:
            annotate_planned_treatment(frame_path, planned_labels[label])
        frames.append(PreviewFrame(label=label, timeline_at=t, image_path=str(frame_path)))
    return PreviewFrameSet(frames=frames)


def build_contact_sheet(frame_set: PreviewFrameSet, out_path: Path, columns: int = 3) -> Path | None:
    """Assembles all preview frames into one grid image for quick visual
    scanning. Returns None (never raises) if Pillow isn't installed or there
    are no frames -- the contact sheet is a convenience, not a hard
    requirement (spec section 6: "contact sheet if practical")."""
    if not frame_set.frames:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    images = []
    for frame in frame_set.frames:
        path = Path(frame.image_path)
        if path.exists():
            images.append((frame.label, Image.open(path)))
    if not images:
        return None

    thumb_w = 360
    thumbs = []
    for label, img in images:
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, round(img.height * ratio)))
        thumbs.append((label, thumb))

    rows = (len(thumbs) + columns - 1) // columns
    row_height = max(t.height for _, t in thumbs) + 24
    sheet = Image.new("RGB", (thumb_w * columns, row_height * rows), color="white")

    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for i, (label, thumb) in enumerate(thumbs):
        col, row = i % columns, i // columns
        x, y = col * thumb_w, row * row_height
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 2), label, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    frame_set.contact_sheet_path = str(out_path)
    return out_path
