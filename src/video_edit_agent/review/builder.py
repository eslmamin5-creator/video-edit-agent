"""Builds every review artifact from the pipeline's own intermediate data --
no new analysis, no re-transcription, no re-planning. Every function here is
a pure, cheap transform of data the pipeline already computed (spec section
12: review artifacts must be cheap, built before any expensive final render).
"""
from __future__ import annotations

from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.prompt import build_broll_prompt
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.schemas import (
    EDL,
    BrollPlanItem,
    BrollSourceKind,
    MotionPlanItem,
    Transcript,
)
from video_edit_agent.review.schemas import (
    SUSPICIOUS_CONFIDENCE_THRESHOLD,
    BrandSummary,
    BrollReview,
    BrollReviewItem,
    CaptionPreview,
    TimelineReview,
    TimelineReviewItem,
    TranscriptCorrection,
    TranscriptReview,
    TranscriptReviewSegment,
    TranscriptReviewWord,
)

# --------------------------------------------------------------------------
# Section 1: transcript review
# --------------------------------------------------------------------------


def _group_caption_lines(text: str, max_chars_per_line: int) -> list[str]:
    """Greedy word-wrap purely for display in the review artifact -- this is
    NOT the real caption-line-breaking logic used by `captions/engine.py`; it
    only gives the user a rough sense of how the line will be split."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len
    if current:
        lines.append(" ".join(current))
    return lines


def build_transcript_review(transcript: Transcript, style: CaptionStyle | None = None) -> TranscriptReview:
    max_chars = style.max_chars_per_line if style else 26
    segments: list[TranscriptReviewSegment] = []
    total_suspicious = 0
    for seg in transcript.segments:
        review_words = []
        seg_suspicious = 0
        for w in seg.words:
            suspicious = w.confidence < SUSPICIOUS_CONFIDENCE_THRESHOLD
            if suspicious:
                seg_suspicious += 1
            review_words.append(
                TranscriptReviewWord(
                    word=w.word, start=w.start, end=w.end, confidence=w.confidence, suspicious=suspicious
                )
            )
        total_suspicious += seg_suspicious
        segments.append(
            TranscriptReviewSegment(
                id=seg.id,
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                caption_lines=_group_caption_lines(seg.text, max_chars),
                words=review_words,
                suspicious_word_count=seg_suspicious,
            )
        )
    return TranscriptReview(
        provider=transcript.provider,
        language=transcript.language,
        segments=segments,
        suspicious_word_count=total_suspicious,
    )


def apply_transcript_corrections(
    transcript: Transcript, corrections: list[TranscriptCorrection]
) -> Transcript:
    """Applies user-supplied `TranscriptCorrection`s to a copy of `transcript`,
    changing only segment text (and its single synthetic word) -- never
    segment/word timing (spec section 1: "Corrections must not change timing
    unnecessarily"). The corrected segment's original per-word timestamps are
    collapsed into one span-covering word, since the exact per-word timing
    of a hand-edited sentence can no longer be trusted to align 1:1 with the
    original ASR words; downstream caption line-breaking only needs a
    segment's total span and text, not the old per-word grid.
    """
    corrections_by_id = {c.segment_id: c.corrected_text for c in corrections}
    if not corrections_by_id:
        return transcript
    new_segments = []
    for seg in transcript.segments:
        corrected_text = corrections_by_id.get(seg.id)
        if corrected_text is None:
            new_segments.append(seg)
            continue
        new_seg = seg.model_copy(
            update={
                "text": corrected_text,
                "words": [
                    type(seg.words[0])(word=corrected_text, start=seg.start, end=seg.end, confidence=1.0)
                ]
                if seg.words
                else [],
            }
        )
        new_segments.append(new_seg)
    return transcript.model_copy(update={"segments": new_segments})


# --------------------------------------------------------------------------
# Section 2: caption preview
# --------------------------------------------------------------------------


def build_caption_preview(style: CaptionStyle, brand: Brand | None = None) -> CaptionPreview:
    safe_zone = None
    if brand and brand.safe_zones:
        safe_zone = brand.safe_zones
    safe_zone_note = (
        f"safe zones: {safe_zone}" if safe_zone else "safe zones: none configured (using preset default position)"
    )
    return CaptionPreview(
        style_name=style.name,
        font_ar=style.font_ar,
        font_en=style.font_en,
        primary_color=style.primary_color,
        highlight_color=style.highlight_color,
        back_color=style.back_color,
        position="bottom-center",
        safe_zone_note=safe_zone_note,
        sample_arabic_line="مرحباً بكم في هذا الفيديو",
        sample_mixed_line="جربوا Discount Code: SAVE20 دلوقتي",
    )


# --------------------------------------------------------------------------
# Section 3: brand summary / brand lock
# --------------------------------------------------------------------------


def build_brand_summary(
    brand: Brand,
    logo_path: str | None,
    cta_text: str,
    caption_style: CaptionStyle | None = None,
) -> BrandSummary:
    warnings: list[str] = []
    accent_fallback_used = brand.colors.accent is None
    accent_color = brand.colors.accent or brand.colors.secondary
    if accent_fallback_used:
        warnings.append(
            f"brand '{brand.name}' has no accent color set; using secondary color "
            f"'{brand.colors.secondary}' as a neutral fallback instead of inventing one"
        )
    if logo_path is None:
        warnings.append(f"brand '{brand.name}' has no logo asset; no logo/watermark will be composited")
    if not brand.fonts:
        warnings.append(f"brand '{brand.name}' has no font list; using caption preset defaults")

    return BrandSummary(
        brand_name=brand.name,
        logo_asset=logo_path,
        primary_color=brand.colors.primary,
        secondary_color=brand.colors.secondary,
        accent_color=accent_color,
        accent_fallback_used=accent_fallback_used,
        arabic_font=caption_style.font_ar if caption_style else (brand.fonts[0] if brand.fonts else None),
        english_font=caption_style.font_en if caption_style else (brand.fonts[0] if brand.fonts else None),
        caption_style=caption_style.name if caption_style else brand.captions.preset,
        cta_text=cta_text,
        cta_style=brand.cta.style,
        motion_accent_style=brand.motion.preferred_engine,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Section 5: timeline / edit-plan review
# --------------------------------------------------------------------------


def build_timeline_review(
    edl: EDL,
    transcript: Transcript,
    broll_plan: list[BrollPlanItem],
    motion_plan: list[MotionPlanItem],
    logo_present: bool,
) -> TimelineReview:
    items: list[TimelineReviewItem] = []
    for clip in edl.clips:
        broll_match = next(
            (
                b
                for b in broll_plan
                if b.timeline_start < clip.timeline_out and b.timeline_end > clip.timeline_in
            ),
            None,
        )
        motion_match = next(
            (
                m
                for m in motion_plan
                if m.spec.timeline_start < clip.timeline_out and m.spec.timeline_end > clip.timeline_in
            ),
            None,
        )
        caption_words = [
            w.word for w in transcript.words if clip.timeline_in <= w.start < clip.timeline_out
        ]
        items.append(
            TimelineReviewItem(
                timeline_start=clip.timeline_in,
                timeline_end=clip.timeline_out,
                mode="broll" if broll_match else "talking_head",
                cut_reason=clip.reason.value if hasattr(clip.reason, "value") else str(clip.reason),
                zoom=clip.zoom,
                caption_text=" ".join(caption_words),
                broll_description=broll_match.recommended_visual if broll_match else None,
                motion_treatment=(
                    motion_match.spec.kind.value
                    if motion_match and hasattr(motion_match.spec.kind, "value")
                    else (str(motion_match.spec.kind) if motion_match else None)
                ),
                behind_subject=motion_match.spec.behind_subject if motion_match else False,
                logo_present=logo_present,
                cta_present=bool(
                    motion_match
                    and (
                        motion_match.spec.kind.value
                        if hasattr(motion_match.spec.kind, "value")
                        else str(motion_match.spec.kind)
                    )
                    == "cta"
                ),
            )
        )
    return TimelineReview(total_duration=edl.total_duration, items=items)


# --------------------------------------------------------------------------
# Section 7: B-roll review
# --------------------------------------------------------------------------


def build_broll_review(broll_plan: list[BrollPlanItem]) -> BrollReview:
    items = []
    for item in sorted(broll_plan, key=lambda b: b.timeline_start):
        source = item.source.value if hasattr(item.source, "value") else str(item.source)
        items.append(
            BrollReviewItem(
                timeline_start=item.timeline_start,
                timeline_end=item.timeline_end,
                spoken_context=item.spoken_concept,
                recommended_visual=item.recommended_visual,
                source=source,
                source_recommendation=(
                    "no local/user asset found; on approval generate one (Gemini image, then Veo) "
                    "or supply your own clip"
                    if item.source == BrollSourceKind.NONE
                    else f"use existing asset ({source})"
                ),
                asset_path=item.asset_path,
                prompt=item.prompt,
                draft_prompt=build_broll_prompt(item),
                confidence=item.confidence,
                quality_gate_passed=None if item.asset_path is None else True,
            )
        )
    return BrollReview(items=items)
