"""Builds every review artifact from the pipeline's own intermediate data --
no new analysis, no re-transcription, no re-planning. Every function here is
a pure, cheap transform of data the pipeline already computed (spec section
12: review artifacts must be cheap, built before any expensive final render).
"""
from __future__ import annotations

from video_edit_agent.brand.logo_policy import LogoPlan
from video_edit_agent.brand.schema import Brand
from video_edit_agent.broll.prompt import build_broll_prompt
from video_edit_agent.broll.treatment import BROLL_TREATMENTS
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core.schemas import (
    EDL,
    BrollPlanItem,
    BrollSourceKind,
    MotionPlanItem,
    Transcript,
)
from video_edit_agent.review.corrections import apply_corrections
from video_edit_agent.review.schemas import (
    SUSPICIOUS_CONFIDENCE_THRESHOLD,
    BrandSummary,
    BrollReview,
    BrollReviewItem,
    CaptionPreview,
    TimelineEndCard,
    TimelinePolicy,
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
    """Applies user-supplied corrections without changing timing; see
    `review.corrections.apply_corrections`."""
    return apply_corrections(transcript, corrections)


# --------------------------------------------------------------------------
# Section 2: caption preview
# --------------------------------------------------------------------------


def build_caption_preview(
    style: CaptionStyle,
    brand: Brand | None = None,
    *,
    sample_lines: tuple[str, str | None, str | None] | None = None,
    notes: list[str] | None = None,
) -> CaptionPreview:
    """`sample_lines` = (normal, multi-line, mixed-script) text taken from the
    real transcript so the preview shows the user's own words -- errors and
    dialect included -- not canned copy."""
    safe_zone = None
    if brand and brand.safe_zones:
        safe_zone = brand.safe_zones
    safe_zone_note = (
        f"safe zones: {safe_zone}" if safe_zone else "safe zones: none configured (using preset default position)"
    )
    normal, multiline, mixed = sample_lines or ("", None, None)
    return CaptionPreview(
        style_name=style.name,
        font_ar=style.font_ar,
        font_en=style.font_en,
        primary_color=style.primary_color,
        highlight_color=style.highlight_color,
        back_color=style.back_color,
        position="bottom-center",
        safe_zone_note=safe_zone_note,
        sample_arabic_line=normal,
        sample_mixed_line=mixed or "",
        sample_multiline=multiline,
        background_mode=style.background,
        outline_color=style.outline_color,
        word_highlight=style.word_highlight,
        line_break_chars=style.line_break_chars,
        max_chars_per_line=style.max_chars_per_line,
        notes=list(notes or []),
    )


# --------------------------------------------------------------------------
# Section 3: brand summary / brand lock
# --------------------------------------------------------------------------


def build_brand_summary(
    brand: Brand,
    logo_path: str | None,
    cta_text: str | None,
    caption_style: CaptionStyle | None = None,
    logo_plan: LogoPlan | None = None,
    style_notes: list[str] | None = None,
) -> BrandSummary:
    warnings: list[str] = list(style_notes or [])
    missing: list[str] = []
    accent_fallback_used = brand.colors.accent is None
    accent_color = brand.colors.accent or brand.colors.secondary
    if accent_fallback_used:
        missing.append("colors.accent")
        warnings.append(
            f"brand '{brand.name}' has no accent color set; using secondary color "
            f"'{brand.colors.secondary}' as a neutral fallback instead of inventing one"
        )
    if logo_path is None:
        missing.append("logo.asset")
        warnings.append(f"brand '{brand.name}' has no logo asset; no logo/watermark will be composited")
    if not (brand.arabic_font or brand.fonts):
        missing.append("typography.arabic")
        warnings.append(f"brand '{brand.name}' has no font list; using caption preset defaults")
    if logo_plan is not None:
        warnings.extend(w for w in logo_plan.warnings if w not in warnings)
    behavior = brand.logo.behavior
    cta = (cta_text or "").strip() or None

    return BrandSummary(
        brand_name=brand.name,
        logo_asset=logo_path,
        primary_color=brand.colors.primary,
        secondary_color=brand.colors.secondary,
        accent_color=accent_color,
        accent_fallback_used=accent_fallback_used,
        arabic_font=caption_style.font_ar if caption_style else brand.arabic_font,
        english_font=caption_style.font_en if caption_style else brand.latin_font,
        caption_style=caption_style.name if caption_style else brand.captions.preset,
        cta_text=cta,
        cta_status="PROVIDED" if cta else "NONE",
        cta_style=brand.cta.style,
        motion_accent_style=brand.motion.preferred_engine,
        logo_mode=(logo_plan.mode if logo_plan else behavior.mode).value,
        logo_duration=behavior.duration,
        logo_reveal=behavior.reveal,
        missing=missing,
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
    logo_present: bool = False,
    *,
    logo_plan: LogoPlan | None = None,
    cta_text: str | None = None,
    caption_note: str = "brand-driven",
    end_card_preview: str | None = None,
) -> TimelineReview:
    if logo_plan is not None:
        logo_present = logo_plan.persistent_bug
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
        # A slot the editor keeps on the speaker (punch-in, typography, ...) is
        # not a cutaway: only local/generated B-roll (or an undecided slot) is.
        treatment = broll_match.treatment if broll_match else None
        is_cutaway = broll_match is not None and (treatment is None or treatment in {t.value for t in BROLL_TREATMENTS})
        items.append(
            TimelineReviewItem(
                timeline_start=clip.timeline_in,
                timeline_end=clip.timeline_out,
                mode="broll" if is_cutaway else "talking_head",
                cut_reason=clip.reason.value if hasattr(clip.reason, "value") else str(clip.reason),
                zoom=clip.zoom,
                caption_text=" ".join(caption_words),
                broll_description=broll_match.recommended_visual if is_cutaway else None,
                editorial_treatment=treatment,
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
    policy = _build_policy(edl, logo_plan, cta_text, caption_note, end_card_preview)
    end = policy.end_card.duration if policy and policy.end_card else 0.0
    intro = logo_plan.intro.duration if logo_plan is not None and logo_plan.intro is not None else 0.0
    return TimelineReview(
        total_duration=edl.total_duration,
        items=items,
        policy=policy,
        total_duration_with_cards=(edl.total_duration + intro + end) if (intro or end) else None,
    )


def _build_policy(
    edl: EDL,
    logo_plan: LogoPlan | None,
    cta_text: str | None,
    caption_note: str,
    end_card_preview: str | None,
) -> TimelinePolicy | None:
    if logo_plan is None:
        return None
    cta = (cta_text or "").strip()
    end_card = None
    if logo_plan.end_card is not None:
        spec = logo_plan.end_card
        background = (
            f"solid brand primary {spec.background_top}"
            if spec.background_top == spec.background_bottom
            else f"gradient {spec.background_top} -> {spec.background_bottom} (brand primary -> secondary)"
        )
        motion = {
            "subtle": "subtle logo reveal (fade + small scale settle)",
            "fade": "logo fade-in",
            "none": "no logo animation",
        }.get(spec.reveal, spec.reveal)
        if spec.accent and spec.accent_style == "underline":
            motion += f"; brand accent {spec.accent} as a thin underline"
        end_card = TimelineEndCard(
            start=edl.total_duration,
            duration=spec.duration,
            logo_asset=str(spec.logo_path) if spec.logo_path else None,
            background=background,
            motion=motion,
            cta=cta or "NONE",
            preview_frame=end_card_preview,
        )
    return TimelinePolicy(
        logo_mode=logo_plan.mode.value,
        persistent_logo_bug="top-right corner (opt-in persistent_bug)" if logo_plan.persistent_bug else "NONE",
        captions=caption_note,
        cta=cta or "NONE",
        intro_card=logo_plan.intro is not None,
        end_card=end_card,
        warnings=list(logo_plan.warnings),
    )


def render_timeline_markdown(review: TimelineReview) -> str:
    """Human-readable version of the timeline review."""
    lines = ["# Timeline review", ""]
    pol = review.policy
    if pol:
        lines += [
            "## Policies",
            f"- Logo mode: {pol.logo_mode}",
            f"- Persistent logo bug (main content): {pol.persistent_logo_bug}",
            f"- Captions: {pol.captions}",
            f"- CTA: {pol.cta}",
            "",
        ]
    lines += [f"## Main content ({review.total_duration:.2f}s)", ""]
    for it in review.items:
        bits = [f"{it.timeline_start:6.2f}-{it.timeline_end:6.2f}s", it.mode, f"zoom x{it.zoom:.2f}"]
        if it.motion_treatment:
            bits.append(f"motion: {it.motion_treatment}{' (behind subject)' if it.behind_subject else ''}")
        if it.editorial_treatment:
            bits.append(f"treatment: {it.editorial_treatment}")
        if it.broll_description:
            bits.append(f"b-roll: {it.broll_description}")
        lines.append("- " + " | ".join(bits))
        if it.caption_text:
            lines.append(f"    caption: {it.caption_text}")
    if pol and pol.end_card:
        e = pol.end_card
        lines += [
            "",
            f"## End ({e.start:.2f}-{e.start + e.duration:.2f}s)",
            f"- Treatment: {e.treatment}",
            f"- Duration: {e.duration:.1f}s",
            f"- Logo: {e.logo}",
            f"- Background: {e.background}",
            f"- Motion: {e.motion}",
            f"- CTA: {e.cta}",
        ]
    if pol:
        lines += [f"- WARNING: {w}" for w in pol.warnings]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Section 7: B-roll review
# --------------------------------------------------------------------------


def build_broll_review(broll_plan: list[BrollPlanItem]) -> BrollReview:
    items = []
    for item in sorted(broll_plan, key=lambda b: b.timeline_start):
        source = item.source.value if hasattr(item.source, "value") else str(item.source)
        if item.source != BrollSourceKind.NONE:
            recommendation = f"use existing asset ({source})"
        elif item.generate_later:
            recommendation = (
                item.source_recommendation
                or "generate only after this plan is approved (Gemini image, then Veo), or supply your own clip"
            )
        else:
            recommendation = item.source_recommendation or "no B-roll needed for this treatment"
        items.append(
            BrollReviewItem(
                timeline_start=item.timeline_start,
                timeline_end=item.timeline_end,
                spoken_context=item.spoken_concept,
                recommended_visual=item.recommended_visual,
                source=source,
                source_recommendation=recommendation,
                asset_path=item.asset_path,
                prompt=item.prompt,
                # Draft prompts exist only for slots that would be generated.
                draft_prompt=build_broll_prompt(item) if item.generate_later else None,
                confidence=item.confidence,
                quality_gate_passed=None if item.asset_path is None else True,
                treatment=item.treatment,
                treatment_reason=item.treatment_reason,
                visual_concept=item.visual_concept,
                generate_later=item.generate_later,
            )
        )
    return BrollReview(items=items)
