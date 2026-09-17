"""Review-First Editing Workflow (spec sections 1-2, 3, 5, 7): each builder
function is a pure transform of pipeline data into a human-reviewable
artifact -- these tests check the pieces the completion standard hinges on:
low-confidence words are flagged (never silently rewritten), corrections
preserve timing, and a missing brand accent is explicitly reported rather
than silently defaulted."""
from __future__ import annotations

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import (
    EDL,
    AnimationKind,
    AnimationSpec,
    BrollPlanItem,
    BrollSourceKind,
    EDLClip,
    MotionPlanItem,
    Segment,
    Transcript,
    Word,
)
from video_edit_agent.review.builder import (
    apply_transcript_corrections,
    build_brand_summary,
    build_broll_review,
    build_timeline_review,
    build_transcript_review,
)
from video_edit_agent.review.schemas import TranscriptCorrection


def _transcript() -> Transcript:
    words = [
        Word(word="مرحبا", start=0.0, end=0.5, confidence=0.95),
        Word(word="بيك", start=0.5, end=1.0, confidence=0.4),
    ]
    segment = Segment(id="seg0", start=0.0, end=1.0, text="مرحبا بيك", words=words)
    return Transcript(provider="test", language="ar", segments=[segment])


def test_low_confidence_words_are_flagged_suspicious_not_rewritten():
    review = build_transcript_review(_transcript())
    seg = review.segments[0]
    assert seg.words[0].suspicious is False
    assert seg.words[1].suspicious is True
    assert seg.words[1].word == "بيك"  # never silently rewritten
    assert review.suspicious_word_count == 1


def test_transcript_review_includes_arabic_questions():
    review = build_transcript_review(_transcript())
    assert any("تعديل" in q for q in review.questions)


def test_apply_transcript_correction_preserves_segment_timing():
    transcript = _transcript()
    corrected = apply_transcript_corrections(
        transcript, [TranscriptCorrection(segment_id="seg0", corrected_text="مرحبا فيك")]
    )
    seg = corrected.segments[0]
    assert seg.text == "مرحبا فيك"
    assert seg.start == 0.0
    assert seg.end == 1.0


def test_apply_transcript_correction_leaves_other_segments_untouched():
    transcript = _transcript()
    corrected = apply_transcript_corrections(transcript, [])
    assert corrected.segments[0].text == transcript.segments[0].text


def test_brand_summary_reports_accent_fallback_explicitly():
    brand = Brand(name="acme")
    brand.colors.secondary = "#00AAFF"
    summary = build_brand_summary(brand, logo_path=None, cta_text="Learn more")
    assert summary.accent_fallback_used is True
    assert summary.accent_color == "#00AAFF"
    assert any("accent" in w for w in summary.warnings)
    assert any("logo" in w for w in summary.warnings)


def test_brand_summary_no_fallback_warning_when_accent_explicit():
    brand = Brand(name="acme")
    brand.colors.accent = "#123456"
    summary = build_brand_summary(brand, logo_path="brands/acme/logos/logo.png", cta_text="Learn more")
    assert summary.accent_fallback_used is False
    assert summary.accent_color == "#123456"
    assert not any("accent" in w for w in summary.warnings)


def test_timeline_review_reflects_broll_and_cut_reason():
    edl = EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[EDLClip(source_file="a.mp4", source_in=0.0, source_out=5.0, timeline_in=0.0, timeline_out=5.0)],
    )
    broll = [
        BrollPlanItem(
            timeline_start=1.0, timeline_end=3.0, purpose="illustrate", spoken_concept="growth",
            recommended_visual="chart", source=BrollSourceKind.LOCAL_LIBRARY, asset_path="chart.mp4",
        )
    ]
    review = build_timeline_review(edl, _transcript(), broll, [], logo_present=False)
    assert review.items[0].mode == "broll"
    assert review.items[0].broll_description == "chart"


def test_timeline_review_marks_cta_present_for_real_animation_kind_enum():
    """Regression test: `AnimationKind` is a `str, Enum` mixin, so
    `str(AnimationKind.CTA)` is "AnimationKind.CTA" (Enum's __str__), not
    "cta" -- a naive `str(kind).endswith("cta")` check silently never
    matches. `cta_present` must be computed from `.value` instead."""
    edl = EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[EDLClip(source_file="a.mp4", source_in=0.0, source_out=5.0, timeline_in=0.0, timeline_out=5.0)],
    )
    motion = [
        MotionPlanItem(
            spec=AnimationSpec(kind=AnimationKind.CTA, timeline_start=1.0, timeline_end=3.0, text="تواصل معنا")
        )
    ]
    review = build_timeline_review(edl, _transcript(), [], motion, logo_present=False)
    assert review.items[0].motion_treatment == "cta"
    assert review.items[0].cta_present is True


def test_broll_review_marks_quality_gate_by_asset_presence():
    items = [
        BrollPlanItem(
            timeline_start=0.0, timeline_end=1.0, purpose="p", spoken_concept="c", recommended_visual="v",
            source=BrollSourceKind.NONE, asset_path=None,
        ),
        BrollPlanItem(
            timeline_start=1.0, timeline_end=2.0, purpose="p", spoken_concept="c", recommended_visual="v",
            source=BrollSourceKind.LOCAL_LIBRARY, asset_path="clip.mp4",
        ),
    ]
    review = build_broll_review(items)
    assert review.items[0].quality_gate_passed is None
    assert review.items[1].quality_gate_passed is True
