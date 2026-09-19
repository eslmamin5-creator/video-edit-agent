"""Typed models for every review artifact (Review-First Editing Workflow
spec sections 1-2, 3, 5-8). Kept separate from `core/schemas.py` because
these are UX/review-layer contracts, not pipeline data contracts -- nothing
downstream of the pipeline should ever need to import from here.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Section 1-2: transcript/caption review
# --------------------------------------------------------------------------

# Confidence below this is flagged as "suspicious" for the user's attention
# -- never silently corrected (spec section 1: "never formalize/translate/
# silently improve Arabic dialect").
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.6

TRANSCRIPT_REVIEW_QUESTIONS_AR = [
    "هل في كلمة أو جملة محتاجة تعديل؟",
    "هل في اسم براند / اسم شخص / مصطلح إنجليزي مكتوب غلط؟",
    "هل تقسيم الكابشن مناسب؟",
]


class TranscriptReviewWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: float
    suspicious: bool = False


class TranscriptReviewSegment(BaseModel):
    id: str
    speaker: str | None = None
    start: float
    end: float
    text: str
    caption_lines: list[str] = Field(default_factory=list)
    words: list[TranscriptReviewWord] = Field(default_factory=list)
    suspicious_word_count: int = 0


class TranscriptReview(BaseModel):
    provider: str
    language: str
    segments: list[TranscriptReviewSegment] = Field(default_factory=list)
    suspicious_word_count: int = 0
    questions: list[str] = Field(default_factory=lambda: list(TRANSCRIPT_REVIEW_QUESTIONS_AR))


class TranscriptCorrection(BaseModel):
    """One user-supplied correction to a single segment's text.

    Applying a correction never changes segment/word timing (spec section 1:
    "Corrections must not change timing unnecessarily") -- only the text
    changes; start/end and word count/spacing are preserved as closely as
    the correction allows."""

    segment_id: str
    corrected_text: str
    # None = `corrected_text` replaces the whole segment; an int replaces only
    # that (0-based) word of the segment, leaving the rest untouched.
    word_index: int | None = None


class CaptionPreview(BaseModel):
    style_name: str
    font_ar: str
    font_en: str
    primary_color: str
    highlight_color: str
    back_color: str
    position: str
    safe_zone_note: str
    sample_arabic_line: str
    sample_mixed_line: str
    background_mode: str = "box"
    outline_color: str | None = None
    word_highlight: bool = False
    line_break_chars: int = 0
    max_chars_per_line: int | None = None
    # Sample chunks taken from the real transcript (not hardcoded copy).
    sample_multiline: str | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Section 3: brand lock / brand summary
# --------------------------------------------------------------------------


class BrandSummary(BaseModel):
    brand_name: str
    logo_asset: str | None = None
    primary_color: str
    secondary_color: str
    accent_color: str | None = None
    accent_fallback_used: bool = False
    arabic_font: str | None = None
    english_font: str | None = None
    caption_style: str | None = None
    cta_text: str | None = None
    cta_status: str = "NONE"
    cta_style: str | None = None
    motion_accent_style: str | None = None
    logo_mode: str = "end_card"
    logo_duration: float | None = None
    logo_reveal: str | None = None
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Section 5: timeline / edit-plan review
# --------------------------------------------------------------------------


class TimelineReviewItem(BaseModel):
    timeline_start: float
    timeline_end: float
    mode: str  # "talking_head" | "broll"
    cut_reason: str
    zoom: float
    caption_text: str
    broll_description: str | None = None
    editorial_treatment: str | None = None  # see broll/treatment.py
    motion_treatment: str | None = None
    behind_subject: bool = False
    logo_present: bool = False
    cta_present: bool = False


class TimelineEndCard(BaseModel):
    """The branded end card appended after the content (logo mode `end_card` /
    `intro_and_end`); every value is derived from the Brand Profile's logo
    behavior."""

    treatment: str = "BRANDED END CARD"
    start: float
    duration: float
    logo: str = "centered"
    logo_asset: str | None = None
    background: str = ""
    motion: str = ""
    cta: str = "NONE"
    preview_frame: str | None = None


class TimelinePolicy(BaseModel):
    logo_mode: str
    persistent_logo_bug: str  # "NONE" or where the bug sits
    captions: str
    cta: str
    intro_card: bool = False
    end_card: TimelineEndCard | None = None
    warnings: list[str] = Field(default_factory=list)


class TimelineReview(BaseModel):
    total_duration: float
    items: list[TimelineReviewItem] = Field(default_factory=list)
    policy: TimelinePolicy | None = None
    total_duration_with_cards: float | None = None


# --------------------------------------------------------------------------
# Section 7: B-roll review
# --------------------------------------------------------------------------


class BrollReviewItem(BaseModel):
    timeline_start: float
    timeline_end: float
    spoken_context: str
    recommended_visual: str = ""
    source: str
    # What approval would do for this slot ("none" means nothing is resolved
    # yet; generation only happens after approval, never during review).
    source_recommendation: str = ""
    asset_path: str | None = None
    prompt: str | None = None
    # The exact prompt a generation provider would receive, shown so the user
    # can correct it before any generation is paid for.
    draft_prompt: str | None = None
    confidence: float = 0.0
    quality_gate_passed: bool | None = None
    # Editorial decision for this slot (one of broll.treatment.Treatment).
    treatment: str | None = None
    treatment_reason: str | None = None
    visual_concept: str | None = None
    generate_later: bool = False


class BrollReview(BaseModel):
    items: list[BrollReviewItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Section 6: preview frames / contact sheet
# --------------------------------------------------------------------------


class PreviewFrame(BaseModel):
    label: str  # e.g. "hook", "caption_ar", "caption_mixed", "logo", ...
    timeline_at: float
    image_path: str


class PreviewFrameSet(BaseModel):
    frames: list[PreviewFrame] = Field(default_factory=list)
    contact_sheet_path: str | None = None


# --------------------------------------------------------------------------
# Section 8: render approval / state machine
# --------------------------------------------------------------------------


class ReviewStage(str, Enum):
    ANALYZE = "ANALYZE"
    REVIEW_TRANSCRIPT = "REVIEW_TRANSCRIPT"
    REVIEW_PLAN = "REVIEW_PLAN"
    REVIEW_VISUALS = "REVIEW_VISUALS"
    APPROVED = "APPROVED"
    RENDERED = "RENDERED"


class UnresolvedTranscriptItem(BaseModel):
    """A transcript segment the reviewer has NOT confirmed. It stays open (and
    blocks approval) until a correction is persisted or it is explicitly
    resolved; nothing is guessed or rewritten in the meantime."""

    segment_id: str
    segment: int | None = None  # 1-based position, as shown in the audio review
    asr_text: str = ""
    reason: str = ""


class ReviewApprovalState(BaseModel):
    stage: ReviewStage = ReviewStage.ANALYZE
    ready_for_final_render: bool = False
    bypassed: bool = False  # true when the user explicitly skipped review (--yes/--no-review)
    # Approval of the B-roll PLAN; provider generation may only run when set.
    broll_generation_approved: bool = False
    # Transcript segments still awaiting the user's confirmation; approval is
    # refused while any remain (see review.state.approve).
    unresolved_transcript: list[UnresolvedTranscriptItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
