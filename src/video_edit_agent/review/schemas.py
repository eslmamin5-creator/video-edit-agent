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
    cta_text: str
    cta_style: str | None = None
    motion_accent_style: str | None = None
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
    motion_treatment: str | None = None
    behind_subject: bool = False
    logo_present: bool = False
    cta_present: bool = False


class TimelineReview(BaseModel):
    total_duration: float
    items: list[TimelineReviewItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Section 7: B-roll review
# --------------------------------------------------------------------------


class BrollReviewItem(BaseModel):
    timeline_start: float
    timeline_end: float
    spoken_context: str
    source: str
    asset_path: str | None = None
    prompt: str | None = None
    confidence: float = 0.0
    quality_gate_passed: bool | None = None


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


class ReviewApprovalState(BaseModel):
    stage: ReviewStage = ReviewStage.ANALYZE
    ready_for_final_render: bool = False
    bypassed: bool = False  # true when the user explicitly skipped review (--yes/--no-review)
    notes: list[str] = Field(default_factory=list)
