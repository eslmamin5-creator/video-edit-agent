"""Typed models for every review artifact (Review-First Editing Workflow
spec sections 1-2, 3, 5-8). Kept separate from `core/schemas.py` because
these are UX/review-layer contracts, not pipeline data contracts -- nothing
downstream of the pipeline should ever need to import from here.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, computed_field

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


class SegmentReviewStatus(str, Enum):
    """Where one transcript segment stands in the chat review.

    approved: the user confirmed it (or typed the exact wording), or it has no
        low-confidence words and was never flagged.
    needs_review: low-confidence words and no decision yet (advisory).
    corrected_pending_approval: a correction exists but the user has not yet
        said the result is right (a partial word/phrase fix). Blocks approval.
    unresolved: flagged and awaiting the user. Blocks approval.
    """

    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    CORRECTED_PENDING_APPROVAL = "corrected_pending_approval"
    UNRESOLVED = "unresolved"


class SegmentDecision(BaseModel):
    """The user's explicit decision on one segment (persisted in the review state)."""

    segment_id: str
    status: SegmentReviewStatus  # only approved / corrected_pending_approval are stored


class ApprovalStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"


class CopySource(str, Enum):
    """The only origins a piece of on-screen copy may have. Raw or unresolved
    ASR text is not among them."""

    APPROVED_TRANSCRIPT = "approved_transcript"  # transcript text the user has confirmed
    USER_SUPPLIED = "user_supplied"  # copy typed/pasted by the user
    APPROVED_REWRITE = "approved_rewrite"  # a rewrite the user explicitly approved


class TextTreatmentReview(BaseModel):
    """Review record of one text-driven motion treatment (a hook title, a
    lower third, ...). The VISUAL treatment (animation, placement, colour,
    contrast) and its COPY (the words) are approved independently, so fixing
    the words never asks the user to re-review the animation."""

    treatment: str  # AnimationKind value, e.g. "hook_title"
    required: bool = True  # a required treatment with unapproved copy blocks final approval
    visual_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
    # The approved visual props (placement/colour/plate/outline...), reused
    # as-is when only the copy changes.
    visual_props: dict | None = None
    copy_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
    copy_source: CopySource | None = None
    approved_copy: str | None = None  # the only text that may reach a final render
    # Where the proposed copy came from in the transcript (never used as copy
    # unless those segments are confirmed).
    source_segment_ids: list[str] = Field(default_factory=list)
    source_segments: list[int] = Field(default_factory=list)  # 1-based, as in the audio review
    proposed_asr_text: str = ""  # reference for the reviewer; never rewritten
    # Wording the agent PROPOSES after the transcript is approved (a rewrite, not
    # a transcript line). Never on screen until the user approves it.
    proposed_copy: str | None = None
    blocking_reason: str | None = None
    placeholder_text: str | None = None
    placeholder_used_in_preview: bool = False
    layout_fit_issue: str | None = None
    layout_adjustments: list[str] = Field(default_factory=list)


class ReviewApprovalState(BaseModel):
    stage: ReviewStage = ReviewStage.ANALYZE
    ready_for_final_render: bool = False
    bypassed: bool = False  # true when the user explicitly skipped review (--yes/--no-review)
    # Approval of the B-roll PLAN; provider generation may only run when set.
    broll_generation_approved: bool = False
    # Transcript segments still awaiting the user's confirmation; approval is
    # refused while any remain (see review.state.approve).
    unresolved_transcript: list[UnresolvedTranscriptItem] = Field(default_factory=list)
    # Explicit per-segment decisions from the chat review (see SegmentReviewStatus).
    segment_reviews: list[SegmentDecision] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Per-treatment visual/copy approval (see TextTreatmentReview).
    text_treatments: list[TextTreatmentReview] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hook_copy_status(self) -> str | None:
        """`copy_status` of the hook title (None when the plan has no hook)."""
        for t in self.text_treatments:
            if t.treatment == "hook_title":
                return t.copy_status.value
        return None

    def reviewed_segment_ids(self) -> set[str]:
        """Segments the user has looked at and decided on in the transcript review."""
        return {d.segment_id for d in self.segment_reviews}

    def treatment(self, name: str) -> TextTreatmentReview | None:
        return next((t for t in self.text_treatments if t.treatment == name), None)
