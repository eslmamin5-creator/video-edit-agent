"""Typed models shared across the whole pipeline.

These are the contracts that let every stage of the pipeline (transcription,
editorial analysis, captions, B-roll, motion, render, QA) stay decoupled from
one another: nothing downstream of transcription needs to know which
transcription provider produced the data, and nothing downstream of the EDL
needs to know how it was constructed.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Unified transcript schema (spec section 7)
# --------------------------------------------------------------------------


class Word(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = 1.0


class Segment(BaseModel):
    id: str
    speaker: Optional[str] = None
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    """Provider-agnostic transcript. Every transcription provider must return
    this shape; nothing else in the program may branch on `provider`."""

    provider: str
    language: str = "auto"
    locale: Optional[str] = None
    duration: float = 0.0
    speakers: list[str] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    verbatim: bool = True
    raw_provider_response_path: Optional[str] = None

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out


# --------------------------------------------------------------------------
# Editorial analysis signals
# --------------------------------------------------------------------------


class CutReason(str, Enum):
    SILENCE = "silence"
    FALSE_START = "false_start"
    REPETITION = "repetition"
    FILLER = "filler"
    MANUAL = "manual"
    PACING = "pacing"


class TransitionType(str, Enum):
    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"
    DIP_TO_BLACK = "dip_to_black"


class EDLClip(BaseModel):
    """One clip in the Edit Decision List (spec section 12)."""

    source_file: str
    source_in: float
    source_out: float
    timeline_in: float
    timeline_out: float
    speaker: Optional[str] = None
    reason: CutReason = CutReason.MANUAL
    transition_in: TransitionType = TransitionType.HARD_CUT
    transition_out: TransitionType = TransitionType.HARD_CUT
    audio_fade_in_ms: int = 0
    audio_fade_out_ms: int = 0
    speed: float = 1.0
    overlay_refs: list[str] = Field(default_factory=list)
    caption_refs: list[str] = Field(default_factory=list)
    broll_refs: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.timeline_out - self.timeline_in


class EDL(BaseModel):
    version: int = 1
    fps: float = 30.0
    width: int = 1080
    height: int = 1920
    clips: list[EDLClip] = Field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return max((c.timeline_out for c in self.clips), default=0.0)


# --------------------------------------------------------------------------
# B-roll plan
# --------------------------------------------------------------------------


class BrollSourceKind(str, Enum):
    USER_ASSET = "user_asset"
    LOCAL_LIBRARY = "local_library"
    PROJECT_ASSET = "project_asset"
    GENERATED_IMAGE = "generated_image"
    GENERATED_VIDEO = "generated_video"
    NONE = "none"


class BrollPlanItem(BaseModel):
    timeline_start: float
    timeline_end: float
    purpose: str
    spoken_concept: str
    recommended_visual: str
    source: BrollSourceKind = BrollSourceKind.NONE
    asset_path: Optional[str] = None
    prompt: Optional[str] = None
    aspect_ratio: str = "9:16"
    duration: float = 0.0
    crop_behavior: str = "center_crop"
    transition: TransitionType = TransitionType.CROSSFADE
    confidence: float = 0.0


# --------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------


class QASeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QAIssue(BaseModel):
    category: str  # technical | language | visual | brand
    severity: QASeverity
    message: str
    timeline_at: Optional[float] = None
    auto_repairable: bool = False


class QAReport(BaseModel):
    issues: list[QAIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == QASeverity.ERROR for i in self.issues)


# --------------------------------------------------------------------------
# Motion graphics (spec sections 17-20)
# --------------------------------------------------------------------------


class AnimationKind(str, Enum):
    HOOK_TITLE = "hook_title"
    LOWER_THIRD = "lower_third"
    STAT_COUNTER = "stat_counter"
    QUOTE = "quote"
    COMPARISON = "comparison"
    FEATURE_CARD = "feature_card"
    CTA = "cta"
    LOGO_REVEAL = "logo_reveal"
    METRIC_HIGHLIGHT = "metric_highlight"
    PRODUCT_CALLOUT = "product_callout"
    TIMELINE_GRAPHIC = "timeline_graphic"
    LABEL = "label"
    ARROW = "arrow"
    BOX = "box"
    DIAGRAM = "diagram"
    DATA_VIZ = "data_viz"


class MotionEngine(str, Enum):
    HYPERFRAMES = "hyperframes"
    REMOTION = "remotion"
    MANIM = "manim"
    SIMPLE = "simple"


class AnimationSpec(BaseModel):
    kind: AnimationKind
    timeline_start: float
    timeline_end: float
    text: str = ""
    subtext: str = ""
    value: Optional[str] = None
    engine_hint: Optional[MotionEngine] = None
    behind_subject: bool = False
    x: str = "(W-w)/2"
    y: str = "(H-h)/2"
    extra: dict = Field(default_factory=dict)


class MotionPlanItem(BaseModel):
    spec: AnimationSpec
    engine_used: Optional[MotionEngine] = None
    output_path: Optional[str] = None
    source_path: Optional[str] = None
    error: Optional[str] = None
