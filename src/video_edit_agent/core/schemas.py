"""Typed models shared across the whole pipeline.

These are the contracts that let every stage of the pipeline (transcription,
editorial analysis, captions, B-roll, motion, render, QA) stay decoupled from
one another: nothing downstream of transcription needs to know which
transcription provider produced the data, and nothing downstream of the EDL
needs to know how it was constructed.
"""
from __future__ import annotations

from enum import Enum

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
    speaker: str | None = None
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    """Provider-agnostic transcript. Every transcription provider must return
    this shape; nothing else in the program may branch on `provider`."""

    provider: str
    language: str = "auto"
    locale: str | None = None
    duration: float = 0.0
    speakers: list[str] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    verbatim: bool = True
    raw_provider_response_path: str | None = None

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


# Fraction of the cropped-away height placed above the crop window when a clip
# is punched in (0.30 keeps a head-and-shoulders subject in frame).
DEFAULT_ZOOM_ANCHOR_Y = 0.30


class ZoomRamp(BaseModel):
    """One eased zoom move inside a clip: the zoom eases to `zoom_to` between
    `start_s` and `end_s` (clip-local seconds; either may fall outside the
    clip, so a ramp centred on a cut is split across both neighbours)."""

    start_s: float
    end_s: float
    zoom_to: float
    easing: str = "smoothstep"  # smoothstep | linear
    # Where the anchor point eases to over the same span (None = it stays where it is). The crop window is then
    # animated in both axes, so a reframe / lower_subject glides instead of jumping.
    anchor_x_to: float | None = None
    anchor_y_to: float | None = None


class Reframe(BaseModel):
    """The single transform description shared by the final render and every
    preview: zoom starts at `zoom_start`, is moved by `ramps` in order, and is
    framed on the anchor point. `face_box` (normalized x, y, w, h of the
    subject's face in the full frame) keeps the face inside the crop window.
    """

    zoom_start: float = 1.0
    ramps: list[ZoomRamp] = Field(default_factory=list)
    anchor_x: float = 0.5
    anchor_y: float = DEFAULT_ZOOM_ANCHOR_Y
    face_box: tuple[float, float, float, float] | None = None
    face_margin: float = 0.04
    owner: str | None = None  # "camera_timeline" when the canonical camera timeline produced it; None = a legacy planner


class EDLClip(BaseModel):
    """One clip in the Edit Decision List (spec section 12)."""

    source_file: str
    source_in: float
    source_out: float
    timeline_in: float
    timeline_out: float
    speaker: str | None = None
    reason: CutReason = CutReason.MANUAL
    transition_in: TransitionType = TransitionType.HARD_CUT
    transition_out: TransitionType = TransitionType.HARD_CUT
    # Real crossfade overlap in seconds for `transition_in`, consumed by the
    # shared render core (spec Phase 2 Finalization section 2-3). Zero means
    # "no real video transition" even if `transition_in` says CROSSFADE --
    # only a caller that deliberately sets this (currently only the
    # Assembler's Transition Director) gets an actually-rendered crossfade,
    # so Editor's/Creator's existing output never changes as a side effect.
    transition_duration_s: float = 0.0
    # Whether this clip's source genuinely contains an audio stream (as
    # opposed to a synthesized silent track muxed in so the shared filter
    # graph always has an audio branch). Used to decide whether a real audio
    # crossfade is legitimate at a boundary (spec section 6).
    has_real_audio: bool = True
    # LUFS-ish target for a real `loudnorm` pass on this clip's own audio
    # (spec section 7). None means no normalization filter is applied.
    loudnorm_target_db: float | None = None
    audio_fade_in_ms: int = 0
    audio_fade_out_ms: int = 0
    speed: float = 1.0
    # Punch-in zoom factor for this clip (Baseline Recovery Milestone item 6,
    # ported from `majedphotos/video-ad-editor`'s `scripts/03_cut_zoom.py`).
    # 1.0 means no zoom. See `editorial/punch_in.py::plan_punch_ins`.
    zoom: float = 1.0
    # Time-varying reframe for this clip (eased punch-in ramps, anchor and
    # face-safe framing). When None the clip uses the static `zoom` above.
    reframe: Reframe | None = None
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
    asset_path: str | None = None
    prompt: str | None = None
    aspect_ratio: str = "9:16"
    duration: float = 0.0
    crop_behavior: str = "center_crop"
    transition: TransitionType = TransitionType.CROSSFADE
    confidence: float = 0.0
    # Editorial decision (see broll/treatment.py). None = not yet decided, in
    # which case the slot is treated as a plain B-roll candidate.
    treatment: str | None = None
    treatment_reason: str | None = None
    visual_concept: str | None = None
    source_recommendation: str | None = None
    generate_later: bool = False


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
    timeline_at: float | None = None
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
    BEHIND_TEXT = "behind_text"
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
    value: str | None = None
    engine_hint: MotionEngine | None = None
    behind_subject: bool = False
    x: str = "(W-w)/2"
    y: str = "(H-h)/2"
    extra: dict = Field(default_factory=dict)


class MotionPlanItem(BaseModel):
    spec: AnimationSpec
    engine_used: MotionEngine | None = None
    output_path: str | None = None
    source_path: str | None = None
    error: str | None = None
    # Records every engine tried and rejected before `engine_used` succeeded
    # (or before every engine failed), so a fallback is always traceable in
    # project metadata rather than silently substituted (spec V1.1 section 7).
    fallback_log: list[str] = Field(default_factory=list)
