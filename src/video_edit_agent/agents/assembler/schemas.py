"""Assembler-only schemas (spec sections 4-19).

Only new concepts that have no shared-core equivalent live here. The
Assembler writes its actual editable timeline exclusively to the shared
`core.schemas.EDL` / `core.timeline.MasterTimeline` -- there is no
Assembler-only timeline format.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class OrderPolicy(str, Enum):
    PRESERVE = "preserve"
    FILENAME = "filename"
    SCRIPT = "script"


class SceneInventoryItem(BaseModel):
    id: str
    filename: str
    path: str
    original_order: int
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None = None
    aspect_ratio: str
    has_audio: bool


class SceneAnalysis(BaseModel):
    scene_id: str
    duration: float
    first_frame_path: str | None = None
    last_frame_path: str | None = None
    dominant_color: str | None = None  # "#rrggbb"
    brightness: float | None = None  # 0-1
    contrast: float | None = None  # 0-1
    motion_direction: str | None = None  # left|right|up|down|static|unknown
    has_audio: bool = False
    is_silent: bool = True
    aspect_ratio: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0


class ScriptAlignmentItem(BaseModel):
    scene_id: str
    script_beat_index: int | None = None
    script_beat_text: str | None = None
    confidence: float = 0.0
    rationale: str = ""


class AspectStrategy(str, Enum):
    LETTERBOX = "letterbox"
    CROP = "crop"
    FIT = "fit"
    BLUR_BACKGROUND = "blur_background"
    BRAND_BACKGROUND = "brand_background"


class NormalizationChoice(BaseModel):
    scene_id: str
    aspect_strategy: AspectStrategy = AspectStrategy.LETTERBOX
    aspect_applied: bool = False
    recommended_brightness_delta: float = 0.0
    recommended_contrast_delta: float = 0.0
    recommended_saturation_delta: float = 0.0
    color_applied: bool = False
    reason: str = ""


class ContinuitySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContinuityFinding(BaseModel):
    from_scene: str
    to_scene: str
    category: str  # brightness|color_temperature|motion_direction|aspect_ratio|resolution
    description: str
    severity: ContinuitySeverity
    confidence: float
    recommendation: str


class TransitionKind(str, Enum):
    HARD_CUT = "hard_cut"
    ACTION_CUT = "action_cut"
    MATCH_CUT = "match_cut"
    J_CUT = "j_cut"
    L_CUT = "l_cut"
    AUDIO_BRIDGE = "audio_bridge"
    SHORT_CROSSFADE = "short_crossfade"
    DISSOLVE = "dissolve"
    BRANDED = "branded"


class TransitionDecision(BaseModel):
    from_scene: str
    to_scene: str
    type: TransitionKind = TransitionKind.HARD_CUT
    duration: float = 0.0
    reason: str = ""
    confidence: float = 1.0
    applied: bool = False


class SoundOperationKind(str, Enum):
    CROSSFADE = "crossfade"
    J_CUT = "j_cut"
    L_CUT = "l_cut"
    AMBIENT_BRIDGE = "ambient_bridge"
    SILENCE_TRIM = "silence_trim"
    LOUDNESS_SMOOTH = "loudness_smooth"
    NONE = "none"


class SoundOperation(BaseModel):
    scene_id: str
    boundary_with: str | None = None  # neighbor scene id this operation relates to, if any
    type: SoundOperationKind = SoundOperationKind.NONE
    params: dict = Field(default_factory=dict)
    reason: str = ""
    applied: bool = False
