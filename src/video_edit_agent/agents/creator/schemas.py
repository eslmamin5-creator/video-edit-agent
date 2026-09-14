"""Creator-specific data contracts (Phase 2 Creator milestone).

These are NEW schemas because Creator's intermediate artifacts (script
analysis, scene breakdown, storyboard, asset plan) have no Editor equivalent.
They deliberately do NOT duplicate shared systems: the final output is still
the shared `core.timeline.MasterTimeline`, motion is still requested through
`motion.router.render_motion` via `AnimationSpec`, B-roll is still resolved
through `broll.providers.local.find_broll` via `BrollPlanItem`, and QA still
uses the shared `QAIssue`/`QASeverity`/`QAReport` types.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CreatorStyle(str, Enum):
    MOTION = "motion"
    CINEMATIC = "cinematic"
    INFOGRAPHIC = "infographic"
    MIXED = "mixed"


class ScriptAnalysis(BaseModel):
    title: str = ""
    topic: str = ""
    language: str = "en"  # "ar" | "en" (Unicode-ratio detected)
    target_tone: str = ""
    hook: str = ""
    key_message: str = ""
    supporting_points: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)  # extracted verbatim from the script only
    cta: str = ""
    emotional_beats: list[str] = Field(default_factory=list)
    estimated_target_duration: float = 0.0
    visual_opportunities: list[str] = Field(default_factory=list)


class ScenePurpose(str, Enum):
    HOOK = "hook"
    PROBLEM = "problem"
    EXPLANATION = "explanation"
    STATISTIC = "statistic"
    EXAMPLE = "example"
    SOLUTION = "solution"
    CTA = "cta"
    OTHER = "other"


class Scene(BaseModel):
    id: str
    script_segment: str
    purpose: ScenePurpose = ScenePurpose.OTHER
    estimated_duration: float = 3.0
    visual_type: str = ""  # e.g. "typography" | "broll" | "chart" | "diagram"
    foreground_concept: str = ""
    background_concept: str = ""
    on_screen_text: str = ""  # verbatim script text -- never paraphrased/translated
    motion_need: bool = False
    broll_need: bool = False
    generated_asset_need: bool = False
    voice_over_text: str = ""
    transition_intent: str = "hard_cut"
    brand_constraints: list[str] = Field(default_factory=list)
    provenance_placeholder: Optional[str] = None


class StoryboardFrame(BaseModel):
    scene_id: str
    on_screen: str = ""
    layout: str = ""
    framing: str = ""
    typography: str = ""
    animation_concept: str = ""
    broll_concept: str = ""
    generated_visual_concept: str = ""
    transition: str = "hard_cut"
    audio_intent: str = ""


class Storyboard(BaseModel):
    frames: list[StoryboardFrame] = Field(default_factory=list)


class AssetTreatment(str, Enum):
    TYPOGRAPHY = "typography"
    REMOTION_COMPOSITION = "remotion_composition"
    SIMPLE_GRAPHICS = "simple_graphics"
    MANIM = "manim"
    LOCAL_IMAGE = "local_image"
    LOCAL_BROLL = "local_broll"
    USER_ASSET = "user_asset"
    GENERATED_IMAGE = "generated_image"
    GENERATED_VIDEO = "generated_video"


class AssetPlanItem(BaseModel):
    scene_id: str
    treatment: AssetTreatment = AssetTreatment.TYPOGRAPHY
    asset_path: Optional[str] = None
    provenance: str = "planned"  # updated to actual engine/source once resolved at render time
    offline_safe: bool = True
