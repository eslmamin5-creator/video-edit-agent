"""MasterTimeline (Phase 2 section 5): the common, engine-agnostic timeline
representation shared by all three workflows -- Editor, Creator, Assembler --
and consumed by every downstream stage (captions, motion, B-roll, audio,
brand, subject compositing, rendering, QA).

The Editor's existing `EDL` (`core/schemas.py`) is NOT replaced. It remains
the source of truth for Editor projects because Editor's render pipeline
(`render/composition.py::build_filter_complex`) is built directly around it
and rewriting that pipeline is explicitly out of scope for this migration
(spec Phase 2 section 4: "Preserve the current Editor" / "Do NOT rewrite it
simply to match a new architecture"). Instead, `edl_to_master_timeline()`
below is a one-way adapter: every Editor render still produces a normal EDL,
and a MasterTimeline is derived from it as an additional, parallel artifact
(`edit/master_timeline.json`) that Creator and Assembler can also produce
directly, without going through an EDL at all.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import EDL, TransitionType


class TrackType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    VOICE_OVER = "voice_over"
    MUSIC = "music"
    SFX = "sfx"
    CAPTIONS = "captions"
    BROLL = "broll"
    MOTION_GRAPHICS = "motion_graphics"
    OVERLAYS = "overlays"
    TRANSITIONS = "transitions"
    SUBJECT_AWARE = "subject_aware"


class ProviderKind(str, Enum):
    """Where an asset originated (spec section 41: provenance)."""

    USER = "user"
    LOCAL_LIBRARY = "local_library"
    GEMINI = "gemini"
    VEO = "veo"
    ELEVENLABS = "elevenlabs"
    REMOTION = "remotion"
    HYPERFRAMES = "hyperframes"
    MANIM = "manim"
    SIMPLE = "simple"
    OTHER = "other"


class Provenance(BaseModel):
    kind: ProviderKind = ProviderKind.USER
    detail: Optional[str] = None  # e.g. model name, prompt id, source filename


class TransitionInfo(BaseModel):
    type: TransitionType = TransitionType.HARD_CUT
    duration: float = 0.0
    reasoning: Optional[str] = None  # Transition Director's recorded reasoning (section 20)


class BrandInfo(BaseModel):
    brand_name: Optional[str] = None
    applied: bool = False


class QAState(BaseModel):
    checked: bool = False
    issues: list[str] = Field(default_factory=list)


class TimelineItem(BaseModel):
    """One item on the MasterTimeline (spec Phase 2 section 5)."""

    id: str
    type: TrackType
    start: float
    duration: float
    source: str
    in_point: float = 0.0
    out_point: float = 0.0
    layer: int = 0
    speaker: Optional[str] = None
    reason: Optional[str] = None
    behind_subject: bool = False
    transition_in: TransitionInfo = Field(default_factory=TransitionInfo)
    transition_out: TransitionInfo = Field(default_factory=TransitionInfo)
    provenance: Provenance = Field(default_factory=Provenance)
    brand: BrandInfo = Field(default_factory=BrandInfo)
    qa: QAState = Field(default_factory=QAState)
    metadata: dict = Field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.start + self.duration


class MasterTimeline(BaseModel):
    """Machine-readable, serializable, re-renderable, versioned (spec section
    5): the common production timeline for Editor, Creator and Assembler."""

    version: int = 1
    workflow: str = "editor"  # "editor" | "creator" | "assembler"
    fps: float = 30.0
    width: int = 1080
    height: int = 1920
    items: list[TimelineItem] = Field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return max((item.end for item in self.items), default=0.0)


def edl_to_master_timeline(edl: EDL, *, workflow: str = "editor") -> MasterTimeline:
    """Adapter (spec section 6): Editor's EDL -> MasterTimeline. One VIDEO
    item per EDL clip, plus a matching AUDIO item since the EDL's ffmpeg
    filter graph always treats video+audio as one concat unit per clip.
    Overlay/motion/B-roll/caption tracks are not represented here because the
    EDL itself doesn't carry them (they live in separate plan files);
    Creator/Assembler timelines populate those tracks directly.
    """
    items: list[TimelineItem] = []
    for i, clip in enumerate(edl.clips):
        base_metadata = {"speed": clip.speed}
        items.append(
            TimelineItem(
                id=f"edl-video-{i}",
                type=TrackType.VIDEO,
                start=clip.timeline_in,
                duration=clip.duration,
                source=clip.source_file,
                in_point=clip.source_in,
                out_point=clip.source_out,
                layer=0,
                speaker=clip.speaker,
                reason=clip.reason.value,
                transition_in=TransitionInfo(type=clip.transition_in),
                transition_out=TransitionInfo(type=clip.transition_out),
                provenance=Provenance(kind=ProviderKind.USER, detail=clip.source_file),
                metadata=base_metadata,
            )
        )
        items.append(
            TimelineItem(
                id=f"edl-audio-{i}",
                type=TrackType.AUDIO,
                start=clip.timeline_in,
                duration=clip.duration,
                source=clip.source_file,
                in_point=clip.source_in,
                out_point=clip.source_out,
                layer=0,
                speaker=clip.speaker,
                reason=clip.reason.value,
                provenance=Provenance(kind=ProviderKind.USER, detail=clip.source_file),
                metadata={
                    "speed": clip.speed,
                    "audio_fade_in_ms": clip.audio_fade_in_ms,
                    "audio_fade_out_ms": clip.audio_fade_out_ms,
                },
            )
        )

    return MasterTimeline(version=1, workflow=workflow, fps=edl.fps, width=edl.width, height=edl.height, items=items)


def save(timeline: MasterTimeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> MasterTimeline:
    return MasterTimeline.model_validate(json.loads(path.read_text(encoding="utf-8")))
