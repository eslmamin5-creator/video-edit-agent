"""SpeakerReplacement: the speaker temporarily disappears, the voice does not.

This is a first-class concept, distinct from B-roll. B-roll is only ONE of the
things that may replace the speaker; a motion graphic, diagram, illustration,
typography scene, a data scene or (only with explicit approval) a generated
visual can do the same job, and for an abstract idea the simplest of them is
usually right. The replacement never touches the audio: the speaker's voice
runs straight through it.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from video_edit_agent.core.timeline import Provenance, ProviderKind, TimelineItem, TrackType
from video_edit_agent.direction.vocabulary import (
    ASSET_TREATMENTS,
    GENERATED_TREATMENTS,
    REPLACEMENT_TREATMENTS,
    ReplacementTreatment,
)


class ReplacementSource(str, Enum):
    FOOTAGE = "footage"  # real/stock/user footage that already exists
    AUTHORED = "authored"  # motion graphic, typography, diagram, illustration built by the pipeline
    GENERATED = "generated"  # an AI-generated visual (explicit approval required)


def source_of(kind: str) -> ReplacementSource:
    if kind in ASSET_TREATMENTS:
        return ReplacementSource.FOOTAGE
    if kind in GENERATED_TREATMENTS:
        return ReplacementSource.GENERATED
    return ReplacementSource.AUTHORED


class SpeakerReplacement(BaseModel):
    start: float
    end: float
    kind: str  # a ReplacementTreatment value
    visual_concept: str | None = None
    voice_continues: bool = True
    fallback: str = "speaker_static"  # what plays when the visual is unavailable
    generation_approved: bool = False
    asset_provided: bool = False  # the footage this replacement needs exists
    entry: str = "direct_cut"
    exit: str = "direct_cut"
    notes: list[str] = Field(default_factory=list)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, kind: str) -> str:
        if kind not in REPLACEMENT_TREATMENTS:
            raise ValueError(f"'{kind}' is not a speaker-replacement treatment")
        return kind

    @field_validator("voice_continues")
    @classmethod
    def _voice_never_stops(cls, value: bool) -> bool:
        if not value:
            raise ValueError("a speaker replacement never interrupts the voice")
        return True

    @model_validator(mode="after")
    def _ordered(self) -> SpeakerReplacement:
        if self.end <= self.start:
            raise ValueError("a replacement needs a positive duration")
        return self

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def source(self) -> ReplacementSource:
        return source_of(self.kind)

    @property
    def needs_asset(self) -> bool:
        return self.kind in ASSET_TREATMENTS

    @property
    def needs_generation(self) -> bool:
        return self.kind in GENERATED_TREATMENTS

    @property
    def executable(self) -> bool:
        """False while an external asset is missing or a generation is not approved (the fallback plays)."""
        if self.needs_generation:
            return self.generation_approved
        if self.needs_asset:
            return self.asset_provided
        return True

    @property
    def effective(self) -> str:
        return self.kind if self.executable else self.fallback

    def to_timeline_item(self, item_id: str) -> TimelineItem:
        """The replacement as a video-layer item above the speaker. The speaker's VIDEO item and its
        audio are left as they are: the voice continues under the replacement."""
        footage = self.source is ReplacementSource.FOOTAGE
        track = TrackType.BROLL if footage or self.source is ReplacementSource.GENERATED else TrackType.MOTION_GRAPHICS
        provider = {
            ReplacementSource.FOOTAGE: ProviderKind.USER,
            ReplacementSource.GENERATED: ProviderKind.OTHER,
            ReplacementSource.AUTHORED: ProviderKind.SIMPLE,
        }[self.source]
        return TimelineItem(
            id=item_id, type=track, start=self.start, duration=self.duration, source="", layer=2,
            reason=self.visual_concept or self.kind, provenance=Provenance(kind=provider, detail=self.kind),
            metadata={
                "speaker_replacement": True, "voice_continues": True, "kind": self.kind,
                "source": self.source.value, "fallback": self.fallback, "executable": self.executable,
                "entry_transition": self.entry, "exit_transition": self.exit,
            },
        )


__all__ = ["ReplacementSource", "ReplacementTreatment", "SpeakerReplacement", "source_of"]
