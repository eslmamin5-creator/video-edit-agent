"""Reference-derived visual direction: a constrained, deterministic edit director.

The AI proposes from a small closed vocabulary (`vocabulary`), deterministic code
executes it (`camera`, `replacement`, `transitions`, `speed`), and gates reject
choices that are unproven or too dense (`suitability`, spacing rules). Sound lives
in the sibling `video_edit_agent.sound` package.
"""
from video_edit_agent.direction.camera import CameraMove, CameraPlan, CameraPolicy, plan_camera
from video_edit_agent.direction.director import (
    Beat,
    BeatKind,
    DirectionResult,
    VisualDecision,
    direct,
)
from video_edit_agent.direction.replacement import SpeakerReplacement
from video_edit_agent.direction.speed import SpeedEvent, plan_speed
from video_edit_agent.direction.suitability import BehindSubjectEvidence, assess_behind_subject
from video_edit_agent.direction.transitions import TransitionStyle, select_transition
from video_edit_agent.direction.vocabulary import VOCABULARY

__all__ = [
    "VOCABULARY",
    "Beat",
    "BeatKind",
    "BehindSubjectEvidence",
    "CameraMove",
    "CameraPlan",
    "CameraPolicy",
    "DirectionResult",
    "SpeakerReplacement",
    "SpeedEvent",
    "TransitionStyle",
    "VisualDecision",
    "assess_behind_subject",
    "direct",
    "plan_camera",
    "plan_speed",
    "select_transition",
]
