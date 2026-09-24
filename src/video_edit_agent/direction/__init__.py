"""Reference-derived visual direction: a constrained, deterministic edit director.

The AI proposes from a small closed vocabulary (`vocabulary`), deterministic code
executes it (`camera`, `replacement`, `transitions`, `speed`), and gates reject
choices that are unproven or too dense (`suitability`, spacing rules). Sound lives
in the sibling `video_edit_agent.sound` package.
"""
from video_edit_agent.direction.camera import CameraMove, CameraPlan, CameraPolicy, plan_camera
from video_edit_agent.direction.camera_timeline import CameraTimeline, build_camera_timeline
from video_edit_agent.direction.director import (
    Beat,
    BeatKind,
    DirectionResult,
    VisualDecision,
    direct,
)
from video_edit_agent.direction.hierarchy import VisualHierarchy, build_hierarchy
from video_edit_agent.direction.history import FatigueReading, TreatmentHistory
from video_edit_agent.direction.replacement import SpeakerReplacement
from video_edit_agent.direction.reset_grammar import CameraStory, camera_story
from video_edit_agent.direction.rhythm import (
    RhythmPlan,
    RhythmPolicy,
    RhythmRow,
    RhythmState,
    plan_rhythm,
    semantic_enhancement_options,
)
from video_edit_agent.direction.semantic_beats import (
    SemanticBeat,
    SemanticKind,
    detect_semantic_beats,
    to_director_beats,
)
from video_edit_agent.direction.speed import SpeedEvent, plan_speed
from video_edit_agent.direction.suitability import (
    BehindSubjectEvidence,
    assess_behind_subject,
    behind_subject_eligibility,
)
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
    "CameraStory",
    "CameraTimeline",
    "DirectionResult",
    "FatigueReading",
    "RhythmPlan",
    "RhythmPolicy",
    "RhythmRow",
    "RhythmState",
    "SemanticBeat",
    "SemanticKind",
    "SpeakerReplacement",
    "SpeedEvent",
    "TransitionStyle",
    "TreatmentHistory",
    "VisualDecision",
    "VisualHierarchy",
    "assess_behind_subject",
    "behind_subject_eligibility",
    "build_camera_timeline",
    "build_hierarchy",
    "camera_story",
    "detect_semantic_beats",
    "direct",
    "plan_camera",
    "plan_rhythm",
    "plan_speed",
    "select_transition",
    "semantic_enhancement_options",
    "to_director_beats",
]
