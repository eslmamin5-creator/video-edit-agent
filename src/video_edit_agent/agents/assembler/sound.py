"""Sound Director (spec section 19): per-boundary audio operations. Never
auto-introduces generated music -- only reacts to what's already there
(silence, abrupt starts/stops implied by the transition plan). J-cut/L-cut
are recorded as designed-but-unapplied (spec section 20) since they need
independent audio/video trim windows the current single-shared-trim-window
`EDLClip` doesn't support at render time yet.
"""
from __future__ import annotations

from video_edit_agent.agents.assembler.schemas import (
    SceneAnalysis,
    SoundOperation,
    SoundOperationKind,
    TransitionDecision,
    TransitionKind,
)

_TRANSITION_TO_SOUND = {
    TransitionKind.SHORT_CROSSFADE: SoundOperationKind.CROSSFADE,
    TransitionKind.AUDIO_BRIDGE: SoundOperationKind.AMBIENT_BRIDGE,
    TransitionKind.DISSOLVE: SoundOperationKind.CROSSFADE,
}


def plan_sound(analyses: list[SceneAnalysis], transitions: list[TransitionDecision]) -> list[SoundOperation]:
    ops: list[SoundOperation] = []

    for t in transitions:
        kind = _TRANSITION_TO_SOUND.get(t.type)
        if kind is None:
            continue
        ops.append(
            SoundOperation(
                scene_id=t.from_scene,
                boundary_with=t.to_scene,
                type=kind,
                params={"duration": t.duration},
                reason=t.reason,
                applied=t.applied,
            )
        )

    for a in analyses:
        if a.has_audio and a.is_silent:
            ops.append(
                SoundOperation(
                    scene_id=a.scene_id,
                    boundary_with=None,
                    type=SoundOperationKind.SILENCE_TRIM,
                    params={},
                    reason="scene audio track is effectively silent throughout",
                    applied=False,
                )
            )

    return ops
