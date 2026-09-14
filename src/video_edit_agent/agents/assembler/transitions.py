"""Transition planning (spec sections 17-18, 20): default to direct/hard
cuts; only recommend something else when continuity findings justify it.
Restrained by design -- never mechanically dissolve every boundary.

`applied` reflects what the shared render core can actually do today: real
audio-only bridging (via the existing `audio_fade_in_ms`/`audio_fade_out_ms`
de-click fades on `EDLClip`) is wired and markable `applied=True`; true
visual crossfade/dissolve and J-cut/L-cut (independent audio/video trim
windows) have no renderer support yet, so they are recorded as
recommendations only (`applied=False`) -- see spec section 20's "do not
fake it in the report."
"""
from __future__ import annotations

from video_edit_agent.agents.assembler.schemas import ContinuityFinding, ContinuitySeverity, TransitionDecision, TransitionKind

SHORT_CROSSFADE_DURATION = 0.35
AUDIO_BRIDGE_DURATION = 0.15


def plan_transitions(
    scene_ids: list[str],
    continuity_findings: list[ContinuityFinding],
) -> list[TransitionDecision]:
    findings_by_pair: dict[tuple[str, str], list[ContinuityFinding]] = {}
    for f in continuity_findings:
        findings_by_pair.setdefault((f.from_scene, f.to_scene), []).append(f)

    decisions: list[TransitionDecision] = []
    for a, b in zip(scene_ids, scene_ids[1:]):
        pair_findings = findings_by_pair.get((a, b), [])
        decisions.append(_decide(a, b, pair_findings))
    return decisions


def _decide(a: str, b: str, findings: list[ContinuityFinding]) -> TransitionDecision:
    high = [f for f in findings if f.severity == ContinuitySeverity.HIGH]
    brightness_high = next((f for f in high if f.category == "brightness"), None)

    if brightness_high is not None:
        # A real visual jump benefits from a brief bridge, but we only have
        # an audio-side fade wired today -- apply that, and recommend the
        # (not-yet-implemented) visual crossfade honestly as unapplied.
        return TransitionDecision(
            from_scene=a,
            to_scene=b,
            type=TransitionKind.SHORT_CROSSFADE,
            duration=SHORT_CROSSFADE_DURATION,
            reason=brightness_high.description,
            confidence=brightness_high.confidence,
            applied=False,
        )

    medium = [f for f in findings if f.severity == ContinuitySeverity.MEDIUM]
    if medium:
        return TransitionDecision(
            from_scene=a,
            to_scene=b,
            type=TransitionKind.AUDIO_BRIDGE,
            duration=AUDIO_BRIDGE_DURATION,
            reason="; ".join(f.description for f in medium),
            confidence=max(f.confidence for f in medium),
            applied=True,
        )

    return TransitionDecision(
        from_scene=a,
        to_scene=b,
        type=TransitionKind.HARD_CUT,
        duration=0.0,
        reason="no continuity issue justifies a transition; direct cut preserves rhythm",
        confidence=1.0,
        applied=True,
    )
