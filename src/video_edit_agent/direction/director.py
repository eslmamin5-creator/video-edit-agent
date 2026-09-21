"""The visual director: a constrained, deterministic edit director.

    Approved Transcript -> Semantic Beats -> Visual Director -> Sound Intent
      -> MasterTimeline -> Deterministic Render -> QA

An AI (or the reviewer) describes each beat as DATA (`Beat`: what kind of idea
it is and how much it matters); this module chooses from the small approved
vocabulary and deterministic code executes it. It contains no language
heuristics, no timestamps and no brand knowledge.

For every beat the first question is *is the speaker still the best visual?*

    yes -> a speaker treatment (static / punch-in / slow push / reframe ...)
    no  -> a speaker REPLACEMENT (motion graphic, typography scene, data scene,
           illustration, footage the project already has ...) while the voice
           continues; the simplest visual that carries the idea wins.

Generated visuals are never chosen by default (they are listed as an
alternative and always need explicit approval), behind-subject text only when
the editorial suitability gate passes, and the camera/transition/sound
grammars below drop anything that would be too dense or unjustified.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from video_edit_agent.direction.camera import (
    CameraMove,
    CameraPlan,
    CameraPolicy,
    CameraRequest,
    plan_camera,
)
from video_edit_agent.direction.suitability import (
    BehindSubjectEvidence,
    Suitability,
    assess_behind_subject,
)
from video_edit_agent.direction.transitions import (
    DIRECT,
    TransitionChoice,
    TransitionReason,
    select_transition,
)
from video_edit_agent.direction.vocabulary import (
    ASSET_TREATMENTS,
    GENERATED_TREATMENTS,
    STRONG_PRIMARY,
    OverlayTreatment,
    ReplacementTreatment,
    SpeakerTreatment,
    default_speaker_visible,
    is_vocabulary,
    treatment_class,
)
from video_edit_agent.sound.intent import DEFAULT_IMPORTANCE, EventType, SoundIntent, parse_intent
from video_edit_agent.sound.planner import SoundDecision, VisualEvent, plan_sound, sfx_availability
from video_edit_agent.sound.profile import SoundProfile
from video_edit_agent.sound.registry import SfxRegistry

PUNCH_IN_EVENT_S = 0.42  # the ease length the sound is aligned to
MIN_PUSH_BEAT_S = 6.0  # a static beat at least this long may get a slow push
MAX_CONSECUTIVE_REPLACEMENTS = 1  # the speaker comes back before the next replacement


class BeatKind(str, Enum):
    PLAIN = "plain"  # ordinary delivery: the speaker is the visual
    PERSONAL = "personal"  # an emotional/personal moment: the speaker is the visual
    EMPHASIS = "emphasis"  # a key claim or punchline
    CONCEPT = "concept"  # an abstract idea with no literal picture
    CONCRETE = "concrete"  # a tangible thing/place/action that could be shown
    DATA = "data"  # a number, comparison or statistic
    KEY_PHRASE = "key_phrase"  # a phrase worth reading (the wording IS the point)
    TOPIC_SHIFT = "topic_shift"  # a new section begins


class Beat(BaseModel):
    """One semantic beat, described by an AI or the reviewer. Data in, decision out."""

    start: float
    end: float
    kind: BeatKind = BeatKind.PLAIN
    importance: float = 0.5  # 0..1
    text: str = ""  # the approved spoken text (never rewritten here)
    visual_concept: str | None = None
    has_footage: bool = False  # real/user footage that fits is already available
    prefer: str | None = None  # a vocabulary treatment the reviewer/AI proposes
    transition_reason: TransitionReason | None = None
    sound_intent: SoundIntent | None = None  # None = the event's default
    sound_locked: bool = False  # the reviewer chose the sound intent explicitly
    evidence: BehindSubjectEvidence | None = None  # measured facts for the behind-subject gate

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


class VisualDecision(BaseModel):
    start: float
    end: float
    treatment: str  # from the vocabulary
    speaker_visible: bool = True
    camera: CameraMove = CameraMove.STATIC
    transition: TransitionChoice = Field(default_factory=lambda: DIRECT.model_copy())
    caption_behavior: str = "normal"  # normal | reduced (never hidden)
    sound: SoundDecision | None = None
    reason: str = ""
    alternatives: list[str] = Field(default_factory=list)
    needs_asset: bool = False
    needs_generation: bool = False
    notes: list[str] = Field(default_factory=list)
    beat_kind: str = BeatKind.PLAIN.value
    suitability: Suitability | None = None

    @property
    def klass(self) -> str:
        return treatment_class(self.treatment)


class DirectionResult(BaseModel):
    decisions: list[VisualDecision] = Field(default_factory=list)
    camera: CameraPlan = Field(default_factory=CameraPlan)
    sound: list[SoundDecision] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Choosing the treatment
# --------------------------------------------------------------------------

_ALT_CONCEPT = [ReplacementTreatment.ILLUSTRATION.value, ReplacementTreatment.KINETIC_TYPOGRAPHY.value, SpeakerTreatment.SPEAKER_STATIC.value]
_ALT_TEXT = [ReplacementTreatment.KINETIC_TYPOGRAPHY.value, SpeakerTreatment.PUNCH_IN.value, SpeakerTreatment.SPEAKER_STATIC.value]


def _choose(beat: Beat) -> tuple[str, str, list[str], Suitability | None]:
    """(treatment, reason, alternatives, behind-subject verdict) for one beat."""
    verdict: Suitability | None = None
    kind = beat.kind
    if beat.prefer:
        if not is_vocabulary(beat.prefer):
            return (SpeakerTreatment.SPEAKER_STATIC.value,
                    f"'{beat.prefer}' is not in the approved vocabulary: the speaker stays", [], None)
        return beat.prefer, "proposed by the reviewer/AI", [], None
    if kind in {BeatKind.PLAIN, BeatKind.PERSONAL}:
        long_enough = beat.duration >= MIN_PUSH_BEAT_S and kind is BeatKind.PERSONAL
        chosen = SpeakerTreatment.SLOW_PUSH.value if long_enough else SpeakerTreatment.SPEAKER_STATIC.value
        why = "the speaker is the best visual for this beat" + (" (a slow push holds a long personal moment)" if long_enough else "")
        return chosen, why, [], None
    if kind is BeatKind.EMPHASIS:
        if beat.evidence is not None:
            verdict = assess_behind_subject(beat.evidence)
            if verdict.suitable:
                return OverlayTreatment.BEHIND_SUBJECT_TEXT.value, "a key word that passes the editorial behind-subject gate", _ALT_TEXT, verdict
        why = "a key claim: the speaker stays, the framing emphasises it"
        if verdict is not None:
            why += f" (behind-subject not used: {verdict.summary})"
        elif beat.evidence is None:
            why += " (behind-subject not used: no evidence it would work here)"
        return SpeakerTreatment.PUNCH_IN.value, why, [ReplacementTreatment.FULL_SCREEN_TEXT_SCENE.value, ReplacementTreatment.KINETIC_TYPOGRAPHY.value], verdict
    if kind is BeatKind.KEY_PHRASE:
        return (ReplacementTreatment.FULL_SCREEN_TEXT_SCENE.value, "the wording itself is the point: read it, full screen",
                [OverlayTreatment.BEHIND_SUBJECT_TEXT.value, ReplacementTreatment.KINETIC_TYPOGRAPHY.value, SpeakerTreatment.PUNCH_IN.value], None)
    if kind is BeatKind.DATA:
        return (ReplacementTreatment.GRAPHIC_DATA_SCENE.value, "numbers read better as a graphic than as footage",
                [ReplacementTreatment.MOTION_GRAPHIC.value, SpeakerTreatment.SPEAKER_STATIC.value], None)
    if kind is BeatKind.CONCEPT:
        return (ReplacementTreatment.MOTION_GRAPHIC.value, "an abstract idea: the simplest graphic beats forced literal footage", _ALT_CONCEPT, None)
    if kind is BeatKind.TOPIC_SHIFT:
        return (ReplacementTreatment.MOTION_GRAPHIC.value, "a new section: a short graphic marks the change",
                [SpeakerTreatment.REFRAME.value, SpeakerTreatment.SPEAKER_STATIC.value], None)
    if kind is BeatKind.CONCRETE:
        if beat.has_footage:
            return (ReplacementTreatment.REAL_BROLL.value, "a tangible subject and matching footage is available",
                    [ReplacementTreatment.MOTION_GRAPHIC.value, SpeakerTreatment.SPEAKER_STATIC.value], None)
        return (SpeakerTreatment.SPEAKER_STATIC.value, "a tangible subject, but no footage exists: stay on the speaker (generation would need your explicit approval)",
                [ReplacementTreatment.USER_BROLL.value, ReplacementTreatment.ILLUSTRATION.value, ReplacementTreatment.GENERATED_VISUAL.value], None)
    return SpeakerTreatment.SPEAKER_STATIC.value, "the speaker is the best visual", [], None


_CAMERA_FOR = {
    SpeakerTreatment.PUNCH_IN.value: CameraMove.PUNCH_IN,
    SpeakerTreatment.SLOW_PUSH.value: CameraMove.SLOW_PUSH,
    SpeakerTreatment.REFRAME.value: CameraMove.REFRAME_RIGHT,
    SpeakerTreatment.PUNCH_OUT.value: CameraMove.PUNCH_OUT,
    SpeakerTreatment.RESET_TO_BASE.value: CameraMove.RESET_TO_BASE,
}


def sound_event_for(
    treatment: str, camera: CameraMove, speaker_visible: bool, start: float, end: float, ident: str,
    *, intent: SoundIntent | None = None, importance: float | None = None, locked: bool = False,
) -> VisualEvent | None:
    """The visual event of a directed beat that could carry a sound, or None when nothing
    visible happens on it (a static speaker shot has no event)."""
    def weight(kind: EventType) -> float | None:
        # a beat can raise an event's semantic weight, never lower it below the event's own default
        return None if importance is None else max(importance, DEFAULT_IMPORTANCE[kind])

    if treatment_class(treatment) == "replacement" and not speaker_visible:
        return VisualEvent(id=ident, type=EventType.REPLACEMENT_IN, start=start, duration=end - start,
                           intent=intent, importance=weight(EventType.REPLACEMENT_IN), user_locked=locked)
    if treatment in STRONG_PRIMARY:
        return VisualEvent(id=ident, type=EventType.KEY_REVEAL, start=start, duration=min(0.6, end - start),
                           intent=intent, importance=weight(EventType.KEY_REVEAL), user_locked=locked)
    if camera is CameraMove.STATIC:
        return None
    kind = {CameraMove.PUNCH_IN: EventType.PUNCH_IN, CameraMove.SLOW_PUSH: EventType.SLOW_PUSH,
            CameraMove.PUNCH_OUT: EventType.PUNCH_OUT, CameraMove.RESET_TO_BASE: EventType.RESET_TO_BASE}.get(camera, EventType.REFRAME)
    return VisualEvent(id=ident, type=kind, start=start, duration=PUNCH_IN_EVENT_S,
                       intent=intent, importance=weight(kind), user_locked=locked)


def _sound_event(decision: VisualDecision, beat: Beat, index: int) -> VisualEvent | None:
    return sound_event_for(
        decision.treatment, decision.camera, decision.speaker_visible, decision.start, decision.end, f"beat{index}",
        intent=beat.sound_intent, importance=beat.importance, locked=beat.sound_locked,
    )


def direct(
    beats: list[Beat],
    *,
    camera_policy: CameraPolicy | None = None,
    sound_profile: SoundProfile | None = None,
    registry: SfxRegistry | None = None,
) -> DirectionResult:
    """Directs `beats` (time-ordered): treatment, camera, transition, caption behaviour and sound.

    Deterministic: the same beats, policy, profile and registry give the same result."""
    ordered = sorted(beats, key=lambda b: b.start)
    decisions: list[VisualDecision] = []
    replacements_in_a_row = 0
    for beat in ordered:
        treatment, reason, alternatives, verdict = _choose(beat)
        notes: list[str] = []
        klass = treatment_class(treatment)
        if klass == "replacement" and replacements_in_a_row >= MAX_CONSECUTIVE_REPLACEMENTS:
            notes.append(f"{treatment} dropped: the speaker must return before another replacement")
            alternatives = [treatment, *[a for a in alternatives if a != treatment]]
            treatment, reason, klass = SpeakerTreatment.SPEAKER_STATIC.value, "kept on the speaker (replacement spacing rule)", "speaker"
        replacements_in_a_row = replacements_in_a_row + 1 if klass == "replacement" and not default_speaker_visible(treatment) else 0
        decisions.append(VisualDecision(
            start=beat.start, end=beat.end, treatment=treatment, speaker_visible=default_speaker_visible(treatment),
            reason=reason, alternatives=[a for a in alternatives if a != treatment],
            needs_asset=treatment in ASSET_TREATMENTS, needs_generation=treatment in GENERATED_TREATMENTS,
            caption_behavior="reduced" if treatment in STRONG_PRIMARY else "normal", notes=notes,
            beat_kind=beat.kind.value, suitability=verdict,
        ))

    # ---- camera: absolute, spaced, reset after emphasis --------------------------------
    requests = [
        CameraRequest(start=d.start, end=d.end, move=_CAMERA_FOR.get(d.treatment, CameraMove.STATIC), importance=b.importance)
        for d, b in zip(decisions, ordered, strict=True) if d.speaker_visible or d.treatment in _CAMERA_FOR
    ]
    camera = plan_camera(requests, camera_policy)
    dropped = {(round(x.start, 3)): x for x in camera.dropped}
    kept = {round(e.beat_start, 3): e.move for e in camera.events if e.beat_start is not None and e.move not in {CameraMove.RESET_TO_BASE, CameraMove.PUNCH_OUT}}
    for d in decisions:
        if not d.speaker_visible or d.klass == "replacement":
            continue
        wanted = _CAMERA_FOR.get(d.treatment)
        if wanted is None:
            continue
        move = kept.get(round(d.start, 3))
        if move is None:
            drop = dropped.get(round(d.start, 3))
            d.notes.append(f"camera move dropped: {drop.reason if drop else 'no room in the camera plan'}")
            d.treatment, d.reason = SpeakerTreatment.SPEAKER_STATIC.value, d.reason + " (camera density rule: stay static)"
        else:
            d.camera = move

    # ---- transitions: direct cut unless justified --------------------------------------
    last_stylized: float | None = None
    for d, b in zip(decisions, ordered, strict=True):
        if d.klass != "replacement" or d.speaker_visible:
            continue
        choice = select_transition(b.transition_reason, at=d.start, previous_stylized_at=last_stylized)
        d.transition = choice
        if choice.stylized:
            last_stylized = d.start

    # ---- sound: intent -> profile -> registry ------------------------------------------
    profile = sound_profile or _silent_profile()
    reg = registry or SfxRegistry()
    events: list[VisualEvent] = []
    owners: dict[str, VisualDecision] = {}
    for i, (d, b) in enumerate(zip(decisions, ordered, strict=True)):
        ev = _sound_event(d, b, i)
        if ev is not None:
            events.append(ev)
            owners[ev.id] = d
    sound = plan_sound(events, profile, reg)
    for s in sound:
        owners[s.event_id].sound = s
    return DirectionResult(decisions=decisions, camera=camera, sound=sound)


def _silent_profile() -> SoundProfile:
    from video_edit_agent.sound.profile import get_profile

    return get_profile("none")


def availability(decision: VisualDecision) -> str:
    return sfx_availability(decision.sound) if decision.sound else "none"


def coerce_intent(value: str | None) -> SoundIntent:
    return parse_intent(value)


__all__ = [
    "MAX_CONSECUTIVE_REPLACEMENTS", "PUNCH_IN_EVENT_S", "Beat", "BeatKind", "DirectionResult", "VisualDecision",
    "availability", "coerce_intent", "direct", "sound_event_for",
]
