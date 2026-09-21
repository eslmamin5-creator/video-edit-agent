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
from video_edit_agent.direction.hierarchy import VisualHierarchy, build_hierarchy
from video_edit_agent.direction.history import FatigueReading, TreatmentHistory
from video_edit_agent.direction.reset_grammar import CameraStory, camera_story
from video_edit_agent.direction.suitability import (
    BehindSubjectEvidence,
    Suitability,
    assess_behind_subject,
)
from video_edit_agent.direction.transitions import (
    DIRECT,
    TransitionChoice,
    TransitionReason,
    TransitionStyle,
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
LOW_CONFIDENCE = 0.45  # a semantic classification below this is a guess: it never buys a treatment
HIGH_CONFIDENCE = 0.7
STRONG_AFTER_STRONG_IMPORTANCE = 0.75  # a beat this important may still follow a strong one
_ZOOMING = frozenset({SpeakerTreatment.PUNCH_IN.value, SpeakerTreatment.SLOW_PUSH.value, SpeakerTreatment.REFRAME.value})
_EMPHASIS_ON_SPEAKER = _ZOOMING | {OverlayTreatment.BEHIND_SUBJECT_TEXT.value}  # emphasis laid over the speaker
_MEANINGFUL_KINDS = frozenset({"key_claim", "payoff", "contrast", "process_list", "topic_shift"})


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
    transition_style: TransitionStyle | None = None  # an explicit style intent for the boundary (e.g. light_leak)
    sound_intent: SoundIntent | None = None  # None = the event's default
    sound_locked: bool = False  # the reviewer chose the sound intent explicitly
    evidence: BehindSubjectEvidence | None = None  # measured facts for the behind-subject gate
    # ---- planning context (all optional; a beat supplied as plain data works as before) ----
    semantic_kind: str | None = None  # from the semantic beat detector
    semantic_confidence: float | None = None
    semantic_cues: list[str] = Field(default_factory=list)
    speaker_motion: float = 0.0  # 0..1, how much the speaker visibly moves (unknown = 0)
    pinned: bool = False  # the reviewer already decided this beat: history records it, nothing overrides it

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
    # ---- semantic / history / reset / hierarchy planning metadata ----
    semantic_kind: str | None = None
    semantic_confidence: float | None = None
    semantic_cues: list[str] = Field(default_factory=list)
    fatigue: FatigueReading | None = None
    variation_desirable: bool = False
    variation_reason: str = ""
    camera_story: CameraStory | None = None
    hierarchy: VisualHierarchy | None = None
    pinned: bool = False

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


def _semantic_reason(beat: Beat) -> bool:
    """Does this beat mean something a treatment could serve? Never true for a low-confidence guess."""
    if beat.semantic_confidence is not None and beat.semantic_confidence < LOW_CONFIDENCE:
        return False
    if beat.semantic_kind is not None:
        return beat.semantic_kind in _MEANINGFUL_KINDS
    return beat.kind is not BeatKind.PLAIN and beat.kind is not BeatKind.PERSONAL


def _alternatives_in_reach(
    treatment: str, alternatives: list[str], beat: Beat, history: TreatmentHistory, *, replacement_blocked: bool, semantic: bool,
) -> list[str]:
    """Alternatives that could really be executed now: in the vocabulary, needing no external asset
    and no generation, not blocked by the replacement-spacing rule and not a recent repeat."""
    pool = list(alternatives)
    if semantic and treatment == SpeakerTreatment.SPEAKER_STATIC.value:
        pool.append(SpeakerTreatment.PUNCH_IN.value)  # a beat that means something can be marked by the framing
    out: list[str] = []
    for alt in dict.fromkeys(pool):
        if alt == treatment or not is_vocabulary(alt) or alt in ASSET_TREATMENTS or alt in GENERATED_TREATMENTS:
            continue
        if treatment_class(alt) == "replacement" and replacement_blocked:
            continue
        if alt in _ZOOMING and history.repetitions(alt):
            continue
        out.append(alt)
    return out


def _soften(beat: Beat, treatment: str, history: TreatmentHistory, reserved: tuple[float, ...] = (), min_gap_s: float = 0.0) -> str | None:
    """Why `treatment` should give way to the calmer speaker shot, given what the viewer just saw
    (None = it stands). A treatment the reviewer chose is never softened."""
    if beat.prefer or beat.pinned:
        return None
    if treatment in _ZOOMING and (ahead := next((r for r in reserved if 0.0 < r - beat.start < min_gap_s), None)) is not None:
        return f"the reviewer's own camera move starts {ahead - beat.start:.1f}s later: two moves this close would crowd it"
    if treatment in _ZOOMING and (n := history.repetitions(treatment)):
        return f"'{treatment}' was already used {n}x in the last few beats: a repeated camera emphasis wears thin"
    static = SpeakerTreatment.SPEAKER_STATIC.value
    if beat.semantic_confidence is not None and beat.semantic_confidence < LOW_CONFIDENCE and treatment != static:
        return f"the classification is low-confidence ({beat.semantic_confidence:.2f}): a guess never buys a treatment"
    prev = history.previous
    if prev is not None and prev.strong and treatment in _EMPHASIS_ON_SPEAKER and beat.importance < STRONG_AFTER_STRONG_IMPORTANCE:
        return "the previous beat already used strong emphasis and this one is not important enough to follow it at the same strength"
    return None


def _variation_text(d: VisualDecision, reading: FatigueReading, wanted: str) -> str:
    """The reviewer-facing sentence for the soft fatigue reading, including what was (not) done about it."""
    applied = d.treatment != SpeakerTreatment.SPEAKER_STATIC.value or d.camera is not CameraMove.STATIC
    if not reading.variation_desirable:
        if applied:  # the reading was about a fresh look; this beat already carries one
            return f"no further variation needed: this beat already carries {d.treatment}" + (" (your choice)" if d.pinned else "")
        return reading.variation_reason
    if applied:
        tail = f" -> applied: {d.treatment}"
    elif d.alternatives:
        tail = f" -> advisory only, not applied (options: {', '.join(d.alternatives[:3])})"
    else:
        tail = " -> advisory only, not applied"
    return reading.variation_reason + tail + (f" [{wanted} was softened]" if wanted != d.treatment else "")


def direct(
    beats: list[Beat],
    *,
    camera_policy: CameraPolicy | None = None,
    sound_profile: SoundProfile | None = None,
    registry: SfxRegistry | None = None,
) -> DirectionResult:
    """Directs `beats` (time-ordered): treatment, camera, transition, caption behaviour and sound.

    Beat by beat it remembers what the viewer has just seen (`TreatmentHistory`): a repeated camera
    emphasis, or a strong beat right after a strong beat, is softened to the speaker, and a SOFT
    fatigue reading (`variation_desirable`) is recorded. Nothing is forced because time passed.
    Every decision also carries its camera story (incoming state, event, release) and hierarchy.

    Deterministic: the same beats, policy, profile and registry give the same result."""
    ordered = sorted(beats, key=lambda b: b.start)
    decisions: list[VisualDecision] = []
    requests: list[CameraRequest] = []
    history = TreatmentHistory(origin=ordered[0].start if ordered else 0.0)
    replacements_in_a_row = 0
    static = SpeakerTreatment.SPEAKER_STATIC.value
    # camera moves the reviewer chose come first: a proposal right before one gives way to it
    reserved = tuple(b.start for b in ordered if b.pinned and b.prefer in _CAMERA_FOR)
    gap = (camera_policy or CameraPolicy()).min_gap_s
    for beat in ordered:
        treatment, reason, alternatives, verdict = _choose(beat)
        wanted = treatment
        notes: list[str] = []
        semantic = _semantic_reason(beat)
        soft = _soften(beat, treatment, history, reserved, gap)
        if soft:
            notes.append(f"{treatment} not used: {soft}")
            alternatives = [treatment, *[a for a in alternatives if a != treatment]]
            treatment, reason = static, f"kept on the speaker ({soft})"
        klass = treatment_class(treatment)
        blocked = replacements_in_a_row >= MAX_CONSECUTIVE_REPLACEMENTS
        if klass == "replacement" and blocked:
            notes.append(f"{treatment} dropped: the speaker must return before another replacement")
            alternatives = [treatment, *[a for a in alternatives if a != treatment]]
            treatment, reason, klass = static, "kept on the speaker (replacement spacing rule)", "speaker"
        replacements_in_a_row = replacements_in_a_row + 1 if klass == "replacement" and not default_speaker_visible(treatment) else 0
        reach = _alternatives_in_reach(treatment, alternatives, beat, history, replacement_blocked=blocked, semantic=semantic)
        reading = history.read(beat.start, wanted, semantic_reason=semantic, alternative_available=bool(reach), speaker_motion=beat.speaker_motion)
        d = VisualDecision(
            start=beat.start, end=beat.end, treatment=treatment, speaker_visible=default_speaker_visible(treatment),
            reason=reason, alternatives=[a for a in dict.fromkeys([*alternatives, *reach]) if a != treatment],
            needs_asset=treatment in ASSET_TREATMENTS, needs_generation=treatment in GENERATED_TREATMENTS,
            notes=notes, beat_kind=beat.kind.value, suitability=verdict,
            semantic_kind=beat.semantic_kind, semantic_confidence=beat.semantic_confidence, semantic_cues=list(beat.semantic_cues),
            fatigue=reading, variation_desirable=reading.variation_desirable, pinned=beat.pinned,
        )
        # camera for this beat: absolute, spaced, reset after emphasis. plan_camera is sequential, so
        # the plan of the beats so far is the prefix of the final plan.
        if d.speaker_visible or treatment in _CAMERA_FOR:
            requests.append(CameraRequest(start=d.start, end=d.end, move=_CAMERA_FOR.get(treatment, CameraMove.STATIC), importance=beat.importance))
        wanted_move = _CAMERA_FOR.get(treatment)
        if d.klass != "replacement" and d.speaker_visible and wanted_move is not None:
            so_far = plan_camera(requests, camera_policy)
            kept = next((e.move for e in so_far.events if e.beat_start is not None and abs(e.beat_start - d.start) < 1e-3
                         and e.move not in {CameraMove.RESET_TO_BASE, CameraMove.PUNCH_OUT}), None)
            if kept is not None:
                d.camera = kept
            else:
                drop = next((x for x in so_far.dropped if abs(x.start - d.start) < 1e-3), None)
                d.notes.append(f"camera move dropped: {drop.reason if drop else 'no room in the camera plan'}")
                d.treatment, d.reason = static, d.reason + " (camera density rule: stay static)"
        d.variation_reason = _variation_text(d, reading, wanted)
        history.record(d.start, d.end, d.treatment, d.camera.value, d.speaker_visible, pinned=beat.pinned)
        decisions.append(d)

    camera = plan_camera(requests, camera_policy)
    for d in decisions:
        replaced = d.klass == "replacement" and not d.speaker_visible
        d.camera_story = camera_story(camera, start=d.start, end=d.end, move=d.camera, replaced=replaced)
        d.hierarchy = build_hierarchy(d.treatment, d.speaker_visible, gate_passed=d.suitability.suitable if d.suitability else None)
        d.caption_behavior = d.hierarchy.caption_role

    # ---- transitions: direct cut unless justified --------------------------------------
    last_stylized: float | None = None
    for d, b in zip(decisions, ordered, strict=True):
        replaced = d.klass == "replacement" and not d.speaker_visible
        if not replaced and b.transition_style is None:
            continue
        choice = select_transition(b.transition_reason, b.transition_style, at=d.start, previous_stylized_at=last_stylized)
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
    "HIGH_CONFIDENCE", "LOW_CONFIDENCE", "MAX_CONSECUTIVE_REPLACEMENTS", "PUNCH_IN_EVENT_S", "Beat", "BeatKind", "DirectionResult", "VisualDecision",
    "availability", "coerce_intent", "direct", "sound_event_for",
]
