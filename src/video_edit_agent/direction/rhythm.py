"""Visual Rhythm Engine: always-on, low-risk framing variation for a talking head.

    Approved transcript -> Visual Rhythm Engine -> Semantic Enhancement Director -> treatment plan -> review

This is the LOW-semantic layer. It keeps the video from sitting on one framing
without inventing meaning: it needs a transcript for phrase boundaries, and
nothing else. Semantic evidence, when there is some, only nudges WHICH state is
picked (a small score term); a beat with no semantic reading gets exactly the
same safe variation. Semantic treatments (headlines, icons, graphics, B-roll,
behind-subject text ...) live in the semantic director, never here: elapsed time
alone can never create one.

Timing is SOFT, not a timer. A refresh is preferred every ~3-5 s and looked at
again around ~6 s, but a state change only ever happens at a safe phrase boundary
(a segment edge, sentence/clause punctuation, a pause, a discourse marker) and
never inside a word. It may come earlier when a natural beat exists and it may run
past the guard when no safe boundary does. Boundary quality, hold length and
history are scored together; the choice is deterministic (no randomness).

Vocabulary: base, punch_in, punch_out, slow_push, slow_pull, reframe_left,
reframe_right, lower_subject, raise_subject (only with a measured composition),
reset_to_base, hold. Every framing value is ABSOLUTE (no accumulation). Every
excursion is planned with its return: base -> move -> hold -> reset.

Two boundary classes. ABRUPT moves (a snap punch_in, a cut-style return) are perceptually discrete: they want a
phrase / clause / sentence boundary or a strong pause. SMOOTH moves (slow_push, slow_pull, a gradual reframe,
lower_subject and their glide returns) may begin at any word boundary, still never inside a word. A hold that keeps
extending with no phrase boundary in reach may therefore end with a smooth Tier-1 move; that is a soft rule about
boundaries and evidence, not a timer. A smooth return is itself a gradual motion (about 0.7-1.2 s) that starts before
the return point and settles on the boundary; nothing snaps.

Anti-pattern memory: the last three excursions are remembered. The same state
twice in a row is never chosen, a repeated family / ABAB / identical cadence is
penalised.

Nothing here is brand- or project-specific, and no reference timestamp is used.
"""
from __future__ import annotations

from enum import Enum
from itertools import pairwise

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import DEFAULT_ZOOM_ANCHOR_Y as DEFAULT_ANCHOR_Y
from video_edit_agent.core.schemas import Transcript, Word
from video_edit_agent.direction.camera import (
    BASE_ANCHOR_X,
    BASE_ZOOM,
    CameraEvent,
    CameraMove,
    CameraPlan,
    CameraPolicy,
    EventStatus,
    MotionClass,
)
from video_edit_agent.direction.director import LOW_CONFIDENCE
from video_edit_agent.direction.semantic_beats import _INTERROGATIVES, _PHRASE_CUES, _fold
from video_edit_agent.direction.suitability import MAX_PHRASE_WORDS
from video_edit_agent.render.reframe import face_frame


class RhythmState(str, Enum):
    BASE = "base"
    PUNCH_IN = "punch_in"
    PUNCH_OUT = "punch_out"
    SLOW_PUSH = "slow_push"
    SLOW_PULL = "slow_pull"
    REFRAME_LEFT = "reframe_left"
    REFRAME_RIGHT = "reframe_right"
    LOWER_SUBJECT = "lower_subject"
    RAISE_SUBJECT = "raise_subject"
    RESET_TO_BASE = "reset_to_base"
    HOLD = "hold"


S = RhythmState
VOCABULARY = tuple(s.value for s in RhythmState)

# the states an excursion can START with, in the fixed order used for tie-breaks
EXCURSIONS = (S.SLOW_PUSH, S.REFRAME_LEFT, S.REFRAME_RIGHT, S.PUNCH_IN, S.LOWER_SUBJECT, S.RAISE_SUBJECT)
ABRUPT_STATES = frozenset({S.PUNCH_IN, S.PUNCH_OUT})
SMOOTH_STATES = frozenset({S.SLOW_PUSH, S.SLOW_PULL, S.REFRAME_LEFT, S.REFRAME_RIGHT, S.LOWER_SUBJECT, S.RAISE_SUBJECT, S.RESET_TO_BASE})
EXTENSION_STATES = (S.SLOW_PUSH, S.REFRAME_LEFT, S.REFRAME_RIGHT)  # the smooth Tier-1 moves an extended hold may use
_MOTION = {S.PUNCH_IN, S.PUNCH_OUT, S.SLOW_PUSH, S.SLOW_PULL, S.REFRAME_LEFT, S.REFRAME_RIGHT, S.LOWER_SUBJECT,
           S.RAISE_SUBJECT, S.RESET_TO_BASE}
_STATIC = {S.BASE, S.HOLD}

# family (for "the same kind of move twice"), energy 0..1, chance of crowding the caption band 0..1
_FAMILY = {S.SLOW_PUSH: "zoom", S.PUNCH_IN: "zoom", S.REFRAME_LEFT: "lateral", S.REFRAME_RIGHT: "lateral",
           S.LOWER_SUBJECT: "vertical", S.RAISE_SUBJECT: "vertical"}
_ENERGY = {S.SLOW_PUSH: 0.3, S.REFRAME_LEFT: 0.45, S.REFRAME_RIGHT: 0.45, S.LOWER_SUBJECT: 0.5, S.RAISE_SUBJECT: 0.4, S.PUNCH_IN: 0.9}
_CAPTION_RISK = {S.SLOW_PUSH: 0.1, S.REFRAME_LEFT: 0.1, S.REFRAME_RIGHT: 0.1, S.LOWER_SUBJECT: 0.3, S.RAISE_SUBJECT: 0.0, S.PUNCH_IN: 0.2}
_ENERGY_TARGET = {"low": 0.3, "medium": 0.5, "high": 0.8}

# which state a confident semantic reading of a phrase leans toward (only ever a small score nudge)
_LEANS: dict[str, tuple[S, ...]] = {
    "key_claim": (S.PUNCH_IN,), "payoff": (S.PUNCH_IN, S.SLOW_PUSH), "contrast": (S.REFRAME_LEFT, S.REFRAME_RIGHT),
    "question": (S.SLOW_PUSH,), "topic_shift": (S.LOWER_SUBJECT, S.REFRAME_LEFT, S.REFRAME_RIGHT), "process_list": (S.REFRAME_LEFT, S.REFRAME_RIGHT),
}

_EXCURSION_NAMES = {s.value for s in EXCURSIONS}
SAFE_QUALITY = 0.5  # a boundary below this is a word gap, not a phrase boundary: never a place to start a move
WEIGHTS = {"boundary": 0.28, "timing": 0.22, "novelty": 0.18, "composition": 0.10, "reset": 0.08, "semantic": 0.06,
           "energy": 0.04, "caption": 0.04}


def motion_class(state: str | S) -> MotionClass:
    """abrupt (perceptually discrete) or smooth (gradual) for a rhythm state; holds and base are neither."""
    st = S(state)
    if st in ABRUPT_STATES:
        return MotionClass.ABRUPT
    return MotionClass.SMOOTH if st in SMOOTH_STATES else MotionClass.NONE


class RhythmPolicy(BaseModel):
    """Soft targets (never exposed as rigid timers) and the absolute framing levels."""

    refresh_min_s: float = 3.0  # the preferred refresh window
    refresh_max_s: float = 5.0
    attention_guard_s: float = 6.0  # where the timing preference starts to fall away
    min_hold_s: float = 2.0  # never change again sooner than this after the last change
    after_emphasis_hold_s: float = 2.5  # emphasis -> calm
    horizon_s: float = 10.0  # how far past the last change a start is looked for before the hold just extends
    excursion_min_s: float = 2.0
    excursion_max_s: float = 4.5
    ease_s: float = 0.3
    slow_ease_s: float = 1.2  # a slow_pull runs slower than a snap reset
    occupied_margin_s: float = 1.2  # calm kept before/after a director-owned window
    history: int = 3
    energy: str = "medium"
    # absolute levels (kept under camera.max_zoom)
    punch_in_zoom: float = 1.08
    slow_push_zoom: float = 1.05
    reframe_zoom: float = 1.05
    reframe_shift: float = 0.05
    lower_zoom: float = 1.12
    raise_zoom: float = 1.06
    max_zoom: float = 1.14
    caption_safe_top: float = 0.70  # the caption band starts below this fraction of the frame height
    min_top_margin: float = 0.03  # headroom a raise_subject must leave
    min_headroom_gain: float = 0.03  # a lower_subject must add at least this much face-top space (fraction of frame height)
    plan_raise_subject: bool = False  # raise_subject is planning-only (the renderer cannot pad above the source frame)
    # smooth (gradual) motion
    smooth_entry_s: float = 0.9  # a smooth move glides in over this long
    return_min_s: float = 0.7  # a smooth return runs between these lengths and settles on its boundary
    return_max_s: float = 1.2
    return_glide_s: float = 0.9  # the preferred length inside that range
    smooth_min_quality: float = 0.3  # a smooth move may start on any word boundary (never inside a word)
    extend_after_s: float | None = None  # an extended hold may end with a smooth move this long after the last change (default refresh_max_s)

    def level(self, state: S) -> float:
        z = {S.PUNCH_IN: self.punch_in_zoom, S.SLOW_PUSH: self.slow_push_zoom, S.REFRAME_LEFT: self.reframe_zoom,
             S.REFRAME_RIGHT: self.reframe_zoom, S.LOWER_SUBJECT: self.lower_zoom, S.RAISE_SUBJECT: self.raise_zoom}.get(state, BASE_ZOOM)
        return round(min(z, self.max_zoom), 4)


# --------------------------------------------------------------------------
# Phrase boundaries
# --------------------------------------------------------------------------

_SENTENCE = ".!?؟…"
_CLAUSE = ",،؛;:"
_LEAD_S = 0.12  # a move starts this long before the first word of the new phrase (never inside the previous word)


class Boundary(BaseModel):
    t: float  # where a state change may start (always on a word edge or in a gap, never inside a word)
    kind: str  # segment_edge | sentence_end | pause | clause | discourse_marker | word_gap
    quality: float
    after: str = ""  # the first words of the phrase that starts here (verbatim)

    @property
    def safe(self) -> bool:
        return self.quality >= SAFE_QUALITY


def _markers() -> set[str]:
    out = set(_INTERROGATIVES)
    for table in _PHRASE_CUES.values():
        out.update(table)
    return out


_MARKERS = _markers()


def _marker_at(words: list[Word], i: int) -> str | None:
    """A discourse marker (contrast, list step, topic shift, payoff, question) that opens the phrase at words[i]."""
    for n in (3, 2, 1):
        chunk = " ".join(w.word for w in words[i:i + n])
        if len(words[i:i + n]) == n and _fold(chunk) in _MARKERS:
            return chunk
    return None


def _words(transcript: Transcript) -> list[tuple[int, Word]]:
    out: list[tuple[int, Word]] = []
    for si, seg in enumerate(transcript.segments):
        out.extend((si, w) for w in sorted(seg.words, key=lambda w: (w.start, w.end)) if w.word.strip())
    return sorted(out, key=lambda p: (p[1].start, p[1].end))


def inside_word(t: float, transcript: Transcript, eps: float = 1e-6) -> bool:
    """True when `t` falls strictly inside a spoken word."""
    return any(w.start + eps < t < w.end - eps for _, w in _words(transcript))


def find_boundaries(transcript: Transcript) -> list[Boundary]:
    """Every place a state change may start, best evidence first per gap: segment edge / jump cut (1.0),
    sentence end (.95), pause (.85 >= .5s, .7 >= .3s, .5 >= .15s), clause punctuation (.65), a discourse marker
    opening the next phrase (.6, +.1 on top of another cue), and a bare word gap (.3, not safe). Deterministic."""
    pairs = _words(transcript)
    words = [w for _, w in pairs]
    found: list[Boundary] = []
    for i in range(1, len(pairs)):
        (sa, a), (sb, b) = pairs[i - 1], pairs[i]
        gap = b.start - a.end
        text = a.word.strip()
        quality, kind = 0.3, "word_gap"
        if sa != sb:
            quality, kind = 1.0, "segment_edge"
        elif text and text[-1] in _SENTENCE:
            quality, kind = 0.95, "sentence_end"
        elif gap >= 0.5:
            quality, kind = 0.85, "pause"
        elif gap >= 0.3:
            quality, kind = 0.7, "pause"
        elif text and text[-1] in _CLAUSE:
            quality, kind = 0.65, "clause"
        elif gap >= 0.15:
            quality, kind = 0.5, "pause"
        marker = _marker_at(words, i)
        if marker is not None:
            if quality >= SAFE_QUALITY:
                quality = min(1.0, quality + 0.1)
            else:
                quality, kind = 0.6, "discourse_marker"
        t = min(max(b.start - _LEAD_S, a.end), b.start)
        if any(x.start + 1e-6 < t < x.end - 1e-6 for x in words[max(0, i - 3):i + 3]):  # overlapping/garbled timing
            continue
        found.append(Boundary(t=round(t, 3), kind=kind, quality=round(quality, 3),
                              after=" ".join(w.word.strip() for w in words[i:i + 3])))
    found.sort(key=lambda x: (x.t, -x.quality))
    dedup: list[Boundary] = []
    for bd in found:
        if dedup and abs(dedup[-1].t - bd.t) < 1e-6:
            continue
        dedup.append(bd)
    return dedup


# --------------------------------------------------------------------------
# Inputs the semantic director / review layer may hand in
# --------------------------------------------------------------------------


class SemanticHint(BaseModel):
    """What the semantic detector believes about a stretch of speech (optional; the engine works without it)."""

    start: float
    end: float
    kind: str
    confidence: float


class Occupied(BaseModel):
    """A window already owned by the director or by the user (a pinned punch, a replacement scene, the hook).
    Rhythm plans around it and never touches its framing."""

    start: float
    end: float
    label: str
    pinned: bool = False
    changes: tuple[float, ...] = ()  # when the picture visibly changes inside it (default: its start and end)


FaceBox = tuple[float, float, float, float]  # normalised x, y, w, h (the Reframe convention)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


class RhythmRow(BaseModel):
    """One planned visual state (a stretch with one framing intent) plus what the review needs to see."""

    number: int = 0
    start: float
    end: float
    state: str
    source: str = "rhythm"  # rhythm | director | pinned
    zoom_from: float = BASE_ZOOM
    zoom_to: float = BASE_ZOOM
    anchor_x: float = BASE_ANCHOR_X
    anchor_y: float | None = None  # None = the renderer default; 0/1 only for lower/raise
    executable: bool = True  # mirrors `status == executable` once the camera timeline has classified it
    # the motion itself, so the timeline can store start / settle and both states
    motion_start: float | None = None
    motion_end: float | None = None
    motion_class: str = "none"  # abrupt | smooth | none
    anchor_x_from: float | None = None
    anchor_y_from: float | None = None
    status: str = "executable"  # planning_only | executable | blocked | superseded (see direction.camera.EventStatus)
    status_reason: str = ""
    safe_headroom: float | None = None
    headroom_gain: float | None = None
    face_bottom: float | None = None
    superseded_legacy: list[str] = Field(default_factory=list)
    semantic_source: str = "auto"  # auto | user_pinned
    behind_subject_candidate: str = ""
    boundary_kind: str = ""
    boundary_quality: float | None = None
    boundary_after: str = ""
    history_reason: str = ""
    composition_reason: str = ""
    reset_plan: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    # filled by the review layer
    transcript_context: str = ""
    semantic_enhancement: str = "none"
    semantic_confidence: float | None = None
    semantic_kind: str | None = None
    semantic_options: list[str] = Field(default_factory=list)  # what the semantic evidence would support (never applied here)
    primary_layer: str = "speaker"
    caption_role: str = "normal"
    speaker_visibility: str = "full"
    behind_subject: str = "not_applicable"  # eligible | not_eligible | not_applicable
    behind_subject_reason: str = ""
    sound_intent: str = "none"
    sound_status: str = "none"
    approval_status: str = "proposed"  # proposed | approved (the user's own decision) | n/a

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def moving(self) -> bool:
        return self.state in {s.value for s in _MOTION}


class RhythmDecision(BaseModel):
    """One choice the engine made, with every candidate it weighed and what it ruled out."""

    t: float
    state: str
    score: float
    breakdown: dict[str, float]
    boundary_kind: str
    hold_before: float
    excluded: dict[str, str] = Field(default_factory=dict)
    runner_up: str | None = None


class RhythmPlan(BaseModel):
    rows: list[RhythmRow] = Field(default_factory=list)
    decisions: list[RhythmDecision] = Field(default_factory=list)
    policy: RhythmPolicy = Field(default_factory=RhythmPolicy)
    start: float = 0.0
    end: float = 0.0
    changes: list[float] = Field(default_factory=list)  # every time the picture visibly changes (rhythm + director)
    notes: list[str] = Field(default_factory=list)

    def excursions(self) -> list[str]:
        return [r.state for r in self.rows if r.source == "rhythm" and r.state in {s.value for s in EXCURSIONS}]

    def longest_unchanged_hold(self) -> tuple[float, float, float]:
        """(seconds, from, to) of the longest stretch with no visible change. Reported, never a target."""
        marks = sorted({round(c, 3) for c in self.changes})
        edges = [self.start, *[m for m in marks if self.start < m < self.end], self.end]
        best = max(((b - a, a, b) for a, b in pairwise(edges)), default=(0.0, self.start, self.end))
        return round(best[0], 3), round(best[1], 3), round(best[2], 3)

    def to_camera_plan(self, base: CameraPolicy | None = None, *, face_box: FaceBox | None = None) -> CameraPlan:
        """The executable part of this plan through the camera model. raise_subject (and a lower_subject with no measured
        face) stay in `rows` as planning-only; every other move becomes an event with both states and its motion span."""
        policy = base or CameraPolicy()
        plan = CameraPlan(policy=policy)
        for r in self.rows:
            if r.source != "rhythm" or not r.moving:
                continue
            if r.state == S.RAISE_SUBJECT.value or (r.state == S.LOWER_SUBJECT.value and face_box is None):
                continue
            plan.events.append(row_event(r, face_box))
        return plan


_BACK = {S.PUNCH_OUT.value, S.SLOW_PULL.value, S.RESET_TO_BASE.value}


def row_event(r: RhythmRow, face_box: FaceBox | None = None) -> CameraEvent:
    """One moving rhythm row as a camera event (start = motion start, end = settle)."""
    move = CameraMove.RESET_TO_BASE if r.state == S.PUNCH_OUT.value else CameraMove(r.state)
    back = r.state in _BACK
    ay = DEFAULT_ANCHOR_Y if r.anchor_y is None else r.anchor_y
    ev = CameraEvent(
        start=r.motion_start if r.motion_start is not None else r.start, end=r.motion_end if r.motion_end is not None else r.end,
        move=move, zoom_to=r.zoom_to, zoom_from=r.zoom_from, anchor_x=BASE_ANCHOR_X if back else r.anchor_x,  # a return targets the canonical base exactly
        anchor_x_from=r.anchor_x_from if r.anchor_x_from is not None else r.anchor_x,
        anchor_y=DEFAULT_ANCHOR_Y if back else ay, anchor_y_from=r.anchor_y_from if r.anchor_y_from is not None else ay,
        motion_class=MotionClass(r.motion_class if r.motion_class in ("abrupt", "smooth") else "smooth" if back else "abrupt"),
        boundary_kind=r.boundary_kind, boundary_quality=r.boundary_quality, number=r.number,
        status=EventStatus.EXECUTABLE if r.executable else EventStatus.PLANNING_ONLY, status_reason=r.status_reason,
    )
    if face_box is not None and not back:
        top, bottom = face_frame(face_box, ev.zoom_to, ev.anchor_x, ev.anchor_y)[1::2]
        ev.safe_headroom, ev.headroom_gain, ev.face_bottom = round(top, 4), round(top - face_box[1], 4), round(bottom, 4)
    return ev


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def timing_score(hold: float, policy: RhythmPolicy) -> float:
    """How well a change after `hold` seconds fits the soft refresh window. Never zero past the minimum: a long
    hold with no better moment is a valid outcome, so the score only falls to a floor."""
    lo, hi, guard = policy.refresh_min_s, policy.refresh_max_s, policy.attention_guard_s
    if hold < policy.min_hold_s:
        return 0.0
    if hold < lo:
        return 0.55 + 0.45 * (hold - policy.min_hold_s) / max(lo - policy.min_hold_s, 1e-6)
    if hold <= hi:
        return 1.0
    if hold <= guard:
        return 1.0 - 0.4 * (hold - hi) / max(guard - hi, 1e-6)
    return max(0.1, 0.6 - 0.15 * (hold - guard))  # keeps falling (a soft pull, not a timer) so a nearer phrase boundary beats a long wait


class _Memory:
    """The last few excursions and the gaps between them (anti-pattern memory)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.states: list[S] = []
        self.gaps: list[float] = []

    def push(self, state: S, gap: float) -> None:
        keep = max(self.size, 6)  # the last `size` drive the penalties; a few more are kept to spot a repeating cycle
        self.states = (self.states + [state])[-keep:]
        self.gaps = (self.gaps + [gap])[-keep:]

    def excluded(self, state: S) -> str | None:
        if self.states and self.states[-1] is state:
            return f"repeats the last state ({state.value} -> {state.value})"
        return None

    def novelty(self, state: S, gap: float) -> tuple[float, str]:
        pen, why = 0.0, []
        recent = self.states[-self.size:]
        if state in recent:
            pen += 0.5
            why.append(f"{state.value} is among the last {len(recent)}")
        elif state in self.states:
            pen += 0.25
            why.append(f"{state.value} would start a repeating cycle")
        fam = _FAMILY[state]
        if self.states and _FAMILY[self.states[-1]] == fam:
            pen += 0.35
            why.append(f"same {fam} family as the last move")
        if len(self.states) >= 3 and self.states[-2] is state and self.states[-3] is self.states[-1]:
            pen += 0.3
            why.append("would close an ABAB loop")
        fams = [_FAMILY[x] for x in self.states]
        if len(fams) >= 3 and fams[-1] != fam and fams[-2] == fam and fams[-3] == fams[-1]:
            pen += 0.3
            why.append("would keep a predictable family alternation going")
        if len(self.gaps) >= 2 and abs(gap - self.gaps[-1]) < 0.35 and abs(self.gaps[-1] - self.gaps[-2]) < 0.35:
            pen += 0.3
            why.append("would repeat the same cadence a third time")
        names = [x.value for x in recent]
        return max(0.0, 1.0 - pen), (f"last {len(names)}: {names}; " if names else "no earlier moves; ") + ("; ".join(why) or "novel")


def _face_checks(state: S, policy: RhythmPolicy, face: FaceBox | None) -> tuple[bool, float, str]:
    """(allowed, composition score, reason) for `state` against the measured face box, if there is one."""
    z = policy.level(state)
    if state in (S.SLOW_PUSH, S.PUNCH_IN, S.REFRAME_LEFT, S.REFRAME_RIGHT):
        return True, 0.85 if state is S.PUNCH_IN else 0.9, "centre-anchored crop; the renderer's face-safe clamp applies"
    if state is S.LOWER_SUBJECT:  # top-anchored crop: the frame keeps its top edge, only the bottom is trimmed
        if face is None:
            return True, 0.75, (f"top-anchored crop at {z:.2f}x trims only the bottom edge so the face stays in frame; "
                               "face box not measured (top space unmeasured, renderer clamp applies)")
        y, h = face[1], face[3]
        top, bottom = y * z, (y + h) * z
        if bottom > policy.caption_safe_top:
            return False, 0.0, f"the lowered face would reach {bottom:.2f} of the frame, inside the caption band (>{policy.caption_safe_top:.2f})"
        if top - y < policy.min_headroom_gain:
            return False, 0.0, f"top space would grow only {top - y:.3f} of the frame (< {policy.min_headroom_gain:.2f}): not a usable lower_subject"
        return True, 1.0, f"top space {y:.3f} -> {top:.3f} of the frame (+{top - y:.3f}); face bottom {bottom:.2f} stays above the caption band"
    if face is None:  # raise_subject
        return False, 0.0, "composition not measured: raising the subject needs a measured face box"
    y, h = face[1], face[3]
    top = y * z - (z - 1.0)
    if top < policy.min_top_margin:
        return False, 0.0, f"the raised face would leave only {top:.3f} of headroom"
    return True, 1.0, f"headroom {y:.3f} -> {top:.3f}; face stays in frame"


def _align(state: S, t: float, hints: list[SemanticHint]) -> tuple[float, str]:
    """A small nudge from a CONFIDENT nearby semantic reading. Without one every candidate scores the same 0.5."""
    for h in hints:
        if h.confidence >= LOW_CONFIDENCE and abs(t - h.start) <= 0.4:
            lean = _LEANS.get(h.kind, ())
            if not lean:
                return 0.5, ""
            return (0.5 + 0.5 * h.confidence, f"{h.kind} ({h.confidence:.2f}) leans to {state.value}") if state in lean else (0.4, "")
    return 0.5, ""


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


def _free_windows(start: float, end: float, occupied: list[Occupied], margin: float) -> list[tuple[float, float, float]]:
    """(window_start, window_end, last_visible_change) for each stretch the engine may plan in. The window starts
    where the owned stretch ends; the hold is counted from the last time its picture actually changed."""
    wins: list[tuple[float, float, float]] = []
    cursor, last = start, start
    for o in sorted(occupied, key=lambda o: (o.start, o.end)):
        if o.end <= start or o.start >= end:
            continue
        if o.start - margin > cursor:
            wins.append((cursor, o.start - margin, last))
        cursor = max(cursor, o.end)
        last = max(last, *(o.changes or (o.end,)))
    if end > cursor:
        wins.append((cursor, end, last))
    return wins


def _return_start(settle: float, bounds: list[Boundary], policy: RhythmPolicy, floor: float) -> float | None:
    """Where a smooth return starts so that it glides for return_min..return_max seconds and settles at `settle`. It
    starts on a word boundary (never inside a word), not before `floor`; None when speech offers no such boundary."""
    pool = [b for b in bounds if b.t >= floor - 1e-9 and policy.return_min_s - 1e-9 <= settle - b.t <= policy.return_max_s + 1e-9]
    if not pool:
        return None
    return min(pool, key=lambda b: (round(abs(settle - b.t - policy.return_glide_s), 1), -b.quality, -b.t)).t


def _release(t: float, bounds: list[Boundary], policy: RhythmPolicy, limit: float, *, smooth: bool = False) -> tuple[Boundary, float] | None:
    """(settle boundary, return start) for the excursion started at `t`.

    A smooth excursion settles between the minimum and maximum excursion length on the boundary nearest the middle of
    that range (a phrase boundary when one fits, else any word boundary) and its return glides in beforehand. An abrupt
    one snaps back ON a phrase boundary and there is no fallback to a word gap."""
    lo, hi = t + policy.excursion_min_s, min(t + policy.excursion_max_s, limit)
    if hi < lo:
        return None
    mid = (policy.excursion_min_s + policy.excursion_max_s) / 2
    pool = [b for b in bounds if lo <= b.t <= hi]
    ranked = sorted((b for b in pool if b.safe or smooth),
                    key=lambda b: (b.safe, 0.6 * b.quality + 0.4 * (1 - abs((b.t - t) - mid) / mid), -b.t), reverse=True)
    for b in ranked:
        if not smooth:
            return b, b.t
        start = _return_start(b.t, bounds, policy, t + policy.smooth_entry_s + 0.2)
        if start is not None:
            return b, start
    return None


def _inside_emphasis(t: float, hints: list[SemanticHint]) -> bool:
    """True when `t` falls inside a stretch the semantic detector is confident about (a smooth move must not interrupt it)."""
    return any(h.confidence >= LOW_CONFIDENCE and h.start < t < h.end for h in hints)


def _extension_pool(bounds: list[Boundary], lo: float, latest: float, cursor: float, policy: RhythmPolicy,
                    hints: list[SemanticHint]) -> list[Boundary]:
    """Word boundaries where an extended hold may end with a smooth move: no phrase boundary is in reach, the hold has
    run past the refresh window, and no confident emphasis is in progress. Called only when the hold is extending."""
    after = policy.extend_after_s if policy.extend_after_s is not None else policy.refresh_max_s
    hi = min(cursor + policy.attention_guard_s, latest)
    return [b for b in bounds if not b.safe and b.quality >= policy.smooth_min_quality and max(lo, cursor + after) <= b.t <= hi
            and not _inside_emphasis(b.t, hints)]


def _rows_for(state: S, t: float, rel: Boundary, ret_start: float, policy: RhythmPolicy, face: FaceBox | None) -> list[RhythmRow]:
    z = policy.level(state)
    smooth = state in SMOOTH_STATES
    cls = (MotionClass.SMOOTH if smooth else MotionClass.ABRUPT).value
    ax, ay = BASE_ANCHOR_X, None
    if state is S.REFRAME_LEFT:
        ax = round(BASE_ANCHOR_X - policy.reframe_shift, 3)
    elif state is S.REFRAME_RIGHT:
        ax = round(BASE_ANCHOR_X + policy.reframe_shift, 3)
    elif state is S.LOWER_SUBJECT:
        ay = 0.0
    elif state is S.RAISE_SUBJECT:
        ay = 1.0
    back = S.SLOW_PULL if state is S.SLOW_PUSH else S.RESET_TO_BASE
    settle = rel.t if smooth else round(rel.t + policy.ease_s, 3)
    rows = []
    if state is S.SLOW_PUSH:  # base -> slow_push -> slow_pull -> base: the push drifts until the pull begins
        rows.append(RhythmRow(start=t, end=ret_start, state=state.value, zoom_from=BASE_ZOOM, zoom_to=z, anchor_x=ax, motion_start=t,
                              motion_end=ret_start, motion_class=cls, anchor_x_from=BASE_ANCHOR_X))
    else:  # base -> move -> hold -> return
        e = round(t + (policy.smooth_entry_s if smooth else policy.ease_s), 3)
        rows.append(RhythmRow(start=t, end=e, state=state.value, zoom_from=BASE_ZOOM, zoom_to=z, anchor_x=ax, anchor_y=ay, motion_start=t,
                              motion_end=e, motion_class=cls, anchor_x_from=BASE_ANCHOR_X))
        rows.append(RhythmRow(start=e, end=ret_start, state=S.HOLD.value, zoom_from=z, zoom_to=z, anchor_x=ax, anchor_y=ay))
    rows.append(RhythmRow(start=ret_start, end=settle, state=back.value, zoom_from=z, zoom_to=BASE_ZOOM, anchor_x=BASE_ANCHOR_X, anchor_y=ay,
                          motion_start=ret_start, motion_end=settle, motion_class=cls, anchor_x_from=ax, anchor_y_from=ay))
    if state is S.RAISE_SUBJECT or (state is S.LOWER_SUBJECT and face is None):
        why = ("raise_subject is planning-only: the renderer cannot pad above the source frame" if state is S.RAISE_SUBJECT
               else "lower_subject needs a measured face box before it can be proven safe")
        for r in rows:
            r.executable, r.status, r.status_reason = False, EventStatus.PLANNING_ONLY.value, why
    return rows


def plan_rhythm(
    transcript: Transcript,
    *,
    start: float = 0.0,
    end: float | None = None,
    policy: RhythmPolicy | None = None,
    occupied: list[Occupied] | None = None,
    hints: list[SemanticHint] | None = None,
    face_box: FaceBox | None = None,
    boundaries: list[Boundary] | None = None,
) -> RhythmPlan:
    """Plans the visual rhythm over [start, end]. Deterministic: the same inputs give the same plan. `hints` and
    `face_box` are optional; without them the plan is the same low-risk variation, just with neutral semantic terms."""
    policy = policy or RhythmPolicy()
    occupied = occupied or []
    hints = hints or []
    end = end if end is not None else max([transcript.duration, *[s.end for s in transcript.segments]])
    bounds = boundaries if boundaries is not None else find_boundaries(transcript)
    plan = RhythmPlan(policy=policy, start=start, end=end)
    memory = _Memory(policy.history)
    target = _ENERGY_TARGET.get(policy.energy, 0.5)
    tiles: list[RhythmRow] = []
    seen: set[int] = set()
    last_excursion_start: float | None = None
    if not bounds:
        plan.notes.append("no word timing in the transcript: no phrase boundary exists, so nothing was planned (the framing stays on base)")

    for w0, w1, last in _free_windows(start, end, occupied, policy.occupied_margin_s):
        for o in sorted(occupied, key=lambda o: (o.start, o.end)):  # what the director / you already put on screen counts as history
            if o.end <= w0 and id(o) not in seen:
                seen.add(id(o))
                if o.label in _EXCURSION_NAMES:
                    memory.push(S(o.label), 0.0)
        cursor = last  # the last visible change: holds are measured from here
        tile_from = max(w0, last)
        while bounds:
            calm = policy.after_emphasis_hold_s if memory.states and memory.states[-1] is S.PUNCH_IN and last_excursion_start is not None \
                and cursor - last_excursion_start < 6.0 else policy.min_hold_s
            lo = max(cursor + calm, w0)
            latest = w1 - policy.excursion_min_s - policy.slow_ease_s
            safe = [b for b in bounds if b.safe and lo <= b.t <= latest]
            extended = not any(b.t - cursor <= policy.attention_guard_s for b in safe)  # no phrase boundary within reach
            ext = _extension_pool(bounds, lo, latest, cursor, policy, hints) if extended else []
            near = [b for b in safe if b.t - cursor <= policy.horizon_s]
            pool = ext or near or safe[:1]  # nothing in reach: the hold extends to the next phrase boundary that exists
            states = EXTENSION_STATES if ext else tuple(x for x in EXCURSIONS if x is not S.RAISE_SUBJECT or policy.plan_raise_subject)
            if not pool:
                break
            best: tuple[float, S, Boundary, Boundary, float, dict[str, float], dict[str, str]] | None = None
            second: tuple[float, S] | None = None
            excluded: dict[str, str] = {}
            if not ext and not policy.plan_raise_subject:
                excluded["raise_subject"] = "planning-only: the renderer cannot pad above the source frame"
            for b in pool:
                hold = b.t - cursor
                gap = b.t - last_excursion_start if last_excursion_start is not None else hold
                for state in states:
                    is_smooth = state in SMOOTH_STATES
                    rel = _release(b.t, bounds, policy, w1 if is_smooth else w1 - policy.ease_s, smooth=is_smooth)
                    banned = memory.excluded(state)
                    ok, comp, comp_why = _face_checks(state, policy, face_box)
                    if rel is None:
                        excluded.setdefault(f"{state.value}@{b.t:.2f}", "no boundary to settle on within the excursion length")
                        continue
                    if banned or not ok:
                        excluded.setdefault(f"{state.value}@{b.t:.2f}", banned or comp_why)
                        continue
                    settle, ret_start = rel
                    nov, nov_why = memory.novelty(state, gap)
                    sem, sem_why = _align(state, b.t, hints)
                    terms = {
                        "boundary": b.quality, "timing": timing_score(hold, policy), "novelty": nov, "composition": comp,
                        "reset": settle.quality, "semantic": sem, "energy": 1.0 - abs(target - _ENERGY[state]),
                        "caption": 1.0 - _CAPTION_RISK[state],
                    }
                    total = sum(WEIGHTS[k] * v for k, v in terms.items())
                    key = (round(total, 6), -b.t, -EXCURSIONS.index(state))  # deterministic tie-break: earlier, then fixed order
                    if best is None or key > (round(best[0], 6), -best[2].t, -EXCURSIONS.index(best[1])):
                        if best is not None:
                            second = (best[0], best[1])
                        best = (total, state, b, settle, ret_start, terms, {"nov": nov_why, "comp": comp_why, "sem": sem_why})
                    elif second is None or total > second[0]:
                        second = (total, state)
            if best is None:
                break
            total, state, b, rel, ret_start, terms, why = best
            excursion = _rows_for(state, b.t, rel, ret_start, policy, face_box)
            if b.t - tile_from > 1e-6:
                tiles.append(RhythmRow(start=round(tile_from, 3), end=b.t, state=S.BASE.value))
            lead = excursion[0]
            lead.boundary_kind, lead.boundary_quality, lead.boundary_after = b.kind, b.quality, b.after
            lead.history_reason = why["nov"] + (f"; {why['sem']}" if why["sem"] else "")
            if ext:
                lead.history_reason += (f"; extended hold: no phrase boundary within {policy.attention_guard_s:g}s, so a smooth {state.value} "
                                        f"starts on a word boundary (quality {b.quality:.2f}), never mid-word")
            lead.composition_reason = why["comp"]
            lead.scores = {k: round(v, 3) for k, v in terms.items()} | {"total": round(total, 3)}
            tail = excursion[-1]
            smooth_tail = tail.motion_class == MotionClass.SMOOTH.value
            how = (f"glides {tail.start:.2f}-{tail.end:.2f}s and settles on base" if smooth_tail
                   else f"snaps back at {tail.start:.2f}s, on base by {tail.end:.2f}s")
            lead.reset_plan = tail.reset_plan = f"{tail.state} {how} at a {rel.kind} boundary (quality {rel.quality:.2f})"
            tiles.extend(excursion)
            plan.decisions.append(RhythmDecision(
                t=b.t, state=state.value, score=round(total, 3), breakdown=lead.scores, boundary_kind=b.kind,
                hold_before=round(b.t - cursor, 3), excluded=dict(sorted(excluded.items())[:6]), runner_up=second[1].value if second else None))
            if state is S.SLOW_PUSH:
                plan.changes.extend([b.t, tail.start, tail.end])
            elif smooth_tail:
                plan.changes.extend([b.t, lead.end, tail.start, tail.end])
            else:
                plan.changes.extend([b.t, tail.start])
            memory.push(state, b.t - last_excursion_start if last_excursion_start is not None else b.t - cursor)
            last_excursion_start = b.t
            cursor = tile_from = tail.end
        if w1 - tile_from > 1e-6:
            tiles.append(RhythmRow(start=round(tile_from, 3), end=round(w1, 3), state=S.BASE.value))
    _finish(plan, tiles, occupied, start, end)
    return plan


def _finish(plan: RhythmPlan, tiles: list[RhythmRow], occupied: list[Occupied], start: float, end: float) -> None:
    """Interleaves the director-owned windows, tiles the timeline, and records every visible change."""
    rows = list(tiles)
    for o in occupied:
        if o.end <= start or o.start >= end:
            continue
        rows.append(RhythmRow(start=o.start, end=o.end, state=o.label, source="pinned" if o.pinned else "director", approval_status="n/a"))
        plan.changes.extend(o.changes or (o.start, o.end))
    rows.sort(key=lambda r: (r.start, r.end))
    merged: list[RhythmRow] = []
    for r in rows:  # a base stretch split by a free window edge is one base stretch
        if merged and merged[-1].state == r.state == S.BASE.value and abs(merged[-1].end - r.start) < 1e-6:
            merged[-1].end = r.end
        else:
            merged.append(r)
    number, grouped = 0, False
    for r in merged:  # one numbered entry per planned state change: an excursion is its move, hold and return together
        if not grouped:
            number += 1
        r.number = number
        if r.source == "rhythm" and r.state in _EXCURSION_NAMES:
            grouped = True
        elif grouped and r.state in (S.RESET_TO_BASE.value, S.SLOW_PULL.value):
            grouped = False
        if r.source != "rhythm":
            r.history_reason = "owned by the semantic director / your earlier decision: rhythm plans around it"
    plan.rows = merged
    plan.changes = sorted({round(c, 3) for c in plan.changes})


def entries(rows: list[RhythmRow]) -> list[list[RhythmRow]]:
    """The rows grouped as the review shows them: one entry per planned state change."""
    out: list[list[RhythmRow]] = []
    for r in rows:
        if out and out[-1][0].number == r.number:
            out[-1].append(r)
        else:
            out.append([r])
    return out


# --------------------------------------------------------------------------
# Semantic side: options only ever come from semantic evidence
# --------------------------------------------------------------------------

SEMANTIC_ENHANCEMENTS = (
    "primary_headline_typography", "icon_overlay", "small_motion_graphic", "diagram_process_scene", "kinetic_typography",
    "speaker_replacement", "behind_subject_text", "full_screen_graphic", "real_broll", "user_broll", "generated_visual",
)
_TEXT_KINDS = {"key_claim", "payoff", "contrast", "question", "topic_shift"}
_PROCESS_KINDS = {"process_list"}


def semantic_enhancement_options(kind: str | None, confidence: float | None, *, phrase: str | None = None) -> list[str]:
    """The semantic treatments a beat's evidence supports. Weak or absent evidence gives NOTHING: enhancements are
    never invented, and elapsed time is not an input here. `generated_visual` is only ever offered as an
    approval-required alternative for a confident, concrete beat, never as a default."""
    if kind is None or confidence is None or confidence < LOW_CONFIDENCE:
        return []
    opts: list[str] = []
    if kind in _TEXT_KINDS and phrase and 0 < len(phrase.split()) <= MAX_PHRASE_WORDS:
        opts.append("primary_headline_typography")
    if kind in _PROCESS_KINDS:
        opts.extend(["diagram_process_scene", "small_motion_graphic"])
    if kind in ("key_claim", "payoff") and confidence >= 0.7:
        opts.append("kinetic_typography")
    return opts


def headline_allowed(state: str, kind: str | None, confidence: float | None, phrase: str | None) -> bool:
    """A headline needs semantic evidence AND a short verbatim phrase. lower_subject only makes room for one; it
    never makes one appear."""
    return "primary_headline_typography" in semantic_enhancement_options(kind, confidence, phrase=phrase) and state != S.HOLD.value


__all__ = [
    "ABRUPT_STATES",
    "EXCURSIONS",
    "EXTENSION_STATES",
    "MAX_PHRASE_WORDS",
    "SAFE_QUALITY",
    "SEMANTIC_ENHANCEMENTS",
    "SMOOTH_STATES",
    "VOCABULARY",
    "WEIGHTS",
    "Boundary",
    "Occupied",
    "RhythmDecision",
    "RhythmPlan",
    "RhythmPolicy",
    "RhythmRow",
    "RhythmState",
    "SemanticHint",
    "entries",
    "find_boundaries",
    "headline_allowed",
    "inside_word",
    "motion_class",
    "plan_rhythm",
    "row_event",
    "semantic_enhancement_options",
    "timing_score",
]
